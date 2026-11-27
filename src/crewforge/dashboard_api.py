"""Read-only trace, replay, topology, and benchmark access layer for CrewForge Mission Control."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crewforge.core.demo_trace import DemoTraceExport, _sanitize_recursive
from crewforge.schemas import (
    AgentMessage,
    AgentSpec,
    Claim,
    ClaimCategory,
    ClaimStatus,
    Evidence,
    EvidenceReliability,
    ExecutionEvent,
    ExecutionPlan,
    HonestyGateRecord,
    ReviewDecision,
    ReviewVerdict,
    RunResult,
    TaskSpec,
    TaskStatus,
    ToolExecutionRecord,
    UsageMetrics,
    VerificationSummary,
)

_RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9_-]{4,64}$")
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.json$")


class DashboardValidationError(ValueError):
    """Raised when an API caller supplies an invalid run_id or replay filename."""


class DashboardNotFoundError(FileNotFoundError):
    """Raised when a requested run or replay file does not exist."""


def validate_run_id(run_id: str) -> str:
    """Validate a run_id against a strict allowlist pattern."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise DashboardValidationError(f"Invalid run_id '{run_id}'.")
    return run_id


def validate_replay_filename(filename: str, replay_dir: Path) -> Path:
    """Validate a replay filename and ensure it resolves inside replay_dir."""
    if not isinstance(filename, str) or not _SAFE_FILENAME_RE.fullmatch(filename):
        raise DashboardValidationError(f"Invalid replay filename '{filename}'.")
    base = replay_dir.resolve()
    candidate = (base / filename).resolve()
    if candidate.parent != base:
        raise DashboardValidationError("Path traversal outside replay directory is forbidden.")
    return candidate


def _categorize_event(event_type: str) -> str:
    """Map persisted event types to Mission Control timeline filter categories."""
    if "plan" in event_type:
        return "planning"
    if "task" in event_type or "worker" in event_type:
        return "task_execution"
    if "message" in event_type:
        return "peer_messages"
    if "tool" in event_type:
        return "tool_calls"
    if "review" in event_type:
        return "review"
    if "revision" in event_type or "claim" in event_type:
        return "revision"
    if "synthesis" in event_type:
        return "synthesis"
    if "honesty_gate" in event_type:
        return "honesty_gate"
    if "budget" in event_type:
        return "budget"
    return "other"


def _agent_status_from_tasks(
    agent_id: str,
    role: str,
    tasks: list[dict[str, Any]],
    run_status: str,
) -> str:
    """Derive an agent's execution state from its assigned tasks and overall run status."""
    assigned = [t for t in tasks if t.get("assigned_agent_id") == agent_id]
    if not assigned:
        if role == "lead":
            return "completed" if run_status in ("completed", "partial") else run_status
        return "completed" if run_status == "completed" else "pending"
    statuses = {str(t.get("status", "pending")).lower() for t in assigned}
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses:
        return "blocked"
    if "running" in statuses or "in_progress" in statuses:
        return "running"
    if statuses <= {"completed"}:
        return "completed"
    return "pending"


def build_agent_topology(trace: dict[str, Any]) -> dict[str, Any]:
    """Construct dynamic agent topology nodes and edges from actual persisted trace data."""
    roster = trace.get("roster") or []
    tasks = trace.get("tasks") or []
    messages = trace.get("peer_messages") or []
    verdicts = trace.get("review_verdicts") or []
    gate = trace.get("honesty_gate")
    run_status = str(trace.get("status", "pending")).lower()

    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for agent in roster:
        aid = str(agent.get("agent_id", ""))
        if not aid or aid in seen_ids:
            continue
        seen_ids.add(aid)
        role = str(agent.get("role", "specialist")).lower()
        assigned_tasks = [t for t in tasks if t.get("assigned_agent_id") == aid]
        node_type = "lead" if role == "lead" else "specialist"
        nodes.append(
            {
                "id": aid,
                "label": agent.get("role") or aid,
                "sublabel": (
                    agent.get("specialty")
                    or agent.get("model_override")
                    or agent.get("model")
                    or ""
                ),
                "node_type": node_type,
                "status": _agent_status_from_tasks(aid, role, tasks, run_status),
                "model": agent.get("model_override") or agent.get("model") or trace.get("model"),
                "task_ids": [t.get("task_id") for t in assigned_tasks],
                "tasks": assigned_tasks,
            }
        )

    if verdicts:
        reviewer_ids = sorted({str(v.get("reviewer_id", "reviewer")) for v in verdicts})
        for rid in reviewer_ids:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            r_verdicts = [v for v in verdicts if str(v.get("reviewer_id", "reviewer")) == rid]
            nodes.append(
                {
                    "id": rid,
                    "label": rid,
                    "sublabel": f"{len(r_verdicts)} verdict(s)",
                    "node_type": "reviewer",
                    "status": "completed",
                    "model": trace.get("model"),
                    "task_ids": [],
                    "tasks": [],
                }
            )

    if gate is not None:
        gate_id = "honesty_gate"
        if gate_id not in seen_ids:
            seen_ids.add(gate_id)
            rmode = str(gate.get("release_mode", "unknown")).lower()
            gate_status = (
                "blocked"
                if (not gate.get("passed") or rmode == "blocked")
                else ("partial" if rmode == "partial" else "completed")
            )
            nodes.append(
                {
                    "id": gate_id,
                    "label": "Honesty Gate",
                    "sublabel": f"mode: {gate.get('release_mode', 'unknown')}",
                    "node_type": "gate",
                    "status": gate_status,
                    "model": None,
                    "task_ids": [],
                    "tasks": [],
                }
            )

    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    lead_id = next((n["id"] for n in nodes if n["node_type"] == "lead"), None)
    task_by_id = {str(t.get("task_id")): t for t in tasks if t.get("task_id")}
    for t in tasks:
        target_agent = str(t.get("assigned_agent_id", ""))
        deps = t.get("depends_on") or t.get("depends_on_task_ids") or []
        if not deps and lead_id and target_agent and lead_id != target_agent:
            key = (lead_id, target_agent, "task_dependency")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    {
                        "source": lead_id,
                        "target": target_agent,
                        "edge_type": "task_dependency",
                        "label": str(t.get("task_id", "")),
                    }
                )
        for dep_tid in deps:
            dep_task = task_by_id.get(str(dep_tid))
            src_agent = str(dep_task.get("assigned_agent_id", "")) if dep_task else ""
            if src_agent and target_agent and src_agent != target_agent:
                key = (src_agent, target_agent, "task_dependency")
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges.append(
                        {
                            "source": src_agent,
                            "target": target_agent,
                            "edge_type": "task_dependency",
                            "label": f"{dep_tid} -> {t.get('task_id')}",
                        }
                    )

    for msg in messages:
        src = str(msg.get("sender_id") or msg.get("sender_agent_id") or "")
        dst = str(msg.get("recipient_id") or msg.get("recipient_agent_id") or "")
        if src in seen_ids and dst in seen_ids and src != dst:
            key = (src, dst, "peer_message")
            if key not in edge_keys:
                edge_keys.add(key)
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "edge_type": "peer_message",
                        "label": str(msg.get("message_type", "message")),
                    }
                )

    return {"nodes": nodes, "edges": edges}


def normalize_trace_payload(
    export: DemoTraceExport,
    *,
    is_historical_replay: bool,
) -> dict[str, Any]:
    """Normalize a sanitized DemoTraceExport into the Mission Control JSON payload."""
    data = export.model_dump(mode="json")
    data["is_historical_replay"] = is_historical_replay
    data["topology"] = build_agent_topology(data)
    events = data.get("execution_events") or []
    for idx, ev in enumerate(events):
        if ev.get("seq") is None:
            ev["seq"] = idx + 1
        ev["category"] = _categorize_event(str(ev.get("event_type", "")))
    return data


def _row_to_claim(row: sqlite3.Row) -> Claim:
    keys = set(row.keys())
    return Claim(
        claim_id=row["claim_id"],
        run_id=row["run_id"],
        statement=row["statement"],
        category=ClaimCategory(row["category"]),
        status=ClaimStatus(row["status"]),
        verification_status=row["verification_status"],
        version=int(row["version"]),
        parent_claim_id=row["parent_claim_id"],
        author_agent_id=row["author_agent_id"],
        task_id=row["task_id"],
        evidence_ids=json.loads(row["evidence_ids_json"]),
        depends_on_claim_ids=json.loads(row["depends_on_claim_ids_json"]),
        premise_refs=json.loads(row["premise_refs_json"]) if "premise_refs_json" in keys else [],
        tool_execution_ids=json.loads(row["tool_execution_ids_json"]),
        requirement_refs=json.loads(row["requirement_refs_json"]),
        assumptions=json.loads(row["assumptions_json"]),
        revision_notes=row["revision_notes"],
        confidence=(float(row["confidence"]) if row["confidence"] is not None else None),
        is_material=bool(row["is_material"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_tool(row: sqlite3.Row) -> ToolExecutionRecord:
    keys = set(row.keys())
    return ToolExecutionRecord(
        tool_execution_id=row["tool_execution_id"],
        run_id=row["run_id"],
        tool_name=row["tool_name"],
        actor=row["actor"],
        task_id=row["task_id"],
        expression=row["expression"],
        variables=json.loads(row["variables_json"]),
        input_provenance_refs=(
            json.loads(row["input_provenance_refs_json"])
            if "input_provenance_refs_json" in keys
            else {}
        ),
        provenance_details=(
            json.loads(row["provenance_details_json"])
            if "provenance_details_json" in keys
            else {}
        ),
        proposed_result=row["proposed_result"],
        computed_result=row["computed_result"],
        units=row["units"],
        assumptions=json.loads(row["assumptions_json"]) if "assumptions_json" in keys else [],
        matches_proposed=(
            bool(row["matches_proposed"]) if row["matches_proposed"] is not None else None
        ),
        inputs_grounded=bool(row["inputs_grounded"]) if "inputs_grounded" in keys else False,
        has_ungrounded_literals=(
            bool(row["has_ungrounded_literals"]) if "has_ungrounded_literals" in keys else False
        ),
        formula_appropriate=(
            bool(row["formula_appropriate"])
            if "formula_appropriate" in keys and row["formula_appropriate"] is not None
            else None
        ),
        success=bool(row["success"]),
        error=row["error"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _aggregate_usage_ro(conn: sqlite3.Connection, run_id: str) -> UsageMetrics:
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(input_tokens), 0) AS input_tokens,
            COALESCE(SUM(output_tokens), 0) AS output_tokens,
            COALESCE(SUM(total_tokens), 0) AS total_tokens,
            COALESCE(SUM(estimated_cost_usd), 0.0) AS estimated_cost_usd,
            COALESCE(SUM(COALESCE(actual_cost_usd, 0.0)), 0.0) AS actual_cost_usd,
            COUNT(*) AS model_calls,
            COALESCE(SUM(
                CASE WHEN outcome IN ('completed', 'completed_unknown_usage')
                    THEN 1 ELSE 0 END
            ), 0) AS successful_operations,
            COALESCE(SUM(retries), 0) AS retries,
            COALESCE(SUM(latency_ms), 0.0) AS latency_ms,
            COALESCE(MIN(
                CASE WHEN outcome = 'completed_unknown_usage' THEN 0 ELSE 1 END
            ), 1) AS all_usage_reported
        FROM usage_records
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None or int(row["model_calls"]) == 0:
        return UsageMetrics(usage_reported=False)
    return UsageMetrics(
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        total_tokens=int(row["total_tokens"]),
        estimated_cost_usd=float(row["estimated_cost_usd"]),
        actual_cost_usd=float(row["actual_cost_usd"]),
        model_calls=int(row["model_calls"]),
        successful_operations=int(row["successful_operations"]),
        retries=int(row["retries"]),
        latency_ms=float(row["latency_ms"]),
        usage_reported=bool(row["all_usage_reported"]),
    )


class ReadOnlyTraceStore:
    """Genuine read-only SQLite trace store (mode=ro + PRAGMA query_only=ON, no WAL/writes)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect_ro(self) -> sqlite3.Connection | None:
        if not self.db_path.is_file():
            return None
        uri = f"file:{self.db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON;")
        return conn

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List runs in the configured demo database without exposing private prompts."""
        conn = self._connect_ro()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                """
                SELECT run_id, mode, status, provider, model, created_at, completed_at
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def _get_run_payload_sync(self, run_id: str) -> dict[str, Any]:
        validate_run_id(run_id)
        conn = self._connect_ro()
        if conn is None:
            raise DashboardNotFoundError(f"Demo database '{self.db_path}' does not exist.")
        try:
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise DashboardNotFoundError(f"Run '{run_id}' not found.")

            plan_row = conn.execute(
                "SELECT * FROM execution_plans WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            agent_rows = conn.execute(
                "SELECT * FROM agents WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            task_rows = conn.execute(
                "SELECT * FROM tasks WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            transition_rows = conn.execute(
                "SELECT * FROM task_transitions WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            evidence_rows = conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            claim_rows = conn.execute(
                "SELECT * FROM claims WHERE run_id = ? ORDER BY created_at ASC, version ASC",
                (run_id,),
            ).fetchall()
            message_rows = conn.execute(
                """
                SELECT m.*,
                    COALESCE((
                        SELECT d.status FROM message_deliveries d
                        WHERE d.message_id = m.message_id
                        ORDER BY CASE d.status
                            WHEN 'consumed' THEN 3
                            WHEN 'delivered' THEN 2
                            ELSE 1 END DESC
                        LIMIT 1
                    ), 'pending') AS agg_status
                FROM agent_messages m
                WHERE m.run_id = ?
                ORDER BY m.created_at ASC
                """,
                (run_id,),
            ).fetchall()
            verdict_rows = conn.execute(
                "SELECT * FROM review_verdicts WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            tool_rows = conn.execute(
                "SELECT * FROM tool_executions WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            gate_row = conn.execute(
                "SELECT * FROM honesty_gate_decisions WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            event_rows = conn.execute(
                "SELECT * FROM execution_events WHERE run_id = ? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()

            result = (
                RunResult.model_validate_json(run_row["result_json"])
                if run_row["result_json"]
                else None
            )
            usage_metrics = (
                result.usage if result and result.usage else _aggregate_usage_ro(conn, run_id)
            )
        except sqlite3.OperationalError as exc:
            raise DashboardNotFoundError(
                f"Demo database '{self.db_path}' is temporarily unavailable: {exc}"
            ) from exc
        finally:
            conn.close()

        agents = [
            AgentSpec(
                agent_id=row["agent_id"],
                agent_ref=row["agent_ref"],
                run_id=row["run_id"],
                role=row["role"],
                instructions=row["instructions"],
                capabilities=json.loads(row["capabilities_json"]),
                model_override=row["model_override"],
                created_by=row["created_by"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in agent_rows
        ]
        tasks = [
            TaskSpec(
                task_id=row["task_id"],
                task_ref=row["task_ref"],
                run_id=row["run_id"],
                plan_id=row["plan_id"],
                title=row["title"],
                description=row["description"],
                assigned_agent_id=row["assigned_agent_id"],
                depends_on=json.loads(row["depends_on_json"]),
                expected_outputs=json.loads(row["expected_outputs_json"]),
                acceptance_criteria=json.loads(row["acceptance_criteria_json"]),
                status=TaskStatus(row["status"]),
                findings=row["findings"],
                error=row["error"],
                created_by=row["created_by"],
                created_at=datetime.fromisoformat(row["created_at"]),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
            )
            for row in task_rows
        ]
        plan = (
            ExecutionPlan(
                plan_id=plan_row["plan_id"],
                run_id=plan_row["run_id"],
                goal=plan_row["goal"],
                user_requirements=json.loads(plan_row["user_requirements_json"]),
                acceptance_criteria=json.loads(plan_row["acceptance_criteria_json"]),
                agents=agents,
                tasks=tasks,
                assumptions=json.loads(plan_row["assumptions_json"]),
                missing_information=json.loads(plan_row["missing_information_json"]),
                created_by=plan_row["created_by"],
                created_at=datetime.fromisoformat(plan_row["created_at"]),
            )
            if plan_row is not None
            else None
        )
        evidence = [
            Evidence(
                evidence_id=row["evidence_id"],
                run_id=row["run_id"],
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                excerpt=row["excerpt"],
                reliability=EvidenceReliability(row["reliability"]),
                verified_in_input=bool(row["verified_in_input"]),
                task_id=row["task_id"],
                created_by=row["created_by"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in evidence_rows
        ]
        claims = [_row_to_claim(row) for row in claim_rows]
        messages = [
            AgentMessage(
                message_id=row["message_id"],
                run_id=row["run_id"],
                sender_id=row["sender_id"],
                recipient_id=row["recipient_id"],
                message_type=row["message_type"],
                content=row["content"],
                related_claim_ids=json.loads(row["related_claim_ids_json"]),
                delivery_status=row["agg_status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in message_rows
        ]
        verdicts = [
            ReviewVerdict(
                verdict_id=row["verdict_id"],
                run_id=row["run_id"],
                claim_id=row["claim_id"],
                claim_version=int(row["claim_version"]),
                reviewer_id=row["reviewer_id"],
                review_dimension=(
                    row["review_dimension"] if "review_dimension" in set(row.keys()) else "general"
                ),
                decision=ReviewDecision(row["decision"]),
                rationale=row["rationale"],
                evidence_refs=(
                    json.loads(row["evidence_refs_json"])
                    if "evidence_refs_json" in set(row.keys())
                    else []
                ),
                required_correction=(
                    row["required_correction"]
                    if "required_correction" in set(row.keys())
                    else None
                ),
                blocking_severity=(
                    row["blocking_severity"] if "blocking_severity" in set(row.keys()) else "none"
                ),
                blocking_issues=json.loads(row["blocking_issues_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in verdict_rows
        ]
        tools = [_row_to_tool(row) for row in tool_rows]
        honesty_gate = (
            HonestyGateRecord(
                gate_id=gate_row["gate_id"],
                run_id=gate_row["run_id"],
                passed=bool(gate_row["passed"]),
                release_mode=gate_row["release_mode"],
                rejected_segments=json.loads(gate_row["rejected_segments_json"]),
                rejected_claim_refs=json.loads(gate_row["rejected_claim_refs_json"]),
                reasons=json.loads(gate_row["reasons_json"]),
                verification_summary=VerificationSummary.model_validate_json(
                    gate_row["verification_summary_json"]
                ),
                created_at=datetime.fromisoformat(gate_row["created_at"]),
            )
            if gate_row is not None
            else None
        )
        events = [
            ExecutionEvent(
                event_id=row["event_id"],
                run_id=row["run_id"],
                seq=row["seq"],
                event_type=row["event_type"],
                actor=row["actor"],
                message=row["message"],
                payload=json.loads(row["payload_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in event_rows
        ]

        export = DemoTraceExport(
            is_historical_replay=False,
            run_id=run_row["run_id"],
            mode=run_row["mode"],
            status=run_row["status"],
            provider=run_row["provider"],
            model=run_row["model"],
            returned_model=result.returned_model if result else None,
            prompt=_sanitize_recursive(run_row["prompt"]),
            error=_sanitize_recursive(result.error if result else None),
            created_at=run_row["created_at"],
            completed_at=run_row["completed_at"],
            latency_ms=result.latency_ms if result else usage_metrics.latency_ms,
            usage=_sanitize_recursive(usage_metrics.model_dump()),
            roster=_sanitize_recursive([a.model_dump() for a in agents]),
            execution_plan=_sanitize_recursive(plan.model_dump() if plan else None),
            tasks=_sanitize_recursive([t.model_dump() for t in tasks]),
            task_transitions=_sanitize_recursive([dict(r) for r in transition_rows]),
            execution_events=_sanitize_recursive([e.model_dump() for e in events]),
            peer_messages=_sanitize_recursive([m.model_dump() for m in messages]),
            evidence=_sanitize_recursive([e.model_dump() for e in evidence]),
            claims=_sanitize_recursive([c.model_dump() for c in claims]),
            review_verdicts=_sanitize_recursive([v.model_dump() for v in verdicts]),
            tool_executions=_sanitize_recursive([t.model_dump() for t in tools]),
            honesty_gate=_sanitize_recursive(honesty_gate.model_dump() if honesty_gate else None),
            final_result=_sanitize_recursive(result.model_dump() if result else None),
        )
        return normalize_trace_payload(export, is_historical_replay=False)

    async def get_run_payload(
        self, run_id: str, *, since_seq: int | None = None
    ) -> dict[str, Any]:
        """Load and sanitize a single run trace from the read-only demo database."""
        payload = await asyncio.to_thread(self._get_run_payload_sync, run_id)
        events = payload.get("execution_events") or []
        latest_seq = max((int(e.get("seq", 0)) for e in events), default=0)
        payload["latest_seq"] = latest_seq
        if since_seq is not None and since_seq > 0:
            payload["execution_events"] = [
                e for e in events if int(e.get("seq", 0)) > since_seq
            ]
            payload["incremental"] = True
        else:
            payload["incremental"] = False
        return payload


class ReplayStore:
    """Validated reader for sanitized DemoTraceExport JSON artifacts."""

    def __init__(self, replay_dir: str | Path) -> None:
        self.replay_dir = Path(replay_dir)

    def list_files(self) -> list[dict[str, Any]]:
        """List valid DemoTraceExport files inside replay_dir."""
        if not self.replay_dir.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for p in sorted(self.replay_dir.glob("*.json")):
            if not _SAFE_FILENAME_RE.fullmatch(p.name):
                continue
            try:
                exp = DemoTraceExport.model_validate_json(p.read_text(encoding="utf-8"))
                items.append(
                    {
                        "filename": p.name,
                        "run_id": exp.run_id,
                        "status": exp.status,
                        "provider": exp.provider,
                        "model": exp.model,
                        "mode": exp.mode,
                        "created_at": exp.created_at,
                    }
                )
            except (OSError, ValidationError, json.JSONDecodeError):
                continue
        return items

    def load_replay(self, filename: str) -> dict[str, Any]:
        """Validate, sanitize, and load a DemoTraceExport JSON file."""
        target = validate_replay_filename(filename, self.replay_dir)
        if not target.is_file():
            raise DashboardNotFoundError(f"Replay file '{filename}' not found.")
        try:
            exp = DemoTraceExport.model_validate_json(target.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise DashboardValidationError(
                f"Malformed DemoTraceExport '{filename}': {exc}"
            ) from exc
        sanitized = DemoTraceExport.model_validate(_sanitize_recursive(exp.model_dump()))
        return normalize_trace_payload(sanitized, is_historical_replay=True)


class BenchmarkStore:
    """Read-only loader for reconciled benchmark reports (without private grading keys)."""

    def __init__(
        self,
        benchmark_path: str | Path,
        dvg_path: str | Path | None = None,
    ) -> None:
        self.benchmark_path = Path(benchmark_path)
        self.dvg_path = Path(dvg_path) if dvg_path else None

    def load(self) -> dict[str, Any]:
        """Load reconciled V2 benchmark summary and David vs Goliath status."""
        if not self.benchmark_path.is_file():
            raise DashboardNotFoundError(f"Benchmark report '{self.benchmark_path}' not found.")
        try:
            data = json.loads(self.benchmark_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DashboardValidationError(f"Malformed benchmark JSON: {exc}") from exc

        if self.dvg_path and self.dvg_path.is_file():
            with contextlib.suppress(json.JSONDecodeError):
                data["david_vs_goliath"] = json.loads(self.dvg_path.read_text(encoding="utf-8"))
        elif "david_vs_goliath" not in data:
            data["david_vs_goliath"] = {
                "status": "Not executed — local readiness blocker.",
                "detail": (
                    "Qwen3-4B failed planner validation twice by generating "
                    "four tasks against a maximum of three."
                ),
            }
        return data
