"""Read-only trace, replay, topology, and benchmark access layer for CrewForge Mission Control."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from crewforge.core.demo_trace import DemoTraceExport, _sanitize_recursive
from crewforge.storage.database import Database

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
                "sublabel": agent.get("specialty") or agent.get("model") or "",
                "node_type": node_type,
                "status": _agent_status_from_tasks(aid, role, tasks, run_status),
                "model": agent.get("model"),
                "task_ids": [t.get("task_id") for t in assigned_tasks],
                "tasks": assigned_tasks,
            }
        )

    # Add verification components only when persisted review/gate artifacts exist
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
            gate_status = "completed" if gate.get("passed") else "blocked"
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

    # Lead -> root tasks (no dependencies)
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

    # Peer message edges
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
        ev["seq"] = idx + 1
        ev["category"] = _categorize_event(str(ev.get("event_type", "")))
    return data


class ReadOnlyTraceStore:
    """Read-only SQLite trace store bound to an explicitly configured demo database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect_ro(self) -> sqlite3.Connection | None:
        if not self.db_path.is_file():
            return None
        uri = f"file:{self.db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
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

    async def get_run_payload(self, run_id: str) -> dict[str, Any]:
        """Load and sanitize a single run trace from the read-only demo database."""
        validate_run_id(run_id)
        if not self.db_path.is_file():
            raise DashboardNotFoundError(f"Demo database '{self.db_path}' does not exist.")
        db = Database(self.db_path)
        trace = await db.get_run_trace(run_id)
        if trace is None:
            raise DashboardNotFoundError(f"Run '{run_id}' not found.")

        usage_metrics = (
            trace.result.usage
            if trace.result and trace.result.usage
            else await db.aggregate_run_usage(run_id)
        )
        raw_plan = trace.plan.model_dump() if trace.plan else None
        raw_final = trace.result.model_dump() if trace.result else None

        export = DemoTraceExport(
            is_historical_replay=False,
            run_id=trace.run_id,
            mode=trace.mode,
            status=trace.status.value,
            provider=trace.provider,
            model=trace.model,
            returned_model=trace.result.returned_model if trace.result else None,
            prompt=_sanitize_recursive(trace.prompt),
            error=_sanitize_recursive(trace.result.error if trace.result else None),
            created_at=trace.created_at.isoformat(),
            completed_at=trace.completed_at.isoformat() if trace.completed_at else None,
            latency_ms=trace.result.latency_ms if trace.result else 0.0,
            usage=_sanitize_recursive(usage_metrics.model_dump()),
            roster=_sanitize_recursive([a.model_dump() for a in trace.agents]),
            execution_plan=_sanitize_recursive(raw_plan),
            tasks=_sanitize_recursive([t.model_dump() for t in trace.tasks]),
            task_transitions=_sanitize_recursive(trace.task_transitions),
            execution_events=_sanitize_recursive([e.model_dump() for e in trace.events]),
            peer_messages=_sanitize_recursive([m.model_dump() for m in trace.messages]),
            evidence=_sanitize_recursive([e.model_dump() for e in trace.evidence]),
            claims=_sanitize_recursive([c.model_dump() for c in trace.claims]),
            review_verdicts=_sanitize_recursive([v.model_dump() for v in trace.review_verdicts]),
            tool_executions=_sanitize_recursive([t.model_dump() for t in trace.tool_executions]),
            honesty_gate=_sanitize_recursive(
                trace.honesty_gate.model_dump() if trace.honesty_gate else None
            ),
            final_result=_sanitize_recursive(raw_final),
        )
        return normalize_trace_payload(export, is_historical_replay=False)


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
