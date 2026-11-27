"""Dedicated unauthenticated public Streamable HTTP MCP service for SEED Hackathon 2026."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from crewforge.config import (
    DEFAULT_MODEL_PRICING,
    BudgetAllocationCategory,
    CrewForgeSettings,
    configure_logging,
    sanitize_error_message,
)
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.providers.base import ModelProvider
from crewforge.schemas import RunResult, RunStatus
from crewforge.storage.database import Database

logger = logging.getLogger("crewforge.public_mcp")

_MAX_PROMPT_CHARS = 32_000
_MAX_REFERENCE_DOCS = 20
_MAX_TOTAL_DOC_CHARS = 200_000
_MAX_PUBLIC_REQUEST_BYTES = 512 * 1024
_TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _enum_val(val: Any) -> str:
    return str(val.value if hasattr(val, "value") else val)


def _normalize_reference_documents(
    raw_docs: list[dict[str, Any] | str] | None,
) -> list[ReferenceDocument]:
    """Validate and normalize caller-supplied reference documents into ReferenceDocument objects."""
    if raw_docs is None:
        return []
    if not isinstance(raw_docs, list):
        raise ValueError("reference_documents must be a list of strings or objects.")
    if len(raw_docs) > _MAX_REFERENCE_DOCS:
        raise ValueError(f"At most {_MAX_REFERENCE_DOCS} reference documents are permitted.")

    docs: list[ReferenceDocument] = []
    total_chars = 0
    for idx, item in enumerate(raw_docs, start=1):
        if isinstance(item, str):
            content = item.strip()
            title = f"Reference Document {idx}"
        elif isinstance(item, dict):
            raw_content = item.get("content") or item.get("text") or ""
            content = str(raw_content).strip()
            raw_title = item.get("title") or item.get("name") or f"Reference Document {idx}"
            title = str(raw_title).strip()[:160] or f"Reference Document {idx}"
        else:
            raise ValueError(f"reference_documents[{idx - 1}] must be a string or object.")
        if not content:
            raise ValueError(f"reference_documents[{idx - 1}] cannot have empty content.")
        total_chars += len(content)
        if total_chars > _MAX_TOTAL_DOC_CHARS:
            raise ValueError(
                f"Combined reference_documents content exceeds {_MAX_TOTAL_DOC_CHARS} characters."
            )
        doc_id = f"doc_{idx}"
        docs.append(
            ReferenceDocument(
                doc_id=doc_id,
                title=title,
                content=content,
                source_ref=doc_id,
            )
        )
    return docs


def _stage_from_events(events: list[Any], fallback_stage: str) -> tuple[str, float]:
    """Derive human-readable stage and approximate progress from persisted execution events."""
    if not events:
        return fallback_stage, 0.05
    types = [str(ev.event_type) for ev in events]
    if "honesty_gate_evaluated" in types or "synthesis_completed" in types:
        return "honesty_gate_and_synthesis", 0.9
    if "synthesis_started" in types:
        return "synthesis", 0.82
    if "claim_revised" in types or "repair_started" in types:
        return "targeted_repair", 0.72
    if "review_completed" in types:
        return "review_completed", 0.65
    if "review_started" in types:
        return "independent_review", 0.55
    if "task_completed" in types or "task_started" in types:
        return "specialist_execution", 0.38
    if "plan_created" in types:
        return "planning_completed", 0.2
    if "baseline_started" in types:
        return "baseline_execution", 0.45
    return fallback_stage, 0.1


class PublicJobManager:
    """Durable, isolated async job queue for the unauthenticated public evaluation MCP service."""

    def __init__(
        self,
        *,
        settings: CrewForgeSettings | None = None,
        db_path: str | Path | None = None,
        budget_db_path: str | Path | None = None,
        provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
        max_concurrent_jobs: int | None = None,
        max_pending_jobs: int | None = None,
    ) -> None:
        base_settings = settings or CrewForgeSettings()
        resolved_db = Path(
            db_path
            if db_path is not None
            else os.environ.get("CREWFORGE_PUBLIC_MCP_DB_PATH", ".crewforge/public_mcp.db")
        )
        resolved_budget_db = Path(
            budget_db_path
            if budget_db_path is not None
            else os.environ.get(
                "CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH",
                str(resolved_db.with_name("public_mcp_budget.db")),
            )
        )
        merged_pricing = {**DEFAULT_MODEL_PRICING, **base_settings.pricing_json}
        self.settings = base_settings.model_copy(
            update={
                "db_path": resolved_db,
                "cloud_db_path": resolved_db,
                "budget_authority_path": resolved_budget_db,
                "budget_allocation_category": BudgetAllocationCategory.GENERAL,
                "enable_http_retrieval": False,
                "pricing_json": merged_pricing,
            }
        )
        self.db_path = resolved_db
        self.budget_db_path = resolved_budget_db
        self._provider_factory = provider_factory
        raw_conc = (
            max_concurrent_jobs
            if max_concurrent_jobs is not None
            else int(os.environ.get("CREWFORGE_PUBLIC_MAX_CONCURRENT_JOBS", "4"))
        )
        self._max_concurrent_jobs = max(1, min(16, int(raw_conc)))
        raw_pend = (
            max_pending_jobs
            if max_pending_jobs is not None
            else int(os.environ.get("CREWFORGE_PUBLIC_MAX_PENDING_JOBS", "12"))
        )
        self._max_pending_jobs = max(1, min(100, int(raw_pend)))
        self._semaphore: asyncio.Semaphore | None = None
        self._db_init_lock: asyncio.Lock | None = None
        self._lock = threading.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._init_job_table()
        Database(self.db_path).initialize_sync()
        self.recover_interrupted_jobs()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_job_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS public_jobs (
                    task_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    reference_documents_json TEXT NOT NULL,
                    idempotency_key TEXT,
                    request_hash TEXT,
                    current_stage TEXT NOT NULL,
                    progress REAL NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            cols = {
                str(r["name"])
                for r in conn.execute("PRAGMA table_info(public_jobs)").fetchall()
            }
            if "idempotency_key" not in cols:
                conn.execute("ALTER TABLE public_jobs ADD COLUMN idempotency_key TEXT")
            if "request_hash" not in cols:
                conn.execute("ALTER TABLE public_jobs ADD COLUMN request_hash TEXT")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_public_jobs_idempotency
                ON public_jobs (mode, idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            conn.commit()
        finally:
            conn.close()

    def recover_interrupted_jobs(self) -> int:
        """Mark any prior in-flight jobs as failed on startup to prevent duplicate execution."""
        now = _utc_iso()
        err_msg = (
            "Server restarted while task was in progress. "
            "Submit a new task; duplicate execution was prevented."
        )
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE public_jobs
                    SET status = 'failed',
                        current_stage = 'interrupted_by_restart',
                        error = ?,
                        updated_at = ?,
                        completed_at = ?
                    WHERE status IN ('queued', 'running')
                    """,
                    (err_msg, now, now),
                )
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

    def _get_job_row(self, task_id: str) -> dict[str, Any] | None:
        cleaned = (task_id or "").strip()
        if not cleaned:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM public_jobs WHERE task_id = ?",
                (cleaned,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def _update_job(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _utc_iso()
        cols = ", ".join(f"{k} = ?" for k in fields)
        vals = [*fields.values(), task_id]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(f"UPDATE public_jobs SET {cols} WHERE task_id = ?", vals)
                conn.commit()
            finally:
                conn.close()

    def active_job_count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM public_jobs WHERE status IN ('queued', 'running')"
            ).fetchone()
            return int(row["cnt"]) if row is not None else 0
        finally:
            conn.close()

    def _ensure_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent_jobs)
        return self._semaphore

    def _ensure_db_init_lock(self) -> asyncio.Lock:
        if self._db_init_lock is None:
            self._db_init_lock = asyncio.Lock()
        return self._db_init_lock

    def _build_provider(self) -> ModelProvider:
        if self._provider_factory is not None:
            return self._provider_factory(self.settings)
        return build_provider_from_settings(self.settings)

    async def enqueue_task(
        self,
        *,
        prompt: str,
        reference_documents: list[dict[str, Any] | str] | None = None,
        mode: Literal["network", "baseline"],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Validate inputs, persist a queued job, launch background execution, and return ID."""
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            raise ValueError("prompt must be a non-empty string.")
        if len(cleaned_prompt) > _MAX_PROMPT_CHARS:
            raise ValueError(f"prompt exceeds {_MAX_PROMPT_CHARS} character limit.")
        cleaned_idem: str | None = None
        if idempotency_key is not None and idempotency_key.strip():
            cleaned_idem = idempotency_key.strip()
            if len(cleaned_idem) > 128:
                raise ValueError("idempotency_key exceeds 128 character limit.")
        docs = _normalize_reference_documents(reference_documents)
        docs_json = json.dumps([d.model_dump(mode="json") for d in docs])
        req_hash = hashlib.sha256(
            f"{mode}:{cleaned_prompt}:{docs_json}".encode()
        ).hexdigest()
        task_id = f"tsk_pub_{secrets.token_hex(16)}"
        now = _utc_iso()
        provider_name = self.settings.provider.value
        model_name = self.settings.model

        with self._lock:
            conn = self._connect()
            try:
                if cleaned_idem is not None:
                    existing = conn.execute(
                        "SELECT * FROM public_jobs WHERE mode = ? AND idempotency_key = ?",
                        (mode, cleaned_idem),
                    ).fetchone()
                    if existing is not None:
                        if existing["request_hash"] != req_hash:
                            return {
                                "accepted": False,
                                "retryable": False,
                                "error_code": "idempotency_conflict",
                                "message": (
                                    "idempotency_key was already used with different task inputs."
                                ),
                            }
                        if existing["status"] in {"queued", "running", "completed", "partial"}:
                            return {
                                "accepted": True,
                                "deduplicated": True,
                                "task_id": existing["task_id"],
                                "status": existing["status"],
                                "mode": mode,
                                "provider": existing["provider"],
                                "model": existing["model"],
                                "notice": (
                                    "Existing task returned for idempotency_key; duplicate "
                                    "execution prevented."
                                ),
                            }
                        conn.execute(
                            "UPDATE public_jobs SET idempotency_key = NULL WHERE task_id = ?",
                            (existing["task_id"],),
                        )

                active_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM public_jobs WHERE status IN ('queued', 'running')"
                ).fetchone()
                active_cnt = int(active_row["cnt"]) if active_row is not None else 0
                if active_cnt >= self._max_pending_jobs:
                    return {
                        "accepted": False,
                        "retryable": True,
                        "error_code": "capacity_exhausted",
                        "status": "rejected_capacity",
                        "retry_after_seconds": 15,
                        "active_jobs": active_cnt,
                        "max_pending_jobs": self._max_pending_jobs,
                        "message": (
                            "Public evaluation queue is temporarily at capacity. "
                            "Please retry shortly."
                        ),
                    }

                conn.execute(
                    """
                    INSERT INTO public_jobs (
                        task_id, mode, status, prompt, reference_documents_json,
                        idempotency_key, request_hash,
                        current_stage, progress, provider, model,
                        result_json, error, created_at, updated_at, completed_at
                    ) VALUES (
                        ?, ?, 'queued', ?, ?, ?, ?, 'queued', 0.0, ?, ?, NULL, NULL, ?, ?, NULL
                    )
                    """,
                    (
                        task_id,
                        mode,
                        cleaned_prompt,
                        docs_json,
                        cleaned_idem,
                        req_hash,
                        provider_name,
                        model_name,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        bg = asyncio.create_task(self._execute_job(task_id, cleaned_prompt, docs, mode))
        self._tasks[task_id] = bg
        bg.add_done_callback(lambda _t: self._tasks.pop(task_id, None))
        logger.info(
            "Enqueued public task %s (mode=%s, provider=%s, model=%s, docs=%d)",
            task_id,
            mode,
            provider_name,
            model_name,
            len(docs),
        )
        return {
            "accepted": True,
            "deduplicated": False,
            "task_id": task_id,
            "status": "queued",
            "mode": mode,
            "provider": provider_name,
            "model": model_name,
            "notice": (
                "Task accepted. Poll get_task_status(task_id) and retrieve output with "
                "get_task_result(task_id) or get_task_trace(task_id). Anonymous evaluation "
                "tasks are not private personal workspaces."
            ),
        }

    async def _execute_job(
        self,
        task_id: str,
        prompt: str,
        docs: list[ReferenceDocument],
        mode: Literal["network", "baseline"],
    ) -> None:
        sem = self._ensure_semaphore()
        db: Database | None = None
        provider: ModelProvider | None = None
        start_perf = time.perf_counter()
        try:
            async with sem:
                row = self._get_job_row(task_id)
                if row is None or row["status"] == "cancelled":
                    return
                logger.info(
                    "Starting public task %s (mode=%s, provider=%s, model=%s)",
                    task_id,
                    mode,
                    self.settings.provider.value,
                    self.settings.model,
                )
                self._update_job(
                    task_id,
                    status="running",
                    current_stage="initializing",
                    progress=0.05,
                )
                db = Database(self.db_path)
                async with self._ensure_db_init_lock():
                    await db.initialize()
                authority = UnifiedBudgetAuthority(
                    settings=self.settings,
                    migrate_default_dbs=False,
                )
                budget = BudgetManager(db, self.settings, authority=authority)
                provider = self._build_provider()

                if mode == "network":
                    runner = NetworkRunner(
                        provider=provider,
                        database=db,
                        budget_manager=budget,
                        settings=self.settings,
                    )
                    run_res = await runner.run(
                        prompt,
                        run_id=task_id,
                        reference_documents=docs or None,
                    )
                else:
                    baseline = BaselineRunner(
                        provider=provider,
                        database=db,
                        budget_manager=budget,
                        settings=self.settings,
                    )
                    run_res = await baseline.run(
                        prompt,
                        run_id=task_id,
                        reference_documents=docs or None,
                        compute_matched=True,
                    )

                latest_row = self._get_job_row(task_id)
                if latest_row is not None and latest_row["status"] == "cancelled":
                    return

                payload = await self._format_result_payload(db, task_id, run_res, docs)
                final_status = self._map_run_status(run_res.status)
                safe_err = sanitize_error_message(run_res.error) if run_res.error else None
                self._update_job(
                    task_id,
                    status=final_status,
                    current_stage="completed" if final_status != "failed" else "failed",
                    progress=1.0,
                    result_json=json.dumps(payload),
                    error=safe_err,
                    completed_at=_utc_iso(),
                )
                elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
                logger.info(
                    "Finished public task %s (mode=%s, status=%s, elapsed_ms=%.1f, error=%s)",
                    task_id,
                    mode,
                    final_status,
                    elapsed_ms,
                    safe_err or "none",
                )
        except asyncio.CancelledError:
            logger.info("Cancelled public task %s (mode=%s)", task_id, mode)
            self._update_job(
                task_id,
                status="cancelled",
                current_stage="cancelled",
                completed_at=_utc_iso(),
            )
        except Exception as exc:
            safe_err = sanitize_error_message(str(exc))
            logger.exception(
                "Unhandled error in public task %s (mode=%s): %s",
                task_id,
                mode,
                safe_err,
            )
            self._update_job(
                task_id,
                status="failed",
                current_stage="failed",
                error=safe_err,
                completed_at=_utc_iso(),
            )
        finally:
            if provider is not None and hasattr(provider, "aclose"):
                with contextlib.suppress(Exception):
                    await provider.aclose()  # type: ignore[attr-defined]
            if db is not None:
                with contextlib.suppress(Exception):
                    await db.close()

    @staticmethod
    def _map_run_status(status: RunStatus) -> str:
        if status == RunStatus.COMPLETED:
            return "completed"
        if status == RunStatus.PARTIAL:
            return "partial"
        if status == RunStatus.CANCELLED:
            return "cancelled"
        return "failed"

    async def _format_result_payload(
        self,
        db: Database,
        task_id: str,
        run_res: RunResult,
        docs: list[ReferenceDocument],
    ) -> dict[str, Any]:
        trace = await db.get_run_trace(task_id)
        evidence_items = []
        seen_sources: dict[str, dict[str, Any]] = {}
        for ev in trace.evidence:
            entry = {
                "evidence_id": ev.evidence_id,
                "source_type": _enum_val(ev.source_type),
                "source_ref": ev.source_ref,
                "excerpt": ev.excerpt,
                "reliability": _enum_val(ev.reliability),
            }
            evidence_items.append(entry)
            if ev.source_ref not in seen_sources:
                seen_sources[ev.source_ref] = {
                    "source_ref": ev.source_ref,
                    "source_type": _enum_val(ev.source_type),
                    "reliability": _enum_val(ev.reliability),
                }
        for doc in docs:
            if doc.source_ref not in seen_sources:
                seen_sources[doc.source_ref] = {
                    "source_ref": doc.source_ref,
                    "source_type": "user_supplied_document",
                    "title": doc.title,
                    "reliability": _enum_val(doc.reliability),
                }

        verification_info: dict[str, Any]
        if run_res.mode == "baseline":
            verification_info = {
                "independent_verification_performed": False,
                "honesty_gate_passed": None,
                "release_mode": "unverified_baseline",
                "note": "Single-agent baseline output is not independently verified.",
            }
        else:
            gate = trace.honesty_gate
            v_sum = run_res.verification
            verification_occurred = gate is not None and (
                len(trace.review_verdicts) > 0
                or (
                    v_sum is not None
                    and v_sum.material_claim_count > 0
                    and v_sum.review_completeness is not None
                )
            )
            verification_info = {
                "independent_verification_performed": verification_occurred,
                "honesty_gate_passed": v_sum.honesty_gate_passed if v_sum else False,
                "release_mode": gate.release_mode if gate else "unknown",
                "execution_completed": v_sum.execution_completed if v_sum else False,
                "content_released_safely": v_sum.content_released_safely if v_sum else False,
                "material_claim_count": v_sum.material_claim_count if v_sum else 0,
                "supported_claim_count": v_sum.supported_claim_count if v_sum else 0,
                "unresolved_claim_count": v_sum.unresolved_claim_count if v_sum else 0,
                "rejected_claim_count": v_sum.rejected_claim_count if v_sum else 0,
                "stale_claim_count": v_sum.stale_claim_count if v_sum else 0,
                "honesty_gate_reasons": v_sum.honesty_gate_reasons if v_sum else [],
            }

        return {
            "task_id": task_id,
            "mode": run_res.mode,
            "status": self._map_run_status(run_res.status),
            "provider": run_res.provider,
            "model": run_res.model,
            "returned_model": run_res.returned_model,
            "answer": run_res.answer,
            "sources": list(seen_sources.values()),
            "evidence": evidence_items,
            "verification": verification_info,
            "unresolved_requirements": list(run_res.incomplete_tasks or ()),
            "missing_information": list(run_res.missing_information or ()),
            "assumptions": list(run_res.assumptions or ()),
            "confidence_notes": run_res.confidence_notes,
            "specialist_roles": list(run_res.specialist_roles or ()),
            "usage": run_res.usage.model_dump(mode="json"),
            "error": sanitize_error_message(run_res.error) if run_res.error else None,
        }

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Return live execution state, stage, and progress without starting inference."""
        row = self._get_job_row(task_id)
        if row is None:
            return {"found": False, "task_id": task_id, "error": "Task not found."}

        stage = str(row["current_stage"])
        progress = float(row["progress"])
        event_count = 0
        if row["status"] == "running":
            db = Database(self.db_path)
            try:
                await db.initialize()
                trace = await db.get_run_trace(row["task_id"])
                event_count = len(trace.events)
                stage, derived_prog = _stage_from_events(trace.events, stage)
                progress = max(progress, derived_prog)
            except Exception:
                pass
            finally:
                await db.close()

        return {
            "found": True,
            "task_id": row["task_id"],
            "mode": row["mode"],
            "status": row["status"],
            "current_stage": stage,
            "progress": round(progress, 2),
            "event_count": event_count,
            "provider": row["provider"],
            "model": row["model"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
        }

    async def get_task_result(self, task_id: str) -> dict[str, Any]:
        """Return final answer, sources, evidence, verification status, and unresolved items."""
        row = self._get_job_row(task_id)
        if row is None:
            return {"found": False, "ready": False, "task_id": task_id, "error": "Task not found."}

        status = str(row["status"])
        if status in {"queued", "running"}:
            live = await self.get_task_status(task_id)
            return {
                "found": True,
                "ready": False,
                "task_id": task_id,
                "mode": row["mode"],
                "status": status,
                "current_stage": live["current_stage"],
                "progress": live["progress"],
                "message": (
                    "Task is still running. Poll get_task_status(task_id) or retry "
                    "get_task_result(task_id) shortly."
                ),
            }

        if row["result_json"]:
            parsed = json.loads(str(row["result_json"]))
            return {"found": True, "ready": True, **parsed}

        return {
            "found": True,
            "ready": True,
            "task_id": task_id,
            "mode": row["mode"],
            "status": status,
            "answer": "",
            "sources": [],
            "evidence": [],
            "verification": {
                "independent_verification_performed": False,
                "honesty_gate_passed": None,
                "release_mode": "not_executed",
            },
            "unresolved_requirements": [],
            "error": row["error"],
        }

    async def get_task_trace(self, task_id: str) -> dict[str, Any]:
        """Return sanitized agent roles, assignments, messages, tools, reviews, and lineage."""
        row = self._get_job_row(task_id)
        if row is None:
            return {"found": False, "task_id": task_id, "error": "Task not found."}

        trace = None
        db = Database(self.db_path)
        try:
            await db.initialize()
            trace = await db.get_run_trace(row["task_id"])
        except Exception:
            trace = None
        finally:
            await db.close()

        if trace is None:
            return {
                "found": True,
                "trace_available": False,
                "task_id": row["task_id"],
                "mode": row["mode"],
                "status": row["status"],
                "current_stage": row["current_stage"],
                "agents": [],
                "assignments": [],
                "messages": [],
                "tool_executions": [],
                "review_decisions": [],
                "claims_and_evidence_lineage": [],
                "honesty_gate": None,
                "events": [],
                "error": row["error"],
                "notice": "Anonymous public evaluation task — not a private personal workspace.",
            }

        plan = trace.plan
        agents = (
            [
                {
                    "agent_id": a.agent_id,
                    "role": a.role,
                    "capabilities": list(a.capabilities),
                }
                for a in plan.agents
            ]
            if plan is not None
            else []
        )
        assignments = (
            [
                {
                    "task_id": t.task_id,
                    "task_ref": t.task_ref,
                    "title": t.title,
                    "assigned_agent_id": t.assigned_agent_id,
                    "depends_on": list(t.depends_on),
                    "status": _enum_val(t.status),
                }
                for t in plan.tasks
            ]
            if plan is not None
            else []
        )
        messages = [
            {
                "message_id": m.message_id,
                "sender_agent_id": m.sender_agent_id,
                "recipient_agent_id": m.recipient_agent_id,
                "message_type": _enum_val(m.message_type),
                "content": sanitize_error_message(m.content),
                "round_index": m.round_index,
                "created_at": m.created_at.isoformat(),
            }
            for m in trace.messages
        ]
        tool_executions = [
            {
                "tool_execution_id": te.tool_execution_id,
                "tool_name": te.tool_name,
                "expression": te.expression,
                "result": te.result,
                "succeeded": te.succeeded,
                "error": sanitize_error_message(te.error) if te.error else None,
            }
            for te in trace.tool_executions
        ]
        review_decisions = [
            {
                "verdict_id": v.verdict_id,
                "claim_id": v.claim_id,
                "claim_version": v.claim_version,
                "reviewer_id": v.reviewer_id,
                "review_dimension": v.review_dimension,
                "decision": _enum_val(v.decision),
                "rationale": sanitize_error_message(v.rationale),
                "blocking_severity": _enum_val(v.blocking_severity),
            }
            for v in trace.review_verdicts
        ]
        evidence_by_id = {
            ev.evidence_id: {
                "evidence_id": ev.evidence_id,
                "source_type": _enum_val(ev.source_type),
                "source_ref": ev.source_ref,
                "excerpt": ev.excerpt,
                "reliability": _enum_val(ev.reliability),
            }
            for ev in trace.evidence
        }
        claims_lineage = [
            {
                "claim_id": c.claim_id,
                "version": c.version,
                "statement": c.statement,
                "category": _enum_val(c.category),
                "status": _enum_val(c.status),
                "author_agent_id": c.author_agent_id,
                "requirement_refs": list(c.requirement_refs),
                "evidence": [
                    evidence_by_id[eid] for eid in c.evidence_ids if eid in evidence_by_id
                ],
            }
            for c in trace.claims
        ]
        gate = (
            {
                "passed": trace.honesty_gate.passed,
                "release_mode": trace.honesty_gate.release_mode,
                "rejected_segments": list(trace.honesty_gate.rejected_segments),
                "rejected_claim_refs": list(trace.honesty_gate.rejected_claim_refs),
            }
            if trace.honesty_gate is not None
            else None
        )
        events = [
            {
                "seq": idx + 1,
                "event_type": ev.event_type,
                "actor": ev.actor,
                "message": sanitize_error_message(ev.message),
                "created_at": ev.created_at.isoformat(),
            }
            for idx, ev in enumerate(trace.events)
        ]
        return {
            "found": True,
            "task_id": row["task_id"],
            "mode": row["mode"],
            "status": row["status"],
            "agents": agents,
            "assignments": assignments,
            "messages": messages,
            "tool_executions": tool_executions,
            "review_decisions": review_decisions,
            "claims_and_evidence_lineage": claims_lineage,
            "honesty_gate": gate,
            "events": events,
            "notice": "Anonymous public evaluation task — not a private personal workspace.",
        }

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Safely cancel a queued or running task."""
        row = self._get_job_row(task_id)
        if row is None:
            return {
                "found": False,
                "cancelled": False,
                "task_id": task_id,
                "error": "Task not found.",
            }
        if row["status"] in _TERMINAL_STATUSES:
            return {
                "found": True,
                "cancelled": False,
                "task_id": task_id,
                "status": row["status"],
                "message": f"Task is already in terminal state '{row['status']}'.",
            }
        self._update_job(
            task_id,
            status="cancelled",
            current_stage="cancelled",
            completed_at=_utc_iso(),
        )
        bg = self._tasks.get(task_id)
        if bg is not None and not bg.done():
            bg.cancel()
        return {
            "found": True,
            "cancelled": True,
            "task_id": task_id,
            "status": "cancelled",
        }

    async def shutdown(self) -> None:
        """Gracefully cancel in-flight background tasks on server shutdown."""
        pending = list(self._tasks.values())
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def create_public_mcp_server(job_manager: PublicJobManager) -> FastMCP:
    """Create the unauthenticated public FastMCP server with the 6 required tools."""
    mcp = FastMCP(
        name="crewforge-public",
        streamable_http_path="/mcp",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @mcp.tool(
        description=(
            "Enqueue a multi-agent CrewForge task using the server-configured model and "
            "immediately return an unguessable task_id without waiting for completion. "
            "Poll get_task_status(task_id) and fetch outputs with get_task_result(task_id) "
            "or get_task_trace(task_id)."
        )
    )
    async def run_task(
        prompt: str,
        reference_documents: list[dict[str, Any] | str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a task to the CrewForge multi-agent network and return task_id immediately."""
        return await job_manager.enqueue_task(
            prompt=prompt,
            reference_documents=reference_documents,
            mode="network",
            idempotency_key=idempotency_key,
        )

    @mcp.tool(
        description=(
            "Return the current execution status (queued, running, completed, partial, failed, "
            "or cancelled), progress (0.0-1.0), and current stage for a submitted task_id."
        )
    )
    async def get_task_status(task_id: str) -> dict[str, Any]:
        """Inspect current status and progress of a submitted task without running inference."""
        return await job_manager.get_task_status(task_id)

    @mcp.tool(
        description=(
            "Return the final answer, sources, persisted evidence, Honesty Gate verification "
            "status, and unresolved requirements for a completed task_id (or readiness status "
            "if the task is still running)."
        )
    )
    async def get_task_result(task_id: str) -> dict[str, Any]:
        """Retrieve the final answer, sources, and verification status for task_id."""
        return await job_manager.get_task_result(task_id)

    @mcp.tool(
        description=(
            "Return the sanitized execution trace for task_id, including agent roles, task "
            "assignments, peer messages, deterministic tool executions, independent review "
            "decisions, and claim-to-evidence lineage."
        )
    )
    async def get_task_trace(task_id: str) -> dict[str, Any]:
        """Retrieve sanitized multi-agent trace and evidence lineage for task_id."""
        return await job_manager.get_task_trace(task_id)

    @mcp.tool(
        description=(
            "Enqueue an equivalent single-agent baseline run using the same server-configured "
            "model, prompts, and reference_documents for fair comparison, returning task_id "
            "immediately."
        )
    )
    async def run_single_agent(
        prompt: str,
        reference_documents: list[dict[str, Any] | str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a task to the single-agent baseline and return task_id immediately."""
        return await job_manager.enqueue_task(
            prompt=prompt,
            reference_documents=reference_documents,
            mode="baseline",
            idempotency_key=idempotency_key,
        )

    @mcp.tool(
        description="Safely cancel a queued or running task by task_id."
    )
    async def cancel_task(task_id: str) -> dict[str, Any]:
        """Cancel a queued or running task."""
        return await job_manager.cancel_task(task_id)

    return mcp


class _PublicRequestLimitMiddleware(BaseHTTPMiddleware):
    """Enforce request payload size limits without restricting Host or requiring auth."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > _MAX_PUBLIC_REQUEST_BYTES:
                    return JSONResponse(
                        {"error": "Request body exceeds 512 KB limit.", "retryable": False},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            if len(body) > _MAX_PUBLIC_REQUEST_BYTES:
                return JSONResponse(
                    {"error": "Request body exceeds 512 KB limit.", "retryable": False},
                    status_code=413,
                )
        return await call_next(request)


def create_public_mcp_app(
    *,
    settings: CrewForgeSettings | None = None,
    db_path: str | Path | None = None,
    budget_db_path: str | Path | None = None,
    provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
    max_concurrent_jobs: int | None = None,
    max_pending_jobs: int | None = None,
) -> Starlette:
    """Create the standalone public unauthenticated Streamable HTTP MCP ASGI app."""
    job_manager = PublicJobManager(
        settings=settings,
        db_path=db_path,
        budget_db_path=budget_db_path,
        provider_factory=provider_factory,
        max_concurrent_jobs=max_concurrent_jobs,
        max_pending_jobs=max_pending_jobs,
    )
    mcp = create_public_mcp_server(job_manager)
    mcp_http_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with mcp.session_manager.run():
            try:
                yield
            finally:
                await job_manager.shutdown()

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "crewforge-public-mcp",
                "provider": job_manager.settings.provider.value,
                "model": job_manager.settings.model,
                "active_jobs": job_manager.active_job_count(),
                "max_pending_jobs": job_manager._max_pending_jobs,
                "notice": (
                    "Public unauthenticated evaluation endpoint. Anonymous evaluation tasks "
                    "are not private personal workspaces."
                ),
            }
        )

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            *mcp_http_app.routes,
        ],
        lifespan=_lifespan,
    )
    app.add_middleware(_PublicRequestLimitMiddleware)
    app.state.job_manager = job_manager
    return app


def main() -> None:
    """CLI entrypoint for the standalone public evaluation MCP service."""
    parser = argparse.ArgumentParser(
        description="CrewForge Public Unauthenticated Evaluation MCP Service"
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    settings = CrewForgeSettings()
    configure_logging(settings.log_level)
    app = create_public_mcp_app(settings=settings, db_path=args.db_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
