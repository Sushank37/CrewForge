"""Transaction-safe SQLite persistence with repeatable migrations and run isolation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from crewforge.schemas import (
    AgentMessage,
    AgentSpec,
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    EvidenceReliability,
    ExecutionEvent,
    ExecutionPlan,
    HonestyGateRecord,
    ReviewDecision,
    ReviewerIdentity,
    ReviewVerdict,
    RunConfig,
    RunResult,
    RunStatus,
    RunTrace,
    TaskSpec,
    TaskStatus,
    ToolExecutionRecord,
    UsageMetrics,
    VerificationStatus,
    VerificationSummary,
)


class BudgetConstraintError(Exception):
    """Raised atomically when a budget reservation violates configured caps."""


class CrossRunReferenceError(Exception):
    """Raised when an operation attempts to reference records from another or nonexistent run."""


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


_MIGRATION_V1_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    prompt TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS execution_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_execution_events_run_seq
    ON execution_events(run_id, seq);

CREATE TABLE IF NOT EXISTS budget_reservations (
    reservation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    reserved_cost_usd REAL NOT NULL,
    actual_cost_usd REAL,
    reserved_calls INTEGER NOT NULL DEFAULT 1,
    retries INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_budget_reservations_run
    ON budget_reservations(run_id, status);

CREATE TABLE IF NOT EXISTS usage_records (
    usage_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    reservation_id TEXT REFERENCES budget_reservations(reservation_id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    operation TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    actual_cost_usd REAL,
    retries INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_records_run
    ON usage_records(run_id);
"""

_MIGRATION_V2_SQL = """
CREATE TABLE IF NOT EXISTS execution_plans (
    plan_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    user_requirements_json TEXT NOT NULL,
    acceptance_criteria_json TEXT NOT NULL,
    assumptions_json TEXT NOT NULL,
    missing_information_json TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent_ref TEXT,
    role TEXT NOT NULL,
    instructions TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    model_override TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agents_run ON agents(run_id);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task_ref TEXT,
    plan_id TEXT NOT NULL REFERENCES execution_plans(plan_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    assigned_agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    depends_on_json TEXT NOT NULL,
    expected_outputs_json TEXT NOT NULL,
    acceptance_criteria_json TEXT NOT NULL,
    status TEXT NOT NULL,
    findings TEXT,
    error TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);

CREATE TABLE IF NOT EXISTS task_transitions (
    transition_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    from_status TEXT,
    to_status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_transitions_run ON task_transitions(run_id, created_at);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    reliability TEXT NOT NULL,
    verified_in_input INTEGER NOT NULL DEFAULT 0,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    parent_claim_id TEXT,
    author_agent_id TEXT NOT NULL,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
    evidence_ids_json TEXT NOT NULL,
    depends_on_claim_ids_json TEXT NOT NULL,
    assumptions_json TEXT NOT NULL,
    revision_notes TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (claim_id, version)
);

CREATE INDEX IF NOT EXISTS idx_claims_run ON claims(run_id, claim_id, version);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sender_id TEXT NOT NULL,
    recipient_id TEXT,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    related_claim_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_run ON agent_messages(run_id, created_at);

CREATE TABLE IF NOT EXISTS message_deliveries (
    message_id TEXT NOT NULL REFERENCES agent_messages(message_id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    recipient_id TEXT NOT NULL,
    status TEXT NOT NULL,
    delivered_at TEXT,
    consumed_at TEXT,
    PRIMARY KEY (message_id, recipient_id)
);

CREATE INDEX IF NOT EXISTS idx_message_deliveries_inbox
    ON message_deliveries(run_id, recipient_id, status);

CREATE TABLE IF NOT EXISTS review_verdicts (
    verdict_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL,
    claim_version INTEGER NOT NULL,
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT NOT NULL,
    blocking_issues_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_id, claim_version) REFERENCES claims(claim_id, version) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_review_verdicts_run ON review_verdicts(run_id);
"""

_MIGRATION_V3_SQL = """
CREATE TABLE IF NOT EXISTS run_reviewers (
    reviewer_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    dimensions_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, reviewer_id)
);

CREATE TABLE IF NOT EXISTS tool_executions (
    tool_execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    actor TEXT NOT NULL,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE SET NULL,
    expression TEXT NOT NULL,
    variables_json TEXT NOT NULL,
    input_provenance_refs_json TEXT NOT NULL,
    proposed_result TEXT,
    computed_result TEXT,
    units TEXT NOT NULL DEFAULT '',
    assumptions_json TEXT NOT NULL,
    matches_proposed INTEGER,
    inputs_grounded INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_executions_run ON tool_executions(run_id, created_at);

CREATE TABLE IF NOT EXISTS honesty_gate_decisions (
    gate_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(run_id) ON DELETE CASCADE,
    passed INTEGER NOT NULL,
    release_mode TEXT NOT NULL,
    rejected_segments_json TEXT NOT NULL,
    rejected_claim_refs_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    verification_summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    (1, "phase1_core_tables", _MIGRATION_V1_SQL),
    (2, "phase2_network_ledger_messaging", _MIGRATION_V2_SQL),
    (3, "phase3_verification_tools_and_gate", _MIGRATION_V3_SQL),
)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")


class Database:
    """Thread-safe and async-friendly SQLite persistence manager."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn = conn
        return self._conn

    @contextmanager
    def _transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._get_connection()
            conn.execute("BEGIN IMMEDIATE;" if immediate else "BEGIN;")
            try:
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def initialize_sync(self) -> None:
        """Apply pending idempotent schema migrations safely."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
            rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
            applied = {int(row["version"]) for row in rows}
            for version, name, sql in _MIGRATIONS:
                if version in applied:
                    continue
                conn.executescript(sql)
                if version == 3:
                    _ensure_column(
                        conn,
                        "claims",
                        "category",
                        "TEXT NOT NULL DEFAULT 'logical_inference'",
                    )
                    _ensure_column(
                        conn,
                        "claims",
                        "verification_status",
                        "TEXT NOT NULL DEFAULT 'unsupported'",
                    )
                    _ensure_column(
                        conn, "claims", "premise_refs_json", "TEXT NOT NULL DEFAULT '[]'"
                    )
                    _ensure_column(
                        conn, "claims", "tool_execution_ids_json", "TEXT NOT NULL DEFAULT '[]'"
                    )
                    _ensure_column(
                        conn, "claims", "requirement_refs_json", "TEXT NOT NULL DEFAULT '[]'"
                    )
                    _ensure_column(conn, "claims", "is_material", "INTEGER NOT NULL DEFAULT 1")
                    _ensure_column(
                        conn,
                        "review_verdicts",
                        "review_dimension",
                        "TEXT NOT NULL DEFAULT 'general'",
                    )
                    _ensure_column(
                        conn, "review_verdicts", "evidence_refs_json", "TEXT NOT NULL DEFAULT '[]'"
                    )
                    _ensure_column(conn, "review_verdicts", "required_correction", "TEXT")
                    _ensure_column(
                        conn,
                        "review_verdicts",
                        "blocking_severity",
                        "TEXT NOT NULL DEFAULT 'none'",
                    )
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, _utc_iso()),
                )

    async def initialize(self) -> None:
        """Asynchronously initialize and migrate the SQLite database schema."""
        await asyncio.to_thread(self.initialize_sync)

    def get_schema_version_sync(self) -> int:
        """Return highest applied schema migration version."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS ver FROM schema_migrations"
            ).fetchone()
            return int(row["ver"]) if row is not None else 0

    async def get_schema_version(self) -> int:
        """Asynchronously return highest applied schema migration version."""
        return await asyncio.to_thread(self.get_schema_version_sync)

    def close_sync(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def close(self) -> None:
        """Asynchronously close the underlying SQLite connection."""
        await asyncio.to_thread(self.close_sync)

    # ------------------------------------------------------------------
    # Run & event persistence
    # ------------------------------------------------------------------

    def _create_run_sync(
        self,
        run_id: str,
        mode: str,
        prompt: str,
        config: RunConfig,
    ) -> None:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, mode, status, prompt, provider, model, config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    mode,
                    RunStatus.RUNNING.value,
                    prompt,
                    config.provider,
                    config.model,
                    config.model_dump_json(),
                    now,
                ),
            )

    async def create_run(
        self,
        run_id: str,
        mode: str,
        prompt: str,
        config: RunConfig,
    ) -> None:
        """Persist a newly started run record."""
        await asyncio.to_thread(self._create_run_sync, run_id, mode, prompt, config)

    def _complete_run_sync(self, result: RunResult) -> None:
        completed_at = (
            result.completed_at.isoformat() if result.completed_at is not None else _utc_iso()
        )
        with self._transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, result_json = ?, error = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (
                    result.status.value,
                    result.model_dump_json(),
                    result.error,
                    completed_at,
                    result.run_id,
                ),
            )

    async def complete_run(self, result: RunResult) -> None:
        """Update a run record with its final status and serialized RunResult."""
        await asyncio.to_thread(self._complete_run_sync, result)

    def _append_event_sync(self, event: ExecutionEvent) -> ExecutionEvent:
        with self._transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM execution_events WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()
            next_seq = int(row["max_seq"]) + 1
            persisted = event.model_copy(update={"seq": next_seq})
            conn.execute(
                """
                INSERT INTO execution_events (
                    event_id, run_id, seq, event_type, actor, message, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.event_id,
                    persisted.run_id,
                    persisted.seq,
                    persisted.event_type,
                    persisted.actor,
                    persisted.message,
                    json.dumps(persisted.payload, ensure_ascii=False),
                    persisted.created_at.isoformat(),
                ),
            )
            return persisted

    async def append_event(self, event: ExecutionEvent) -> ExecutionEvent:
        """Append a sequentially numbered execution event for a run."""
        return await asyncio.to_thread(self._append_event_sync, event)

    # ------------------------------------------------------------------
    # Budget reservation & reconciliation
    # ------------------------------------------------------------------

    def _reserve_budget_sync(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        estimated_cost_usd: float,
        max_model_calls_per_run: int,
        max_cost_per_run_usd: float,
        max_cumulative_cost_usd: float,
        call_units: int = 1,
    ) -> dict[str, Any]:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            run_row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'reserved' THEN reserved_cost_usd
                            WHEN status IN ('reconciled', 'unknown')
                                THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                            ELSE 0.0
                        END
                    ), 0.0) AS committed_cost,
                    COALESCE(SUM(
                        CASE
                            WHEN status IN ('reserved', 'reconciled', 'unknown', 'released')
                                THEN reserved_calls
                            ELSE 0
                        END
                    ), 0) AS dispatched_calls
                FROM budget_reservations
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            cum_row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'reserved' THEN reserved_cost_usd
                            WHEN status IN ('reconciled', 'unknown')
                                THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                            ELSE 0.0
                        END
                    ), 0.0) AS cumulative_cost
                FROM budget_reservations
                """
            ).fetchone()

            run_cost = float(run_row["committed_cost"])
            run_calls = int(run_row["dispatched_calls"])
            cum_cost = float(cum_row["cumulative_cost"])

            if run_calls + call_units > max_model_calls_per_run:
                raise BudgetConstraintError(
                    f"Run model-call limit exceeded for {run_id}: "
                    f"{run_calls} + {call_units} > {max_model_calls_per_run}."
                )
            if run_cost + estimated_cost_usd > max_cost_per_run_usd + 1e-12:
                raise BudgetConstraintError(
                    f"Run cost limit exceeded for {run_id}: "
                    f"${run_cost:.6f} + ${estimated_cost_usd:.6f} > ${max_cost_per_run_usd:.6f}."
                )
            if cum_cost + estimated_cost_usd > max_cumulative_cost_usd + 1e-12:
                raise BudgetConstraintError(
                    f"Cumulative budget limit exceeded: "
                    f"${cum_cost:.6f} + ${estimated_cost_usd:.6f} > "
                    f"${max_cumulative_cost_usd:.6f}."
                )

            conn.execute(
                """
                INSERT INTO budget_reservations (
                    reservation_id, run_id, provider, model, reserved_cost_usd,
                    actual_cost_usd, reserved_calls, retries, status, notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, 0, 'reserved', NULL, ?, ?)
                """,
                (
                    reservation_id,
                    run_id,
                    provider,
                    model,
                    estimated_cost_usd,
                    call_units,
                    now,
                    now,
                ),
            )
            return {
                "reservation_id": reservation_id,
                "run_id": run_id,
                "run_committed_cost_usd": run_cost + estimated_cost_usd,
                "run_committed_calls": run_calls + call_units,
                "cumulative_committed_cost_usd": cum_cost + estimated_cost_usd,
            }

    async def reserve_budget(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        estimated_cost_usd: float,
        max_model_calls_per_run: int,
        max_cost_per_run_usd: float,
        max_cumulative_cost_usd: float,
        call_units: int = 1,
    ) -> dict[str, Any]:
        """Atomically verify caps and create a budget reservation."""
        return await asyncio.to_thread(
            self._reserve_budget_sync,
            reservation_id=reservation_id,
            run_id=run_id,
            provider=provider,
            model=model,
            estimated_cost_usd=estimated_cost_usd,
            max_model_calls_per_run=max_model_calls_per_run,
            max_cost_per_run_usd=max_cost_per_run_usd,
            max_cumulative_cost_usd=max_cumulative_cost_usd,
            call_units=call_units,
        )

    def _reconcile_budget_sync(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        operation: str,
        status: str,
        actual_cost_usd: float | None,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
        retries: int,
        latency_ms: float,
        outcome: str,
        notes: str | None = None,
    ) -> None:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            cur = conn.execute(
                """
                UPDATE budget_reservations
                SET status = ?, actual_cost_usd = ?, retries = ?, notes = ?, updated_at = ?
                WHERE reservation_id = ? AND status = 'reserved'
                """,
                (status, actual_cost_usd, retries, notes, now, reservation_id),
            )
            if cur.rowcount == 0:
                raise RuntimeError(
                    f"Reservation '{reservation_id}' could not be reconciled "
                    "(missing or already finalized)."
                )
            conn.execute(
                """
                INSERT INTO usage_records (
                    usage_id, run_id, reservation_id, provider, model, operation,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost_usd, actual_cost_usd, retries, latency_ms,
                    outcome, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"usg_{uuid4().hex[:12]}",
                    run_id,
                    reservation_id,
                    provider,
                    model,
                    operation,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    estimated_cost_usd,
                    actual_cost_usd,
                    retries,
                    latency_ms,
                    outcome,
                    now,
                ),
            )

    async def reconcile_budget(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        operation: str,
        status: str,
        actual_cost_usd: float | None,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
        retries: int,
        latency_ms: float,
        outcome: str,
        notes: str | None = None,
    ) -> None:
        """Reconcile a reservation exactly once and record token/cost accounting atomically."""
        await asyncio.to_thread(
            self._reconcile_budget_sync,
            reservation_id=reservation_id,
            run_id=run_id,
            provider=provider,
            model=model,
            operation=operation,
            status=status,
            actual_cost_usd=actual_cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            retries=retries,
            latency_ms=latency_ms,
            outcome=outcome,
            notes=notes,
        )

    def _get_budget_summary_sync(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            cum_row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'reserved' THEN reserved_cost_usd
                            WHEN status IN ('reconciled', 'unknown')
                                THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                            ELSE 0.0
                        END
                    ), 0.0) AS cumulative_committed_cost_usd,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'reconciled' THEN COALESCE(actual_cost_usd, 0.0)
                            ELSE 0.0
                        END
                    ), 0.0) AS cumulative_reconciled_cost_usd,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'unknown'
                                THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                            ELSE 0.0
                        END
                    ), 0.0) AS cumulative_unknown_cost_usd,
                    COALESCE(SUM(
                        CASE
                            WHEN status IN ('reserved', 'reconciled', 'unknown', 'released')
                                THEN reserved_calls
                            ELSE 0
                        END
                    ), 0) AS cumulative_dispatched_calls,
                    COALESCE(SUM(retries), 0) AS cumulative_retries
                FROM budget_reservations
                """
            ).fetchone()

            summary: dict[str, Any] = {
                "cumulative_committed_cost_usd": float(cum_row["cumulative_committed_cost_usd"]),
                "cumulative_reconciled_cost_usd": float(cum_row["cumulative_reconciled_cost_usd"]),
                "cumulative_unknown_cost_usd": float(cum_row["cumulative_unknown_cost_usd"]),
                "cumulative_dispatched_calls": int(cum_row["cumulative_dispatched_calls"]),
                "cumulative_retries": int(cum_row["cumulative_retries"]),
            }

            if run_id is not None:
                run_row = conn.execute(
                    """
                    SELECT
                        COALESCE(SUM(
                            CASE
                                WHEN status = 'reserved' THEN reserved_cost_usd
                                WHEN status IN ('reconciled', 'unknown')
                                    THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                ELSE 0.0
                            END
                        ), 0.0) AS run_committed_cost_usd,
                        COALESCE(SUM(
                            CASE
                                WHEN status IN ('reserved', 'reconciled', 'unknown', 'released')
                                    THEN reserved_calls
                                ELSE 0
                            END
                        ), 0) AS run_committed_calls,
                        COALESCE(SUM(
                            CASE
                                WHEN status = 'reconciled' THEN reserved_calls
                                ELSE 0
                            END
                        ), 0) AS run_successful_calls,
                        COALESCE(SUM(retries), 0) AS run_retries
                    FROM budget_reservations
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                summary["run_id"] = run_id
                summary["run_committed_cost_usd"] = float(run_row["run_committed_cost_usd"])
                summary["run_committed_calls"] = int(run_row["run_committed_calls"])
                summary["run_successful_calls"] = int(run_row["run_successful_calls"])
                summary["run_retries"] = int(run_row["run_retries"])

            return summary

    async def get_budget_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Return committed, reconciled, dispatched-call, and unknown-outcome budget totals."""
        return await asyncio.to_thread(self._get_budget_summary_sync, run_id)

    def _aggregate_run_usage_sync(self, run_id: str) -> UsageMetrics:
        with self._lock:
            conn = self._get_connection()
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

    async def aggregate_run_usage(self, run_id: str) -> UsageMetrics:
        """Aggregate usage metrics across every dispatched model attempt in a run."""
        return await asyncio.to_thread(self._aggregate_run_usage_sync, run_id)

    # ------------------------------------------------------------------
    # Phase 2 Plan, Agent, Task, Ledger & Messaging persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_agent_in_run(conn: sqlite3.Connection, run_id: str, agent_id: str) -> None:
        if agent_id in {"lead", "baseline"}:
            return
        row = conn.execute(
            "SELECT run_id FROM agents WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        if row is not None and row["run_id"] == run_id:
            return
        rev_row = conn.execute(
            "SELECT run_id FROM run_reviewers WHERE run_id = ? AND reviewer_id = ?",
            (run_id, agent_id),
        ).fetchone()
        if rev_row is not None:
            return
        raise CrossRunReferenceError(
            f"Agent or actor '{agent_id}' does not belong to run '{run_id}'."
        )

    @staticmethod
    def _assert_task_in_run(conn: sqlite3.Connection, run_id: str, task_id: str) -> None:
        row = conn.execute(
            "SELECT run_id FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None or row["run_id"] != run_id:
            raise CrossRunReferenceError(f"Task '{task_id}' does not belong to run '{run_id}'.")

    def _save_execution_plan_sync(self, plan: ExecutionPlan) -> None:
        with self._transaction(immediate=True) as conn:
            run_row = conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (plan.run_id,),
            ).fetchone()
            if run_row is None:
                raise CrossRunReferenceError(f"Run '{plan.run_id}' does not exist.")

            conn.execute(
                """
                INSERT INTO execution_plans (
                    plan_id, run_id, goal, user_requirements_json, acceptance_criteria_json,
                    assumptions_json, missing_information_json, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    plan.run_id,
                    plan.goal,
                    json.dumps(plan.user_requirements, ensure_ascii=False),
                    json.dumps(plan.acceptance_criteria, ensure_ascii=False),
                    json.dumps(plan.assumptions, ensure_ascii=False),
                    json.dumps(plan.missing_information, ensure_ascii=False),
                    plan.created_by,
                    plan.created_at.isoformat(),
                ),
            )

            agent_ids_in_plan: set[str] = set()
            for agent in plan.agents:
                if agent.run_id not in (None, plan.run_id):
                    raise CrossRunReferenceError(
                        f"Agent '{agent.agent_id}' run_id mismatch "
                        f"({agent.run_id} != {plan.run_id})."
                    )
                agent_ids_in_plan.add(agent.agent_id)
                conn.execute(
                    """
                    INSERT INTO agents (
                        agent_id, run_id, agent_ref, role, instructions,
                        capabilities_json, model_override, created_by, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        agent.agent_id,
                        plan.run_id,
                        agent.agent_ref,
                        agent.role,
                        agent.instructions,
                        json.dumps(agent.capabilities, ensure_ascii=False),
                        agent.model_override,
                        agent.created_by,
                        agent.created_at.isoformat(),
                    ),
                )

            for task in plan.tasks:
                if task.run_id not in (None, plan.run_id):
                    raise CrossRunReferenceError(
                        f"Task '{task.task_id}' run_id mismatch ({task.run_id} != {plan.run_id})."
                    )
                if task.assigned_agent_id not in agent_ids_in_plan:
                    raise CrossRunReferenceError(
                        f"Task '{task.task_id}' assigned to unknown agent "
                        f"'{task.assigned_agent_id}'."
                    )
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, run_id, task_ref, plan_id, title, description,
                        assigned_agent_id, depends_on_json, expected_outputs_json,
                        acceptance_criteria_json, status, findings, error,
                        created_by, created_at, started_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.task_id,
                        plan.run_id,
                        task.task_ref,
                        plan.plan_id,
                        task.title,
                        task.description,
                        task.assigned_agent_id,
                        json.dumps(task.depends_on, ensure_ascii=False),
                        json.dumps(task.expected_outputs, ensure_ascii=False),
                        json.dumps(task.acceptance_criteria, ensure_ascii=False),
                        task.status.value,
                        task.findings,
                        task.error,
                        task.created_by,
                        task.created_at.isoformat(),
                        task.started_at.isoformat() if task.started_at else None,
                        task.completed_at.isoformat() if task.completed_at else None,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO task_transitions (
                        transition_id, run_id, task_id, from_status, to_status, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"trn_{uuid4().hex[:12]}",
                        plan.run_id,
                        task.task_id,
                        None,
                        task.status.value,
                        "Plan initialized",
                        task.created_at.isoformat(),
                    ),
                )

    async def save_execution_plan(self, plan: ExecutionPlan) -> None:
        """Persist a validated ExecutionPlan, its specialist agents, and its tasks."""
        await asyncio.to_thread(self._save_execution_plan_sync, plan)

    def _transition_task_sync(
        self,
        *,
        run_id: str,
        task_id: str,
        to_status: TaskStatus,
        reason: str = "",
        findings: str | None = None,
        error: str | None = None,
    ) -> None:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT run_id, status, started_at FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None or row["run_id"] != run_id:
                raise CrossRunReferenceError(f"Task '{task_id}' does not belong to run '{run_id}'.")

            from_status = row["status"]
            started_at = row["started_at"]
            completed_at: str | None = None
            if to_status == TaskStatus.RUNNING and started_at is None:
                started_at = now
            if to_status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED}:
                completed_at = now

            conn.execute(
                """
                UPDATE tasks
                SET status = ?,
                    findings = COALESCE(?, findings),
                    error = COALESCE(?, error),
                    started_at = ?,
                    completed_at = COALESCE(?, completed_at)
                WHERE task_id = ? AND run_id = ?
                """,
                (
                    to_status.value,
                    findings,
                    error,
                    started_at,
                    completed_at,
                    task_id,
                    run_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_transitions (
                    transition_id, run_id, task_id, from_status, to_status, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"trn_{uuid4().hex[:12]}",
                    run_id,
                    task_id,
                    from_status,
                    to_status.value,
                    reason or None,
                    now,
                ),
            )

    async def transition_task(
        self,
        *,
        run_id: str,
        task_id: str,
        to_status: TaskStatus,
        reason: str = "",
        findings: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record an atomic task status transition inside the task's run."""
        await asyncio.to_thread(
            self._transition_task_sync,
            run_id=run_id,
            task_id=task_id,
            to_status=to_status,
            reason=reason,
            findings=findings,
            error=error,
        )

    def _insert_evidence_sync(self, evd: Evidence) -> None:
        with self._transaction(immediate=True) as conn:
            self._assert_agent_in_run(conn, evd.run_id, evd.created_by)
            if evd.task_id is not None:
                self._assert_task_in_run(conn, evd.run_id, evd.task_id)
            conn.execute(
                """
                INSERT INTO evidence (
                    evidence_id, run_id, source_type, source_ref, excerpt,
                    reliability, verified_in_input, task_id, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evd.evidence_id,
                    evd.run_id,
                    evd.source_type,
                    evd.source_ref,
                    evd.excerpt,
                    evd.reliability.value,
                    1 if evd.verified_in_input else 0,
                    evd.task_id,
                    evd.created_by,
                    evd.created_at.isoformat(),
                ),
            )

    async def insert_evidence(self, evd: Evidence) -> None:
        """Persist a provenance-checked evidence record."""
        await asyncio.to_thread(self._insert_evidence_sync, evd)

    def _insert_claim_sync(self, claim: Claim) -> None:
        with self._transaction(immediate=True) as conn:
            self._assert_agent_in_run(conn, claim.run_id, claim.author_agent_id)
            if claim.task_id is not None:
                self._assert_task_in_run(conn, claim.run_id, claim.task_id)

            for evd_id in claim.evidence_ids:
                row = conn.execute(
                    "SELECT run_id FROM evidence WHERE evidence_id = ?",
                    (evd_id,),
                ).fetchone()
                if row is None or row["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Evidence '{evd_id}' does not belong to run '{claim.run_id}'."
                    )

            if claim.parent_claim_id is not None:
                parent_row = conn.execute(
                    "SELECT run_id FROM claims WHERE claim_id = ? LIMIT 1",
                    (claim.parent_claim_id,),
                ).fetchone()
                if parent_row is None or parent_row["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Parent claim '{claim.parent_claim_id}' does not belong "
                        f"to run '{claim.run_id}'."
                    )

            for dep_claim_id in claim.depends_on_claim_ids:
                dep_row = conn.execute(
                    "SELECT run_id FROM claims WHERE claim_id = ? LIMIT 1",
                    (dep_claim_id,),
                ).fetchone()
                if dep_row is None or dep_row["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Dependency claim '{dep_claim_id}' does not belong "
                        f"to run '{claim.run_id}'."
                    )

            for pref in claim.premise_refs:
                pref_row = conn.execute(
                    "SELECT run_id FROM claims WHERE claim_id = ? AND version = ?",
                    (pref.claim_id, pref.claim_version),
                ).fetchone()
                if pref_row is None or pref_row["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Premise claim '{pref.claim_id}' v{pref.claim_version} "
                        f"does not belong to run '{claim.run_id}'."
                    )

            for tlx_id in claim.tool_execution_ids:
                tlx_row = conn.execute(
                    "SELECT run_id FROM tool_executions WHERE tool_execution_id = ?",
                    (tlx_id,),
                ).fetchone()
                if tlx_row is None or tlx_row["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Tool execution '{tlx_id}' does not belong to run '{claim.run_id}'."
                    )

            existing_for_id = conn.execute(
                "SELECT run_id, MAX(version) AS max_ver FROM claims WHERE claim_id = ?",
                (claim.claim_id,),
            ).fetchone()
            if existing_for_id is not None and existing_for_id["run_id"] is not None:
                if existing_for_id["run_id"] != claim.run_id:
                    raise CrossRunReferenceError(
                        f"Claim ID '{claim.claim_id}' already belongs to another run."
                    )
                expected_ver = int(existing_for_id["max_ver"]) + 1
                if claim.version != expected_ver:
                    raise ValueError(
                        f"Immutable claim version violation for '{claim.claim_id}': "
                        f"expected version {expected_ver}, got {claim.version}."
                    )

            conn.execute(
                """
                INSERT INTO claims (
                    claim_id, version, run_id, statement, category, status, verification_status,
                    parent_claim_id, author_agent_id, task_id, evidence_ids_json,
                    depends_on_claim_ids_json, premise_refs_json, tool_execution_ids_json,
                    requirement_refs_json, assumptions_json, revision_notes, confidence,
                    is_material, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    claim.version,
                    claim.run_id,
                    claim.statement,
                    claim.category.value,
                    claim.status.value,
                    claim.verification_status.value,
                    claim.parent_claim_id,
                    claim.author_agent_id,
                    claim.task_id,
                    json.dumps(claim.evidence_ids, ensure_ascii=False),
                    json.dumps(claim.depends_on_claim_ids, ensure_ascii=False),
                    json.dumps([p.model_dump() for p in claim.premise_refs], ensure_ascii=False),
                    json.dumps(claim.tool_execution_ids, ensure_ascii=False),
                    json.dumps(claim.requirement_refs, ensure_ascii=False),
                    json.dumps(claim.assumptions, ensure_ascii=False),
                    claim.revision_notes,
                    claim.confidence,
                    1 if claim.is_material else 0,
                    claim.created_at.isoformat(),
                    claim.updated_at.isoformat(),
                ),
            )

    async def insert_claim(self, claim: Claim) -> None:
        """Persist a claim version row with strict cross-run validation."""
        await asyncio.to_thread(self._insert_claim_sync, claim)

    def _update_claim_verification_sync(
        self,
        *,
        run_id: str,
        claim_id: str,
        version: int,
        status: ClaimStatus,
        verification_status: VerificationStatus,
        revision_notes: str | None = None,
    ) -> None:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            cur = conn.execute(
                """
                UPDATE claims
                SET status = ?,
                    verification_status = ?,
                    revision_notes = COALESCE(?, revision_notes),
                    updated_at = ?
                WHERE run_id = ? AND claim_id = ? AND version = ?
                """,
                (
                    status.value,
                    verification_status.value,
                    revision_notes,
                    now,
                    run_id,
                    claim_id,
                    version,
                ),
            )
            if cur.rowcount == 0:
                raise CrossRunReferenceError(
                    f"Claim '{claim_id}' v{version} does not exist in run '{run_id}'."
                )

    async def update_claim_verification_state(
        self,
        *,
        run_id: str,
        claim_id: str,
        version: int,
        status: ClaimStatus,
        verification_status: VerificationStatus,
        revision_notes: str | None = None,
    ) -> None:
        """Update verification state of an existing claim version without modifying statement."""
        await asyncio.to_thread(
            self._update_claim_verification_sync,
            run_id=run_id,
            claim_id=claim_id,
            version=version,
            status=status,
            verification_status=verification_status,
            revision_notes=revision_notes,
        )

    def _register_reviewer_sync(self, reviewer: ReviewerIdentity) -> None:
        with self._transaction(immediate=True) as conn:
            run_row = conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (reviewer.run_id,),
            ).fetchone()
            if run_row is None:
                raise CrossRunReferenceError(f"Run '{reviewer.run_id}' does not exist.")
            conn.execute(
                """
                INSERT OR REPLACE INTO run_reviewers (
                    reviewer_id, run_id, role, dimensions_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    reviewer.reviewer_id,
                    reviewer.run_id,
                    reviewer.role,
                    json.dumps(reviewer.dimensions, ensure_ascii=False),
                    reviewer.created_at.isoformat(),
                ),
            )

    async def register_reviewer(self, reviewer: ReviewerIdentity) -> None:
        """Register an independent reviewer identity in the run."""
        await asyncio.to_thread(self._register_reviewer_sync, reviewer)

    def _insert_tool_execution_sync(self, record: ToolExecutionRecord) -> None:
        with self._transaction(immediate=True) as conn:
            self._assert_agent_in_run(conn, record.run_id, record.actor)
            if record.task_id is not None:
                self._assert_task_in_run(conn, record.run_id, record.task_id)
            conn.execute(
                """
                INSERT INTO tool_executions (
                    tool_execution_id, run_id, tool_name, actor, task_id, expression,
                    variables_json, input_provenance_refs_json, proposed_result, computed_result,
                    units, assumptions_json, matches_proposed, inputs_grounded, success, error,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.tool_execution_id,
                    record.run_id,
                    record.tool_name,
                    record.actor,
                    record.task_id,
                    record.expression,
                    json.dumps(record.variables, ensure_ascii=False),
                    json.dumps(record.input_provenance_refs, ensure_ascii=False),
                    record.proposed_result,
                    record.computed_result,
                    record.units,
                    json.dumps(record.assumptions, ensure_ascii=False),
                    (
                        None
                        if record.matches_proposed is None
                        else (1 if record.matches_proposed else 0)
                    ),
                    1 if record.inputs_grounded else 0,
                    1 if record.success else 0,
                    record.error,
                    record.created_at.isoformat(),
                ),
            )

    async def insert_tool_execution(self, record: ToolExecutionRecord) -> None:
        """Persist an immutable deterministic tool execution record."""
        await asyncio.to_thread(self._insert_tool_execution_sync, record)

    def _get_tool_executions_sync(self, run_id: str) -> list[ToolExecutionRecord]:
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM tool_executions WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
        return [self._row_to_tool_execution(row) for row in rows]

    async def get_tool_executions_for_run(self, run_id: str) -> list[ToolExecutionRecord]:
        """Return all deterministic tool execution records for a run."""
        return await asyncio.to_thread(self._get_tool_executions_sync, run_id)

    def _save_honesty_gate_sync(self, gate: HonestyGateRecord) -> None:
        with self._transaction(immediate=True) as conn:
            run_row = conn.execute(
                "SELECT run_id FROM runs WHERE run_id = ?",
                (gate.run_id,),
            ).fetchone()
            if run_row is None:
                raise CrossRunReferenceError(f"Run '{gate.run_id}' does not exist.")
            conn.execute(
                """
                INSERT OR REPLACE INTO honesty_gate_decisions (
                    gate_id, run_id, passed, release_mode, rejected_segments_json,
                    rejected_claim_refs_json, reasons_json, verification_summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate.gate_id,
                    gate.run_id,
                    1 if gate.passed else 0,
                    gate.release_mode,
                    json.dumps(gate.rejected_segments, ensure_ascii=False),
                    json.dumps(gate.rejected_claim_refs, ensure_ascii=False),
                    json.dumps(gate.reasons, ensure_ascii=False),
                    gate.verification_summary.model_dump_json(),
                    gate.created_at.isoformat(),
                ),
            )

    async def save_honesty_gate_decision(self, gate: HonestyGateRecord) -> None:
        """Persist the final deterministic Honesty Gate decision for a run."""
        await asyncio.to_thread(self._save_honesty_gate_sync, gate)

    def _insert_message_sync(
        self,
        msg: AgentMessage,
        delivery_recipients: list[str],
    ) -> None:
        with self._transaction(immediate=True) as conn:
            self._assert_agent_in_run(conn, msg.run_id, msg.sender_id)
            if msg.recipient_id is not None:
                self._assert_agent_in_run(conn, msg.run_id, msg.recipient_id)
            for target_id in delivery_recipients:
                self._assert_agent_in_run(conn, msg.run_id, target_id)
            for claim_id in msg.related_claim_ids:
                row = conn.execute(
                    "SELECT run_id FROM claims WHERE claim_id = ? LIMIT 1",
                    (claim_id,),
                ).fetchone()
                if row is None or row["run_id"] != msg.run_id:
                    raise CrossRunReferenceError(
                        f"Message related claim '{claim_id}' does not belong to run '{msg.run_id}'."
                    )

            conn.execute(
                """
                INSERT INTO agent_messages (
                    message_id, run_id, sender_id, recipient_id, message_type,
                    content, related_claim_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.message_id,
                    msg.run_id,
                    msg.sender_id,
                    msg.recipient_id,
                    msg.message_type,
                    msg.content,
                    json.dumps(msg.related_claim_ids, ensure_ascii=False),
                    msg.created_at.isoformat(),
                ),
            )
            for target_id in delivery_recipients:
                conn.execute(
                    """
                    INSERT INTO message_deliveries (
                        message_id, run_id, recipient_id, status, delivered_at, consumed_at
                    ) VALUES (?, ?, ?, 'pending', NULL, NULL)
                    """,
                    (msg.message_id, msg.run_id, target_id),
                )

    async def insert_message(
        self,
        msg: AgentMessage,
        delivery_recipients: list[str],
    ) -> None:
        """Persist an agent message and initialize per-recipient delivery rows."""
        await asyncio.to_thread(self._insert_message_sync, msg, delivery_recipients)

    def _get_pending_messages_sync(
        self,
        *,
        run_id: str,
        recipient_id: str,
        mark_delivered: bool = True,
    ) -> list[AgentMessage]:
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            self._assert_agent_in_run(conn, run_id, recipient_id)
            rows = conn.execute(
                """
                SELECT m.*, d.status AS delivery_status
                FROM agent_messages m
                JOIN message_deliveries d ON m.message_id = d.message_id
                WHERE d.run_id = ? AND d.recipient_id = ? AND d.status IN ('pending', 'delivered')
                ORDER BY m.created_at ASC
                """,
                (run_id, recipient_id),
            ).fetchall()

            if mark_delivered and rows:
                msg_ids = [row["message_id"] for row in rows if row["delivery_status"] == "pending"]
                for mid in msg_ids:
                    conn.execute(
                        """
                        UPDATE message_deliveries
                        SET status = 'delivered', delivered_at = COALESCE(delivered_at, ?)
                        WHERE message_id = ? AND recipient_id = ?
                        """,
                        (now, mid, recipient_id),
                    )

            return [
                AgentMessage(
                    message_id=row["message_id"],
                    run_id=row["run_id"],
                    sender_id=row["sender_id"],
                    recipient_id=row["recipient_id"],
                    message_type=row["message_type"],
                    content=row["content"],
                    related_claim_ids=json.loads(row["related_claim_ids_json"]),
                    delivery_status="delivered" if mark_delivered else row["delivery_status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    async def get_pending_messages_for_agent(
        self,
        *,
        run_id: str,
        recipient_id: str,
        mark_delivered: bool = True,
    ) -> list[AgentMessage]:
        """Fetch unconsumed inbox messages for an agent in the same run."""
        return await asyncio.to_thread(
            self._get_pending_messages_sync,
            run_id=run_id,
            recipient_id=recipient_id,
            mark_delivered=mark_delivered,
        )

    def _mark_messages_consumed_sync(
        self,
        *,
        run_id: str,
        recipient_id: str,
        message_ids: list[str],
    ) -> None:
        if not message_ids:
            return
        now = _utc_iso()
        with self._transaction(immediate=True) as conn:
            for mid in message_ids:
                conn.execute(
                    """
                    UPDATE message_deliveries
                    SET status = 'consumed',
                        delivered_at = COALESCE(delivered_at, ?),
                        consumed_at = ?
                    WHERE message_id = ? AND run_id = ? AND recipient_id = ?
                    """,
                    (now, now, mid, run_id, recipient_id),
                )

    async def mark_messages_consumed(
        self,
        *,
        run_id: str,
        recipient_id: str,
        message_ids: list[str],
    ) -> None:
        """Mark delivered messages as consumed by the target agent."""
        await asyncio.to_thread(
            self._mark_messages_consumed_sync,
            run_id=run_id,
            recipient_id=recipient_id,
            message_ids=message_ids,
        )

    def _list_agents_with_pending_messages_sync(self, run_id: str) -> list[str]:
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT DISTINCT d.recipient_id
                FROM message_deliveries d
                JOIN agents a ON a.agent_id = d.recipient_id AND a.run_id = d.run_id
                WHERE d.run_id = ? AND d.status IN ('pending', 'delivered')
                ORDER BY d.recipient_id ASC
                """,
                (run_id,),
            ).fetchall()
            return [str(row["recipient_id"]) for row in rows]

    async def list_agents_with_pending_messages(self, run_id: str) -> list[str]:
        """Return specialist agent IDs in a run that still have unconsumed messages."""
        return await asyncio.to_thread(self._list_agents_with_pending_messages_sync, run_id)

    def _insert_review_verdict_sync(self, verdict: ReviewVerdict) -> None:
        with self._transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT run_id, author_agent_id FROM claims WHERE claim_id = ? AND version = ?",
                (verdict.claim_id, verdict.claim_version),
            ).fetchone()
            if row is None or row["run_id"] != verdict.run_id:
                raise CrossRunReferenceError(
                    f"Claim '{verdict.claim_id}' v{verdict.claim_version} "
                    f"does not belong to run '{verdict.run_id}'."
                )
            if row["author_agent_id"] == verdict.reviewer_id:
                raise ValueError(
                    f"Independence violation: reviewer '{verdict.reviewer_id}' cannot "
                    f"review its own claim '{verdict.claim_id}'."
                )
            for evd_id in verdict.evidence_refs:
                evd_row = conn.execute(
                    "SELECT run_id FROM evidence WHERE evidence_id = ?",
                    (evd_id,),
                ).fetchone()
                if evd_row is None or evd_row["run_id"] != verdict.run_id:
                    raise CrossRunReferenceError(
                        f"Verdict evidence '{evd_id}' does not belong to run '{verdict.run_id}'."
                    )

            conn.execute(
                """
                INSERT INTO review_verdicts (
                    verdict_id, run_id, claim_id, claim_version, reviewer_id,
                    review_dimension, decision, rationale, evidence_refs_json,
                    required_correction, blocking_severity, blocking_issues_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    verdict.verdict_id,
                    verdict.run_id,
                    verdict.claim_id,
                    verdict.claim_version,
                    verdict.reviewer_id,
                    verdict.review_dimension,
                    verdict.decision.value,
                    verdict.rationale,
                    json.dumps(verdict.evidence_refs, ensure_ascii=False),
                    verdict.required_correction,
                    verdict.blocking_severity,
                    json.dumps(verdict.blocking_issues, ensure_ascii=False),
                    verdict.created_at.isoformat(),
                ),
            )

    async def insert_review_verdict(self, verdict: ReviewVerdict) -> None:
        """Persist an independent reviewer verdict targeting a specific claim version."""
        await asyncio.to_thread(self._insert_review_verdict_sync, verdict)

    # ------------------------------------------------------------------
    # Query helpers & RunTrace assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_tool_execution(row: sqlite3.Row) -> ToolExecutionRecord:
        matches_raw = row["matches_proposed"]
        return ToolExecutionRecord(
            tool_execution_id=row["tool_execution_id"],
            run_id=row["run_id"],
            tool_name=row["tool_name"],
            actor=row["actor"],
            task_id=row["task_id"],
            expression=row["expression"],
            variables=json.loads(row["variables_json"]),
            input_provenance_refs=json.loads(row["input_provenance_refs_json"]),
            proposed_result=row["proposed_result"],
            computed_result=row["computed_result"],
            units=row["units"],
            assumptions=json.loads(row["assumptions_json"]),
            matches_proposed=None if matches_raw is None else bool(matches_raw),
            inputs_grounded=bool(row["inputs_grounded"]),
            success=bool(row["success"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        keys = set(row.keys())
        premise_raw = json.loads(row["premise_refs_json"]) if "premise_refs_json" in keys else []
        return Claim(
            claim_id=row["claim_id"],
            version=int(row["version"]),
            run_id=row["run_id"],
            statement=row["statement"],
            category=ClaimCategory(row["category"])
            if "category" in keys
            else ClaimCategory.INFERENCE,
            status=ClaimStatus(row["status"]),
            verification_status=(
                VerificationStatus(row["verification_status"])
                if "verification_status" in keys
                else VerificationStatus.UNSUPPORTED
            ),
            parent_claim_id=row["parent_claim_id"],
            author_agent_id=row["author_agent_id"],
            task_id=row["task_id"],
            evidence_ids=json.loads(row["evidence_ids_json"]),
            depends_on_claim_ids=json.loads(row["depends_on_claim_ids_json"]),
            premise_refs=[ClaimVersionRef.model_validate(item) for item in premise_raw],
            tool_execution_ids=(
                json.loads(row["tool_execution_ids_json"])
                if "tool_execution_ids_json" in keys
                else []
            ),
            requirement_refs=(
                json.loads(row["requirement_refs_json"]) if "requirement_refs_json" in keys else []
            ),
            assumptions=json.loads(row["assumptions_json"]),
            revision_notes=row["revision_notes"],
            confidence=row["confidence"],
            is_material=bool(row["is_material"]) if "is_material" in keys else True,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _get_claims_for_run_sync(self, run_id: str, latest_only: bool = False) -> list[Claim]:
        with self._lock:
            conn = self._get_connection()
            if latest_only:
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM claims c
                    JOIN (
                        SELECT claim_id, MAX(version) AS max_ver
                        FROM claims
                        WHERE run_id = ?
                        GROUP BY claim_id
                    ) latest ON c.claim_id = latest.claim_id AND c.version = latest.max_ver
                    WHERE c.run_id = ?
                    ORDER BY c.created_at ASC
                    """,
                    (run_id, run_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM claims WHERE run_id = ? ORDER BY created_at ASC, version ASC",
                    (run_id,),
                ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    async def get_claims_for_run(self, run_id: str, *, latest_only: bool = False) -> list[Claim]:
        """Return all claim versions (or latest version per claim_id) for a run."""
        return await asyncio.to_thread(self._get_claims_for_run_sync, run_id, latest_only)

    def _get_run_trace_sync(self, run_id: str) -> RunTrace | None:
        with self._lock:
            conn = self._get_connection()
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                return None

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
            reviewer_rows = conn.execute(
                "SELECT * FROM run_reviewers WHERE run_id = ? ORDER BY created_at ASC",
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
            usage_rows = conn.execute(
                "SELECT * FROM usage_records WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            res_rows = conn.execute(
                "SELECT * FROM budget_reservations WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()

        config = RunConfig.model_validate_json(run_row["config_json"])
        result = (
            RunResult.model_validate_json(run_row["result_json"])
            if run_row["result_json"]
            else None
        )
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
                started_at=(
                    datetime.fromisoformat(row["started_at"]) if row["started_at"] else None
                ),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
            )
            for row in task_rows
        ]
        plan: ExecutionPlan | None = None
        if plan_row is not None:
            plan = ExecutionPlan(
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
        claims = [self._row_to_claim(row) for row in claim_rows]
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
        reviewers = [
            ReviewerIdentity(
                reviewer_id=row["reviewer_id"],
                run_id=row["run_id"],
                role=row["role"],
                dimensions=json.loads(row["dimensions_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in reviewer_rows
        ]
        tool_executions = [self._row_to_tool_execution(row) for row in tool_rows]
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
                    row["required_correction"] if "required_correction" in set(row.keys()) else None
                ),
                blocking_severity=(
                    row["blocking_severity"] if "blocking_severity" in set(row.keys()) else "none"
                ),
                blocking_issues=json.loads(row["blocking_issues_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in verdict_rows
        ]
        honesty_gate: HonestyGateRecord | None = None
        if gate_row is not None:
            honesty_gate = HonestyGateRecord(
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
        return RunTrace(
            run_id=run_row["run_id"],
            mode=run_row["mode"],
            status=RunStatus(run_row["status"]),
            prompt=run_row["prompt"],
            provider=run_row["provider"],
            model=run_row["model"],
            config=config,
            result=result,
            plan=plan,
            agents=agents,
            reviewers=reviewers,
            tasks=tasks,
            task_transitions=[dict(row) for row in transition_rows],
            claims=claims,
            evidence=evidence,
            tool_executions=tool_executions,
            messages=messages,
            review_verdicts=verdicts,
            honesty_gate=honesty_gate,
            events=events,
            usage_records=[dict(row) for row in usage_rows],
            budget_reservations=[dict(row) for row in res_rows],
            created_at=datetime.fromisoformat(run_row["created_at"]),
            completed_at=(
                datetime.fromisoformat(run_row["completed_at"]) if run_row["completed_at"] else None
            ),
        )

    async def get_run_trace(self, run_id: str) -> RunTrace | None:
        """Fetch full run trace without requiring any model provider."""
        return await asyncio.to_thread(self._get_run_trace_sync, run_id)
