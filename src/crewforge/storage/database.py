"""Transaction-safe SQLite persistence for runs, events, usage, and budget reservations."""

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
    ExecutionEvent,
    RunConfig,
    RunResult,
    RunStatus,
    RunTrace,
)


class BudgetConstraintError(Exception):
    """Raised atomically when a budget reservation violates configured caps."""


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


_SCHEMA_SQL = """
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
        """Create schema tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            conn.executescript(_SCHEMA_SQL)

    async def initialize(self) -> None:
        """Asynchronously initialize the SQLite database schema."""
        await asyncio.to_thread(self.initialize_sync)

    def close_sync(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def close(self) -> None:
        """Asynchronously close the underlying SQLite connection."""
        await asyncio.to_thread(self.close_sync)

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
                            WHEN status IN ('reserved', 'reconciled', 'unknown')
                                THEN reserved_calls
                            ELSE 0
                        END
                    ), 0) AS committed_calls
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
            run_calls = int(run_row["committed_calls"])
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
            conn.execute(
                """
                UPDATE budget_reservations
                SET status = ?, actual_cost_usd = ?, retries = ?, notes = ?, updated_at = ?
                WHERE reservation_id = ?
                """,
                (status, actual_cost_usd, retries, notes, now, reservation_id),
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
        """Reconcile a reservation and record token/cost accounting atomically."""
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
                    COALESCE(SUM(retries), 0) AS cumulative_retries
                FROM budget_reservations
                """
            ).fetchone()

            summary: dict[str, Any] = {
                "cumulative_committed_cost_usd": float(cum_row["cumulative_committed_cost_usd"]),
                "cumulative_reconciled_cost_usd": float(
                    cum_row["cumulative_reconciled_cost_usd"]
                ),
                "cumulative_unknown_cost_usd": float(cum_row["cumulative_unknown_cost_usd"]),
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
                                WHEN status IN ('reserved', 'reconciled', 'unknown')
                                    THEN reserved_calls
                                ELSE 0
                            END
                        ), 0) AS run_committed_calls,
                        COALESCE(SUM(retries), 0) AS run_retries
                    FROM budget_reservations
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                summary["run_id"] = run_id
                summary["run_committed_cost_usd"] = float(run_row["run_committed_cost_usd"])
                summary["run_committed_calls"] = int(run_row["run_committed_calls"])
                summary["run_retries"] = int(run_row["run_retries"])

            return summary

    async def get_budget_summary(self, run_id: str | None = None) -> dict[str, Any]:
        """Return committed, reconciled, and unknown-outcome budget totals."""
        return await asyncio.to_thread(self._get_budget_summary_sync, run_id)

    def _get_run_trace_sync(self, run_id: str) -> RunTrace | None:
        with self._lock:
            conn = self._get_connection()
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                return None

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
            events=events,
            usage_records=[dict(row) for row in usage_rows],
            budget_reservations=[dict(row) for row in res_rows],
            created_at=datetime.fromisoformat(run_row["created_at"]),
            completed_at=(
                datetime.fromisoformat(run_row["completed_at"])
                if run_row["completed_at"]
                else None
            ),
        )

    async def get_run_trace(self, run_id: str) -> RunTrace | None:
        """Fetch full run trace, events, usage records, and budget reservations."""
        return await asyncio.to_thread(self._get_run_trace_sync, run_id)
