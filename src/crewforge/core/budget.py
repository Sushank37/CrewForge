"""Persistent, concurrency-safe budget manager for CrewForge.

IMPORTANT: Application-level cost estimation and pre-call reservations are
software safeguards based on configured token pricing and max-token bounds.
They are NOT a guaranteed provider-side billing cap; always configure hard
spending limits in your cloud provider dashboard as well.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewforge.config import CrewForgeSettings, ModelPricing, sanitize_error_message
from crewforge.schemas import UsageMetrics
from crewforge.storage.database import BudgetConstraintError, Database


class BudgetError(Exception):
    """Base error for budget enforcement failures."""


class BudgetExceededError(BudgetError):
    """Raised when a model call would exceed per-run or cumulative caps."""


class UnknownModelPricingError(BudgetError):
    """Raised when a paid provider is invoked without configured pricing (fail-closed)."""


class BudgetReservation(BaseModel):
    """Active pre-call budget reservation for a single model attempt."""

    reservation_id: str = Field(default_factory=lambda: f"res_{uuid4().hex[:12]}")
    run_id: str
    provider: str
    model: str
    is_paid_provider: bool
    reserved_cost_usd: float = Field(ge=0.0)
    estimated_input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=1)
    call_units: int = Field(default=1, ge=1)
    allocation_category: str = Field(default="general")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UnifiedBudgetAuthority:
    """Canonical cross-database persistent budget authority and spend guard."""

    _instances: dict[str, UnifiedBudgetAuthority] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        authority_path: str | Path | None = None,
        settings: CrewForgeSettings | None = None,
        *,
        source_dbs: list[Path | str] | None = None,
        migrate_default_dbs: bool | None = None,
    ) -> None:
        self._settings = settings or CrewForgeSettings()
        raw_path = authority_path or self._settings.budget_authority_path
        self.authority_path = Path(raw_path)
        self._source_dbs = [Path(p) for p in source_dbs] if source_dbs is not None else None
        self._migrate_default_dbs = migrate_default_dbs
        self._db_lock = threading.Lock()
        self._initialize_sync()

    @classmethod
    def reset_instances(cls) -> None:
        """Clear cached authority singletons (intended for test isolation)."""
        with cls._lock:
            cls._instances.clear()

    @classmethod
    def get_instance(
        cls,
        settings: CrewForgeSettings | None = None,
        authority_path: str | Path | None = None,
    ) -> UnifiedBudgetAuthority:
        cfg = settings or CrewForgeSettings()
        raw_path = authority_path or cfg.budget_authority_path
        path_str = ":memory:" if str(raw_path) == ":memory:" else str(Path(raw_path).resolve())
        with cls._lock:
            if path_str not in cls._instances:
                cls._instances[path_str] = cls(authority_path=raw_path, settings=cfg)
            return cls._instances[path_str]

    def _initialize_sync(self) -> None:
        with self._db_lock:
            if str(self.authority_path) != ":memory:":
                self.authority_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.authority_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS canonical_budget_ledger (
                    reservation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_cost_usd REAL NOT NULL,
                    actual_cost_usd REAL,
                    reserved_calls INTEGER NOT NULL DEFAULT 1,
                    retries INTEGER NOT NULL DEFAULT 0,
                    allocation_category TEXT NOT NULL DEFAULT 'general',
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_canonical_budget_status
                ON canonical_budget_ledger(status, provider, allocation_category);
                """
            )
            # Reconcile historical reservations only from explicitly supplied sources
            # or when running against the canonical production authority path.
            if self._source_dbs is not None:
                candidate_dbs = self._source_dbs
            else:
                prod_auth = Path(".crewforge/budget_authority.db").resolve()
                is_prod_auth = (
                    str(self.authority_path) != ":memory:"
                    and self.authority_path.resolve() == prod_auth
                )
                should_migrate = (
                    self._migrate_default_dbs
                    if self._migrate_default_dbs is not None
                    else is_prod_auth
                )
                candidate_dbs = (
                    [Path(".crewforge/crewforge.db"), Path(".crewforge/cloud_dev.db")]
                    if should_migrate
                    else []
                )
            for c_path in candidate_dbs:
                if c_path.exists() and str(c_path.resolve()) != str(self.authority_path.resolve()):
                    try:
                        src_conn = sqlite3.connect(str(c_path))
                        src_conn.row_factory = sqlite3.Row
                        tbl_exists = src_conn.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' "
                            "AND name='budget_reservations'"
                        ).fetchone()
                        if tbl_exists:
                            rows = src_conn.execute(
                                """
                                SELECT reservation_id, run_id, provider, model, reserved_cost_usd,
                                       actual_cost_usd, reserved_calls, retries, status, notes,
                                       created_at, updated_at
                                FROM budget_reservations
                                """
                            ).fetchall()
                            for r in rows:
                                conn.execute(
                                    """
                                    INSERT INTO canonical_budget_ledger (
                                        reservation_id, run_id, provider, model, status,
                                        reserved_cost_usd, actual_cost_usd, reserved_calls,
                                        retries, allocation_category, notes, created_at, updated_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'phase4_pilots', ?, ?, ?)
                                    ON CONFLICT(reservation_id) DO UPDATE SET
                                        actual_cost_usd = COALESCE(
                                            excluded.actual_cost_usd,
                                            canonical_budget_ledger.actual_cost_usd
                                        ),
                                        status = excluded.status,
                                        updated_at = excluded.updated_at
                                    """,
                                    (
                                        r["reservation_id"],
                                        r["run_id"],
                                        r["provider"],
                                        r["model"],
                                        r["status"],
                                        float(r["reserved_cost_usd"]),
                                        float(r["actual_cost_usd"])
                                        if r["actual_cost_usd"] is not None
                                        else None,
                                        int(r["reserved_calls"]),
                                        int(r["retries"]),
                                        r["notes"],
                                        r["created_at"],
                                        r["updated_at"],
                                    ),
                                )
                        src_conn.close()
                    except Exception:
                        pass
            conn.commit()
            conn.close()

    def reserve_sync(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        is_paid_provider: bool,
        estimated_cost_usd: float,
        call_units: int = 1,
        allocation_category: str = "general",
    ) -> None:
        """Atomically enforce protected reserve and allocation caps in canonical ledger."""
        now = datetime.now(UTC).isoformat()
        with self._db_lock:
            conn = sqlite3.connect(str(self.authority_path))
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE;")
                if is_paid_provider:
                    # 1. Global protected reserve check ($6.10 - $4.00 = $2.10 spendable ceiling)
                    max_spendable = max(
                        0.0,
                        self._settings.total_budget_ceiling_usd
                        - self._settings.protected_reserve_usd,
                    )
                    cum_row = conn.execute(
                        """
                        SELECT COALESCE(SUM(
                            CASE
                                WHEN status = 'reserved' THEN reserved_cost_usd
                                WHEN status IN ('reconciled', 'unknown')
                                    THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                ELSE 0.0
                            END
                        ), 0.0) AS total_committed
                        FROM canonical_budget_ledger
                        WHERE provider != 'ollama'
                        """
                    ).fetchone()
                    total_committed = float(cum_row["total_committed"]) if cum_row else 0.0

                    if total_committed + estimated_cost_usd > max_spendable + 1e-12:
                        res_usd = self._settings.protected_reserve_usd
                        commit_val = total_committed + estimated_cost_usd
                        raise BudgetConstraintError(
                            "Protected global reserve violation: commitment "
                            f"${commit_val:.6f} would exceed allowable spend "
                            f"${max_spendable:.2f} (leaving less than ${res_usd:.2f} reserve)."
                        )

                    # 2. Allocation category caps
                    if allocation_category == "phase5_same_model":
                        cat_row = conn.execute(
                            """
                            SELECT COALESCE(SUM(
                                CASE
                                    WHEN status = 'reserved' THEN reserved_cost_usd
                                    WHEN status IN ('reconciled', 'unknown')
                                        THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                    ELSE 0.0
                                END
                            ), 0.0) AS cat_committed
                            FROM canonical_budget_ledger
                            WHERE allocation_category = 'phase5_same_model' AND provider != 'ollama'
                            """
                        ).fetchone()
                        cat_committed = float(cat_row["cat_committed"]) if cat_row else 0.0
                        if (
                            cat_committed + estimated_cost_usd
                            > self._settings.phase5_same_model_budget_cap_usd + 1e-12
                        ):
                            raise BudgetConstraintError(
                                f"Phase 5 same-model benchmarking allocation cap exceeded: "
                                f"${cat_committed:.6f} + ${estimated_cost_usd:.6f} > "
                                f"${self._settings.phase5_same_model_budget_cap_usd:.2f}."
                            )
                    elif allocation_category == "david_vs_goliath":
                        cat_row = conn.execute(
                            """
                            SELECT COALESCE(SUM(
                                CASE
                                    WHEN status = 'reserved' THEN reserved_cost_usd
                                    WHEN status IN ('reconciled', 'unknown')
                                        THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                    ELSE 0.0
                                END
                            ), 0.0) AS cat_committed
                            FROM canonical_budget_ledger
                            WHERE allocation_category = 'david_vs_goliath' AND provider != 'ollama'
                            """
                        ).fetchone()
                        cat_committed = float(cat_row["cat_committed"]) if cat_row else 0.0
                        if (
                            cat_committed + estimated_cost_usd
                            > self._settings.david_vs_goliath_budget_cap_usd + 1e-12
                        ):
                            raise BudgetConstraintError(
                                f"David vs Goliath allocation cap exceeded: "
                                f"${cat_committed:.6f} + ${estimated_cost_usd:.6f} > "
                                f"${self._settings.david_vs_goliath_budget_cap_usd:.2f}."
                            )

                conn.execute(
                    """
                    INSERT INTO canonical_budget_ledger (
                        reservation_id, run_id, provider, model, status,
                        reserved_cost_usd, actual_cost_usd, reserved_calls,
                        retries, allocation_category, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'reserved', ?, NULL, ?, 0, ?, NULL, ?, ?)
                    ON CONFLICT(reservation_id) DO UPDATE SET
                        reserved_cost_usd = excluded.reserved_cost_usd,
                        reserved_calls = excluded.reserved_calls,
                        allocation_category = excluded.allocation_category,
                        updated_at = excluded.updated_at
                    """,
                    (
                        reservation_id,
                        run_id,
                        provider,
                        model,
                        estimated_cost_usd,
                        call_units,
                        allocation_category,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def reconcile_sync(
        self,
        *,
        reservation_id: str,
        status: str,
        actual_cost_usd: float | None,
        retries: int = 0,
        notes: str | None = None,
    ) -> None:
        """Update reservation outcome atomically in the canonical ledger."""
        now = datetime.now(UTC).isoformat()
        with self._db_lock:
            conn = sqlite3.connect(str(self.authority_path))
            try:
                conn.execute(
                    """
                    UPDATE canonical_budget_ledger
                    SET status = ?, actual_cost_usd = ?, retries = ?, notes = ?, updated_at = ?
                    WHERE reservation_id = ?
                    """,
                    (status, actual_cost_usd, retries, notes, now, reservation_id),
                )
                conn.commit()
            finally:
                conn.close()

    def get_summary_sync(self) -> dict[str, Any]:
        """Compute canonical spending totals, reserve health, and category breakdown."""
        with self._db_lock:
            conn = sqlite3.connect(str(self.authority_path))
            conn.row_factory = sqlite3.Row
            try:
                totals = conn.execute(
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
                            CASE WHEN status = 'reconciled'
                                THEN COALESCE(actual_cost_usd, 0.0)
                                ELSE 0.0
                            END
                        ), 0.0) AS reconciled_cost,
                        COALESCE(SUM(
                            CASE WHEN status = 'unknown'
                                THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                ELSE 0.0
                            END
                        ), 0.0) AS unknown_cost,
                        COALESCE(SUM(reserved_calls), 0) AS total_calls,
                        COUNT(*) AS total_records
                    FROM canonical_budget_ledger
                    WHERE provider != 'ollama'
                    """
                ).fetchone()

                cat_rows = conn.execute(
                    """
                    SELECT
                        allocation_category,
                        COUNT(*) AS cnt,
                        COALESCE(SUM(
                            CASE
                                WHEN status = 'reserved' THEN reserved_cost_usd
                                WHEN status IN ('reconciled', 'unknown')
                                    THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                                ELSE 0.0
                            END
                        ), 0.0) AS committed
                    FROM canonical_budget_ledger
                    WHERE provider != 'ollama'
                    GROUP BY allocation_category
                    """
                ).fetchall()

                category_spending = {
                    r["allocation_category"]: round(float(r["committed"]), 6) for r in cat_rows
                }

                committed = float(totals["committed_cost"]) if totals else 0.0
                reconciled = float(totals["reconciled_cost"]) if totals else 0.0
                unknown = float(totals["unknown_cost"]) if totals else 0.0
                calls = int(totals["total_calls"]) if totals else 0
                records = int(totals["total_records"]) if totals else 0

                max_spendable = max(
                    0.0,
                    self._settings.total_budget_ceiling_usd - self._settings.protected_reserve_usd,
                )
                spendable_remaining = max(0.0, max_spendable - committed)
                protected_reserve = self._settings.protected_reserve_usd

                return {
                    "authority_db": str(self.authority_path),
                    "total_budget_ceiling_usd": self._settings.total_budget_ceiling_usd,
                    "protected_reserve_usd": protected_reserve,
                    "max_spendable_usd": max_spendable,
                    "cumulative_committed_cost_usd": round(committed, 6),
                    "cumulative_reconciled_cost_usd": round(reconciled, 6),
                    "cumulative_unknown_cost_usd": round(unknown, 6),
                    "spendable_remaining_usd": round(spendable_remaining, 6),
                    "cumulative_dispatched_calls": calls,
                    "canonical_ledger_records": records,
                    "category_spending": category_spending,
                }
            finally:
                conn.close()

    async def reserve(
        self,
        *,
        reservation_id: str,
        run_id: str,
        provider: str,
        model: str,
        is_paid_provider: bool,
        estimated_cost_usd: float,
        call_units: int = 1,
        allocation_category: str = "general",
    ) -> None:
        await asyncio.to_thread(
            self.reserve_sync,
            reservation_id=reservation_id,
            run_id=run_id,
            provider=provider,
            model=model,
            is_paid_provider=is_paid_provider,
            estimated_cost_usd=estimated_cost_usd,
            call_units=call_units,
            allocation_category=allocation_category,
        )

    async def reconcile(
        self,
        *,
        reservation_id: str,
        status: str,
        actual_cost_usd: float | None,
        retries: int = 0,
        notes: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.reconcile_sync,
            reservation_id=reservation_id,
            status=status,
            actual_cost_usd=actual_cost_usd,
            retries=retries,
            notes=notes,
        )

    async def get_summary(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.get_summary_sync)


def estimate_token_count(text: str) -> int:
    """Conservative character-based token estimate (~3 chars/token + overhead)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3)) + 16


def serialize_schema_for_estimation(response_model: type[BaseModel] | None) -> str:
    """Serialize Pydantic JSON schema so structured-output prompt overhead is budgeted."""
    if response_model is None:
        return ""
    return json.dumps(response_model.model_json_schema(), separators=(",", ":"), sort_keys=True)


class BudgetManager:
    """Enforces atomic pre-call cost/call reservations and per-attempt reconciliation."""

    def __init__(
        self,
        database: Database,
        settings: CrewForgeSettings,
        pricing_overrides: dict[str, ModelPricing] | None = None,
        authority: UnifiedBudgetAuthority | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._pricing: dict[str, ModelPricing] = dict(settings.pricing_json)
        if pricing_overrides:
            self._pricing.update(pricing_overrides)
        self._authority = authority or UnifiedBudgetAuthority.get_instance(settings=settings)

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Return configured pricing for a model if available."""
        return self._pricing.get(model)

    def estimate_worst_case_cost_usd(
        self,
        *,
        provider: str,
        model: str,
        is_paid_provider: bool,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> float:
        """Compute conservative pre-call USD cost reservation.

        Fails closed with UnknownModelPricingError for paid providers without
        explicit pricing configuration. Local/free providers reserve $0.00.
        """
        if not is_paid_provider:
            return 0.0

        pricing = self.get_pricing(model)
        if pricing is None:
            raise UnknownModelPricingError(
                f"Fail-closed budget protection: pricing is unknown for paid model '{model}' "
                f"(provider='{provider}'). Configure CREWFORGE_PRICING_JSON before invoking."
            )

        # Conservative 25% + 64-token margin on estimated input tokens plus full max_output_tokens.
        buffered_input = math.ceil(estimated_input_tokens * 1.25) + 64
        return pricing.estimate_cost_usd(buffered_input, max_output_tokens)

    def compute_actual_cost_usd(
        self,
        *,
        model: str,
        is_paid_provider: bool,
        usage: UsageMetrics,
        reserved_cost_usd: float = 0.0,
    ) -> tuple[float, bool]:
        """Calculate actual cost from observed usage.

        Returns (cost_usd, usage_was_known). If a paid provider omits token usage,
        retains reserved_cost_usd rather than assuming zero cost.
        """
        if not is_paid_provider:
            return 0.0, usage.usage_reported
        if usage.actual_cost_usd is not None:
            return usage.actual_cost_usd, True
        if not usage.usage_reported:
            return reserved_cost_usd, False
        pricing = self.get_pricing(model)
        if pricing is None:
            raise UnknownModelPricingError(
                f"Cannot reconcile actual cost without pricing for paid model '{model}'."
            )
        return pricing.estimate_cost_usd(usage.input_tokens, usage.output_tokens), True

    async def reserve_call(
        self,
        *,
        run_id: str,
        provider: str,
        model: str,
        is_paid_provider: bool,
        prompt_text: str,
        system_prompt: str | None = None,
        response_model: type[BaseModel] | None = None,
        response_schema_json: str | None = None,
        max_output_tokens: int | None = None,
        call_units: int = 1,
        allocation_category: str = "general",
    ) -> BudgetReservation:
        """Atomically check run/cumulative caps and persist a single-attempt reservation."""
        if is_paid_provider and not self._settings.allow_paid_runs:
            raise BudgetError(
                "Paid model execution is blocked by default. Set CREWFORGE_ALLOW_PAID_RUNS=true "
                "explicitly to authorize paid API expenditure."
            )
        max_out = max_output_tokens or self._settings.max_output_tokens
        schema_text = (
            response_schema_json
            if response_schema_json is not None
            else serialize_schema_for_estimation(response_model)
        )
        combined_input = "\n".join(
            part for part in (system_prompt or "", prompt_text, schema_text) if part
        )
        estimated_input = estimate_token_count(combined_input)

        reserved_cost = self.estimate_worst_case_cost_usd(
            provider=provider,
            model=model,
            is_paid_provider=is_paid_provider,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_out,
        )

        reservation = BudgetReservation(
            run_id=run_id,
            provider=provider,
            model=model,
            is_paid_provider=is_paid_provider,
            reserved_cost_usd=reserved_cost,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_out,
            call_units=call_units,
            allocation_category=allocation_category,
        )

        # 1. Authority-level check across all trace databases and protected reserve
        if is_paid_provider:
            try:
                await self._authority.reserve(
                    reservation_id=reservation.reservation_id,
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    is_paid_provider=True,
                    estimated_cost_usd=reserved_cost,
                    call_units=call_units,
                    allocation_category=allocation_category,
                )
            except BudgetConstraintError as exc:
                raise BudgetExceededError(str(exc)) from exc

        # 2. Local trace-level reservation
        try:
            await self._db.reserve_budget(
                reservation_id=reservation.reservation_id,
                run_id=run_id,
                provider=provider,
                model=model,
                estimated_cost_usd=reserved_cost,
                max_model_calls_per_run=self._settings.max_model_calls_per_run,
                max_cost_per_run_usd=self._settings.max_cost_per_run_usd,
                max_cumulative_cost_usd=self._settings.max_cumulative_cost_usd,
                call_units=call_units,
            )
        except BudgetConstraintError as exc:
            if is_paid_provider:
                await self._authority.reconcile(
                    reservation_id=reservation.reservation_id,
                    status="released",
                    actual_cost_usd=0.0,
                    notes="Rejected by run-local budget caps.",
                )
            raise BudgetExceededError(str(exc)) from exc

        return reservation

    async def reconcile_success(
        self,
        reservation: BudgetReservation,
        usage: UsageMetrics,
        *,
        operation: str = "generate",
    ) -> UsageMetrics:
        """Reconcile a completed attempt with actual usage (or conservative fallback if missing)."""
        actual_cost, usage_known = self.compute_actual_cost_usd(
            model=reservation.model,
            is_paid_provider=reservation.is_paid_provider,
            usage=usage,
            reserved_cost_usd=reservation.reserved_cost_usd,
        )
        status = "reconciled" if (not reservation.is_paid_provider or usage_known) else "unknown"
        outcome = "completed" if usage_known else "completed_unknown_usage"
        notes = (
            None
            if usage_known
            else "Provider omitted usage metrics; retained conservative reserved cost."
        )

        reconciled_usage = usage.model_copy(
            update={
                "estimated_cost_usd": reservation.reserved_cost_usd,
                "actual_cost_usd": actual_cost,
                "model_calls": max(1, usage.model_calls),
                "successful_operations": 1,
                "usage_reported": usage.usage_reported,
            }
        )
        await self._db.reconcile_budget(
            reservation_id=reservation.reservation_id,
            run_id=reservation.run_id,
            provider=reservation.provider,
            model=reservation.model,
            operation=operation,
            status=status,
            actual_cost_usd=actual_cost,
            input_tokens=reconciled_usage.input_tokens,
            output_tokens=reconciled_usage.output_tokens,
            total_tokens=reconciled_usage.total_tokens,
            estimated_cost_usd=reservation.reserved_cost_usd,
            retries=reconciled_usage.retries,
            latency_ms=reconciled_usage.latency_ms,
            outcome=outcome,
            notes=notes,
        )
        if reservation.is_paid_provider:
            await self._authority.reconcile(
                reservation_id=reservation.reservation_id,
                status=status,
                actual_cost_usd=actual_cost,
                retries=reconciled_usage.retries,
                notes=notes,
            )
        return reconciled_usage

    async def reconcile_failure(
        self,
        reservation: BudgetReservation,
        *,
        billing_known_unbilled: bool,
        usage: UsageMetrics | None = None,
        retries: int = 0,
        latency_ms: float = 0.0,
        operation: str = "generate",
        reason: str = "",
    ) -> None:
        """Reconcile a failed or cancelled attempt.

        - If billing is known unbilled: releases reservation ($0.00).
        - If actual provider usage is reported on the failed call: reconciles exact
          token counts and monetary cost while keeping outcome='failed' (never
          incrementing successful_operations).
        - If usage is unavailable: conservatively retains reserved cost as unknown.
        """
        in_tok = 0
        out_tok = 0
        tot_tok = 0
        if billing_known_unbilled:
            status = "released"
            actual_cost = 0.0
            outcome = "failed"
        elif usage is not None and usage.usage_reported:
            status = "reconciled"
            actual_cost, _ = self.compute_actual_cost_usd(
                model=reservation.model,
                is_paid_provider=reservation.is_paid_provider,
                usage=usage,
                reserved_cost_usd=reservation.reserved_cost_usd,
            )
            outcome = "failed"
            in_tok = usage.input_tokens
            out_tok = usage.output_tokens
            tot_tok = usage.total_tokens
        else:
            status = "unknown"
            actual_cost = reservation.reserved_cost_usd
            outcome = "unknown_billing"

        safe_reason = sanitize_error_message(reason) if reason else None
        await self._db.reconcile_budget(
            reservation_id=reservation.reservation_id,
            run_id=reservation.run_id,
            provider=reservation.provider,
            model=reservation.model,
            operation=operation,
            status=status,
            actual_cost_usd=actual_cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=tot_tok,
            estimated_cost_usd=reservation.reserved_cost_usd,
            retries=retries,
            latency_ms=latency_ms,
            outcome=outcome,
            notes=safe_reason,
        )
        if reservation.is_paid_provider:
            await self._authority.reconcile(
                reservation_id=reservation.reservation_id,
                status=status,
                actual_cost_usd=actual_cost,
                retries=retries,
                notes=safe_reason,
            )

    async def get_summary(self, run_id: str | None = None) -> dict[str, object]:
        """Return persistent budget summary for a run and cumulatively."""
        local_sum = await self._db.get_budget_summary(run_id=run_id)
        auth_sum = await self._authority.get_summary()
        local_sum["unified_authority"] = auth_sum
        return local_sum
