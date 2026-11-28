"""Tests for persistent budget manager, ModelExecutor defects, and SQLite migrations."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings, ModelPricing
from crewforge.core.budget import (
    BudgetExceededError,
    BudgetManager,
    UnknownModelPricingError,
    serialize_schema_for_estimation,
)
from crewforge.core.executor import ModelExecutor
from crewforge.providers.base import ProviderRateLimitError, ProviderTimeoutError
from crewforge.schemas import RunConfig, UsageMetrics
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider


class _SmallSchema(BaseModel):
    value: str


class _LargeSchema(BaseModel):
    alpha: str
    beta: list[str]
    gamma: dict[str, int]
    delta: float | None = None
    epsilon_description_field: str


def _make_settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "allow_paid_runs": True,
        "max_cost_per_run_usd": 0.05,
        "max_cumulative_cost_usd": 0.10,
        "max_model_calls_per_run": 5,
        "max_output_tokens": 1000,
        "max_retries": 0,
        "pricing_json": {
            "paid-test-model": ModelPricing(input_per_1m=1.0, output_per_1m=10.0),
        },
    }
    defaults.update(overrides)
    return CrewForgeSettings(**defaults)  # type: ignore[arg-type]


async def _register_run(db: Database, run_id: str, settings: CrewForgeSettings) -> None:
    cfg: RunConfig = settings.to_run_config()
    await db.create_run(run_id=run_id, mode="baseline", prompt="budget test", config=cfg)


async def test_paid_model_fails_closed_without_pricing(tmp_path: Path) -> None:
    db_path = tmp_path / "budget.db"
    settings = _make_settings(db_path, pricing_json={})
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_fail_closed", settings)
        budget = BudgetManager(db, settings)
        with pytest.raises(UnknownModelPricingError, match="Fail-closed"):
            await budget.reserve_call(
                run_id="run_fail_closed",
                provider="openai",
                model="unpriced-model",
                is_paid_provider=True,
                prompt_text="hello",
            )
    finally:
        await db.close()


async def test_concurrent_reservations_atomic_enforcement(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    settings = _make_settings(
        db_path,
        max_cost_per_run_usd=0.04,
        max_cumulative_cost_usd=0.10,
        max_model_calls_per_run=16,
        max_output_tokens=1000,
    )
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_conc", settings)
        budget = BudgetManager(db, settings)

        async def try_reserve() -> str:
            res = await budget.reserve_call(
                run_id="run_conc",
                provider="openai",
                model="paid-test-model",
                is_paid_provider=True,
                prompt_text="short prompt",
            )
            return res.reservation_id

        results = await asyncio.gather(*(try_reserve() for _ in range(10)), return_exceptions=True)
        succeeded = [r for r in results if isinstance(r, str)]
        rejected = [r for r in results if isinstance(r, BudgetExceededError)]
        assert len(succeeded) == 3
        assert len(rejected) == 7
    finally:
        await db.close()


async def test_persistent_budget_reconciliation_and_unknown_outcome(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.db"
    settings = _make_settings(
        db_path,
        max_cost_per_run_usd=0.025,
        max_cumulative_cost_usd=0.025,
        max_model_calls_per_run=10,
        max_output_tokens=1000,
    )

    db1 = Database(db_path)
    await db1.initialize()
    try:
        await _register_run(db1, "run_1", settings)
        budget1 = BudgetManager(db1, settings)

        res1 = await budget1.reserve_call(
            run_id="run_1",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="hello",
        )
        usage1 = await budget1.reconcile_success(
            res1,
            UsageMetrics(input_tokens=100, output_tokens=10, retries=1, latency_ms=25.0),
        )
        assert usage1.actual_cost_usd == pytest.approx(0.0002)

        res2 = await budget1.reserve_call(
            run_id="run_1",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="hello",
        )
        await budget1.reconcile_failure(
            res2,
            billing_known_unbilled=False,
            retries=2,
            reason="timeout mid-flight",
        )

        res3 = await budget1.reserve_call(
            run_id="run_1",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="hello",
        )
        await budget1.reconcile_failure(
            res3,
            billing_known_unbilled=True,
            retries=0,
            reason="connection refused",
        )
    finally:
        await db1.close()

    db2 = Database(db_path)
    await db2.initialize()
    try:
        await _register_run(db2, "run_2", settings)
        budget2 = BudgetManager(db2, settings)
        summary = await budget2.get_summary()
        assert summary["cumulative_retries"] == 3
        assert summary["cumulative_unknown_cost_usd"] > 0.01
        assert summary["cumulative_committed_cost_usd"] == pytest.approx(
            summary["cumulative_reconciled_cost_usd"] + summary["cumulative_unknown_cost_usd"]
        )

        await budget2.reserve_call(
            run_id="run_2",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="hello",
        )
        with pytest.raises(BudgetExceededError, match="Cumulative budget limit exceeded"):
            await budget2.reserve_call(
                run_id="run_2",
                provider="openai",
                model="paid-test-model",
                is_paid_provider=True,
                prompt_text="hello",
            )
    finally:
        await db2.close()


async def test_local_provider_enforces_model_call_limit(tmp_path: Path) -> None:
    db_path = tmp_path / "local_calls.db"
    settings = _make_settings(db_path, max_model_calls_per_run=2)
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_local", settings)
        budget = BudgetManager(db, settings)
        for _ in range(2):
            res = await budget.reserve_call(
                run_id="run_local",
                provider="ollama",
                model="qwen3:4b",
                is_paid_provider=False,
                prompt_text="local prompt",
            )
            assert res.reserved_cost_usd == 0.0
            await budget.reconcile_success(res, UsageMetrics(input_tokens=20, output_tokens=15))

        with pytest.raises(BudgetExceededError, match="model-call limit exceeded"):
            await budget.reserve_call(
                run_id="run_local",
                provider="ollama",
                model="qwen3:4b",
                is_paid_provider=False,
                prompt_text="third call",
            )
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Phase 2 regression tests for Task 1 defects (A, B, C, D)
# ---------------------------------------------------------------------------


async def test_executor_reserves_each_retry_attempt_separately(tmp_path: Path) -> None:
    """Regression A: every retry attempt gets its own atomic reservation and reconciliation."""
    db_path = tmp_path / "retry_reserve.db"
    settings = _make_settings(
        db_path,
        max_retries=2,
        max_model_calls_per_run=5,
        max_cost_per_run_usd=0.10,
        max_cumulative_cost_usd=0.20,
    )
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_retry_sep", settings)
        attempts = 0

        def handler(_p: str, _m: type[BaseModel] | None) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderTimeoutError(
                    "first timeout", provider="openai", model="paid-test-model"
                )
            if attempts == 2:
                raise ProviderRateLimitError(
                    "rate limited", provider="openai", model="paid-test-model"
                )
            return {"value": "recovered"}

        # Intentionally set provider internal max_retries=5 to prove ModelExecutor overrides to 0.
        provider = FakeProvider(
            provider_name="openai",
            model_name="paid-test-model",
            is_paid_provider=True,
            max_retries=5,
            handler=handler,
        )
        budget = BudgetManager(db, settings)
        executor = ModelExecutor(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
            base_backoff_seconds=0.005,
        )

        res = await executor.execute_structured(
            run_id="run_retry_sep",
            actor="worker_1",
            operation="worker",
            prompt="recover on third try",
            response_model=_SmallSchema,
        )
        assert res.parsed.value == "recovered"
        assert attempts == 3

        trace = await db.get_run_trace("run_retry_sep")
        assert trace is not None
        assert len(trace.budget_reservations) == 3
        statuses = [r["status"] for r in trace.budget_reservations]
        assert statuses == ["unknown", "released", "reconciled"]

        agg = await db.aggregate_run_usage("run_retry_sep")
        assert agg.model_calls == 3
        assert agg.successful_operations == 1
    finally:
        await db.close()


async def test_released_unbilled_calls_still_count_toward_call_limit(tmp_path: Path) -> None:
    """Regression B: released (known-unbilled) attempts count against max_model_calls_per_run."""
    db_path = tmp_path / "released_calls.db"
    settings = _make_settings(db_path, max_model_calls_per_run=2, max_retries=2)
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_released_cap", settings)

        def always_rate_limit(_p: str, _m: type[BaseModel] | None) -> Any:
            raise ProviderRateLimitError("429", provider="openai", model="paid-test-model")

        provider = FakeProvider(
            provider_name="openai",
            model_name="paid-test-model",
            is_paid_provider=True,
            handler=always_rate_limit,
        )
        budget = BudgetManager(db, settings)
        executor = ModelExecutor(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
            base_backoff_seconds=0.005,
        )

        with pytest.raises(BudgetExceededError, match="model-call limit exceeded"):
            await executor.execute_structured(
                run_id="run_released_cap",
                actor="baseline",
                operation="baseline",
                prompt="trigger rate limits",
                response_model=_SmallSchema,
            )

        summary = await db.get_budget_summary("run_released_cap")
        assert summary["run_committed_calls"] == 2
        assert summary["run_successful_calls"] == 0
    finally:
        await db.close()


async def test_missing_paid_provider_usage_never_creates_zero_cost(tmp_path: Path) -> None:
    """Regression C: missing usage on a paid provider retains reserved cost, not $0.00."""
    db_path = tmp_path / "missing_usage.db"
    settings = _make_settings(db_path)
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_missing_usage", settings)
        provider = FakeProvider(
            provider_name="openai",
            model_name="paid-test-model",
            is_paid_provider=True,
            usage_reported=False,
            handler=lambda _p, _m: {"value": "ok"},
        )
        budget = BudgetManager(db, settings)
        executor = ModelExecutor(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
        )

        res = await executor.execute_structured(
            run_id="run_missing_usage",
            actor="baseline",
            operation="baseline",
            prompt="missing usage test",
            response_model=_SmallSchema,
        )
        assert res.usage.usage_reported is False
        assert res.usage.actual_cost_usd is not None
        assert res.usage.actual_cost_usd > 0.0
        assert res.usage.actual_cost_usd == pytest.approx(res.usage.estimated_cost_usd)

        trace = await db.get_run_trace("run_missing_usage")
        assert trace is not None
        assert trace.usage_records[0]["outcome"] == "completed_unknown_usage"
        assert trace.budget_reservations[0]["status"] == "unknown"
    finally:
        await db.close()


async def test_input_estimate_includes_response_schema_overhead(tmp_path: Path) -> None:
    """Regression D: larger structured-output schemas increase reserved token & USD estimates."""
    db_path = tmp_path / "schema_est.db"
    settings = _make_settings(db_path, max_cost_per_run_usd=0.09, max_cumulative_cost_usd=0.10)
    db = Database(db_path)
    await db.initialize()
    try:
        await _register_run(db, "run_schema_est", settings)
        budget = BudgetManager(db, settings)

        small_res = await budget.reserve_call(
            run_id="run_schema_est",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="prompt",
            response_model=_SmallSchema,
        )
        large_res = await budget.reserve_call(
            run_id="run_schema_est",
            provider="openai",
            model="paid-test-model",
            is_paid_provider=True,
            prompt_text="prompt",
            response_model=_LargeSchema,
        )
        assert len(serialize_schema_for_estimation(_LargeSchema)) > len(
            serialize_schema_for_estimation(_SmallSchema)
        )
        assert large_res.estimated_input_tokens > small_res.estimated_input_tokens
        assert large_res.reserved_cost_usd > small_res.reserved_cost_usd
    finally:
        await db.close()


async def test_phase1_database_migrates_cleanly_to_phase2(tmp_path: Path) -> None:
    """Verify existing Phase 1 SQLite databases remain readable and migrate to v2."""
    db_path = tmp_path / "phase1_legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE runs (
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
        CREATE TABLE execution_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            actor TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
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
        CREATE TABLE usage_records (
            usage_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            reservation_id TEXT,
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
        INSERT INTO runs (
            run_id, mode, status, prompt, provider, model, config_json, created_at
        ) VALUES (
            'run_legacy_1',
            'baseline',
            'completed',
            'legacy prompt',
            'ollama',
            'qwen3:4b',
            '{"provider":"ollama","model":"qwen3:4b","request_timeout_seconds":60.0,"max_retries":0,"max_concurrency":1,"max_output_tokens":1024,"max_model_calls_per_run":16,"max_cost_per_run_usd":0.05,"max_cumulative_cost_usd":1.0}',
            '2026-11-25T10:00:00+00:00'
        );
        """
    )
    conn.close()

    db = Database(db_path)
    await db.initialize()
    try:
        assert await db.get_schema_version() == 5
        trace = await db.get_run_trace("run_legacy_1")
        assert trace is not None
        assert trace.prompt == "legacy prompt"
        assert trace.plan is None
        assert trace.claims == []
    finally:
        await db.close()
