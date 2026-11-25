"""Tests for persistent, concurrency-safe SQLite budget manager."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from crewforge.config import CrewForgeSettings, ModelPricing
from crewforge.core.budget import (
    BudgetExceededError,
    BudgetManager,
    UnknownModelPricingError,
)
from crewforge.schemas import RunConfig, UsageMetrics
from crewforge.storage.database import Database


def _make_settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "max_cost_per_run_usd": 0.05,
        "max_cumulative_cost_usd": 0.10,
        "max_model_calls_per_run": 5,
        "max_output_tokens": 1000,
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
    # Each reservation with 1000 output tokens at $10/1M costs ~$0.01+ so at most 3 fit under $0.04.
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
    # Reservation cost per call is ~0.0101; cumulative cap 0.025 allows 2 active reservations.
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

        # Call 1 succeeds with tiny actual usage (~$0.0002), freeing most of the reservation.
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

        # Call 2 fails with unknown billing outcome -> retains full reserved cost conservatively.
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

        # Call 3 fails before request dispatch -> released to $0.00.
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

    # Re-open database in a brand-new Database + BudgetManager instance to verify persistence.
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

        # Trying two more reservations in run_2 should hit cumulative cap because run_1's
        # reconciled + unknown cost (~0.0103) persists across restarts.
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
