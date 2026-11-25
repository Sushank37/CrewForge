"""Tests for single-agent baseline execution, run isolation, and trace retrieval."""

from __future__ import annotations

from pathlib import Path

from crewforge.config import CrewForgeSettings
from crewforge.core.baseline import BaselineRunner
from crewforge.core.budget import BudgetManager
from crewforge.providers.base import ProviderTimeoutError
from crewforge.schemas import RunStatus
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider


async def test_baseline_execution_and_run_isolation(tmp_path: Path) -> None:
    db_path = tmp_path / "baseline.db"
    settings = CrewForgeSettings(_env_file=None, db_path=db_path)
    db = Database(db_path)
    await db.initialize()

    try:
        provider = FakeProvider(provider_name="fake", model_name="fake-baseline-model")
        budget = BudgetManager(db, settings)
        runner = BaselineRunner(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
        )

        res1 = await runner.run("Prompt alpha")
        res2 = await runner.run("Prompt beta")

        assert res1.run_id != res2.run_id
        assert res1.status == RunStatus.COMPLETED
        assert res2.status == RunStatus.COMPLETED
        assert "Prompt alpha" in res1.answer
        assert "Prompt beta" in res2.answer
        assert res1.missing_information
        assert res1.usage.total_tokens == 66
        assert res1.latency_ms >= 0.0

        trace1 = await db.get_run_trace(res1.run_id)
        trace2 = await db.get_run_trace(res2.run_id)
        assert trace1 is not None
        assert trace2 is not None
        assert trace1.prompt == "Prompt alpha"
        assert trace2.prompt == "Prompt beta"
        assert [e.seq for e in trace1.events] == [1, 2, 3, 4]
        assert [e.seq for e in trace2.events] == [1, 2, 3, 4]
        assert len(trace1.usage_records) == 1
        assert len(trace2.usage_records) == 1
        assert await db.get_run_trace("run_nonexistent") is None
    finally:
        await db.close()


async def test_baseline_records_provider_failure_in_trace(tmp_path: Path) -> None:
    db_path = tmp_path / "baseline_fail.db"
    settings = CrewForgeSettings(_env_file=None, db_path=db_path, max_retries=0)
    db = Database(db_path)
    await db.initialize()

    try:
        def timeout_handler(_p: str, _m: object) -> None:
            raise ProviderTimeoutError(
                "simulated timeout",
                provider="fake",
                model="fake-model",
                retries_attempted=0,
            )

        provider = FakeProvider(max_retries=0, handler=timeout_handler)
        budget = BudgetManager(db, settings)
        runner = BaselineRunner(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
        )

        res = await runner.run("Prompt timeout")
        assert res.status == RunStatus.FAILED
        assert res.error is not None
        assert "simulated timeout" in res.error

        trace = await db.get_run_trace(res.run_id)
        assert trace is not None
        assert trace.status == RunStatus.FAILED
        assert trace.events[-1].event_type == "model_call_failed"
        assert trace.budget_reservations[0]["status"] == "unknown"
    finally:
        await db.close()
