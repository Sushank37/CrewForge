"""Offline regression tests for Groq quota-aware scheduling and arithmetic grounding."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.providers.base import ProviderRateLimitError
from crewforge.providers.groq_quota import (
    extract_retry_after_seconds,
    get_groq_scheduler,
    reset_groq_schedulers,
    run_deadline_var,
    wait_observer_var,
)
from crewforge.providers.openai_provider import OpenAIProvider
from crewforge.schemas import (
    ArithmeticRequest,
    Claim,
    ClaimCategory,
    ClaimStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import (
    DeterministicArithmeticTool,
    build_provenance_catalog,
    extract_numbers_from_text,
)


def test_groq_quota_scheduler_shared_and_counts_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify shared Groq scheduler parses reset hints, notifies UI wait, and counts attempts."""
    reset_groq_schedulers()

    # Shared across callers (Chat + MCP) for the same endpoint & model
    s1 = get_groq_scheduler(base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b")
    s2 = get_groq_scheduler(base_url="https://api.groq.com/openai/v1/", model="openai/gpt-oss-120b")
    assert s1 is s2

    msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000, Used 7153, "
        "Requested 3890. Please try again in 22.8225s.'}}"
    )
    assert extract_retry_after_seconds({"x-ratelimit-reset-tokens": "12.5s"}, msg) == pytest.approx(
        22.8225
    )

    provider = OpenAIProvider(
        api_key="gsk-test",
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        use_chat_completions=True,
        is_free_tier=True,
        provider_label="openai_compatible",
    )

    waits: list[tuple[float, str]] = []
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        # Advance scheduler's next_available_at as if time elapsed
        s1._next_available_at = 0.0

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    attempts = 0

    async def flaky_call() -> tuple[SimpleNamespace, dict[str, str]]:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ProviderRateLimitError(
                "Please try again in 3.5s",
                provider="openai_compatible",
                model="openai/gpt-oss-120b",
                response_headers={"retry-after": "3.5"},
            )
        resp = SimpleNamespace(
            id="resp_1",
            model="openai/gpt-oss-120b",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        return resp, {"x-ratelimit-limit-tokens": "8000"}

    token_obs = wait_observer_var.set(lambda w, r: waits.append((w, r)))
    token_dl = run_deadline_var.set(time.monotonic() + 120.0)
    try:
        (resp, _hdrs), retries, _lat = asyncio.run(
            provider._run_with_rate_limit_backoff(flaky_call, timeout_seconds=10.0, retries_cap=0)
        )
    finally:
        wait_observer_var.reset(token_obs)
        run_deadline_var.reset(token_dl)

    assert attempts == 3
    assert retries == 2
    assert len(waits) == 2
    assert len(sleep_calls) == 2
    usage = provider._extract_usage(resp, retries, 100.0)
    assert usage.retries == 2
    assert usage.model_calls == 3


def test_groq_quota_scheduler_respects_deadline() -> None:
    """Scheduler must not sleep past run_deadline_var."""
    reset_groq_schedulers()
    sched = get_groq_scheduler(
        base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b"
    )
    sched.record_rate_limit(message="Please try again in 45s")
    with pytest.raises(TimeoutError):
        asyncio.run(sched.wait_for_slot(1000, deadline_monotonic=time.monotonic() + 5.0))


def test_arithmetic_grounding_words_dependencies_and_invalid_cases(tmp_path) -> None:
    """Verify word numbers, verified claim dependencies, conversions, and strict rejections."""

    async def _run() -> None:
        db = Database(tmp_path / "grounding.db")
        await db.initialize()
        await db.create_run("run_test", "network", "test", CrewForgeSettings().to_run_config())
        try:
            tool = DeterministicArithmeticTool(db)
            prompt = (
                "A center processes 450 orders per hour over an eight-hour shift, "
                "with one and a half hours of maintenance and 850,000 requests at "
                "$0.12 per 1,000 requests."
            )
            nums = extract_numbers_from_text(prompt)
            assert any(str(n) == "8" for n in nums)
            assert any(str(n) == "1.5" for n in nums)

            # 1. Valid: word numbers ("eight", "one and a half")
            rec_words = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="rate * (shift - maint)",
                    variables={"rate": "450", "shift": "8", "maint": "1.5"},
                    input_provenance_refs={"rate": "up_1", "shift": "up_1", "maint": "up_1"},
                    proposed_result="2925",
                    units="orders",
                ),
                grounded_texts=[prompt],
            )
            assert rec_words.success is True
            assert rec_words.inputs_grounded is True
            assert rec_words.has_ungrounded_literals is False

            # 2. Valid: legitimate unit conversion constant (1000 with "per 1,000" in prompt)
            rec_conv = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="(requests / 1000) * price",
                    variables={"requests": "850000", "price": "0.12"},
                    input_provenance_refs={"requests": "up_1", "price": "up_1"},
                    proposed_result="102",
                    units="USD",
                ),
                grounded_texts=[prompt],
            )
            assert rec_conv.success is True
            assert rec_conv.inputs_grounded is True
            assert rec_conv.has_ungrounded_literals is False

            # 3. Valid: dependency on an ASSERTED claim backed by a grounded ToolExecutionRecord
            verified_asserted = Claim(
                claim_id="clm_verified_gross",
                run_id="run_test",
                version=1,
                statement="Net capacity is 2925 orders.",
                category=ClaimCategory.COMPUTATION,
                author_agent_id="a1",
                status=ClaimStatus.ASSERTED,
                tool_execution_ids=[rec_words.tool_execution_id],
            )
            unverified_asserted = Claim(
                claim_id="clm_unverified_fake",
                run_id="run_test",
                version=1,
                statement="Unverified guess is 500 orders.",
                category=ClaimCategory.COMPUTATION,
                author_agent_id="a1",
                status=ClaimStatus.ASSERTED,
                tool_execution_ids=[],
            )
            catalog = build_provenance_catalog(
                original_prompt=prompt,
                all_claims=[verified_asserted, unverified_asserted],
                tool_records=[rec_words, rec_conv],
            )
            rec_dep_ok = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="net - 0",
                    variables={"net": "2925"},
                    input_provenance_refs={"net": "clm_verified_gross"},
                    proposed_result="2925",
                    units="orders",
                ),
                catalog=catalog,
            )
            assert rec_dep_ok.inputs_grounded is True

            # 4. Invalid (must fail!): dependency on unverified ASSERTED claim with no tool record
            rec_dep_bad = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="guess + 10",
                    variables={"guess": "500"},
                    input_provenance_refs={"guess": "clm_unverified_fake"},
                    proposed_result="510",
                    units="orders",
                ),
                catalog=catalog,
            )
            assert rec_dep_bad.inputs_grounded is False

            # 5. Invalid (must fail!): arbitrary ungrounded inline constant (42)
            rec_inline_bad = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="rate * 42",
                    variables={"rate": "450"},
                    input_provenance_refs={"rate": "up_1"},
                    proposed_result="18900",
                    units="orders",
                ),
                catalog=catalog,
            )
            assert rec_inline_bad.has_ungrounded_literals is True

            # 6. Invalid (must fail!): fabricated number not in prompt
            rec_fab_bad = await tool.execute(
                run_id="run_test",
                actor="lead",
                request=ArithmeticRequest(
                    expression="bogus * 1",
                    variables={"bogus": "999"},
                    input_provenance_refs={"bogus": "up_1"},
                    proposed_result="999",
                    units="orders",
                ),
                catalog=catalog,
            )
            assert rec_fab_bad.inputs_grounded is False
        finally:
            await db.close()

    asyncio.run(_run())


def test_judge_byok_isolated_from_groq_free_tier_restrictions() -> None:
    """Verify BYOK / judge profiles are isolated from Groq free-tier scheduler and token caps."""
    reset_groq_schedulers()
    byok_groq = OpenAIProvider(
        api_key="gsk-paid-dev-tier",
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        use_chat_completions=True,
        is_free_tier=False,
        provider_label="openai_compatible",
    )
    assert byok_groq._is_groq is True
    assert byok_groq._is_groq_free_tier is False
    assert byok_groq._quota_scheduler is None
    assert byok_groq._effective_chat_max_tokens(8192) == 8192
    compact_full = byok_groq._compact_schema_json(ArithmeticRequest, strip_descriptions=False)
    assert "description" in compact_full


def test_invalid_settings_do_not_leave_chat_permanently_running(tmp_path) -> None:
    """Invalid settings must fail fast (400) and _execute_run must mark failed, never running."""
    from starlette.testclient import TestClient

    from crewforge.dashboard import create_dashboard_app
    from crewforge.web_api import ChatMessageRequest, JudgeSessionManager, JudgeSettingsUpdate

    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
        replay_dir="demo/traces",
        benchmark_path="demo/v2_reconciled_scores.json",
        require_auth=False,
    )
    with TestClient(app) as client:
        client.post(
            "/api/settings",
            json={
                "billing_mode": "free",
                "free_tier_verified": False,
                "provider": "openai_compatible",
                "model": "openai/gpt-oss-120b",
                "openai_base_url": "https://api.groq.com/openai/v1",
                "api_key": "gsk-test-1234",
            },
        )
        chat = client.post("/api/chats", json={"title": "Invalid Settings Test"}).json()
        res = client.post(
            f"/api/chats/{chat['chat_id']}/messages",
            json={"prompt": "Compute 2 + 2", "mode": "network"},
        )
        assert res.status_code == 400
        detail = client.get(f"/api/chats/{chat['chat_id']}").json()
        assert detail["messages"] == []

    # Also verify _execute_run itself marks assistant message failed if settings fail mid-task
    mgr = JudgeSessionManager(
        db_path=tmp_path / "mgr_session.db",
        budget_db_path=tmp_path / "mgr_budget.db",
    )
    mgr.update_settings(
        JudgeSettingsUpdate(
            billing_mode="free",
            free_tier_verified=True,
            provider="openai_compatible",
            model="openai/gpt-oss-120b",
            openai_base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test-1234",
        )
    )
    c2 = mgr.create_chat("Mid-run invalid")
    # Manually insert running assistant message then invalidate settings before _execute_run
    now = "2026-11-27T00:00:00Z"
    conn = mgr._connect()
    conn.execute(
        "INSERT INTO chat_messages (message_id, chat_id, role, mode, content, run_id, "
        "status, attachments_json, error, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "cmsg_test_asst",
            c2["chat_id"],
            "assistant",
            "network",
            "",
            "run_mid",
            "running",
            "[]",
            None,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    mgr._config["free_tier_verified"] = False
    asyncio.run(
        mgr._execute_run(
            run_id="run_mid",
            assistant_msg_id="cmsg_test_asst",
            prompt=ChatMessageRequest(prompt="test").prompt,
            mode="network",
            ref_docs=[],
        )
    )
    chat_after = mgr.get_chat(c2["chat_id"])
    assert chat_after is not None
    assert chat_after["messages"][0]["status"] == "failed"
    assert "verified" in (chat_after["messages"][0]["error"] or "").lower()


def test_engine_level_request_attempt_accounting(tmp_path) -> None:
    """Verify ModelExecutor + BudgetManager preserve provider rate-limit retries in model_calls."""
    from pydantic import BaseModel

    from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
    from crewforge.core.executor import ModelExecutor
    from crewforge.providers.base import StructuredGenerationResult
    from crewforge.schemas import UsageMetrics

    class DummyOut(BaseModel):
        value: int

    class RetryReportingProvider:
        provider_name = "openai_compatible"
        model_name = "openai/gpt-oss-120b"
        is_paid_provider = False

        async def generate_structured(self, *_args, **_kwargs):
            return StructuredGenerationResult(
                provider=self.provider_name,
                model=self.model_name,
                returned_model=self.model_name,
                raw_text='{"value": 42}',
                parsed=DummyOut(value=42),
                usage=UsageMetrics(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                    model_calls=3,
                    retries=2,
                    latency_ms=250.0,
                    usage_reported=True,
                ),
            )

    async def _run() -> None:
        settings = CrewForgeSettings(
            db_path=tmp_path / "eng.db",
            budget_authority_path=tmp_path / "eng_budget.db",
            provider="openai_compatible",
            model="openai/gpt-oss-120b",
            openai_base_url="https://api.groq.com/openai/v1",
            allow_paid_runs=False,
        )
        db = Database(tmp_path / "eng.db")
        await db.initialize()
        await db.create_run("run_eng", "network", "test", settings.to_run_config())
        auth = UnifiedBudgetAuthority(settings=settings, migrate_default_dbs=False)
        bm = BudgetManager(database=db, settings=settings, authority=auth)
        ex = ModelExecutor(
            provider=RetryReportingProvider(),  # type: ignore[arg-type]
            database=db,
            budget_manager=bm,
            settings=settings,
        )
        res = await ex.execute_structured(
            run_id="run_eng",
            actor="lead",
            operation="test_op",
            prompt="Return 42",
            response_model=DummyOut,
        )
        assert res.usage.retries == 2
        assert res.usage.model_calls == 3
        await db.close()

    asyncio.run(_run())


def test_negative_arithmetic_fabricated_numbers_and_constants(tmp_path) -> None:
    """Negative tests: revised grounding rules must reject fabricated words, constants, and refs."""

    async def _run() -> None:
        db = Database(tmp_path / "neg.db")
        await db.initialize()
        await db.create_run("run_neg", "network", "neg", CrewForgeSettings().to_run_config())
        try:
            tool = DeterministicArithmeticTool(db)
            prompt = "A warehouse ships 100 boxes over an eight-hour shift."

            # 1. Fabricated word number ('fifty' = 50 not in prompt)
            r1 = await tool.execute(
                run_id="run_neg",
                actor="lead",
                request=ArithmeticRequest(
                    expression="boxes * fabricated",
                    variables={"boxes": "100", "fabricated": "50"},
                    input_provenance_refs={"boxes": "up_1", "fabricated": "up_1"},
                    proposed_result="5000",
                    units="boxes",
                ),
                grounded_texts=[prompt],
            )
            assert r1.inputs_grounded is False

            # 2. Ungrounded conversion constant (1000 without any kilo/per-1000 context)
            r2 = await tool.execute(
                run_id="run_neg",
                actor="lead",
                request=ArithmeticRequest(
                    expression="boxes * 1000",
                    variables={"boxes": "100"},
                    input_provenance_refs={"boxes": "up_1"},
                    proposed_result="100000",
                    units="boxes",
                ),
                grounded_texts=[prompt],
            )
            assert r2.has_ungrounded_literals is True

            # 3. Fabricated sequential claim ref ('clm_1') with no prior valid tool record
            r3 = await tool.execute(
                run_id="run_neg",
                actor="lead",
                request=ArithmeticRequest(
                    expression="prev + 8",
                    variables={"prev": "800", "hours": "8"},
                    input_provenance_refs={"prev": "clm_1", "hours": "up_1"},
                    proposed_result="808",
                    units="boxes",
                ),
                grounded_texts=[prompt],
            )
            assert r3.inputs_grounded is False

            # 4. Rejected claim dependency must still be rejected
            rejected_claim = Claim(
                claim_id="clm_rej",
                run_id="run_neg",
                version=1,
                statement="Wrong total is 999 boxes.",
                category=ClaimCategory.COMPUTATION,
                author_agent_id="a1",
                status=ClaimStatus.REJECTED,
            )
            cat = build_provenance_catalog(original_prompt=prompt, all_claims=[rejected_claim])
            r4 = await tool.execute(
                run_id="run_neg",
                actor="lead",
                request=ArithmeticRequest(
                    expression="bad + 1",
                    variables={"bad": "999"},
                    input_provenance_refs={"bad": "clm_rej"},
                    proposed_result="1000",
                    units="boxes",
                ),
                catalog=cat,
            )
            assert r4.inputs_grounded is False
        finally:
            await db.close()

    asyncio.run(_run())
