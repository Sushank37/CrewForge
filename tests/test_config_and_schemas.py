"""Tests for CrewForge configuration and Pydantic v2 data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crewforge.config import CrewForgeSettings, ModelPricing, ProviderType
from crewforge.schemas import (
    AgentMessage,
    AgentSpec,
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceReliability,
    ExecutionEvent,
    ExecutionPlan,
    ReviewDecision,
    ReviewVerdict,
    RunConfig,
    RunResult,
    RunStatus,
    TaskSpec,
    TaskStatus,
    UsageMetrics,
)


def test_settings_defaults_and_placeholder_key_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWFORGE_OPENAI_API_KEY", "your-openai-api-key-here")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = CrewForgeSettings(_env_file=None)

    assert settings.provider == ProviderType.OLLAMA
    assert settings.model == "qwen3:4b"
    assert settings.openai_api_key is None
    assert settings.max_cost_per_run_usd == pytest.approx(0.05)
    assert settings.max_cumulative_cost_usd == pytest.approx(1.00)
    assert settings.max_model_calls_per_run == 16
    assert settings.max_concurrency == 3
    assert settings.effective_max_concurrency == 1
    assert settings.max_retries == 0

    cloud_settings = CrewForgeSettings(_env_file=None, provider=ProviderType.OPENAI)
    assert cloud_settings.effective_max_concurrency == 3


def test_settings_pricing_json_and_cost_cap_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CREWFORGE_PRICING_JSON",
        '{"test-model": {"input_per_1m": 0.5, "output_per_1m": 2.0}}',
    )
    settings = CrewForgeSettings(_env_file=None)
    pricing = settings.get_model_pricing("test-model")
    assert pricing is not None
    assert pricing.estimate_cost_usd(1_000_000, 500_000) == pytest.approx(1.5)

    with pytest.raises(ValidationError, match="max_cost_per_run_usd"):
        CrewForgeSettings(
            _env_file=None,
            max_cost_per_run_usd=2.0,
            max_cumulative_cost_usd=1.0,
        )

    with pytest.raises(ValidationError):
        ModelPricing(input_per_1m=-0.1, output_per_1m=1.0)


def test_usage_metrics_validation() -> None:
    usage = UsageMetrics(input_tokens=100, output_tokens=50)
    assert usage.total_tokens == 150

    with pytest.raises(ValidationError, match="total_tokens"):
        UsageMetrics(input_tokens=100, output_tokens=50, total_tokens=120)


def test_all_core_schemas_and_claim_states() -> None:
    agent = AgentSpec(role="Analyst", instructions="Analyze numerical claims.")
    task = TaskSpec(
        title="Verify totals",
        description="Check line items.",
        assigned_agent_id=agent.agent_id,
        depends_on=["tsk_prev"],
        status=TaskStatus.PENDING,
    )
    plan = ExecutionPlan(
        run_id="run_test1",
        goal="Evaluate budget math",
        agents=[agent],
        tasks=[task],
        assumptions=["Currency is USD"],
    )
    evidence = Evidence(
        run_id="run_test1",
        source_type="document",
        source_ref="section-2",
        excerpt="Total equals 42.",
        reliability=EvidenceReliability.HIGH,
        created_by=agent.agent_id,
    )

    asserted = Claim(
        run_id="run_test1",
        statement="Total is 40.",
        status=ClaimStatus.ASSERTED,
        version=1,
        author_agent_id=agent.agent_id,
        task_id=task.task_id,
    )
    rejected = asserted.model_copy(update={"status": ClaimStatus.REJECTED})
    revised = Claim(
        run_id="run_test1",
        statement="Total is 42.",
        status=ClaimStatus.SUPPORTED,
        version=2,
        parent_claim_id=asserted.claim_id,
        author_agent_id=agent.agent_id,
        evidence_ids=[evidence.evidence_id],
        depends_on_claim_ids=[asserted.claim_id],
        confidence=0.95,
    )
    unresolved = Claim(
        run_id="run_test1",
        statement="Future tax rate is 10%.",
        status=ClaimStatus.UNRESOLVED,
        author_agent_id=agent.agent_id,
    )

    assert {c.status for c in (asserted, rejected, revised, unresolved)} == {
        ClaimStatus.ASSERTED,
        ClaimStatus.REJECTED,
        ClaimStatus.SUPPORTED,
        ClaimStatus.UNRESOLVED,
    }
    assert revised.version == 2
    assert revised.parent_claim_id == asserted.claim_id

    msg = AgentMessage(
        run_id="run_test1",
        sender_id=agent.agent_id,
        recipient_id="reviewer_1",
        content="Submitted revised claim.",
        related_claim_ids=[revised.claim_id],
    )
    verdict = ReviewVerdict(
        run_id="run_test1",
        claim_id=revised.claim_id,
        claim_version=revised.version,
        reviewer_id="reviewer_1",
        decision=ReviewDecision.APPROVE,
        rationale="Matches cited evidence.",
    )
    run_cfg = RunConfig(
        provider="ollama",
        model="qwen3:4b",
        request_timeout_seconds=30.0,
        max_retries=2,
        max_concurrency=3,
        max_output_tokens=512,
        max_model_calls_per_run=16,
        max_cost_per_run_usd=0.05,
        max_cumulative_cost_usd=1.0,
    )
    event = ExecutionEvent(
        run_id="run_test1",
        seq=1,
        event_type="plan_created",
        actor="lead",
        message="Created plan.",
        payload={"plan_id": plan.plan_id},
    )
    result = RunResult(
        run_id="run_test1",
        mode="baseline",
        status=RunStatus.COMPLETED,
        prompt="Compute total",
        provider="ollama",
        model="qwen3:4b",
        answer="42",
        missing_information=["Tax rate"],
    )

    assert msg.related_claim_ids == [revised.claim_id]
    assert verdict.decision == ReviewDecision.APPROVE
    assert run_cfg.max_concurrency == 3
    assert event.seq == 1
    assert result.status == RunStatus.COMPLETED
