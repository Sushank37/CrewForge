"""Regression tests for combinatorics, HonestyGate blocking, >6 workers, and incremental traces."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.core.budget import BudgetManager
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.orchestrator import NetworkRunner
from crewforge.dashboard import create_dashboard_app
from crewforge.schemas import (
    AgentSpec,
    Claim,
    ClaimCategory,
    ClaimStatus,
    ExecutionPlan,
    ReviewDecision,
    ReviewVerdict,
    RunStatus,
    SynthesisOutput,
    TaskSpec,
    TaskStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import evaluate_decimal_expression
from tests.fake_provider import ChecklistFixtureProvider as FakeProvider


def test_deterministic_combinatorics_and_coin_toss_probability() -> None:
    # Exact binomial survival P(X >= 6) for n=10, p=0.5: 386/1024 = 0.376953125
    sf_val = evaluate_decimal_expression(
        "binomial_sf(n, k, p)", {"n": 10, "k": 6, "p": "0.5"}
    )
    assert sf_val == Decimal("0.376953125")

    sum_val = evaluate_decimal_expression(
        "sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))",
        {"n": 10, "k": 6, "p": "0.5"},
    )
    assert sum_val == Decimal("0.376953125")

    assert evaluate_decimal_expression("comb(10, 6)", {}) == Decimal("210")
    assert evaluate_decimal_expression("perm(5, 2)", {}) == Decimal("20")
    assert evaluate_decimal_expression("factorial(6)", {}) == Decimal("720")
    assert evaluate_decimal_expression(
        "binomial_pmf(10, 6, 0.5)", {}
    ) == Decimal("0.205078125")
    assert evaluate_decimal_expression(
        "binomial_cdf(10, 5, 0.5)", {}
    ) == Decimal("0.623046875")


async def test_coin_toss_probability_network_run_verified_end_to_end(tmp_path: Path) -> None:
    db = Database(tmp_path / "coin_toss.db")
    await db.initialize()

    def _handler(_user_prompt: str, response_model: Any) -> dict[str, Any]:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Compute P(X >= 6) for 10 fair independent coin tosses",
                "user_requirements": [
                    "Calculate the probability of at least 6 heads in 10 fair coin tosses"
                ],
                "acceptance_criteria": ["Provide exact deterministic binomial probability"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Probability Specialist",
                        "specialty": "Combinatorics",
                        "instructions": "Compute exact binomial probability.",
                        "capabilities": ["reasoning"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute binomial survival probability",
                        "description": "Compute P(X >= 6) for n=10, p=0.5",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Exact probability 0.376953125"],
                    }
                ],
                "assumptions": ["Coins are fair and independent"],
                "missing_information": [],
            }
        if schema_name in {"WorkerOutput", "SpecialistOutput"}:
            return {
                "findings": "For X ~ Binomial(10, 0.5), P(X >= 6) = 386/1024 = 0.376953125.",
                "evidence": [
                    {
                        "source_type": "user_prompt",
                        "source_ref": "up_1",
                        "excerpt": (
                            "toss 10 fair independent coins together what is the probability "
                            "i will get "
                            "atleast 6 heads"
                        ),
                    }
                ],
                "claims": [
                    {
                        "statement": (
                            "For 10 fair independent coin tosses, the probability of at "
                            "least 6 heads is 0.376953125."
                        ),
                        "category": "computation",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_1"],
                        "computation": {
                            "expression": "binomial_sf(n, k, p)",
                            "variable_bindings": [
                                {"name": "n", "value": "10", "provenance_ref": "up_1"},
                                {"name": "k", "value": "6", "provenance_ref": "up_1"},
                                {
                                    "name": "p",
                                    "value": "0.5",
                                    "provenance_ref": "const:fair_coin_probability",
                                },
                            ],
                            "proposed_result": "0.376953125",
                            "units": "probability",
                        },
                    }
                ],
                "assumptions": ["Fair independent coins"],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
            }
        if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
            import re

            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", _user_prompt)))
            return {
                "verdicts": [
                    {
                        "claim_id": cid,
                        "decision": "approve",
                        "rationale": (
                            "Verified deterministically via binomial_sf(10, 6, 0.5) = 0.376953125."
                        ),
                        "evidence_refs": [],
                        "required_correction": None,
                        "blocking_severity": "none",
                        "blocking_issues": [],
                    }
                    for cid in cids
                ]
            }
        if schema_name == "SynthesisOutput":
            # Claim ref is dynamically resolved from the user prompt claim ID
            import re

            m = re.search(r"(clm_[A-Za-z0-9_]+)@v(\d+)", _user_prompt)
            cid = m.group(1) if m else "clm_missing"
            ver = int(m.group(2)) if m else 1
            return {
                "direct_summary": (
                    "For 10 fair independent coin tosses, the probability of at "
                    "least 6 heads is 0.376953125."
                ),
                "accepted_segments": [
                    {
                        "segment_type": "accepted_claim",
                        "text": (
                            "For 10 fair independent coin tosses, the probability of at "
                            "least 6 heads is 0.376953125."
                        ),
                        "claim_refs": [{"claim_id": cid, "claim_version": ver}],
                        "requirement_ref": "req_1",
                    }
                ],
                "conditional_segments": [],
                "recommendations": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": [cid],
                "assumptions": [],
                "missing_information": [],
            }
        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        provider=ProviderType.OLLAMA,
        model="fake-math",
        db_path=tmp_path / "coin_toss.db",
        budget_authority_path=tmp_path / "coin_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        pricing_json={"fake-math": {"input_per_1m": 0.0, "output_per_1m": 0.0}},
    )
    runner = NetworkRunner(
        provider=FakeProvider(handler=_handler, provider_name="fake", model_name="fake-math"),
        database=db,
        budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    res = await runner.run(
        "If i toss 10 fair independent coins together what is the probability "
        "i will get atleast 6 heads"
    )
    await db.close()
    assert res.status == RunStatus.COMPLETED, f"err={res.error} notes={res.confidence_notes}"
    assert res.verification is not None
    assert res.verification.honesty_gate_passed is True
    assert "0.376953125" in res.answer


async def test_honesty_gate_zero_released_claims_is_blocked_not_partial(tmp_path: Path) -> None:
    db = Database(tmp_path / "gate_blocked.db")
    await db.initialize()
    gate = HonestyGate(db)

    plan = ExecutionPlan(
        run_id="run_zero_rel",
        goal="Test zero released claims",
        user_requirements=["Req 1"],
        acceptance_criteria=["AC 1"],
        agents=[AgentSpec(agent_id="agt_1", role="Analyst", instructions="Analyze")],
        tasks=[
            TaskSpec(
                task_id="tsk_1",
                title="Task 1",
                description="Compute",
                assigned_agent_id="agt_1",
                status=TaskStatus.COMPLETED,
            )
        ],
    )
    rejected_claim = Claim(
        claim_id="clm_rej1",
        run_id="run_zero_rel",
        version=1,
        statement="Bad computation",
        category=ClaimCategory.COMPUTATION,
        status=ClaimStatus.REJECTED,
        author_agent_id="agt_1",
    )
    verdict = ReviewVerdict(
        run_id="run_zero_rel",
        claim_id="clm_rej1",
        claim_version=1,
        reviewer_id="reviewer_numbers",
        review_dimension="numbers",
        decision=ReviewDecision.REJECT,
        rationale="Tool failed",
    )
    out = gate.evaluate(
        run_id="run_zero_rel",
        original_prompt="Test zero released claims",
        plan=plan,
        all_claims=[rejected_claim],
        evidence_list=[],
        tool_executions=[],
        verdicts=[verdict],
        incomplete_tasks=[],
        extra_assumptions=[],
        extra_missing=[],
        uncertainties=[],
        is_partial_run=False,
        synthesis=SynthesisOutput(
            direct_summary="",
            accepted_segments=[],
            conditional_segments=[],
            recommendations=[],
        ),
    )
    await db.close()
    assert out.gate_record.passed is False
    assert out.gate_record.release_mode == "blocked"


async def test_scalable_eight_specialist_team_and_incremental_events(tmp_path: Path) -> None:
    db_path = tmp_path / "scale8.db"
    db = Database(db_path)
    await db.initialize()

    def _handler(user_prompt: str, response_model: Any) -> dict[str, Any]:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            agents = [
                {
                    "agent_ref": f"a{i}",
                    "role": f"Specialist {i}",
                    "specialty": f"Domain {i}",
                    "instructions": f"Handle domain {i}.",
                    "capabilities": ["reasoning"],
                }
                for i in range(1, 9)
            ]
            tasks = [
                {
                    "task_ref": f"t{i}",
                    "title": f"Task {i}",
                    "description": f"Analyze domain {i}.",
                    "assigned_agent_ref": f"a{i}",
                    "depends_on": ["t1"] if i > 4 else [],
                    "expected_deliverables": [f"Finding {i}"],
                }
                for i in range(1, 9)
            ]
            return {
                "goal": "Large 8-specialist parallel + dependent DAG",
                "user_requirements": ["Cover all 8 domains"],
                "acceptance_criteria": ["All 8 domains verified"],
                "agents": agents,
                "tasks": tasks,
                "assumptions": [],
                "missing_information": [],
            }
        if schema_name in {"WorkerOutput", "SpecialistOutput"}:
            return {
                "findings": "Verified domain analysis.",
                "evidence": [
                    {
                        "source_type": "user_prompt",
                        "source_ref": "up_1",
                        "excerpt": "Analyze 8 domains",
                    }
                ],
                "claims": [
                    {
                        "statement": "Domain assessment completed using prompt constraints.",
                        "category": "given_fact",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_1"],
                    }
                ],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
            }
        if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
            return {
                "verdicts": [
                    {
                        "claim_index": idx,
                        "decision": "accept",
                        "rationale": "Grounded in prompt.",
                        "evidence_refs": [],
                        "required_correction": None,
                        "blocking_severity": "none",
                        "blocking_issues": [],
                    }
                    for idx in range(user_prompt.count("claim_id="))
                ]
            }
        if schema_name == "SynthesisOutput":
            import re

            refs = re.findall(r"(clm_[A-Za-z0-9_]+)\s+@v(\d+)", user_prompt)
            c_refs = [{"claim_id": cid, "claim_version": int(ver)} for cid, ver in refs[:1]]
            return {
                "direct_summary": "All 8 domains verified.",
                "accepted_segments": [
                    {
                        "text": "Domain assessment completed using prompt constraints.",
                        "claim_refs": c_refs,
                        "requirement_ref": "req_1",
                    }
                ],
                "conditional_segments": [],
                "recommendations": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": [r[0] for r in refs[:1]],
                "assumptions": [],
                "missing_information": [],
            }
        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        provider=ProviderType.OPENAI,
        model="fake-large",
        openai_api_key="sk-fake",
        db_path=db_path,
        budget_authority_path=tmp_path / "scale8_budget.db",
        allow_paid_runs=True,
        max_workers=10,
        max_tasks=12,
        max_concurrency=4,
        max_model_calls_per_run=32,
        max_cost_per_run_usd=5.0,
        max_cumulative_cost_usd=10.0,
    )
    runner = NetworkRunner(
        provider=FakeProvider(handler=_handler, provider_name="fake", model_name="fake-large"),
        database=db,
        budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    res = await runner.run("Analyze 8 domains")
    await db.close()
    assert len(res.specialist_roles) == 8

    # Verify incremental since_seq event delivery on /api/live/runs/{run_id}
    app = create_dashboard_app(db_path=db_path, require_auth=False, require_react=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        full_resp = await client.get(f"/api/live/runs/{res.run_id}")
        assert full_resp.status_code == 200
        full_data = full_resp.json()
        total_events = len(full_data["execution_events"])
        assert total_events > 5
        mid_seq = int(full_data["execution_events"][2]["seq"])
        inc_resp = await client.get(f"/api/live/runs/{res.run_id}?since_seq={mid_seq}")
        assert inc_resp.status_code == 200
        inc_data = inc_resp.json()
        assert inc_data["incremental"] is True
        assert all(int(e["seq"]) > mid_seq for e in inc_data["execution_events"])
        assert len(inc_data["execution_events"]) < total_events


def test_adaptive_team_capacity_and_arithmetic_cumulative_limit(tmp_path: Path) -> None:
    import pytest

    from crewforge.tools.arithmetic import UnsafeExpressionError
    from crewforge.web_api import JudgeSessionManager, JudgeSettingsUpdate

    # Adaptive capacity scales within configurable model-call resource limits
    low_budget = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="gpt-5.4",
        max_model_calls_per_run=6,
    )
    high_budget = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="gpt-5.4",
        max_model_calls_per_run=20,
    )
    assert low_budget.effective_max_workers == 3
    assert high_budget.effective_max_workers == 17
    assert high_budget.effective_max_tasks == 17

    session = JudgeSessionManager(db_path=tmp_path / "adapt.db")
    pub = session.get_public_settings()
    assert pub["adaptive_team_capacity"] is True
    assert pub["max_workers"] == 3
    session.update_settings(
        JudgeSettingsUpdate(
            billing_mode="byok",
            provider="openai",
            model="gpt-5.4",
            max_model_calls_per_run=24,
        )
    )
    assert session.get_public_settings()["max_workers"] == 21

    # Cumulative computation safety limit rejects nested/excessive ops
    with pytest.raises(UnsafeExpressionError, match="cumulative computation limit"):
        evaluate_decimal_expression(
            "sum(sum(1 for j in range(100)) for i in range(100))",
            {},
        )


async def test_incomplete_execution_with_zero_verified_claims_is_blocked(tmp_path: Path) -> None:
    db = Database(tmp_path / "incomplete_zero.db")
    await db.initialize()
    gate = HonestyGate(db)
    plan = ExecutionPlan(
        run_id="run_inc_zero",
        goal="Test incomplete execution without verified claims",
        user_requirements=["Req 1"],
        acceptance_criteria=["AC 1"],
        agents=[AgentSpec(agent_id="agt_1", role="Analyst", instructions="Analyze")],
        tasks=[
            TaskSpec(
                task_id="tsk_1",
                title="Task 1",
                description="Incomplete",
                assigned_agent_id="agt_1",
                status=TaskStatus.PENDING,
            )
        ],
    )
    synth = SynthesisOutput(
        direct_summary="Execution stopped before any verified claims were produced.",
        accepted_segments=[],
        conditional_segments=[],
        recommendations=[],
        new_factual_assertions=[],
        referenced_claim_ids=[],
        assumptions=[],
        missing_information=["Task 1 did not complete"],
    )
    gate_res = gate.evaluate(
        run_id="run_inc_zero",
        original_prompt="Test incomplete execution",
        plan=plan,
        all_claims=[],
        evidence_list=[],
        tool_executions=[],
        verdicts=[],
        synthesis=synth,
        incomplete_tasks=plan.tasks,
        is_partial_run=True,
        extra_assumptions=[],
        extra_missing=["Task 1 did not complete"],
        uncertainties=[],
    )
    await db.close()
    assert gate_res.gate_record.passed is False
    assert gate_res.gate_record.release_mode == "blocked"

