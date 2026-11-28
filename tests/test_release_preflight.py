"""Offline regressions for the final public release contract."""

import asyncio

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetExceededError, BudgetManager
from crewforge.core.executor import ModelExecutor
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.planner import LeadPlanner, PlanValidationError
from crewforge.public_mcp_service import PublicJobManager
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    ExecutionPlan,
    PlannerOutput,
    ReleaseSegment,
    RequirementDeliverable,
    ReviewerBatchOutput,
    ReviewVerdict,
    SpecialistOutput,
    SynthesisOutput,
    ToolExecutionRecord,
    VerificationStatus,
)
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider
from tests.test_requirement_recovery import approved, one_task_plan


@pytest.mark.parametrize("invalid", ["missing", "null", "empty", "uncovered"])
async def test_new_plan_checklist_is_repaired_once(tmp_path, invalid):
    valid = one_task_plan(["Report storage", "Report transit"], [
        ("req_1", "Calculate storage duration", "calculation"),
        ("req_2", "Explain transit uncertainty", "explanation"),
    ])
    broken = dict(valid)
    if invalid == "missing":
        del broken["deliverables"]
    else:
        invalid_checklists = {"null": None, "empty": [], "uncovered": valid["deliverables"][:1]}
        broken["deliverables"] = invalid_checklists[invalid]
    calls = []

    def script(prompt, schema):
        assert schema is PlannerOutput
        calls.append(prompt)
        return broken if len(calls) == 1 else valid

    db = Database(tmp_path / "planner.db")
    await db.initialize()
    settings = CrewForgeSettings(_env_file=None, allow_paid_runs=False, max_model_calls_per_run=5)
    await db.create_run("plan", "network", "Report storage and transit.", settings.to_run_config())
    executor = ModelExecutor(
        provider=FakeProvider(handler=script), database=db, settings=settings,
        budget_manager=BudgetManager(db, settings),
    )
    try:
        planner = LeadPlanner(executor=executor, database=db, settings=settings)
        plan = await planner.create_plan(run_id="plan", user_prompt="Report storage and transit.")
        assert len(calls) == 2 and "deliverable" in calls[1].lower()
        assert {item.requirement_ref for item in plan.deliverables} == {"req_1", "req_2"}
        trace = await db.get_run_trace("plan")
        assert trace is not None and trace.plan == plan
        assert [e.event_type for e in trace.events].count("plan_repaired") == 1
    finally:
        await db.close()


@pytest.mark.parametrize("max_calls,error,calls", [
    (4, BudgetExceededError, 1),
    (5, PlanValidationError, 2),
])
async def test_invalid_plan_cannot_be_persisted_or_spend_reserved_calls(
    tmp_path, max_calls, error, calls,
):
    broken = one_task_plan(["Report storage"], [])
    db = Database(tmp_path / "planner.db")
    await db.initialize()
    settings = CrewForgeSettings(
        _env_file=None, allow_paid_runs=False, max_model_calls_per_run=max_calls,
    )
    await db.create_run("plan", "network", "Report storage.", settings.to_run_config())
    provider = FakeProvider(handler=lambda _prompt, _schema: broken)
    executor = ModelExecutor(
        provider=provider, database=db, settings=settings,
        budget_manager=BudgetManager(db, settings),
    )
    try:
        planner = LeadPlanner(executor=executor, database=db, settings=settings)
        with pytest.raises(error):
            await planner.create_plan(run_id="plan", user_prompt="Report storage.")
        assert len(provider.calls) == calls
        trace = await db.get_run_trace("plan")
        assert trace is not None and trace.plan is None
    finally:
        await db.close()


async def test_public_partial_release_distinguishes_approval_and_coverage(tmp_path):
    relevant = "Aster fits the weight limit."
    complete = "The driver confirmed receipt."
    irrelevant = "The dispatch form is blue."
    reviewed = {}

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan(["Find the best carrier", "Confirm delivery"], [
                ("req_1", "Prove the optimal carrier after comparing prices", "comparison"),
                ("req_2", "Confirm whether the driver received the parcel", "explanation"),
            ])
        if schema is SpecialistOutput:
            return {"findings": relevant, "claims": [
                {"statement": relevant, "category": "logical_inference",
                 "requirement_refs": ["deliv_1"]},
                {"statement": complete, "category": "logical_inference",
                 "requirement_refs": ["deliv_2"]},
                {"statement": irrelevant, "category": "logical_inference"},
            ]}
        if schema is ReviewerBatchOutput:
            return approved(prompt, {
                relevant: ([], "Weight feasibility is supported. No prices were compared."),
                complete: (["deliv_2"], "The receipt is explicit."),
                irrelevant: ([], "The source confirms the color."),
            }, reviewed)
        assert schema is SynthesisOutput
        return {}  # Exercise the bounded ledger-backed selection.

    settings = CrewForgeSettings(
        _env_file=None, allow_paid_runs=False, max_model_calls_per_run=4,
        max_retries=0, enable_deterministic_synthesis_fast_path=False,
    )
    manager = PublicJobManager(
        settings=settings, db_path=tmp_path / "public.db",
        budget_db_path=tmp_path / "budget.db",
        provider_factory=lambda _: FakeProvider(handler=script),
    )
    try:
        submission = await manager.enqueue_task(
            prompt=(
                "Aster carries 45 kg within a 50 kg limit. The driver confirmed receipt. "
                "The dispatch form is blue. Compare prices to identify the best carrier and "
                "confirm delivery."
            ),
            mode="network",
        )
        assert submission["accepted"]
        task_id = submission["task_id"]
        for _ in range(80):
            status = await manager.get_task_status(task_id)
            if status["status"] in {"partial", "completed", "failed"}:
                break
            await asyncio.sleep(0.025)
        result = await manager.get_task_result(task_id)
        trace = await manager.get_task_trace(task_id)
        assert result["status"] == "partial"
        assert relevant in result["answer"] and complete in result["answer"]
        assert irrelevant not in result["answer"]
        verification = result["verification"]
        assert verification["execution_completed"] is True
        assert verification["content_released_safely"] is True
        assert verification["answer_complete"] is False
        first, second = verification["requirement_statuses"]
        assert first["status"] == "unresolved" and first["verified_deliverable_refs"] == []
        assert second["status"] == "answered" and second["verified_deliverable_refs"] == ["deliv_2"]
        assert [d["deliverable_id"] for d in trace["deliverables"]] == ["deliv_1", "deliv_2"]
        verdicts = {r["claim_id"]: r for r in trace["review_decisions"]}
        assert verdicts[reviewed[relevant][0]]["decision"] == "approve"
        assert verdicts[reviewed[relevant][0]]["covered_deliverable_refs"] == []
        assert verdicts[reviewed[complete][0]]["covered_deliverable_refs"] == ["deliv_2"]
        assert trace["honesty_gate"]["answer_complete"] is False
        assert trace["honesty_gate"]["execution_completed"] is True
        assert trace["honesty_gate"]["content_released_safely"] is True
    finally:
        await manager.shutdown()


@pytest.mark.parametrize("superseded", [False, True])
async def test_advice_and_condition_check_transitive_arithmetic_lineage(tmp_path, superseded):
    db = Database(tmp_path / "gate.db")
    premise = Claim(
        claim_id="clm_input", run_id="run", statement="The measured amount is 30.",
        category=ClaimCategory.GIVEN_FACT, author_agent_id="specialist",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.PROVENANCE_VALIDATED, version=2,
    )
    record = ToolExecutionRecord(
        tool_execution_id="tlx_derived", run_id="run", actor="specialist",
        expression="measured + fee", variables={"measured": "30", "fee": "5"},
        provenance_details={
            "measured": {
                "kind": "supported_claim",
                "resolved_id": f"clm_input@v{1 if superseded else 2}",
                "grounded": True,
            },
        },
        computed_result="35", inputs_grounded=True, matches_proposed=True, success=True,
    )
    derived = Claim(
        claim_id="clm_derived", run_id="run", statement="The loaded amount is 35.",
        category=ClaimCategory.COMPUTATION, author_agent_id="specialist",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.COMPUTATIONALLY_VERIFIED,
        tool_execution_ids=[record.tool_execution_id],
        premise_refs=[ClaimVersionRef(claim_id=premise.claim_id, claim_version=2)],
    )
    intermediate = Claim(
        claim_id="clm_intermediate", run_id="run", statement="The load fits the capacity.",
        category=ClaimCategory.INFERENCE, author_agent_id="specialist",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
        premise_refs=[ClaimVersionRef(claim_id=derived.claim_id, claim_version=1)],
    )
    advice = intermediate.model_copy(update={
        "claim_id": "clm_advice", "category": ClaimCategory.RECOMMENDATION, "is_material": False,
        "statement": "Use the standard vehicle.", "requirement_refs": ["deliv_1"],
        "premise_refs": [ClaimVersionRef(claim_id=intermediate.claim_id, claim_version=1)],
    })
    conditional = advice.model_copy(update={
        "claim_id": "clm_conditional", "category": ClaimCategory.INFERENCE, "is_material": True,
        "statement": "If the standard vehicle is available, the load fits.",
        "requirement_refs": ["deliv_2"],
    })
    all_claims = [premise, derived, intermediate, advice, conditional]
    verdicts = [
        ReviewVerdict(
            run_id="run", claim_id=c.claim_id, claim_version=c.version, reviewer_id="reviewer",
            decision="approve", rationale="Matches the stated source and calculation.",
            covered_deliverable_refs=c.requirement_refs,
        )
        for c in all_claims
    ]
    plan = ExecutionPlan(
        run_id="run", goal="Advise on transport", user_requirements=["Advise", "State outcome"],
        deliverables=[
            RequirementDeliverable(
                deliverable_id="deliv_1", requirement_ref="req_1", task_ref="t1",
                kind="recommendation", text="Advise on transport",
            ),
            RequirementDeliverable(
                deliverable_id="deliv_2", requirement_ref="req_2", task_ref="t1",
                kind="conditional", text="State conditional outcome",
            ),
        ],
    )
    report = HonestyGate(db).evaluate(
        run_id="run", original_prompt="Load 30, handling adds 5, capacity is 40.",
        plan=plan, all_claims=all_claims, evidence_list=[], tool_executions=[record],
        verdicts=verdicts, incomplete_tasks=[], extra_assumptions=[], extra_missing=[],
        uncertainties=[], is_partial_run=False,
        synthesis=SynthesisOutput(
            recommendations=[ReleaseSegment(
                segment_type="recommendation", text=advice.statement,
                claim_refs=[ClaimVersionRef(claim_id=advice.claim_id, claim_version=1)],
            )],
            conditional_conclusions=[ReleaseSegment(
                segment_type="conditional_conclusion", text=conditional.statement,
                claim_refs=[ClaimVersionRef(claim_id=conditional.claim_id, claim_version=1)],
            )],
        ),
    )
    assert report.verification.answer_complete is not superseded
    assert (advice.statement in report.rendered_answer) is not superseded
    assert (conditional.statement in report.rendered_answer) is not superseded
    assert report.gate_record is not None
    if superseded:
        assert len(report.gate_record.rejected_claim_refs) == 2
        assert all(
            "invalidated arithmetic" in ref["reason"]
            for ref in report.gate_record.rejected_claim_refs
        )
