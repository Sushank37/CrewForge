"""Adversarial Phase 3 tests: arithmetic, evidence, reviewer independence, repair, graph, Gate."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.core.baseline import BaselineRunner
from crewforge.core.budget import BudgetManager
from crewforge.core.claim_graph import ClaimCycleError, ClaimDependencyGraph
from crewforge.core.evidence_verifier import (
    UnavailableExternalRetriever,
    verify_evidence_proposal,
)
from crewforge.core.executor import ModelExecutor
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.ledger import ClaimLedger, ClaimRevisionError
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.planner import validate_and_materialize_plan
from crewforge.core.reviewer import REVIEWER_COMBINED_ID, ReviewerPool
from crewforge.schemas import (
    ArithmeticRequest,
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    EvidenceReliability,
    PlannerOutput,
    ProposedClaim,
    ProposedEvidence,
    ReleaseSegment,
    ReviewDecision,
    ReviewerBatchOutput,
    ReviewVerdict,
    RevisionOutput,
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
    VerificationStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database
from crewforge.tools.arithmetic import (
    DeterministicArithmeticTool,
    MalformedExpressionError,
    UnsafeExpressionError,
    evaluate_decimal_expression,
    format_decimal,
)
from tests.fake_provider import FakeProvider

_CLAIM_HEADER_RE = re.compile(r"Claim (clm_[a-f0-9]+) \(version=(\d+),")
_SUPPORTED_LINE_RE = re.compile(r"- (clm_[a-f0-9]+) \(version=(\d+),.*?status=supported")


def _settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "max_concurrency": 1,
        "max_retries": 0,
        "max_model_calls_per_run": 16,
        "run_deadline_seconds": 60.0,
        "max_communication_rounds": 0,
        "max_revision_rounds": 1,
    }
    defaults.update(overrides)
    return CrewForgeSettings(**defaults)  # type: ignore[arg-type]


async def test_arithmetic_tool_precision_safety_and_grounding(tmp_path: Path) -> None:
    db_path = tmp_path / "arith.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()
    try:
        await db.create_run("run_arith", "network", "Compute", settings.to_run_config())
        tool = DeterministicArithmeticTool(db)

        # Exact Decimal evaluation + aggregates + comparisons
        assert format_decimal(evaluate_decimal_expression("120000 / (10000 - 5000)", {})) == "24"
        assert (
            format_decimal(evaluate_decimal_expression("avg([10, 20, 30]) + max(1, 4)", {})) == "24"
        )
        assert format_decimal(evaluate_decimal_expression("((75 - 60) / 60) * 100", {})) == "25"
        assert format_decimal(evaluate_decimal_expression("2 ** 10", {})) == "1024"
        assert format_decimal(evaluate_decimal_expression("10 > 5", {})) == "1"

        # Unsafe & malformed expressions rejected rather than silently repaired
        for bad_expr in [
            "__import__('os').system('echo hi')",
            "eval('1+1')",
            "a.__class__",
            "[1, 2][0]",
            "2 ** 1000",
        ]:
            with pytest.raises(UnsafeExpressionError):
                evaluate_decimal_expression(bad_expr, {"a": 1})

        for malformed in ["1 / 0", "1 + * 2", "", "unknown_var + 5"]:
            with pytest.raises(MalformedExpressionError):
                evaluate_decimal_expression(malformed, {})

        # Correct arithmetic with grounded inputs
        grounded_rec = await tool.execute(
            run_id="run_arith",
            actor="lead",
            request=ArithmeticRequest(
                expression="inv / (rev - exp)",
                variables={"inv": "120000", "rev": "10000", "exp": "5000"},
                input_provenance_refs={
                    "inv": "user_prompt",
                    "rev": "user_prompt",
                    "exp": "user_prompt",
                },
                proposed_result="24",
                units="months",
            ),
            grounded_texts=["Initial investment 120,000, revenue 10,000, expenses 5,000."],
        )
        assert grounded_rec.success is True
        assert grounded_rec.matches_proposed is True
        assert grounded_rec.inputs_grounded is True

        # Correct arithmetic with UNGROUNDED inputs -> inputs_grounded=False
        ungrounded_rec = await tool.execute(
            run_id="run_arith",
            actor="lead",
            request=ArithmeticRequest(
                expression="a + b",
                variables={"a": 999, "b": 1},
                input_provenance_refs={"a": "user_prompt", "b": "user_prompt"},
                proposed_result=1000,
            ),
            grounded_texts=["The numbers in the prompt are 10 and 20."],
        )
        assert ungrounded_rec.success is True
        assert ungrounded_rec.matches_proposed is True
        assert ungrounded_rec.inputs_grounded is False

        # Incorrect proposed result detected deterministically
        wrong_rec = await tool.execute(
            run_id="run_arith",
            actor="lead",
            request=ArithmeticRequest(
                expression="120000 / 5000",
                variables={},
                input_provenance_refs={},
                proposed_result="12",
            ),
            grounded_texts=[],
        )
        assert wrong_rec.success is True
        assert wrong_rec.matches_proposed is False
    finally:
        await db.close()


def test_evidence_verification_negation_and_external_citations() -> None:
    retriever = UnavailableExternalRetriever()
    assert retriever.retrieval_available is False

    # Substring dropping preceding negation marker must fail verification
    neg_check = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="user_prompt",
            source_ref="prompt",
            excerpt="positive cash flow guarantees attractiveness",
        ),
        original_prompt=(
            "Explain whether it is not true that positive cash flow guarantees attractiveness."
        ),
    )
    assert neg_check.verified_in_input is False
    assert neg_check.reliability == EvidenceReliability.UNVERIFIED

    # Irrelevant / non-matching excerpt rejected
    irrel_check = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="user_prompt",
            source_ref="prompt",
            excerpt="profit margin is 95%",
        ),
        original_prompt="Revenue is 10,000 and expenses are 5,000.",
    )
    assert irrel_check.verified_in_input is False

    # URL-like external citation remains UNVERIFIED without active retriever
    url_check = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_unverified",
            source_ref="https://example.com/finance-study-2026",
            excerpt="Studies show 80% payback improvement.",
        ),
        original_prompt="Evaluate investment.",
    )
    assert url_check.verified_in_input is False
    assert url_check.reliability == EvidenceReliability.UNVERIFIED


async def test_reviewer_independence_missing_verdicts_and_three_level_invalidation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "graph_review.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()

    try:
        await db.create_run("run_G", "network", "Chain test", settings.to_run_config())
        raw = PlannerOutput.model_validate(
            {
                "goal": "Chain test",
                "user_requirements": ["Check 3-level chain"],
                "acceptance_criteria": ["Verify chain"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "instructions": "Work.",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "T1",
                        "description": "D1",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Out"],
                    }
                ],
            }
        )
        plan = validate_and_materialize_plan(run_id="run_G", raw_plan=raw, settings=settings)
        await db.save_execution_plan(plan)
        agent_id = plan.agents[0].agent_id
        task_id = plan.tasks[0].task_id

        ledger = ClaimLedger(db)
        graph = ClaimDependencyGraph(db)

        _, claims, _ = await ledger.record_specialist_outputs(
            run_id="run_G",
            agent_id=agent_id,
            task_id=task_id,
            original_prompt="Base premise alpha is given.",
            dependency_findings_text="",
            evidence_proposals=[
                ProposedEvidence(
                    source_type="user_prompt",
                    source_ref="prompt",
                    excerpt="Base premise alpha is given.",
                )
            ],
            claim_proposals=[
                ProposedClaim(
                    claim_ref="c1",
                    statement="Base premise alpha is given.",
                    category=ClaimCategory.GIVEN_FACT,
                    evidence_indices=[0],
                ),
                ProposedClaim(
                    claim_ref="c2",
                    statement="Intermediate inference from c1.",
                    category=ClaimCategory.INFERENCE,
                    premise_claim_ids=["c1"],
                ),
                ProposedClaim(
                    claim_ref="c3",
                    statement="Final conclusion derived from c2.",
                    category=ClaimCategory.INFERENCE,
                    premise_claim_ids=["c2"],
                ),
            ],
        )
        c1, c2, c3 = claims
        assert c2.premise_refs == [ClaimVersionRef(claim_id=c1.claim_id, claim_version=1)]
        assert c3.premise_refs == [ClaimVersionRef(claim_id=c2.claim_id, claim_version=1)]

        # Self-approval prevention in Database
        with pytest.raises(ValueError, match="Independence violation"):
            await db.insert_review_verdict(
                ReviewVerdict(
                    run_id="run_G",
                    claim_id=c1.claim_id,
                    claim_version=1,
                    reviewer_id=agent_id,
                    review_dimension="evidence",
                    decision=ReviewDecision.APPROVE,
                    rationale="Self-approval attempt.",
                )
            )

        # Cross-run verdict rejection
        await db.create_run("run_OTHER", "network", "Other", settings.to_run_config())
        with pytest.raises(CrossRunReferenceError):
            await db.insert_review_verdict(
                ReviewVerdict(
                    run_id="run_OTHER",
                    claim_id=c1.claim_id,
                    claim_version=1,
                    reviewer_id=REVIEWER_COMBINED_ID,
                    review_dimension="evidence",
                    decision=ReviewDecision.APPROVE,
                    rationale="Wrong run.",
                )
            )

        # Cycle validation check
        cyclic_c1 = c1.model_copy(
            update={"premise_refs": [ClaimVersionRef(claim_id=c3.claim_id, claim_version=1)]}
        )
        with pytest.raises(ClaimCycleError):
            graph.validate_graph([cyclic_c1, c2, c3], run_id="run_G")

        # Mark C2 & C3 initially approved, then invalidate C1 -> C2 & C3 must become STALE!
        await db.insert_review_verdict(
            ReviewVerdict(
                run_id="run_G",
                claim_id=c2.claim_id,
                claim_version=1,
                reviewer_id=REVIEWER_COMBINED_ID,
                review_dimension="consistency",
                decision=ReviewDecision.APPROVE,
                rationale="Approved prior to premise rejection.",
            )
        )
        await db.update_claim_verification_state(
            run_id="run_G",
            claim_id=c2.claim_id,
            version=1,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
        )
        await db.update_claim_verification_state(
            run_id="run_G",
            claim_id=c3.claim_id,
            version=1,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
        )

        # Revise c1 to v2 and cascade invalidation
        c1_v2 = await ledger.apply_targeted_revision(
            run_id="run_G",
            target_claim_id=c1.claim_id,
            expected_version=1,
            new_statement="Revised base premise alpha.",
            author_agent_id=agent_id,
            original_prompt="Base premise alpha is given.",
            revision_notes="Corrected premise.",
        )
        assert c1_v2.status == ClaimStatus.ASSERTED  # Never inherits approval!

        # Wrong expected_version or unauthorized author rejected
        with pytest.raises(ClaimRevisionError, match="Stale revision target"):
            await ledger.apply_targeted_revision(
                run_id="run_G",
                target_claim_id=c1.claim_id,
                expected_version=1,
                new_statement="Stale target",
                author_agent_id=agent_id,
                original_prompt="Base premise alpha is given.",
            )
        with pytest.raises(ClaimRevisionError, match="Unauthorized revision"):
            await ledger.apply_targeted_revision(
                run_id="run_G",
                target_claim_id=c1.claim_id,
                expected_version=2,
                new_statement="Wrong author",
                author_agent_id="other_agent",
                original_prompt="Base premise alpha is given.",
            )

        invalidated = await graph.invalidate_dependents(
            run_id="run_G",
            root_claim_id=c1.claim_id,
            reason="Upstream premise revised.",
            superseded_version=1,
        )
        assert {c.claim_id for c in invalidated} == {c2.claim_id, c3.claim_id}
        latest_after = {c.claim_id: c for c in await ledger.get_claims("run_G", latest_only=True)}
        assert latest_after[c2.claim_id].status == ClaimStatus.STALE
        assert latest_after[c3.claim_id].status == ClaimStatus.STALE
        assert (
            latest_after[c2.claim_id].verification_status == VerificationStatus.STALE_PENDING_REVIEW
        )

        # Honesty Gate must refuse stale c2/c3 despite historical v1 approval verdict
        trace = await db.get_run_trace("run_G")
        assert trace is not None
        gate = HonestyGate(db)
        gate_res = gate.evaluate(
            run_id="run_G",
            original_prompt="Base premise alpha is given.",
            plan=plan,
            synthesis=SynthesisOutput(
                accepted_segments=[
                    ReleaseSegment(
                        segment_type="accepted_claim",
                        text="Final conclusion derived from c2.",
                        claim_refs=[ClaimVersionRef(claim_id=c3.claim_id, claim_version=1)],
                    )
                ]
            ),
            all_claims=trace.claims,
            evidence_list=trace.evidence,
            tool_executions=trace.tool_executions,
            verdicts=trace.review_verdicts,
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=False,
        )
        assert gate_res.gate_record is not None
        assert gate_res.gate_record.passed is False
        assert gate_res.verification.stale_claim_count == 2
        assert gate_res.gate_record.rejected_segments
    finally:
        await db.close()


async def test_missing_reviewer_verdict_leaves_claim_unresolved(tmp_path: Path) -> None:
    db_path = tmp_path / "missing_verdict.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()
    try:
        await db.create_run("run_mv", "network", "Prompt", settings.to_run_config())
        raw = PlannerOutput.model_validate(
            {
                "goal": "G",
                "user_requirements": ["R"],
                "acceptance_criteria": ["C"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "instructions": "Work.",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "T1",
                        "description": "D1",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["O1"],
                    }
                ],
            }
        )
        plan = validate_and_materialize_plan(run_id="run_mv", raw_plan=raw, settings=settings)
        await db.save_execution_plan(plan)
        ledger = ClaimLedger(db)
        await ledger.record_specialist_outputs(
            run_id="run_mv",
            agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
            original_prompt="Prompt",
            dependency_findings_text="",
            evidence_proposals=[],
            claim_proposals=[
                ProposedClaim(
                    statement="Inference claim awaiting review.",
                    category=ClaimCategory.INFERENCE,
                )
            ],
        )
        # Reviewer returns empty verdict list -> claim MUST become UNRESOLVED!
        executor = ModelExecutor(
            provider=FakeProvider(
                max_retries=0,
                handler=lambda _p, _m: {"verdicts": [], "missing_requirement_coverage": []},
            ),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        pool = ReviewerPool(executor=executor, database=db, settings=settings)
        outcome = await pool.review_claims(run_id="run_mv", original_prompt="Prompt", plan=plan)
        assert outcome.reviewer_calls == 1
        latest = await ledger.get_claims("run_mv", latest_only=True)
        assert latest[0].status == ClaimStatus.UNRESOLVED
    finally:
        await db.close()


async def test_honesty_gate_blocks_bypass_vectors() -> None:
    """Verify empty refs, Lead factual assertions, and ungrounded numbers fail Gate."""
    db = Database(":memory:")
    await db.initialize()
    try:
        settings = CrewForgeSettings(_env_file=None, db_path=Path(":memory:"))
        await db.create_run("run_bypass", "network", "Check rate", settings.to_run_config())
        raw = PlannerOutput.model_validate(
            {
                "goal": "Check rate",
                "user_requirements": ["Verify rate"],
                "acceptance_criteria": ["Verified rate"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "instructions": "Work.",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "T1",
                        "description": "D1",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Out"],
                    }
                ],
            }
        )
        plan = validate_and_materialize_plan(run_id="run_bypass", raw_plan=raw, settings=settings)
        await db.save_execution_plan(plan)

        c1 = Claim(
            claim_id="clm_valid1",
            run_id="run_bypass",
            statement="Verified base assumption.",
            category=ClaimCategory.GIVEN_FACT,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.PROVENANCE_VALIDATED,
            version=1,
            author_agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
        )
        await db.insert_claim(c1)
        await db.insert_review_verdict(
            ReviewVerdict(
                run_id="run_bypass",
                claim_id=c1.claim_id,
                claim_version=1,
                reviewer_id=REVIEWER_COMBINED_ID,
                review_dimension="evidence",
                decision=ReviewDecision.APPROVE,
                rationale="Approved",
            )
        )

        gate = HonestyGate(db)
        syn = SynthesisOutput(
            user_requirements=["Verify rate"],
            accepted_segments=[
                # Bypass 1: empty claim_refs
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text="The rate is 42%.",
                    claim_refs=[],
                    is_factual_assertion=True,
                ),
                # Bypass 2: unrelated text + extra unverified number hiding behind valid claim_id
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text="Rocket fuel efficiency improved by 99.9%.",
                    claim_refs=[ClaimVersionRef(claim_id="clm_valid1", claim_version=1)],
                    is_factual_assertion=True,
                ),
            ],
            conditional_conclusions=[
                # Bypass 3: ungrounded empirical number in conditional commentary without claim ref
                ReleaseSegment(
                    segment_type="conditional_conclusion",
                    text="Output reached 88.5 units yesterday.",
                    claim_refs=[],
                    is_factual_assertion=True,
                )
            ],
            new_factual_assertions=["Lead invented an unreviewed 500% gain."],
        )
        res = gate.evaluate(
            run_id="run_bypass",
            original_prompt="Check rate",
            plan=plan,
            synthesis=syn,
            all_claims=[c1],
            evidence_list=[],
            tool_executions=[],
            verdicts=[
                ReviewVerdict(
                    run_id="run_bypass",
                    claim_id=c1.claim_id,
                    claim_version=1,
                    reviewer_id=REVIEWER_COMBINED_ID,
                    review_dimension="evidence",
                    decision=ReviewDecision.APPROVE,
                    rationale="Approved",
                )
            ],
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=False,
        )
        assert res.gate_record is not None
        assert res.gate_record.passed is False
        assert len(res.gate_record.rejected_segments) == 3
        assert "42%" not in res.rendered_answer
        assert "99.9%" not in res.rendered_answer
        assert "88.5" not in res.rendered_answer
        assert "Unreviewed synthesis assertion" in res.rendered_answer
    finally:
        await db.close()


async def test_baseline_and_network_arithmetic_tool_parity(tmp_path: Path) -> None:
    db_path = tmp_path / "parity.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()

    try:
        call_num = 0

        def baseline_handler(_prompt: str, _model: Any) -> dict[str, Any]:
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                return {
                    "answer": "Drafting calculation.",
                    "computations": [
                        {
                            "expression": "inv / (rev - exp)",
                            "variables": {"inv": "120000", "rev": "10000", "exp": "5000"},
                            "input_provenance_refs": {
                                "inv": "user_prompt",
                                "rev": "user_prompt",
                                "exp": "user_prompt",
                            },
                            "proposed_result": "24",
                            "units": "months",
                        }
                    ],
                    "needs_tool_followup": True,
                }
            return {
                "answer": "The deterministic simple payback period is 24 months.",
                "missing_information": ["Project lifespan"],
                "assumptions": ["Constant cash flows"],
                "needs_tool_followup": False,
            }

        runner = BaselineRunner(
            provider=FakeProvider(handler=baseline_handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        res = await runner.run(
            "Investment 120,000, monthly revenue 10,000, monthly expenses 5,000."
        )
        assert res.status == RunStatus.COMPLETED
        assert res.usage.model_calls == 2
        assert "24 months" in res.answer

        trace = await db.get_run_trace(res.run_id)
        assert trace is not None
        assert len(trace.tool_executions) == 1
        assert trace.tool_executions[0].computed_result == "24"
        assert trace.tool_executions[0].inputs_grounded is True
    finally:
        await db.close()


async def test_scripted_end_to_end_rejection_targeted_repair_and_release(
    tmp_path: Path,
) -> None:
    """Scripted adversarial E2E test (Task 12 #1-7):

    1. Specialist produces an incorrect calculation claim (v1: 12 months).
    2. Deterministic Number Checker rejects v1 with blocking objection.
    3. Originating worker receives rejection message via MessageBus.
    4. Worker submits corrected version (v2: 24 months) targeting exact claim_id & expected_version.
    5. v2 is independently reviewed and approved.
    6. Final answer releases corrected 24-month claim.
    7. Original incorrect v1 remains visible in trace.claims with status=rejected.
    """
    db_path = tmp_path / "e2e_repair.db"
    settings = _settings(db_path, max_revision_rounds=1)
    db = Database(db_path)
    await db.initialize()

    revision_prompts: list[str] = []

    def handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        if response_model is PlannerOutput:
            return {
                "goal": "Calculate simple payback period accurately",
                "user_requirements": ["Calculate the simple payback period in months"],
                "acceptance_criteria": ["Payback period matches exact deterministic arithmetic"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Financial Calculator",
                        "instructions": "Compute simple payback period.",
                        "capabilities": ["calculation"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute payback period",
                        "description": "Compute 120000 / (10000 - 5000).",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Simple payback period"],
                    }
                ],
                "assumptions": ["Constant cash flows"],
                "missing_information": ["Project lifespan"],
            }
        if response_model is SpecialistOutput:
            # Step 1: Intentionally emit incorrect proposed_result=12!
            return {
                "findings": "Initial calculation suggests 12 months.",
                "claims": [
                    {
                        "statement": "The simple payback period is 12 months.",
                        "category": "computation",
                        "evidence_indices": [0],
                        "computation": {
                            "expression": "inv / (rev - exp)",
                            "variables": {"inv": "120000", "rev": "10000", "exp": "5000"},
                            "input_provenance_refs": {
                                "inv": "user_prompt",
                                "rev": "user_prompt",
                                "exp": "user_prompt",
                            },
                            "proposed_result": "12",
                            "units": "months",
                        },
                        "requirement_refs": ["Calculate the simple payback period in months"],
                        "confidence": 0.7,
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "calculation",
                        "source_ref": "arithmetic_tool",
                        "excerpt": "inv / (rev - exp)",
                    }
                ],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
                "task_completed": True,
            }
        if response_model is RevisionOutput:
            revision_prompts.append(prompt)
            match = re.search(r"claim_id=(clm_[a-f0-9]+),\s*expected_version=(\d+)", prompt)
            assert match is not None
            cid, cver = match.group(1), int(match.group(2))
            # Step 4: Submit corrected v2 targeting exact claim_id & expected_version
            return {
                "updated_findings": "Corrected simple payback period to 24 months.",
                "revisions": [
                    {
                        "target_claim_id": cid,
                        "expected_version": cver,
                        "new_statement": "The simple payback period is 24 months.",
                        "category": "computation",
                        "evidence_indices": [0],
                        "computation": {
                            "expression": "inv / (rev - exp)",
                            "variables": {"inv": "120000", "rev": "10000", "exp": "5000"},
                            "input_provenance_refs": {
                                "inv": "user_prompt",
                                "rev": "user_prompt",
                                "exp": "user_prompt",
                            },
                            "proposed_result": "24",
                            "units": "months",
                        },
                        "revision_notes": "Corrected division from 12 to 24 months per tool check.",
                        "confidence": 0.99,
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "calculation",
                        "source_ref": "arithmetic_tool",
                        "excerpt": "120000 / (10000 - 5000) = 24",
                    }
                ],
                "uncertainties": [],
                "outgoing_messages": [],
            }
        if response_model is ReviewerBatchOutput:
            verdicts = []
            for cid, cver in _CLAIM_HEADER_RE.findall(prompt):
                verdicts.append(
                    {
                        "claim_id": cid,
                        "claim_version": int(cver),
                        "review_dimension": "combined_evidence_consistency",
                        "decision": "approve",
                        "rationale": (
                            "Expression inv/(rev-exp) with units 'months' matches requirement."
                        ),
                        "evidence_refs": [],
                        "blocking_severity": "none",
                    }
                )
            return {"verdicts": verdicts, "missing_requirement_coverage": []}

        # Synthesis selects only the reviewed supported v2 claim
        supported = _SUPPORTED_LINE_RE.findall(prompt)
        assert supported, "Corrected v2 claim must be SUPPORTED in synthesis input"
        cid, cver = supported[0]
        return {
            "user_requirements": ["Calculate the simple payback period in months"],
            "accepted_segments": [
                {
                    "segment_type": "accepted_claim",
                    "text": "The simple payback period is 24 months.",
                    "claim_refs": [{"claim_id": cid, "claim_version": int(cver)}],
                    "requirement_ref": "Calculate the simple payback period in months",
                    "is_factual_assertion": True,
                }
            ],
            "conditional_conclusions": [],
            "recommendations": [],
            "assumptions": ["Constant cash flows"],
            "unresolved_issues": [],
            "missing_information": ["Project lifespan"],
            "new_factual_assertions": [],
            "referenced_claim_ids": [cid],
        }

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        res = await runner.run(
            "Initial investment 120,000, monthly revenue 10,000, monthly expenses 5,000."
        )
        assert res.status == RunStatus.COMPLETED
        assert "24 months" in res.answer
        assert "12 months" not in res.answer
        assert res.verification is not None
        assert res.verification.honesty_gate_passed is True
        assert res.verification.revision_count == 1
        assert res.verification.supported_claim_count == 1
        assert len(revision_prompts) == 1
        assert (
            "Proposed result '12' does not match deterministic computation '24'"
            in revision_prompts[0]
        )

        trace = await db.get_run_trace(res.run_id)
        assert trace is not None
        # Step 7: Original incorrect v1 AND corrected v2 both preserved in trace.claims!
        assert len(trace.claims) == 2
        v1, v2 = trace.claims[0], trace.claims[1]
        assert v1.claim_id == v2.claim_id
        assert v1.version == 1 and v1.status == ClaimStatus.REJECTED
        assert v2.version == 2 and v2.status == ClaimStatus.SUPPORTED
        assert v2.verification_status == VerificationStatus.COMPUTATIONALLY_VERIFIED
        assert len(trace.messages) == 1
        assert trace.messages[0].delivery_status == "consumed"
    finally:
        await db.close()
