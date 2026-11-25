"""Phase 4 regression tests: claim-lock, lineage, reviewer dedup, retrieval, portability, guard."""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings, ModelPricing, ProviderType
from crewforge.core.baseline import BaselineRunner
from crewforge.core.budget import BudgetError, BudgetManager
from crewforge.core.executor import ModelExecutor
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.ledger import ClaimLedger
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.planner import validate_and_materialize_plan
from crewforge.core.retrieval import (
    AllowlistedHttpRetriever,
    LocalDocumentRetriever,
    ReferenceDocument,
    validate_allowlisted_https_url,
)
from crewforge.core.reviewer import (
    REVIEWER_COMBINED_ID,
    REVIEWER_CONSISTENCY_ID,
    ReviewerPool,
)
from crewforge.providers.base import ProviderConfigurationError
from crewforge.providers.ollama_provider import OllamaProvider
from crewforge.providers.openai_provider import OpenAIProvider
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
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog
from tests.fake_provider import FakeProvider


def _base_plan_output(reqs: list[str] | None = None) -> PlannerOutput:
    return PlannerOutput.model_validate(
        {
            "goal": "Verified analysis",
            "user_requirements": reqs or ["Evaluate cash flow attractiveness"],
            "acceptance_criteria": ["All released statements verified"],
            "agents": [
                {
                    "agent_ref": "a1",
                    "role": "Analyst",
                    "instructions": "Analyze.",
                    "capabilities": ["analysis", "calculation"],
                }
            ],
            "tasks": [
                {
                    "task_ref": "t1",
                    "title": "Analyze prompt",
                    "description": "Perform task.",
                    "assigned_agent_ref": "a1",
                    "depends_on": [],
                    "expected_deliverables": ["Verified claims"],
                }
            ],
        }
    )


async def test_claim_locked_release_blocks_negation_reversal_and_laundering() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        settings = CrewForgeSettings(_env_file=None, db_path=Path(":memory:"))
        await db.create_run("run_lock", "network", "Assess cash flow.", settings.to_run_config())
        plan = validate_and_materialize_plan(
            run_id="run_lock",
            raw_plan=_base_plan_output(["Explain investment attractiveness"]),
            settings=settings,
        )
        await db.save_execution_plan(plan)

        approved = Claim(
            claim_id="clm_pos",
            run_id="run_lock",
            statement="Positive monthly cash flow does not establish investment attractiveness.",
            category=ClaimCategory.INFERENCE,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
            version=1,
            author_agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
            requirement_refs=["req_1"],
        )
        rejected = Claim(
            claim_id="clm_rej",
            run_id="run_lock",
            statement="Project guarantees 400% internal rate of return.",
            category=ClaimCategory.INFERENCE,
            status=ClaimStatus.REJECTED,
            verification_status=VerificationStatus.CONTRADICTED,
            version=1,
            author_agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
        )
        await db.insert_claim(approved)
        await db.insert_claim(rejected)

        verdict = ReviewVerdict(
            run_id="run_lock",
            claim_id=approved.claim_id,
            claim_version=1,
            reviewer_id=REVIEWER_COMBINED_ID,
            review_dimension="combined_evidence_consistency",
            decision=ReviewDecision.APPROVE,
            rationale="Valid caveat",
        )
        await db.insert_review_verdict(verdict)

        syn = SynthesisOutput(
            user_requirements=["Explain investment attractiveness"],
            accepted_segments=[
                # Negation reversal sharing identical keywords!
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text="Positive monthly cash flow guarantees investment attractiveness.",
                    claim_refs=[ClaimVersionRef(claim_id="clm_pos", claim_version=1)],
                    requirement_ref="req_1",
                    is_factual_assertion=True,
                )
            ],
            conditional_conclusions=[
                ReleaseSegment(
                    segment_type="conditional_conclusion",
                    text="Cash flow guarantees return without risk.",
                    claim_refs=[],
                    is_factual_assertion=True,
                )
            ],
            recommendations=[
                # Laundering rejected factual premise in recommendations!
                ReleaseSegment(
                    segment_type="recommendation",
                    text="Invest immediately because 400% return is assured.",
                    claim_refs=[],
                    is_factual_assertion=False,
                )
            ],
            assumptions=["Project guarantees 400% internal rate of return."],
            missing_information=["Actual discount rate is missing, proved 88% higher elsewhere."],
            unresolved_issues=["Project definitely caused 77% revenue spike."],
        )

        gate = HonestyGate(db)
        res = gate.evaluate(
            run_id="run_lock",
            original_prompt="Assess cash flow.",
            plan=plan,
            synthesis=syn,
            all_claims=[approved, rejected],
            evidence_list=[],
            tool_executions=[],
            verdicts=[verdict],
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=False,
        )
        assert res.gate_record is not None
        assert res.gate_record.passed is False
        assert "guarantees investment attractiveness" not in res.rendered_answer
        assert "400%" not in res.rendered_answer
        assert "88%" not in res.rendered_answer
        assert "77%" not in res.rendered_answer
        assert res.assumptions == []
        assert res.missing_information == []
        assert res.verification.content_released_safely is False
        # Repeating the question / unlinked requirement gets 0 coverage
        assert res.verification.requirement_coverage == 0.0
        assert res.verification.requirements_missing == 1
    finally:
        await db.close()


async def test_honest_coverage_and_partial_safety_distinction() -> None:
    db = Database(":memory:")
    await db.initialize()
    try:
        settings = CrewForgeSettings(_env_file=None, db_path=Path(":memory:"))
        await db.create_run("run_cov", "network", "Two questions.", settings.to_run_config())
        plan = validate_and_materialize_plan(
            run_id="run_cov",
            raw_plan=_base_plan_output(["Question one", "Question two"]),
            settings=settings,
        )
        await db.save_execution_plan(plan)

        c1 = Claim(
            claim_id="clm_q1",
            run_id="run_cov",
            statement="Answer one is verified.",
            category=ClaimCategory.INFERENCE,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
            version=1,
            author_agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
            requirement_refs=["req_1"],
        )
        await db.insert_claim(c1)
        verdict = ReviewVerdict(
            run_id="run_cov",
            claim_id="clm_q1",
            claim_version=1,
            reviewer_id=REVIEWER_COMBINED_ID,
            review_dimension="combined_evidence_consistency",
            decision=ReviewDecision.APPROVE,
            rationale="Verified",
        )
        await db.insert_review_verdict(verdict)

        syn = SynthesisOutput(
            user_requirements=["Question one", "Question two"],
            accepted_segments=[
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text="",  # Empty triggers direct verbatim ledger rendering
                    claim_refs=[ClaimVersionRef(claim_id="clm_q1", claim_version=1)],
                    requirement_ref="req_1",
                    is_factual_assertion=True,
                )
            ],
        )
        gate = HonestyGate(db)
        res = gate.evaluate(
            run_id="run_cov",
            original_prompt="Two questions.",
            plan=plan,
            synthesis=syn,
            all_claims=[c1],
            evidence_list=[],
            tool_executions=[],
            verdicts=[verdict],
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=True,  # Partial execution whose released content passed checks
        )
        assert "Answer one is verified." in res.rendered_answer
        assert res.verification.content_released_safely is True
        assert res.verification.execution_completed is False
        assert res.verification.honesty_gate_passed is False
        assert res.verification.requirements_answered == 1
        assert res.verification.requirements_missing == 1
        assert res.verification.requirement_coverage == pytest.approx(0.5)
        assert res.verification.independent_grade is None
    finally:
        await db.close()


async def test_arithmetic_lineage_rejects_swaps_fabricated_unrelated_stale_and_literals(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "lineage.db"
    settings = CrewForgeSettings(_env_file=None, db_path=db_path)
    db = Database(db_path)
    await db.initialize()
    try:
        await db.create_run("run_lin", "network", "Test lineage", settings.to_run_config())
        plan = validate_and_materialize_plan(
            run_id="run_lin",
            raw_plan=_base_plan_output(["Compute"]),
            settings=settings,
        )
        await db.save_execution_plan(plan)
        agent_id = plan.agents[0].agent_id
        task_id = plan.tasks[0].task_id

        prompt = (
            "The initial investment is $120,000, monthly revenue is $10,000, and monthly "
            "expenses are $5,000. Unrelated server latency log showed 5000 ms."
        )
        c_v1 = Claim(
            claim_id="clm_prem",
            run_id="run_lin",
            statement="Net monthly cash flow is 4000 dollars.",
            category=ClaimCategory.COMPUTATION,
            status=ClaimStatus.REJECTED,
            verification_status=VerificationStatus.CONTRADICTED,
            version=1,
            author_agent_id=agent_id,
            task_id=task_id,
        )
        c_v2 = Claim(
            claim_id="clm_prem",
            run_id="run_lin",
            statement="Net monthly cash flow is 5000 USD.",
            category=ClaimCategory.COMPUTATION,
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.COMPUTATIONALLY_VERIFIED,
            version=2,
            parent_claim_id="clm_prem",
            author_agent_id=agent_id,
            task_id=task_id,
        )
        await db.insert_claim(c_v1)
        await db.insert_claim(c_v2)

        tool = DeterministicArithmeticTool(db)
        catalog = build_provenance_catalog(
            original_prompt=prompt,
            all_claims=[c_v1, c_v2],
            assumptions=["12 months per year"],
        )

        # 1. Wrong variable binding (swapped 10000 as inv and 120000 as rev in up_1)
        swapped = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="inv / (rev - exp)",
                variables={"inv": "10000", "rev": "120000", "exp": "5000"},
                input_provenance_refs={"inv": "up_1", "rev": "up_1", "exp": "up_1"},
                proposed_result="1",
            ),
        )
        assert swapped.inputs_grounded is False

        # 2. Fabricated provenance reference
        fabricated = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="inv / 2",
                variables={"inv": "120000"},
                input_provenance_refs={"inv": "fabricated_table_3"},
            ),
        )
        assert fabricated.inputs_grounded is False

        # 3. Matching number in unrelated sentence (up_2 has 5000 ms latency)
        wrong_sentence = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="inv / (rev - exp)",
                variables={"inv": "120000", "rev": "10000", "exp": "5000"},
                input_provenance_refs={"inv": "up_1", "rev": "up_1", "exp": "up_2"},
                units="months",
            ),
        )
        assert wrong_sentence.inputs_grounded is False

        # 4. Correct expression using ungrounded inline constant (5000 hardcoded inside AST)
        inline_const = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="inv / 5000",
                variables={"inv": "120000"},
                input_provenance_refs={"inv": "up_1"},
                proposed_result="24",
                units="months",
            ),
        )
        assert inline_const.has_ungrounded_literals is True
        assert inline_const.inputs_grounded is False

        # 5. Stale premise reference (`clm_prem@v1` when `v2` exists)
        stale_prem = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="inv / cash",
                variables={"inv": "120000", "cash": "4000"},
                input_provenance_refs={"inv": "up_1", "cash": "clm_prem@v1"},
            ),
        )
        assert stale_prem.inputs_grounded is False

        # 6. Valid chained computation using supported claim v2 + explicit constant
        chained = await tool.execute(
            run_id="run_lin",
            actor=agent_id,
            catalog=catalog,
            request=ArithmeticRequest(
                expression="(inv / cash) / months_per_yr",
                variables={"inv": "120000", "cash": "5000", "months_per_yr": "12"},
                input_provenance_refs={
                    "inv": "up_1",
                    "cash": "clm_prem@v2",
                    "months_per_yr": "const:months_per_year",
                },
                proposed_result="2",
                units="years",
            ),
        )
        assert chained.success is True
        assert chained.inputs_grounded is True
        assert chained.has_ungrounded_literals is False
        assert chained.matches_proposed is True
    finally:
        await db.close()


async def test_reviewer_dedup_missing_dimension_and_stale_reapproval_block(tmp_path: Path) -> None:
    db_path = tmp_path / "rev.db"
    settings = CrewForgeSettings(
        _env_file=None,
        db_path=db_path,
        separate_review_dimensions=True,
        max_retries=0,
    )
    db = Database(db_path)
    await db.initialize()
    try:
        await db.create_run("run_rv", "network", "Check logic", settings.to_run_config())
        plan = validate_and_materialize_plan(
            run_id="run_rv",
            raw_plan=_base_plan_output(["Check"]),
            settings=settings,
        )
        await db.save_execution_plan(plan)

        ledger = ClaimLedger(db)
        evds, claims, _ = await ledger.record_specialist_outputs(
            run_id="run_rv",
            agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
            original_prompt="Check logic",
            dependency_findings_text="",
            evidence_proposals=[
                ProposedEvidence(
                    source_type="user_prompt", source_ref="up_1", excerpt="Check logic"
                )
            ],
            claim_proposals=[
                ProposedClaim(
                    statement="Logic follows.",
                    category=ClaimCategory.INFERENCE,
                    evidence_indices=[0],
                )
            ],
        )
        claim = claims[0]

        call_count = 0

        def handler(_prompt: str, _rm: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Duplicate approve entries in pass 1 (evidence); pass 2 fails!
                return {
                    "verdicts": [
                        {
                            "claim_id": "c1",  # alias support!
                            "decision": "approve",
                            "rationale": "First approval",
                            "evidence_refs": [evds[0].evidence_id, "evd_wrong_claim"],
                            "blocking_severity": "none",
                        },
                        {
                            "claim_id": claim.claim_id,
                            "claim_version": 1,
                            "decision": "approve",
                            "rationale": "Duplicate attempt",
                            "evidence_refs": [],
                            "blocking_severity": "none",
                        },
                    ],
                    "missing_requirement_coverage": [],
                }
            # Pass 2 returns empty (missing dimension verdict)
            return {"verdicts": [], "missing_requirement_coverage": []}

        executor = ModelExecutor(
            provider=FakeProvider(handler=handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        pool = ReviewerPool(executor=executor, database=db, settings=settings)
        outcome = await pool.review_claims(
            run_id="run_rv", original_prompt="Check logic", plan=plan
        )
        assert len(outcome.verdicts) == 1  # Duplicate filtered out!
        assert outcome.verdicts[0].evidence_refs == [evds[0].evidence_id]
        # Missing second required dimension leaves claim UNRESOLVED, never SUPPORTED!
        assert outcome.updated_claims[claim.claim_id].status == ClaimStatus.UNRESOLVED

        # Verify stale claim cannot be overwritten to SUPPORTED by an old batch
        await db.update_claim_verification_state(
            run_id="run_rv",
            claim_id=claim.claim_id,
            version=1,
            status=ClaimStatus.STALE,
            verification_status=VerificationStatus.STALE_PENDING_REVIEW,
        )
        c_dep = Claim(
            claim_id="clm_dep_stale",
            run_id="run_rv",
            statement="Derived on stale claim.",
            category=ClaimCategory.INFERENCE,
            status=ClaimStatus.ASSERTED,
            version=1,
            author_agent_id=plan.agents[0].agent_id,
            task_id=plan.tasks[0].task_id,
            premise_refs=[ClaimVersionRef(claim_id=claim.claim_id, claim_version=1)],
        )
        await db.insert_claim(c_dep)
        outcome2 = await pool.review_claims(
            run_id="run_rv",
            original_prompt="Check logic",
            plan=plan,
            target_claims=[c_dep],
        )
        assert outcome2.updated_claims[c_dep.claim_id].status == ClaimStatus.STALE
        assert any(v.reviewer_id == REVIEWER_CONSISTENCY_ID for v in outcome2.verdicts)
    finally:
        await db.close()


async def test_document_ingestion_ssrf_blocking_and_provider_portability(tmp_path: Path) -> None:
    # 1. Genuine local document ingestion + equal Baseline vs Network access
    doc = ReferenceDocument(
        doc_id="policy_doc",
        title="SLA Policy",
        content="Maximum allowed latency p99 is 250 ms under regional failover.",
        source_ref="sla_policy.md",
        reliability=EvidenceReliability.HIGH,
    )
    retriever = LocalDocumentRetriever([doc])
    assert retriever.retrieval_available is True
    hits = await retriever.retrieve("latency failover", run_id="run_doc")
    assert len(hits) == 1
    assert hits[0].content_hash
    assert "sla_policy.md#chars:" in hits[0].source_location

    db_path = tmp_path / "doc.db"
    db = Database(db_path)
    await db.initialize()
    try:
        settings = CrewForgeSettings(_env_file=None, db_path=db_path)
        await db.create_run("run_doc", "network", "Check SLA", settings.to_run_config())
        evds = await retriever.ingest_into_ledger(database=db, run_id="run_doc")
        assert len(evds) == 1
        assert evds[0].verified_in_input is True
        assert evds[0].content_hash == hits[0].content_hash
        assert evds[0].retrieval_status == "ok"
    finally:
        await db.close()

    # 2. SSRF & disabled HTTP retrieval controls
    disabled_http = AllowlistedHttpRetriever(enabled=False)
    res_disabled = await disabled_http.fetch_url("https://docs.example.com/guide")
    assert res_disabled.status == "unavailable"

    def private_dns(_host: str, _port: int, type: int = 0) -> list[Any]:  # noqa: A002
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    ok_ssrf, err_ssrf = validate_allowlisted_https_url(
        "https://docs.example.com/secret",
        allowed_hosts=["docs.example.com"],
        resolver=private_dns,
    )
    assert ok_ssrf is False
    assert "SSRF" in (err_ssrf or "")

    def redirect_handler(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/admin"})

    http_redir = AllowlistedHttpRetriever(
        enabled=True,
        allowed_hosts=["docs.example.com"],
        transport=httpx.MockTransport(redirect_handler),
        dns_resolver=lambda _h, _p, type=0: [  # noqa: A002
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    redir_res = await http_redir.fetch_url("https://docs.example.com/page")
    assert redir_res.status == "blocked"

    # 3. Provider parameter compatibility & returned_model tracking
    seen_kwargs: list[dict[str, Any]] = []

    class FakeReasoningClient:
        async def parse(self, **kwargs: Any) -> Any:
            seen_kwargs.append(kwargs)
            return SimpleNamespace(
                id="resp_luna",
                model="gpt-6-luna-2026-10-01",
                output_text='{"verdicts":[],"missing_requirement_coverage":[]}',
                output_parsed=ReviewerBatchOutput(),
                usage=SimpleNamespace(input_tokens=15, output_tokens=8, total_tokens=23),
            )

    luna_provider = OpenAIProvider(
        model="gpt-6-luna",
        api_key="sk-test",
        reasoning_effort="low",
        client=SimpleNamespace(responses=FakeReasoningClient()),  # type: ignore[arg-type]
    )
    assert luna_provider.supports_reasoning_effort is True
    assert luna_provider.supports_temperature is False
    luna_res = await luna_provider.generate_structured(
        "Check", ReviewerBatchOutput, temperature=0.2
    )
    assert seen_kwargs[0]["reasoning"] == {"effort": "low"}
    assert "temperature" not in seen_kwargs[0]
    assert luna_res.returned_model == "gpt-6-luna-2026-10-01"

    # OpenAI-compatible requires base_url
    compat_settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI_COMPATIBLE,
        model="local-compat",
        openai_api_key="sk-compat",
        openai_base_url=None,
    )
    with pytest.raises(ProviderConfigurationError, match="CREWFORGE_OPENAI_BASE_URL"):
        OpenAIProvider.from_settings(compat_settings)

    # Ollama records returned_model cleanly
    def ollama_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:4b",
                "message": {
                    "role": "assistant",
                    "content": '{"verdicts":[],"missing_requirement_coverage":[]}',
                },
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(ollama_handler)
    ) as o_client:
        ol = OllamaProvider(model="qwen3:4b", client=o_client)
        o_res = await ol.generate_structured("Hi", ReviewerBatchOutput)
        assert o_res.returned_model == "qwen3:4b"


async def test_fast_path_baseline_compute_matched_and_paid_guard(tmp_path: Path) -> None:
    # 1. Paid run authorization guard defaults to False
    default_settings = CrewForgeSettings(_env_file=None, provider=ProviderType.OPENAI)
    assert default_settings.allow_paid_runs is False
    assert default_settings.effective_db_path == Path(".crewforge/cloud_dev.db")
    assert default_settings.effective_max_concurrency == 3

    db_guard = Database(tmp_path / "guard.db")
    await db_guard.initialize()
    try:
        await db_guard.create_run("run_guard", "baseline", "Hi", default_settings.to_run_config())
        guard_bm = BudgetManager(
            db_guard,
            default_settings.model_copy(
                update={
                    "pricing_json": {
                        "gpt-6-luna": ModelPricing(input_per_1m=0.1, output_per_1m=0.5)
                    }
                }
            ),
        )
        with pytest.raises(BudgetError, match="CREWFORGE_ALLOW_PAID_RUNS=true"):
            await guard_bm.reserve_call(
                run_id="run_guard",
                provider="openai",
                model="gpt-6-luna",
                is_paid_provider=True,
                prompt_text="Hi",
            )
    finally:
        await db_guard.close()

    # 2. Deterministic synthesis fast path skips synthesis call & runs Gate
    db_fp = Database(tmp_path / "fp.db")
    await db_fp.initialize()
    fp_settings = CrewForgeSettings(
        _env_file=None,
        db_path=tmp_path / "fp.db",
        max_communication_rounds=0,
        enable_deterministic_synthesis_fast_path=True,
    )
    seen_models: list[type[BaseModel] | None] = []

    def fp_handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        seen_models.append(response_model)
        if response_model is PlannerOutput:
            return _base_plan_output(["State the initial cost"]).model_dump()
        if response_model is SpecialistOutput:
            return {
                "findings": "Initial cost is 100 dollars.",
                "claims": [
                    {
                        "statement": "Initial cost is 100 dollars.",
                        "category": "given_fact",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_1"],
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "user_prompt",
                        "source_ref": "up_1",
                        "excerpt": "initial cost is 100 dollars",
                    }
                ],
                "task_completed": True,
            }
        # Reviewer approves c1
        return {
            "verdicts": [
                {
                    "claim_id": "c1",
                    "decision": "approve",
                    "rationale": "Exact match.",
                    "blocking_severity": "none",
                }
            ],
            "missing_requirement_coverage": [],
        }

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=fp_handler, max_retries=0),
            database=db_fp,
            budget_manager=BudgetManager(db_fp, fp_settings),
            settings=fp_settings,
        )
        res = await runner.run("The initial cost is 100 dollars.")
        assert res.status == RunStatus.COMPLETED
        assert res.synthesis_mode == "deterministic_fast_path"
        assert SynthesisOutput not in seen_models
        assert res.usage.model_calls == 3  # planner + worker + reviewer only!
        assert "Initial cost is 100 dollars." in res.answer

        # 3. Compute-matched baseline remains single actor="baseline" with self-check pass
        b_calls = 0

        def b_handler(_prompt: str, _rm: Any) -> dict[str, Any]:
            nonlocal b_calls
            b_calls += 1
            return {
                "answer": f"Single-agent pass #{b_calls} verified 100 dollars.",
                "missing_information": [],
                "assumptions": [],
                "computations": [],
                "needs_tool_followup": False,
            }

        b_runner = BaselineRunner(
            provider=FakeProvider(handler=b_handler, max_retries=0),
            database=db_fp,
            budget_manager=BudgetManager(db_fp, fp_settings),
            settings=fp_settings,
        )
        b_res = await b_runner.run("The initial cost is 100 dollars.", compute_matched=True)
        assert b_res.status == RunStatus.COMPLETED
        assert b_res.baseline_mode == "compute_matched"
        assert b_res.usage.model_calls == 2
        b_trace = await db_fp.get_run_trace(b_res.run_id)
        assert b_trace is not None
        assert b_trace.agents == []
        assert b_trace.review_verdicts == []
    finally:
        await db_fp.close()
