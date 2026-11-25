"""Tests for Phase 5 competitive evaluation, unified budget safeguards, and stabilization."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import (
    BudgetConstraintError,
    UnifiedBudgetAuthority,
)
from crewforge.core.demo_trace import export_demo_trace
from crewforge.core.honesty_gate import (
    HonestyGate,
    _detect_auxiliary_laundering,
)
from crewforge.core.retrieval import (
    ReferenceDocument,
    _chunk_document,
)
from crewforge.evaluation.anonymize import (
    anonymize_pair,
    export_evaluator_packet,
)
from crewforge.evaluation.contracts import (
    EvaluationRubric,
    ExpectedLogicalCheck,
    ExpectedMissingInfoCheck,
    ExpectedNumericalCheck,
    ExperimentResult,
    PromptDomain,
)
from crewforge.evaluation.scoring import compute_evaluation_score
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    EvidenceReliability,
    ExecutionPlan,
    ReleaseSegment,
    RunResult,
    RunStatus,
    SynthesisOutput,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import build_provenance_catalog


@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize_sync()
    return db


def test_honesty_gate_category_enforcement(temp_db: Database) -> None:
    """Verify unsupported factual claims cannot be laundered as recommendations/conditions."""
    gate = HonestyGate(temp_db)
    run_id = f"run_{uuid4().hex[:12]}"

    premise_claim = Claim(
        claim_id="clm_premise_bad",
        version=1,
        run_id=run_id,
        statement="Database writes were lost during partition.",
        category=ClaimCategory.EXTERNAL_FACT,
        status=ClaimStatus.ASSERTED,  # Not supported!
        verification_status=VerificationStatus.UNSUPPORTED,
        author_agent_id="agent_1",
        task_id="t1",
        is_material=True,
    )

    unsupported_fact_as_rec = Claim(
        claim_id="clm_unsupported_fact",
        version=1,
        run_id=run_id,
        statement="Database throughput dropped by 50% permanently.",
        category=ClaimCategory.EXTERNAL_FACT,  # Factual assertion
        status=ClaimStatus.ASSERTED,
        verification_status=VerificationStatus.UNSUPPORTED,
        author_agent_id="agent_1",
        task_id="t1",
        is_material=True,
    )

    advisory_rec_with_bad_premise = Claim(
        claim_id="clm_advisory_rec",
        version=1,
        run_id=run_id,
        statement="Fail over to secondary replica immediately.",
        category=ClaimCategory.RECOMMENDATION,
        status=ClaimStatus.ASSERTED,
        verification_status=VerificationStatus.UNSUPPORTED,
        author_agent_id="agent_1",
        task_id="t1",
        premise_refs=[ClaimVersionRef(claim_id="clm_premise_bad", claim_version=1)],
        is_material=False,
    )

    all_claims = [premise_claim, unsupported_fact_as_rec, advisory_rec_with_bad_premise]
    plan = ExecutionPlan(
        run_id=run_id,
        goal="Evaluate failure.",
        user_requirements=["Evaluate failure."],
    )

    # 1. Attempt to release unsupported fact under recommendations
    synth_leak = SynthesisOutput(
        user_requirements=plan.user_requirements,
        accepted_segments=[],
        conditional_conclusions=[],
        recommendations=[
            ReleaseSegment(
                segment_type="recommendation",
                text="Database throughput dropped by 50% permanently.",
                claim_refs=[ClaimVersionRef(claim_id="clm_unsupported_fact", claim_version=1)],
            )
        ],
    )
    res_leak = gate.evaluate(
        run_id=run_id,
        original_prompt="Evaluate failure.",
        plan=plan,
        synthesis=synth_leak,
        all_claims=all_claims,
        evidence_list=[],
        tool_executions=[],
        verdicts=[],
        incomplete_tasks=[],
        extra_assumptions=[],
        extra_missing=[],
        uncertainties=[],
        is_partial_run=False,
    )
    assert not res_leak.gate_record.passed
    assert any("Unsupported factual claim" in r for r in res_leak.gate_record.reasons)

    # 2. Attempt to release recommendation with unsupported premise
    synth_rec_bad_premise = SynthesisOutput(
        user_requirements=plan.user_requirements,
        accepted_segments=[],
        conditional_conclusions=[],
        recommendations=[
            ReleaseSegment(
                segment_type="recommendation",
                text="Fail over to secondary replica immediately.",
                claim_refs=[ClaimVersionRef(claim_id="clm_advisory_rec", claim_version=1)],
            )
        ],
    )
    res_rec = gate.evaluate(
        run_id=run_id,
        original_prompt="Evaluate failure.",
        plan=plan,
        synthesis=synth_rec_bad_premise,
        all_claims=all_claims,
        evidence_list=[],
        tool_executions=[],
        verdicts=[],
        incomplete_tasks=[],
        extra_assumptions=[],
        extra_missing=[],
        uncertainties=[],
        is_partial_run=False,
    )
    assert not res_rec.gate_record.passed
    assert any("depends on unsupported material premise" in r for r in res_rec.gate_record.reasons)


def test_auxiliary_laundering_and_number_containment() -> None:
    """Verify auxiliary prose cannot introduce unapproved numbers or echo rejected claims."""
    rejected = Claim(
        claim_id="clm_bad_rej",
        version=1,
        run_id="run_1",
        statement="The program guarantees 100% pass rates across all cohorts.",
        category=ClaimCategory.EXTERNAL_FACT,
        status=ClaimStatus.REJECTED,
        verification_status=VerificationStatus.CONTRADICTED,
        author_agent_id="a1",
        task_id="t1",
    )
    allowed_numbers = {Decimal("15"), Decimal("0.15"), Decimal("25"), Decimal("0.25")}

    # Introduces extra unapproved number
    err_num = _detect_auxiliary_laundering(
        "We also observed an increase of 42 students.",
        allowed_numbers=allowed_numbers,
        rejected_claims=[rejected],
    )
    assert err_num is not None
    assert "unverified number" in err_num

    # Echoes rejected claim
    err_echo = _detect_auxiliary_laundering(
        "It is assumed the program guarantees 100% pass rates across all cohorts.",
        allowed_numbers=allowed_numbers | {Decimal("100"), Decimal("1")},
        rejected_claims=[rejected],
    )
    assert err_echo is not None
    assert "echoes rejected claim" in err_echo or "mirrors rejected claim" in err_echo

    # Clean text with allowed number
    ok_text = _detect_auxiliary_laundering(
        "Relative increase was calculated as 25%.",
        allowed_numbers=allowed_numbers,
        rejected_claims=[rejected],
    )
    assert ok_text is None


def test_document_chunking_and_multi_chunk_provenance() -> None:
    """Verify clean document chunking (no mid-word slicing) and provenance catalog resolution."""
    doc_text = (
        "# Service SLA Policy\n\n"
        "Provisioned capacity is 1,200 QPS.\n"
        "Peak holiday load is 3,000 QPS across checkout (40%) and catalog (60%).\n"
        "Replication lag under heavy write bursts above 500 writes/sec has not been benchmarked."
    )
    doc = ReferenceDocument(
        doc_id="doc_file_1",
        title="policy.md",
        content=doc_text,
        source_ref="policy.md",
        reliability=EvidenceReliability.HIGH,
    )

    # Chunking: under max_chunk_chars, should produce exactly 1 whole chunk
    chunks = _chunk_document(doc, max_chunk_chars=2000)
    assert len(chunks) == 1
    assert "benchmarked." in chunks[0]["excerpt"]
    assert not chunks[0]["excerpt"].endswith("be")

    # Provenance Catalog resolution
    evd = Evidence(
        evidence_id="evd_policy_1",
        run_id="run_1",
        source_type="external_retrieved",
        source_ref="policy.md",
        excerpt=chunks[0]["excerpt"],
        created_by="operator",
        verified_in_input=True,
    )
    catalog = build_provenance_catalog(
        original_prompt="Calculate checkout peak.",
        verified_evidence=[evd],
    )
    assert "doc_1" in catalog.targets_by_key
    assert "policy.md" in catalog.targets_by_key
    assert catalog.targets_by_key["doc_1"].is_trusted
    assert "3,000" in catalog.targets_by_key["doc_1"].text


def test_unified_budget_authority_migration_and_caps(tmp_path: Path) -> None:
    """Verify historical migration without double-counting and protected reserve enforcement."""
    # 1. Create a dummy historical database with 2 reservations
    hist_db_path = tmp_path / "crewforge.db"
    hist_db = Database(hist_db_path)
    hist_db.initialize_sync()
    with hist_db._lock:
        conn = hist_db._get_connection()
        conn.execute(
            """
            INSERT INTO runs (
                run_id, mode, status, prompt, provider, model, config_json, created_at
            ) VALUES (
                'run_h1', 'baseline', 'completed', 'prompt',
                'openai', 'gpt-6-luna', '{}', '2026-11-25T12:00:00Z'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO budget_reservations (
                reservation_id, run_id, provider, model, reserved_cost_usd,
                actual_cost_usd, reserved_calls, retries, status, created_at, updated_at
            ) VALUES (
                'res_hist_01', 'run_h1', 'openai', 'gpt-6-luna', 0.001, 0.0005, 1, 0,
                'reconciled', '2026-11-25T12:00:00Z', '2026-11-25T12:00:01Z'
            )
            """
        )
        conn.commit()

    # 2. Initialize UnifiedBudgetAuthority pointing to a clean authority DB
    auth_db_path = tmp_path / "budget_authority.db"
    settings = CrewForgeSettings(
        budget_authority_path=auth_db_path,
        total_budget_ceiling_usd=6.10,
        protected_reserve_usd=4.00,  # Max spendable = $2.10
        phase5_same_model_budget_cap_usd=0.75,
        david_vs_goliath_budget_cap_usd=1.10,
        allow_paid_runs=True,
    )

    authority = UnifiedBudgetAuthority(authority_path=auth_db_path, settings=settings)
    summary = authority.get_summary_sync()
    assert summary["protected_reserve_usd"] == 4.00
    assert summary["max_spendable_usd"] == pytest.approx(2.10)

    # 3. Test allocation category enforcement
    # Should succeed: $0.10 in phase5_same_model
    authority.reserve_sync(
        reservation_id="res_p5_01",
        run_id="run_p5_1",
        provider="openai",
        model="gpt-6-luna",
        is_paid_provider=True,
        estimated_cost_usd=0.10,
        allocation_category="phase5_same_model",
    )

    # Should fail: exceeding $0.75 phase 5 cap ($0.10 + $0.70 = $0.80 > $0.75)
    with pytest.raises(
        BudgetConstraintError, match="Phase 5 same-model benchmarking allocation cap exceeded"
    ):
        authority.reserve_sync(
            reservation_id="res_p5_02",
            run_id="run_p5_2",
            provider="openai",
            model="gpt-6-luna",
            is_paid_provider=True,
            estimated_cost_usd=0.70,
            allocation_category="phase5_same_model",
        )

    # Should fail: breaching the $2.10 global spendable ceiling ($4.00 protected reserve)
    with pytest.raises(BudgetConstraintError, match="Protected global reserve violation"):
        authority.reserve_sync(
            reservation_id="res_huge_01",
            run_id="run_huge_1",
            provider="openai",
            model="gpt-6.2-astra",
            is_paid_provider=True,
            estimated_cost_usd=2.50,
            allocation_category="general",
        )


def test_objective_checks_and_scoring() -> None:
    """Verify diagnostics stay separate from semantic grading until human grade is supplied."""
    from crewforge.evaluation.contracts import HumanGradeInput
    from crewforge.evaluation.reporting import generate_markdown_report
    from crewforge.evaluation.scoring import finalize_human_score

    rubric = EvaluationRubric(
        prompt_id="test_01",
        domain=PromptDomain.CAUSATION,
        numerical_checks=[
            ExpectedNumericalCheck(
                variable_name="pp_increase",
                expected_value=15.0,
                is_percentage_points=True,
                points=15.0,
            ),
            ExpectedNumericalCheck(
                variable_name="rel_increase",
                expected_value=25.0,
                is_percentage=True,
                points=15.0,
            ),
        ],
        logical_checks=[
            ExpectedLogicalCheck(
                check_id="no_causation",
                description="Does not claim proof of causation.",
                required_phrases=["does not prove"],
                forbidden_phrases=["proves causation"],
                points=20.0,
            )
        ],
        missing_info_checks=[
            ExpectedMissingInfoCheck(
                check_id="control",
                description="Identifies control group requirement.",
                required_missing_concepts=["control group"],
                points=10.0,
            )
        ],
    )

    good_answer = (
        "The pass rate increased by 15 percentage points and by 25% relative increase. "
        "This does not prove the program caused the improvement because confounding was unmanaged. "
        "Evidence from a randomized control group is needed."
    )
    # Without human grade, 100-pt semantic score MUST remain pending (no fabricated grade!)
    pending_score = compute_evaluation_score(
        rubric=rubric,
        system_label="test_system",
        answer_text=good_answer,
        missing_information=["A randomized control group is missing."],
    )
    assert pending_score.grading_status == "pending_independent_grading"
    assert pending_score.total_score is None
    assert pending_score.factual_correctness is None
    assert pending_score.objective_score == 60.0

    # Weak keywords like "data", "policy", "based on" must NOT fabricate a score
    keyword_only = "Based on policy data, the result is unknown."
    weak_score = compute_evaluation_score(
        rubric=rubric,
        system_label="weak_system",
        answer_text=keyword_only,
    )
    assert weak_score.grading_status == "pending_independent_grading"
    assert weak_score.total_score is None
    assert weak_score.evidence_grounding is None

    # Contradictory/negated numerical match must NOT pass numerical check
    negated_answer = "The pass rate did not increase by 15 percentage points; 25% was rejected."
    neg_score = compute_evaluation_score(
        rubric=rubric,
        system_label="neg_system",
        answer_text=negated_answer,
    )
    num_checks = [c for c in neg_score.objective_checks if c.check_name.startswith("num_")]
    assert all(not c.passed for c in num_checks)

    # Ambiguous unit match (unit in different clause) -> ambiguous_manual_review
    unit_rubric = EvaluationRubric(
        prompt_id="unit_01",
        domain=PromptDomain.FINANCIAL,
        numerical_checks=[
            ExpectedNumericalCheck(
                variable_name="payback",
                expected_value=24.0,
                expected_units="months",
                points=10.0,
            )
        ],
    )
    ambig_score = compute_evaluation_score(
        rubric=unit_rubric,
        system_label="ambig",
        answer_text="The value is 24. Later we discuss months in general.",
    )
    assert ambig_score.objective_checks[0].status == "ambiguous_manual_review"
    assert ambig_score.objective_checks[0].needs_manual_review is True
    assert ambig_score.objective_checks[0].passed is False

    # Approved rubric human grade calculation (30/25/20/15/10, 5pt penalty capped at 20, >= 0)
    hg = HumanGradeInput(
        prompt_id="test_01",
        system_label="network",
        factual_correctness=28.0,
        requirement_coverage=22.0,
        evidence_grounding=18.0,
        internal_consistency=14.0,
        usefulness_and_uncertainty=9.0,
        unsupported_material_assertion_count=5,  # 5 * 5 = 25 -> capped at 20!
        evaluator_id="judge_anon_01",
        grading_method="human_blind_rubric",
    )
    graded = finalize_human_score(
        rubric=rubric,
        system_label="network",
        human_grade=hg,
        objective_checks=pending_score.objective_checks,
    )
    assert graded.grading_status == "graded"
    assert graded.unsupported_assertion_penalty == 20.0
    assert graded.total_score == 71.0
    assert graded.evaluator_id == "judge_anon_01"

    # Non-negative floor when penalty exceeds raw points
    low_hg = HumanGradeInput(
        prompt_id="test_01",
        system_label="baseline",
        factual_correctness=5.0,
        requirement_coverage=5.0,
        evidence_grounding=0.0,
        internal_consistency=0.0,
        usefulness_and_uncertainty=0.0,
        unsupported_material_assertion_count=4,
        evaluator_id="judge_anon_01",
    )
    low_graded = finalize_human_score(rubric=rubric, system_label="baseline", human_grade=low_hg)
    assert low_graded.total_score == 0.0

    # Reporting shows "pending independent grading" when ungraded
    ungraded_exp = ExperimentResult(
        prompt_id="test_01",
        domain=PromptDomain.CAUSATION,
        baseline_run_id="r1",
        network_run_id="r2",
        baseline_result=RunResult(
            run_id="r1",
            mode="baseline",
            status=RunStatus.COMPLETED,
            prompt="p",
            provider="ollama",
            model="qwen3:4b",
            answer="a",
        ),
        network_result=RunResult(
            run_id="r2",
            mode="network",
            status=RunStatus.COMPLETED,
            prompt="p",
            provider="ollama",
            model="qwen3:4b",
            answer="b",
        ),
        baseline_score=pending_score,
        network_score=pending_score,
    )
    md = generate_markdown_report([ungraded_exp])
    assert "pending independent grading" in md
    assert "Mean Baseline Score" not in md


def test_anonymize_pair_and_blind_key(tmp_path: Path) -> None:
    """Verify private-seed blinding, multi-ref/confidence-note stripping, and MD completeness."""
    from crewforge.evaluation.anonymize import export_blind_key_file, generate_blinding_seed

    base_res = RunResult(
        run_id="run_base_abcdef12",
        mode="baseline",
        status=RunStatus.COMPLETED,
        prompt="Solve problem X.",
        provider="openai",
        model="gpt-6.2-astra",
        answer="Direct answer text with 24 months payback.",
        assumptions=["Constant monthly revenue."],
        confidence_notes="BaselineRunner ran with gpt-6.2-astra; 100% confident.",
    )
    net_res = RunResult(
        run_id="run_net_12345678",
        mode="network",
        status=RunStatus.COMPLETED,
        prompt="Solve problem X.",
        provider="ollama",
        model="qwen3:4b",
        answer=(
            "Verified Findings:\n"
            "- Payback is 24 months. [clm_123456@v1, clm_abcdef@v2] [req_1, req_2]\n"
            "- See [advisory; clm_999999@v1] from run_net_12345678."
        ),
        assumptions=["Constant monthly revenue [clm_123456@v1]."],
        missing_information=["Project lifespan missing [req_2]."],
        confidence_notes=(
            "HonestyGate passed (supported_claim_count=2, gate_passed=true). "
            "Payback depends on constant monthly cash flow."
        ),
    )

    seed = generate_blinding_seed()
    assert len(seed) == 64

    item, key = anonymize_pair(
        prompt_id="prompt_01",
        domain=PromptDomain.FINANCIAL,
        prompt="Solve problem X.",
        baseline_result=base_res,
        network_result=net_res,
        reference_documents=["policy.md"],
        reference_document_contents={"policy.md": "Full SLA policy document text here."},
        seed=seed,
    )

    for ans in (item.answer_a, item.answer_b):
        assert "[clm_" not in ans.content
        assert "clm_" not in ans.content
        assert "[req_" not in ans.content
        assert "run_net_" not in ans.content
        assert "HonestyGate" not in (ans.confidence_notes or "")
        assert "supported_claim_count" not in (ans.confidence_notes or "")
        assert "gpt-6.2-astra" not in (ans.confidence_notes or "")
        assert "BaselineRunner" not in (ans.confidence_notes or "")

    # Separate packet and blind key files
    packet_path = tmp_path / "evaluator_packet.json"
    key_path = tmp_path / "blind_key.json"
    export_evaluator_packet([item], packet_path)
    export_blind_key_file([key], blinding_seed=seed, output_path=key_path)

    packet_md = packet_path.with_suffix(".md").read_text(encoding="utf-8")
    assert seed not in packet_md
    assert seed not in packet_path.read_text(encoding="utf-8")
    assert "Assumptions (A):" in packet_md
    assert "Assumptions (B):" in packet_md
    assert "Full SLA policy document text here." in packet_md
    assert seed in key_path.read_text(encoding="utf-8")


def test_demo_trace_export(temp_db: Database, tmp_path: Path) -> None:
    """Verify demo trace export defaults to redacted prompt and recursively strips secrets."""
    import asyncio

    from crewforge.schemas import ExecutionEvent, ExecutionPlan

    run_id = f"run_{uuid4().hex[:12]}"
    secret_prompt = "Analyze secret outage at acme-internal."
    config = CrewForgeSettings(provider="ollama", model="qwen3:4b").to_run_config()
    temp_db._create_run_sync(
        run_id=run_id,
        mode="network",
        prompt=secret_prompt,
        config=config,
    )
    temp_db._save_execution_plan_sync(
        ExecutionPlan(run_id=run_id, goal=secret_prompt, user_requirements=["Find root cause"])
    )
    temp_db._append_event_sync(
        ExecutionEvent(
            run_id=run_id,
            event_type="provider_debug",
            actor="executor",
            message="Header Bearer sk-secret1234567890abcdef failed",
            payload={
                "authorization": "Bearer secret_token_xyz",
                "nested": {"api_key": "sk-proj-1234567890123456", "prompt": secret_prompt},
            },
        )
    )
    temp_db._complete_run_sync(
        result=RunResult(
            run_id=run_id,
            mode="network",
            status=RunStatus.COMPLETED,
            prompt=secret_prompt,
            provider="ollama",
            model="qwen3:4b",
            answer="Safe analyzed answer.",
            error="api_key=sk-abc1234567890123",
        )
    )
    out_file = tmp_path / "trace_export.json"
    export = asyncio.run(export_demo_trace(temp_db, run_id, out_file))
    raw_json = out_file.read_text(encoding="utf-8")

    assert export.run_id == run_id
    assert export.is_historical_replay is True
    assert export.prompt == "[REDACTED]"
    assert secret_prompt not in raw_json
    assert "sk-secret" not in raw_json
    assert "sk-proj-" not in raw_json
    assert "sk-abc" not in raw_json
    assert "secret_token_xyz" not in raw_json
    assert export.execution_plan is not None
    assert export.execution_plan["goal"] == "[REDACTED_PROMPT]"


def test_manifest_document_validation_and_fingerprint_isolation(tmp_path: Path) -> None:
    """Verify missing/mismatched docs fail explicitly and fingerprints isolate modes/models."""
    from crewforge.evaluation.contracts import BenchmarkItem, BenchmarkManifest
    from crewforge.evaluation.runner import (
        BenchmarkRunner,
        build_david_vs_goliath_settings,
        load_and_validate_manifest,
        resolve_item_documents,
    )
    from tests.fake_provider import FakeProvider

    doc_file = tmp_path / "docs" / "sla.md"
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("Uptime SLA is 99.9%.", encoding="utf-8")

    # 1. Missing document fails explicitly
    bad_item = BenchmarkItem(
        prompt_id="p_miss",
        domain=PromptDomain.DOCUMENT_GROUNDED,
        prompt="Check SLA.",
        reference_documents=["docs/missing.md"],
    )
    with pytest.raises(FileNotFoundError):
        resolve_item_documents(bad_item, manifest_dir=tmp_path)

    # 2. Hash mismatch fails explicitly
    mismatch_item = BenchmarkItem(
        prompt_id="p_hash",
        domain=PromptDomain.DOCUMENT_GROUNDED,
        prompt="Check SLA.",
        reference_documents=["docs/sla.md"],
        document_hashes={"docs/sla.md": "0000deadbeef"},
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_item_documents(mismatch_item, manifest_dir=tmp_path)

    # 3. Valid external manifest loads relative to manifest_dir
    import hashlib

    sla_hash = hashlib.sha256(b"Uptime SLA is 99.9%.").hexdigest()
    manifest = BenchmarkManifest(
        manifest_id="ext_01",
        name="External Manifest",
        items=[
            BenchmarkItem(
                prompt_id="p1",
                domain=PromptDomain.DOCUMENT_GROUNDED,
                prompt="Check SLA.",
                reference_documents=["docs/sla.md"],
                document_hashes={"docs/sla.md": sla_hash},
            )
        ],
    )
    m_path = tmp_path / "manifest.json"
    m_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    loaded_m, loaded_dir = load_and_validate_manifest(m_path)
    assert loaded_dir == tmp_path.resolve()
    assert len(loaded_m.items) == 1

    # 4. David vs Goliath provider configs are strictly distinct
    goliath_cfg, david_cfg = build_david_vs_goliath_settings(CrewForgeSettings(_env_file=None))
    assert david_cfg.provider.value == "ollama"
    assert david_cfg.model == "qwen3:4b"
    assert goliath_cfg.provider.value == "openai"
    assert goliath_cfg.model == "gpt-6.2-astra"
    assert goliath_cfg.max_cost_per_run_usd == 0.20

    # 5. Standard vs compute-budget-matched runs isolate caches by fingerprint
    import asyncio

    def fake_handler(prompt: str, response_model: Any) -> dict[str, Any]:
        name = getattr(response_model, "__name__", "")
        if "Planner" in name:
            return {
                "goal": "Check SLA",
                "user_requirements": ["Check SLA"],
                "acceptance_criteria": ["Accurate SLA"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "instructions": "Check SLA.",
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
        if "Specialist" in name:
            return {
                "findings": "SLA is 99.9%.",
                "claims": [
                    {
                        "statement": "SLA is 99.9%.",
                        "category": "given_fact",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_1"],
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "sla.md",
                        "excerpt": "Uptime SLA is 99.9%.",
                    }
                ],
                "task_completed": True,
            }
        if "Reviewer" in name:
            return {
                "verdicts": [
                    {
                        "claim_id": "c1",
                        "decision": "approve",
                        "rationale": "Matches doc.",
                        "blocking_severity": "none",
                    }
                ],
                "missing_requirement_coverage": [],
            }
        return {
            "answer": "SLA is 99.9%.",
            "refined_answer": "SLA is 99.9%.",
            "is_fully_satisfied": True,
            "missing_information": [],
            "assumptions": [],
            "computations": [],
            "needs_tool_followup": False,
        }

    cfg = CrewForgeSettings(_env_file=None, db_path=tmp_path / "bench.db")
    runner = BenchmarkRunner(
        settings=cfg,
        baseline_provider=FakeProvider(handler=fake_handler, max_retries=0),
        network_provider=FakeProvider(handler=fake_handler, max_retries=0),
        output_dir=tmp_path / "out",
    )
    res_std, _ = asyncio.run(
        runner.evaluate_manifest(loaded_m, manifest_dir=loaded_dir, mode="baseline_vs_network")
    )
    res_cm, _ = asyncio.run(
        runner.evaluate_manifest(
            loaded_m, manifest_dir=loaded_dir, mode="compute_matched_vs_network"
        )
    )
    assert res_std[0].baseline_run_id != res_cm[0].baseline_run_id
    assert res_std[0].grading_status == "pending_independent_grading"
    assert res_std[0].baseline_score is None


def test_real_ablation_stage_switches(tmp_path: Path) -> None:
    """Verify no_reviewer, no_repair, and no_peer_messages actually skip disabled stages."""
    import asyncio

    from crewforge.core.budget import BudgetManager
    from crewforge.core.orchestrator import NetworkRunner
    from crewforge.evaluation.ablation import AblationConfig
    from tests.fake_provider import FakeProvider

    async def run_variant(variant_name: str) -> tuple[list[str], int, int]:
        db = Database(tmp_path / f"{variant_name}.db")
        await db.initialize()
        base_settings = CrewForgeSettings(_env_file=None, db_path=tmp_path / f"{variant_name}.db")
        ab_settings = AblationConfig.from_variant(variant_name).apply_to_settings(base_settings)

        seen_ops: list[str] = []

        def handler(prompt: str, response_model: Any) -> dict[str, Any]:
            name = getattr(response_model, "__name__", "")
            seen_ops.append(name)
            if "Planner" in name:
                return {
                    "goal": "Test",
                    "user_requirements": ["R1"],
                    "acceptance_criteria": ["A1"],
                    "agents": [
                        {
                            "agent_ref": "a1",
                            "role": "Worker1",
                            "instructions": "Work",
                            "capabilities": ["analysis"],
                        },
                        {
                            "agent_ref": "a2",
                            "role": "Worker2",
                            "instructions": "Work",
                            "capabilities": ["analysis"],
                        },
                    ],
                    "tasks": [
                        {
                            "task_ref": "t1",
                            "title": "T1",
                            "description": "D1",
                            "assigned_agent_ref": "a1",
                            "depends_on": [],
                            "expected_deliverables": ["Out1"],
                        },
                        {
                            "task_ref": "t2",
                            "title": "T2",
                            "description": "D2",
                            "assigned_agent_ref": "a2",
                            "depends_on": [],
                            "expected_deliverables": ["Out2"],
                        },
                    ],
                }
            if "Specialist" in name:
                return {
                    "findings": "Finding X.",
                    "claims": [
                        {
                            "statement": "Finding X.",
                            "category": "given_fact",
                            "evidence_indices": [0],
                            "requirement_refs": ["req_1"],
                        }
                    ],
                    "evidence_proposals": [
                        {
                            "source_type": "user_prompt",
                            "source_ref": "up_1",
                            "excerpt": "Test",
                        }
                    ],
                    "outgoing_messages": [
                        {
                            "recipient_ref": "a2",
                            "message_type": "finding",
                            "content": "Peer note",
                        }
                    ],
                    "task_completed": True,
                }
            if "Reviewer" in name:
                return {
                    "verdicts": [
                        {
                            "claim_id": "c1",
                            "decision": "reject",
                            "rationale": "Needs revision.",
                            "blocking_severity": "blocking",
                        }
                    ],
                    "missing_requirement_coverage": [],
                }
            if "Revision" in name:
                return {
                    "revisions": [
                        {
                            "target_claim_id": "c1",
                            "expected_version": 1,
                            "revised_statement": "Revised Finding X.",
                            "category": "given_fact",
                            "evidence_indices": [0],
                            "revision_notes": "Fixed.",
                        }
                    ]
                }
            return {
                "user_requirements": ["R1"],
                "accepted_segments": [],
                "conditional_conclusions": [],
                "recommendations": [],
                "assumptions": [],
                "unresolved_issues": [],
                "missing_information": [],
            }

        runner = NetworkRunner(
            provider=FakeProvider(handler=handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, ab_settings),
            settings=ab_settings,
        )
        res = await runner.run("Test")
        trace = await db.get_run_trace(res.run_id)
        await db.close()
        assert trace is not None
        peer_msgs = [m for m in trace.messages if m.message_type == "finding"]
        return seen_ops, len(trace.review_verdicts), len(peer_msgs)

    ops_no_rev, verdicts_no_rev, _ = asyncio.run(run_variant("no_reviewer"))
    assert "ReviewerBatchOutput" not in ops_no_rev
    assert verdicts_no_rev == 0

    ops_no_rep, _, _ = asyncio.run(run_variant("no_repair"))
    assert "ReviewerBatchOutput" in ops_no_rep
    assert "RevisionOutput" not in ops_no_rep

    ops_no_peer, _, msgs_no_peer = asyncio.run(run_variant("no_peer_messages"))
    assert "FollowUpOutput" not in ops_no_peer
    assert msgs_no_peer == 0
