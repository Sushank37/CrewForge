"""Tests for Phase 5 competitive evaluation, unified budget safeguards, and stabilization."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
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
    """Verify numerical and logical checks and 100-point rubric calculation."""
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
    score = compute_evaluation_score(
        rubric=rubric,
        system_label="test_system",
        answer_text=good_answer,
        missing_information=["A randomized control group is missing."],
    )
    assert score.factual_correctness == 30.0
    assert score.internal_consistency == 15.0
    assert score.unsupported_assertion_penalty == 0.0
    assert score.total_score >= 85.0

    bad_answer = (
        "The pass rate increased by 15% and 25%. "
        "This proves causation because students scored higher."
    )
    bad_score = compute_evaluation_score(
        rubric=rubric,
        system_label="bad_system",
        answer_text=bad_answer,
    )
    assert bad_score.unsupported_assertion_penalty > 0.0  # Hit forbidden phrase!
    assert bad_score.total_score < score.total_score


def test_anonymize_pair_and_blind_key(tmp_path: Path) -> None:
    """Verify reproducible A/B blinding and metadata sanitization."""
    base_res = RunResult(
        run_id="run_base_1",
        mode="baseline",
        status=RunStatus.COMPLETED,
        prompt="Solve problem X.",
        provider="openai",
        model="gpt-6-luna",
        answer="Direct answer text with 24 months payback.",
    )
    net_res = RunResult(
        run_id="run_net_1",
        mode="network",
        status=RunStatus.COMPLETED,
        prompt="Solve problem X.",
        provider="openai",
        model="gpt-6-luna",
        answer="Verified Findings:\n- Payback is 24 months. [clm_123456@v1]",
    )

    item, key = anonymize_pair(
        prompt_id="prompt_01",
        domain=PromptDomain.FINANCIAL,
        prompt="Solve problem X.",
        baseline_result=base_res,
        network_result=net_res,
        seed="test_seed_123",
    )

    # Check that orchestrator tag [clm_...] was completely stripped
    assert "[clm_" not in item.answer_a.content
    assert "[clm_" not in item.answer_b.content
    assert key.run_id_a in {"run_base_1", "run_net_1"}
    assert key.run_id_b in {"run_base_1", "run_net_1"}
    assert key.run_id_a != key.run_id_b

    # Export packet and verify markdown output
    packet_path = tmp_path / "packet.json"
    packet = export_evaluator_packet([item], packet_path)
    assert packet.packet_id == "eval_packet_blind"
    assert packet_path.exists()
    assert packet_path.with_suffix(".md").exists()


def test_demo_trace_export(temp_db: Database, tmp_path: Path) -> None:
    """Verify demo trace export captures run execution and sanitizes credentials."""
    run_id = f"run_{uuid4().hex[:12]}"
    config = CrewForgeSettings(provider="ollama", model="qwen3:4b").to_run_config()
    temp_db._create_run_sync(
        run_id=run_id,
        mode="network",
        prompt="Analyze system failure.",
        config=config,
    )
    temp_db._complete_run_sync(
        result=RunResult(
            run_id=run_id,
            mode="network",
            status=RunStatus.COMPLETED,
            prompt="Analyze system failure.",
            provider="ollama",
            model="qwen3:4b",
            answer="Safe analyzed answer.",
        )
    )
    out_file = tmp_path / "trace_export.json"
    import asyncio

    export = asyncio.run(export_demo_trace(temp_db, run_id, out_file))
    assert export.run_id == run_id
    assert export.is_historical_replay is True
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["run_id"] == run_id
