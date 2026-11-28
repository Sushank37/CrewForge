"""Read-only historical input with fresh, explicit scripted review of the missing selection."""
import asyncio
import shutil
from pathlib import Path

import pytest

from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.ledger import ClaimLedger
from crewforge.core.repair import _rebind_calculation
from crewforge.core.reviewer import numerical_challenge
from crewforge.public_mcp_service import PublicJobManager
from crewforge.schemas import (
    ClaimStatus,
    ClaimVersionRef,
    ReleaseSegment,
    ReviewVerdict,
    RunTrace,
    SolutionSelection,
    SynthesisOutput,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog

ARCHIVE = Path(".crewforge/v5_live_diagnostic/attempt")


@pytest.mark.skipif(not ARCHIVE.exists(), reason="local live export is unavailable")
async def test_release_saved_live_run_and_reconstructed_selection(tmp_path):
    raw = await asyncio.to_thread((ARCHIVE / "full_trace.json").read_text)
    trace = RunTrace.model_validate_json(raw)
    plan = trace.plan
    assert plan and trace.result and trace.honesty_gate
    assert len(trace.tool_executions) == 22
    assert len(plan.deliverables) == 15
    [rejection] = trace.honesty_gate.rejected_claim_refs
    assert rejection["claim_id"] == "clm_20f5eadc7a05"
    assert "205" in rejection["reason"]
    for event in (e for e in trace.events if e.seq in {33, 48}):
        [challenge] = event.payload["solution_challenges"]
        assert numerical_challenge(challenge["challenge"])
        assert "repair_hypothesis" not in challenge

    file = tmp_path / "replay.db"
    await asyncio.to_thread(shutil.copyfile, ARCHIVE / "diagnostic.db", file)
    db = Database(file)
    try:
        gate = HonestyGate(db)
        latest = {c.claim_id: c for c in sorted(trace.claims, key=lambda c: c.version)}
        rec = latest["clm_20f5eadc7a05"]
        assert not rec.is_material and rec.tool_execution_ids == ["tlx_ab83f5fd1f04"]
        assert any(v.verdict_id == "vrd_36ed2e52833b" and v.decision == "approve"
                   for v in trace.review_verdicts)
        refs = [{"claim_id": rec.claim_id, "claim_version": rec.version}]
        synthesis = SynthesisOutput(recommendations=[
            ReleaseSegment(segment_type="recommendation", claim_refs=refs),
        ])
        args = dict(
            run_id=trace.run_id, original_prompt=trace.prompt, plan=plan,
            synthesis=synthesis, all_claims=trace.claims, evidence_list=trace.evidence,
            tool_executions=trace.tool_executions, verdicts=trace.review_verdicts,
            incomplete_tasks=[], extra_assumptions=[], extra_missing=[], uncertainties=[],
            is_partial_run=False,
        )
        released = gate.evaluate(**args)
        assert rec.claim_id in released.referenced_claim_ids
        assert rec.statement in released.rendered_answer
        assert not released.verification.answer_complete

        wrong_pin = rec.model_copy(update={
            "premise_refs": [
                ClaimVersionRef(claim_id=rec.premise_refs[0].claim_id, claim_version=2),
                rec.premise_refs[1],
            ],
        })
        stale_release = gate.evaluate(**(args | {
            "all_claims": [wrong_pin if c.claim_id == rec.claim_id else c for c in trace.claims],
        }))
        assert rec.claim_id not in stale_release.referenced_claim_ids

        tools = {t.tool_execution_id: t for t in trace.tool_executions}
        owners = {tid: c for c in trace.claims for tid in c.tool_execution_ids}
        catalog = build_provenance_catalog(
            original_prompt=trace.prompt, all_claims=trace.claims,
            tool_records=trace.tool_executions, run_id=trace.run_id,
        )
        targets = {
            "clm_ab2802c807b5": "260",
            "clm_a6bcfa11f1be": "-4940",
            "clm_b5309f8703b8": "5200",
        }
        for cid, expected in targets.items():
            old = next(c for c in trace.claims if c.claim_id == cid and c.version == 1)
            request = _rebind_calculation(old, tools, owners, latest, refresh_named_operands=True)
            assert request is not None
            calc = DeterministicArithmeticTool.evaluate(
                run_id=trace.run_id, actor=old.author_agent_id, request=request, catalog=catalog,
            )
            assert calc.inputs_grounded and calc.computed_result == expected
            assert calc.matches_proposed is False

        # Original lead payload was unavailable. Reconstruct this scenario explanation from
        # supported totals and the reviewed ceiling conclusions, with no new prose or values.
        members = [latest[cid] for cid in (
            "clm_daa7d93ffcd2", "clm_9f8fb941b402", *targets,
        )]
        selected, rejected = await ClaimLedger(db).record_solution_selections(
            trace.run_id, [
                SolutionSelection(deliverable_id="deliv_15", claim_refs=[
                    f"{c.claim_id}@v{c.version}" for c in members
                ]),
                SolutionSelection(deliverable_id="deliv_15", claim_refs=[
                    "clm_a6bcfa11f1be@v1",
                ]),
            ],
            {"deliv_15": members[0].task_id}, {"deliv_12", "deliv_13"},
        )
        assert [r["reason"] for r in rejected] == ["missing_stale_or_unsupported_member"]
        [selection] = selected
        assert selection.statement == " ".join(c.statement for c in members)
        assert all(not c.assumptions for c in members)
        approval = ReviewVerdict(
            run_id=trace.run_id, claim_id=selection.claim_id, claim_version=1,
            reviewer_id="reviewer_combined", review_dimension="combined", decision="approve",
            rationale="Both case totals use the quoted rates and both exceed the ceiling.",
            covered_deliverable_refs=["deliv_15"],
        )
        args["all_claims"] = [*trace.claims, selection.model_copy(update={
            "status": ClaimStatus.SUPPORTED,
            "verification_status": VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
        })]
        args["verdicts"] = [*trace.review_verdicts, approval]
        gate_output = gate.evaluate(**args)
        assert gate_output.verification.answer_complete
        assert gate_output.verification.requirement_coverage == 1.0
        assert gate_output.gate_record.release_mode == "full"
        summary = gate_output.rendered_answer.split("**Calculations and Sources**")[0]
        assert summary.count(latest["clm_ffc5a7891af0"].statement) == 1
        assert summary.count(latest["clm_a6bcfa11f1be"].statement) == 1
        payload = await object.__new__(PublicJobManager)._format_result_payload(
            db, trace.run_id, trace.result.model_copy(update={
                "referenced_claim_ids": gate_output.referenced_claim_ids,
            }), [],
        )
        assert payload["sources"] and payload["evidence"]
        assert all(e["excerpt"] in trace.prompt for e in payload["evidence"])
    finally:
        await db.close()
