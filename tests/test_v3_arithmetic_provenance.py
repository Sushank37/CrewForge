import re
from pathlib import Path

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.executor import ModelExecutor
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.ledger import ClaimLedger, ClaimRevisionError
from crewforge.core.messaging import MessageBus
from crewforge.core.repair import TargetedRepairEngine
from crewforge.core.reviewer import ReviewerPool
from crewforge.schemas import (
    AgentSpec,
    ArithmeticRequest,
    ArithmeticVariableBinding,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    ExecutionPlan,
    ProposedClaim,
    ProposedEvidence,
    ReleaseSegment,
    ReviewDecision,
    ReviewerBatchOutput,
    RevisionOutput,
    SynthesisOutput,
    TaskSpec,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog
from tests.fake_provider import FakeProvider


@pytest.fixture
async def run_db(tmp_path: Path):
    database = Database(tmp_path / "arithmetic.db")
    settings = CrewForgeSettings(_env_file=None)
    await database.initialize()
    await database.create_run("offline", "network", "Offline regression", settings.to_run_config())
    try:
        yield database, settings
    finally:
        await database.close()


def binding(name, value, ref="up_1", *, quote=None, units=""):
    return ArithmeticVariableBinding(
        name=name, value=value, provenance_ref=ref, source_quote=quote, units=units,
    )


def computation(expression, result, units, *bindings):
    return ArithmeticRequest(
        expression=expression, proposed_result=result, units=units, variable_bindings=bindings,
    )


def claim(statement, request, *, alias=None):
    return ProposedClaim(
        statement=statement, category="computation", computation=request, claim_ref=alias,
    )


async def test_document_tiers_and_compound_dependent_calculation(run_db):
    db, _ = run_db
    prompt = "Estimate the cost for 940 inspection samples."
    document = (
        "Charge $2.40 each for the first 600 samples, then charge $0.90 each. "
        "A 7 percent rebate applies to all sample charges."
    )
    await db.insert_evidence(Evidence(
        run_id="offline", source_type="external_retrieved", source_ref="doc_1",
        excerpt=document, verified_in_input=True, created_by="lead",
    ))
    ledger = ClaimLedger(db)
    _, claims, records = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The sample charge is $1746.", computation(
                "min(samples, cap) * first_rate + max(samples - cap, 0) * later_rate",
                "1746", "USD",
                binding("samples", "940"),
                binding("cap", "600", "doc_1", quote="first 600 samples"),
                binding("first_rate", "2.40", "doc_1", quote="Charge $2.40 each"),
                binding("later_rate", "0.90", "doc_1", quote="then charge $0.90 each"),
            ), alias="sample_charge"),
            claim("After the rebate the charge is $1623.78.", computation(
                "subtotal * (1 - rebate)", "1623.78", "USD",
                binding("subtotal", "1746", "sample_charge", units="USD"),
                binding("rebate", "0.07", "doc_1", quote="7 percent rebate", units="ratio"),
            )),
        ],
    )
    assert all(record.inputs_grounded and record.matches_proposed for record in records)
    assert records[0].provenance_details["cap"]["source_token"] == "600"
    assert records[1].provenance_details["rebate"]["source_units"] == "%"
    assert records[1].provenance_details["subtotal"]["source_tool_execution_id"] == (
        records[0].tool_execution_id
    )
    assert claims[1].premise_refs[0].claim_id == claims[0].claim_id
    assert all(c.status == ClaimStatus.ASSERTED for c in claims)


async def test_aggregate_rates_and_scaled_currency_operands(run_db):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    rates = await tool.execute(
        run_id="offline", actor="lead",
        grounded_texts=[
            "Batch East had 180 requests and returned 9 failures.",
            "Batch West had 220 requests and returned 17 failures.",
        ],
        request=computation(
            "sum(east_fails, west_fails) / sum(east_total, west_total) * scale",
            "6.5%", "%",
            binding("east_fails", 9, quote="returned 9 failures"),
            binding("west_fails", 17, "up_2", quote="returned 17 failures"),
            binding("east_total", 180, quote="180 requests"),
            binding("west_total", 220, "up_2", quote="220 requests"),
            binding("scale", 100, "const:percent_scale"),
        ),
    )
    assert rates.inputs_grounded and rates.matches_proposed
    scaled = await tool.execute(
        run_id="offline", actor="lead", grounded_texts=["Budget is $46k and grant is $8k."],
        request=computation(
            "budget + grant", 54000, "USD",
            binding("budget", 46000, units="USD"),
            binding("grant", 8000, units="USD"),
        ),
    )
    assert scaled.inputs_grounded and scaled.matches_proposed
    assert scaled.provenance_details["budget"]["source_token"] == "$46k"


@pytest.mark.parametrize(
    ("prompt", "request_args", "operand", "code"),
    [
        ("The fee is $27.", ("x * 2", 54, "USD", binding("x", 27)),
         "__expression_literals__", "unbound_literal"),
        ("The fee is $27.", ("x * 27", 729, "USD", binding("x", 27)),
         "__expression_literals__", "unbound_literal"),
        ("The fee is $27.", ("x", 27, "USD", binding("x", 27, "const:standard_fee")),
         "x", "unknown_constant"),
        ("The fee is $27.", ("x", 27, "USD", binding("x", 27, "const:percent_scale")),
         "x", "constant_mismatch"),
        ("The fee is $27.", ("x", 27, "USD", binding("x", 27, "up_9")),
         "x", "unknown_reference"),
        ("The fee is $127.", ("x", 27, "USD", binding("x", 27, quote="27.")),
         "x", "value_not_found"),
        ("The fee is $27.", ("x", 27, "USD", binding("x", 27, quote="tax is $27")),
         "x", "quote_not_found"),
        ("The fee is $27.", ("x", 27, "USD", binding("x", 27, units="EUR")),
         "x", "unit_conflict"),
        ("The rebate is 7%.", ("x", 7, "%", binding("x", 7, units="ratio")),
         "x", "value_not_found"),
        ("Alpha fee is $27, Beta fee is $39.",
         ("alpha", 39, "USD", binding("alpha", 39, quote="Beta fee is $39")),
         "alpha", "role_conflict"),
        ("The balance is −27 USD.", ("x", 27, "USD", binding("x", 27)),
         "x", "value_not_found"),
        ("The fee is $27000000.", ("x", 27000001, "USD", binding("x", 27000001)),
         "x", "value_not_found"),
        ("The tank holds 42 litres.", ("x", 42, "", binding("x", 42, "assumption:42 litres")),
         "x", "untrusted_reference"),
    ],
)
async def test_correct_arithmetic_with_invalid_provenance_is_rejected(
    run_db, prompt, request_args, operand, code,
):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    result = await tool.execute(
        run_id="offline", actor="lead", grounded_texts=[prompt],
        request=computation(*request_args),
    )
    assert result.success and result.matches_proposed
    assert not result.inputs_grounded
    assert result.provenance_details[operand]["code"] == code


async def test_changed_premise_recovers_two_dependent_calculations_and_releases(run_db):
    db, settings = run_db
    # Exercise the production call ceiling and one permitted revision per claim.
    assert settings.max_revision_rounds == 1
    assert settings.max_model_calls_per_run == 16
    ledger = ClaimLedger(db)
    prompt = (
        "The initial estimate is $75. The corrected amount is $91. The fee is $6. "
        "The deposit is $20. Use the corrected amount to calculate the total and balance."
    )
    agent = AgentSpec(run_id="offline", role="Analyst", instructions="Calculate charges.")
    task = TaskSpec(
        run_id="offline", title="Calculate charges", description="Use the corrected amount.",
        assigned_agent_id=agent.agent_id,
    )
    plan = ExecutionPlan(run_id="offline", goal="Calculate charges", agents=[agent], tasks=[task])
    await db.save_execution_plan(plan)
    _, claims, records = await ledger.record_specialist_outputs(
        run_id="offline", agent_id=agent.agent_id, task_id=task.task_id, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The estimate is $75.", computation(
                "estimate", 75, "USD", binding("estimate", 75),
            )),
            claim("With the fee the charge is $81.", computation(
                "estimate + fee", 81, "USD",
                binding("estimate", 75, "clm_1"), binding("fee", 6, "up_3"),
            )),
            claim("After the deposit the balance is $61.", computation(
                "charge - deposit", 61, "USD",
                binding("charge", 81, "clm_2"), binding("deposit", 20, "up_4"),
            )),
        ],
    )
    assert all(record.inputs_grounded for record in records)
    upstream, dependent, balance = claims
    revisions = {
        (upstream.claim_id, 1): (
            "The corrected amount is $91.",
            computation("amount", 91, "USD", binding("amount", 91, "up_2")),
            "Use the corrected amount",
        ),
        (dependent.claim_id, 2): (
            "With the fee the charge is $97.",
            computation(
                "estimate + fee", 97, "USD",
                binding("estimate", 91, f"{upstream.claim_id}@v2"), binding("fee", 6, "up_3"),
            ),
            "Candidates: 91 USD.",
        ),
        (balance.claim_id, 2): (
            "After the deposit the balance is $77.",
            computation(
                "charge - deposit", 77, "USD",
                binding("charge", 97, f"{dependent.claim_id}@v3"), binding("deposit", 20, "up_4"),
            ),
            "Candidates: 97 USD.",
        ),
    }
    approvals = {
        (upstream.claim_id, 2): ("The corrected amount is $91.", "amount=91 [up_2", "91"),
        (dependent.claim_id, 3): ("With the fee the charge is $97.", "fee=6 [up_3", "97"),
        (balance.claim_id, 3): ("After the deposit the balance is $77.",
                                "deposit=20 [up_4", "77"),
    }
    seen_repairs = []
    seen_reviews = []

    def scripted_provider(model_prompt, schema):
        if schema is RevisionOutput:
            targets = re.findall(r"Target claim: claim_id=(clm_\w+), expected_version=(\d+)",
                                 model_prompt)
            assert len(targets) == 1
            cid, version = targets[0][0], int(targets[0][1])
            statement, request, expected_diagnostic = revisions[cid, version]
            assert expected_diagnostic in model_prompt
            seen_repairs.append((cid, version))
            return {"updated_findings": statement, "revisions": [{
                "target_claim_id": cid, "expected_version": version,
                "new_statement": statement, "computation": request.model_dump(),
                "revision_notes": "Recomputed from the current supplied source.",
            }]}
        assert schema is ReviewerBatchOutput
        targets = re.findall(r"Claim (clm_\w+) \(version=(\d+),", model_prompt)
        assert len(targets) == 1
        cid, version = targets[0][0], int(targets[0][1])
        seen_reviews.append((cid, version))
        if (cid, version) == (upstream.claim_id, 1):
            assert "Statement: The estimate is $75." in model_prompt
            assert "Use the corrected amount" in model_prompt
            verdict = {"decision": "needs_revision", "blocking_severity": "blocking",
                       "rationale": "Use the corrected amount in up_2 in place of the estimate."}
        else:
            statement, operand, value = approvals[cid, version]
            assert f"Statement: {statement}" in model_prompt
            assert f"computed={value}," in model_prompt
            assert operand in model_prompt
            verdict = {
                "decision": "approve", "rationale": "Matches the scripted source and formula.",
            }
        return {"verdicts": [{"claim_id": cid, "claim_version": version, **verdict}]}

    provider = FakeProvider(handler=scripted_provider, max_retries=0)
    executor = ModelExecutor(
        provider=provider, database=db, budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    reviewer = ReviewerPool(executor=executor, database=db, settings=settings)
    initial_review = await reviewer.review_claims(
        run_id="offline", original_prompt=prompt, plan=plan, target_claims=[upstream],
    )
    assert [c.claim_id for c in initial_review.blocking_claims] == [upstream.claim_id]
    repair = TargetedRepairEngine(
        executor=executor, database=db, ledger=ledger, message_bus=MessageBus(db),
        reviewer_pool=reviewer, settings=settings,
    )
    round_outcome = await repair.resolve_disputes(
        run_id="offline", original_prompt=prompt, plan=plan,
        blocking_claims=initial_review.blocking_claims, verdicts=initial_review.verdicts,
    )
    assert seen_repairs == list(revisions)
    assert seen_reviews == [(upstream.claim_id, 1), *approvals]
    assert round_outcome.repair_calls == round_outcome.re_review_calls == 3

    trace = await db.get_run_trace("offline")
    assert trace is not None
    all_claims = {(c.claim_id, c.version): c for c in trace.claims}
    tools = {r.tool_execution_id: r for r in trace.tool_executions}
    accepted = []
    for (cid, version), (statement, _, value) in approvals.items():
        current = all_claims[cid, version]
        assert current.status == ClaimStatus.SUPPORTED
        assert current.verification_status == VerificationStatus.COMPUTATIONALLY_VERIFIED
        [tid] = current.tool_execution_ids
        assert tools[tid].computed_result == value and tools[tid].inputs_grounded
        assert tid not in all_claims[cid, 1].tool_execution_ids
        assert any(
            v.claim_id == cid and v.claim_version == version
            and v.decision == ReviewDecision.APPROVE and v.reviewer_id != agent.agent_id
            for v in trace.review_verdicts
        )
        accepted.append(ReleaseSegment(
            segment_type="accepted_claim", text=statement,
            claim_refs=[ClaimVersionRef(claim_id=cid, claim_version=version)],
        ))

    for original, old_value, new_value, operand in [
        (dependent, "81", "91", "estimate"),
        (balance, "61", "97", "charge"),
    ]:
        old = all_claims[original.claim_id, 2]
        assert old.status == ClaimStatus.REJECTED
        [tid] = old.tool_execution_ids
        assert tools[tid].computed_result == old_value and not tools[tid].inputs_grounded
        assert tools[tid].provenance_details[operand]["candidates"] == [f"{new_value} USD"]

    gate = HonestyGate(db)
    gate_args = dict(
        run_id="offline", original_prompt=prompt, plan=plan, all_claims=trace.claims,
        evidence_list=trace.evidence, tool_executions=trace.tool_executions,
        verdicts=trace.review_verdicts, incomplete_tasks=[], extra_assumptions=[],
        extra_missing=[], uncertainties=[], is_partial_run=False,
    )
    release = gate.evaluate(synthesis=SynthesisOutput(accepted_segments=accepted), **gate_args)
    assert release.verification.honesty_gate_passed
    assert all(value in release.rendered_answer for value in ("$91", "$97", "$77"))
    assert all(value not in release.rendered_answer for value in ("$75", "$81", "$61"))
    stale_release = gate.evaluate(
        synthesis=SynthesisOutput(accepted_segments=[ReleaseSegment(
            segment_type="accepted_claim", text=dependent.statement,
            claim_refs=[ClaimVersionRef(claim_id=dependent.claim_id, claim_version=1)],
        )]),
        **gate_args,
    )
    assert not stale_release.verification.honesty_gate_passed
    assert "$81" not in stale_release.rendered_answer

    catalog = build_provenance_catalog(
        original_prompt=prompt, all_claims=trace.claims, tool_records=trace.tool_executions,
        run_id="offline",
    )
    for record in records:
        assert not catalog.targets_by_key[record.tool_execution_id].is_trusted
    assert not catalog.targets_by_key[f"{dependent.claim_id}@v1"].is_trusted


async def test_invalid_intermediate_and_computation_evidence_cannot_source_operands(run_db):
    db, _ = run_db
    ledger = ClaimLedger(db)
    prompt = "The two charges are $31 and $42."
    _, _, initial = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The subtotal is $74.", computation(
                "a + b", 74, "USD", binding("a", 31), binding("b", 42),
            )),
        ],
    )
    assert initial[0].inputs_grounded and initial[0].matches_proposed is False
    copied = Evidence(
        run_id="offline", source_type="calculation", source_ref="calculation",
        excerpt="The subtotal is $73.", verified_in_input=True, created_by="lead",
    )
    await db.insert_evidence(copied)
    _, _, attempted = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The subtotal is $73.", computation(
                "subtotal", 73, "USD", binding("subtotal", 73, initial[0].tool_execution_id),
            )),
            claim("The subtotal is $73.", computation(
                "subtotal", 73, "USD", binding("subtotal", 73, copied.evidence_id),
            )),
        ],
    )
    for record in attempted:
        assert record.success and record.matches_proposed and not record.inputs_grounded
        assert record.provenance_details["subtotal"]["code"] == "untrusted_reference"


@pytest.mark.parametrize(("expression", "result", "units", "operand", "other"), [
    ("fee", 12, "USD", binding("fee", 12, "const:months_per_year"), None),
    ("unit_cost * item_count", 9, "USD", binding("item_count", 1, "const:one"),
     binding("unit_cost", 9)),
    ("p", 0.5, "probability", binding("p", "0.5", "const:fair_coin_probability"), None),
    ("budget - unit_cost", 91, "USD", binding("budget", 100, "const:percent_scale"),
     binding("unit_cost", 9)),
    ("unit_cost * multiplier", 900, "USD", binding("multiplier", 100, "const:percent_scale"),
     binding("unit_cost", 9)),
    ("unit_cost + factor", 21, "USD", binding("factor", 12, "const:months_per_year"),
     binding("unit_cost", 9)),
])
async def test_registered_constants_cannot_supply_empirical_operands(
    run_db, expression, result, units, operand, other,
):
    db, _ = run_db
    record = await DeterministicArithmeticTool(db).execute(
        run_id="offline", actor="lead", grounded_texts=["The unit cost is $9."],
        request=computation(expression, result, units, *([operand, other] if other else [operand])),
    )
    assert record.success and record.matches_proposed and not record.inputs_grounded
    assert record.provenance_details[operand.name]["code"] == "invalid_constant_usage"


@pytest.mark.parametrize("inline", [0, 1])
async def test_inline_identity_is_not_an_empirical_result(run_db, inline):
    db, _ = run_db
    result = await DeterministicArithmeticTool(db).execute(
        run_id="offline", actor="lead", grounded_texts=["The count was unavailable."],
        request=computation(str(inline), inline, "items"),
    )
    assert result.matches_proposed and result.has_ungrounded_literals
    assert not result.inputs_grounded


async def test_plan_assumptions_cannot_supply_numbers_or_fairness(run_db):
    db, _ = run_db
    prompt = "There were 6 tosses with 3 heads in a field trial. The fee is unavailable."
    assumptions = ["The fee is $14", "Coins are fair and independent"]
    catalog = build_provenance_catalog(original_prompt=prompt, assumptions=assumptions)
    tool = DeterministicArithmeticTool(db)
    assumed_fee = await tool.execute(
        run_id="offline", actor="lead", catalog=catalog,
        request=computation("fee", 14, "USD", binding("fee", 14, "assumption:The fee is $14")),
    )
    assumed_prior = await tool.execute(
        run_id="offline", actor="lead", catalog=catalog,
        request=computation(
            "binomial_pmf(n, k, p)", "0.3125", "probability",
            binding("n", 6), binding("k", 3), binding("p", "0.5", "const:fair_coin_probability"),
        ),
    )
    assert assumed_fee.matches_proposed and not assumed_fee.inputs_grounded
    assert assumed_fee.provenance_details["fee"]["code"] == "untrusted_reference"
    assert assumed_prior.matches_proposed and not assumed_prior.inputs_grounded
    assert assumed_prior.provenance_details["p"]["code"] == "invalid_constant_usage"


@pytest.mark.parametrize(("prompt", "operand", "code"), [
    ("Alpha fee is $34. Beta fee is $27.", binding("alpha_fee", 27, "up_2"), "role_conflict"),
    ("Alpha fee is $27. Beta fee is $27.", binding("alpha_fee", 27, "up_2"), "role_conflict"),
    ("Alpha fee is $34. Beta fee is $27.",
     binding("alpha_fee", 27, "up_2", quote="Beta fee is $27"), "role_conflict"),
    ("Alpha fee is $27, Beta fee is $27.",
     binding("alpha_fee", 27, quote="Beta fee is $27"), "role_conflict"),
    ("Base-case volume is 400 units. Stress-case volume is 540 units.",
     binding("base_case_volume", 540, "up_2"), "role_conflict"),
    ("Base-case volume is 540 units. Stress-case volume is 540 units.",
     binding("stress_case_volume", 540, quote="Base-case volume is 540 units"), "role_conflict"),
    ("Alpha processing time is 30 seconds. Beta processing time is 30 minutes.",
     binding("alpha_time", 30, units="minutes"), "unit_conflict"),
    ("Alpha fee is $27. Beta fee is $27.", binding("alpha_fee", 27), "matched"),
    ("Alpha fee is $27, Beta fee is $27.", binding("beta_fee", 27), "matched"),
    ("Base-case volume is 540 units. Stress-case volume is 540 units.",
     binding("stress_case_volume", 540, "up_2", quote="Stress-case volume is 540"), "matched"),
    ("A centre completed 18 inspections and found 4 errors.",
     binding("centre_errors", 4, quote="found 4 errors"), "matched"),
])
async def test_operand_context_distinguishes_entities_and_scenarios(run_db, prompt, operand, code):
    db, _ = run_db
    record = await DeterministicArithmeticTool(db).execute(
        run_id="offline", actor="lead", grounded_texts=[prompt],
        request=computation(operand.name, operand.value, "", operand),
    )
    assert record.success and record.matches_proposed
    assert record.inputs_grounded == (code == "matched")
    assert record.provenance_details[operand.name]["code"] == code
    if code == "matched" and operand.name == "beta_fee":
        assert "beta" in record.provenance_details[operand.name]["source_context"]


async def test_failed_revision_remains_auditable_but_cannot_source_operands(run_db):
    db, _ = run_db
    ledger = ClaimLedger(db)
    prompt = "The base charge is $47. The additional charge is $12."
    _, claims, _ = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The base charge is $47.", computation("base", 47, "USD", binding("base", 47))),
            claim("The total is $59.", computation(
                "base + extra", 59, "USD", binding("base", 47, "clm_1"),
                binding("extra", 12, "up_2"),
            )),
        ],
    )
    original, downstream = claims
    with pytest.raises(ClaimRevisionError, match="cycle"):
        await ledger.apply_targeted_revision(
            run_id="offline", target_claim_id=original.claim_id, expected_version=1,
            new_statement="The base charge is $59.", author_agent_id="lead",
            original_prompt=prompt, computation_request=computation(
                "total", 59, "USD", binding("total", 59, downstream.claim_id),
            ),
        )

    latest = await db.get_claims_for_run("offline", latest_only=True)
    records = await db.get_tool_executions_for_run("offline")
    attached = {tid for c in latest for tid in c.tool_execution_ids}
    [orphan] = [r for r in records if r.tool_execution_id not in attached]
    assert orphan.success and orphan.inputs_grounded and orphan.matches_proposed
    assert all(c.version == 1 for c in latest)
    catalog = build_provenance_catalog(
        original_prompt=prompt, all_claims=latest, tool_records=records, run_id="offline",
    )
    result = await DeterministicArithmeticTool(db).execute(
        run_id="offline", actor="lead", catalog=catalog,
        request=computation("total", 59, "USD", binding("total", 59, orphan.tool_execution_id)),
    )
    assert result.success and result.matches_proposed and not result.inputs_grounded
    assert "not attached" in result.provenance_details["total"]["note"]

    # The catalogue must reject the downstream result even before the graph
    # has propagated the rejected status to the dependent claim.
    await db.update_claim_verification_state(
        run_id="offline", claim_id=original.claim_id, version=1,
        status=ClaimStatus.REJECTED, verification_status=VerificationStatus.CONTRADICTED,
    )
    after_rejection = build_provenance_catalog(
        original_prompt=prompt, all_claims=await db.get_claims_for_run("offline", latest_only=True),
        tool_records=records, run_id="offline",
    )
    assert not after_rejection.targets_by_key[downstream.tool_execution_ids[0]].is_trusted
    assert not after_rejection.targets_by_key[downstream.claim_id].is_trusted



@pytest.mark.parametrize("source_type", ["dependency_finding", "user_prompt"])
async def test_assumed_value_cannot_be_released_as_a_given_fact(run_db, source_type):
    db, settings = run_db
    prompt = "Check whether the fee is known."
    plan = ExecutionPlan(run_id="offline", goal="Check fee", assumptions=["The fee is $14."])
    ledger = ClaimLedger(db)
    evidence, claims, _ = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="The fee is $14.",
        evidence_proposals=[ProposedEvidence(
            source_type=source_type, source_ref="citation",
            excerpt=prompt if source_type == "user_prompt" else "The fee is $14.",
        )],
        claim_proposals=[ProposedClaim(
            statement="The fee is $14.", category=ClaimCategory.GIVEN_FACT, evidence_indices=[0],
        )],
    )
    assert evidence[0].verified_in_input
    provider = FakeProvider()
    executor = ModelExecutor(
        provider=provider, database=db, budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    review = await ReviewerPool(executor=executor, database=db, settings=settings).review_claims(
        run_id="offline", original_prompt=prompt, plan=plan,
    )
    assert not provider.calls
    assert review.blocking_claims[0].status == ClaimStatus.REJECTED
    report = HonestyGate(db).evaluate(
        run_id="offline", original_prompt=prompt, plan=plan,
        synthesis=SynthesisOutput(accepted_segments=[ReleaseSegment(
            segment_type="accepted_claim", text=claims[0].statement,
            claim_refs=[ClaimVersionRef(claim_id=claims[0].claim_id, claim_version=1)],
        )]),
        all_claims=review.blocking_claims, evidence_list=evidence, tool_executions=[],
        verdicts=review.verdicts, incomplete_tasks=[], extra_assumptions=[], extra_missing=[],
        uncertainties=[], is_partial_run=False,
    )
    assert not report.verification.honesty_gate_passed
    assert not report.assumptions
    assert "$14" not in report.rendered_answer


async def test_provenance_accepts_time_conversion_and_observed_probability(run_db):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    duration = await tool.execute(
        run_id="offline", actor="lead", grounded_texts=["Elapsed time is 2 hours."],
        request=computation(
            "elapsed * minutes_per_hour", 120, "minutes", binding("elapsed", 2, units="hours"),
            binding("minutes_per_hour", 60, "const:minutes_per_hour", units="minutes/hour"),
        ),
    )
    empirical = await tool.execute(
        run_id="offline", actor="lead",
        grounded_texts=["There are 6 trials and 1 target success. Probability per trial is 0.2."],
        request=computation(
            "binomial_pmf(n, k, p)", "0.393216", "probability",
            binding("n", 6), binding("k", 1), binding("p", "0.2", "up_2", units="ratio"),
        ),
    )
    assert duration.inputs_grounded and duration.matches_proposed
    assert empirical.inputs_grounded and empirical.matches_proposed


async def test_invalid_declared_premise_cannot_seed_a_same_response_intermediate(run_db):
    db, _ = run_db
    ledger = ClaimLedger(db)
    _, _, records = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None,
        original_prompt="The package weighs 42 kg.", dependency_findings_text="",
        evidence_proposals=[],
        claim_proposals=[
            claim("The package weighs 42 kg.", computation(
                "weight", 42, "kg", binding("weight", 42),
            )).model_copy(update={"premise_claim_ids": ["clm_unknown"]}),
            claim("The shipment weighs 42 kg.", computation(
                "weight", 42, "kg", binding("weight", 42, "clm_1"),
            )),
        ],
    )
    assert all(record.success and record.matches_proposed for record in records)
    assert all(not record.inputs_grounded for record in records)
    assert records[0].provenance_details["__claim_premises__"]["values"] == ["clm_unknown"]
    assert records[1].provenance_details["weight"]["code"] == "untrusted_reference"
    assert records[1].provenance_details["weight"]["source_tool_execution_id"] == (
        records[0].tool_execution_id
    )


async def test_elapsed_days_requires_two_grounded_calendar_dates(run_db):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    request = computation(
        "days_between(start, end)", 4, "days",
        binding("start", 20280227, quote="2028-02-27", units="date"),
        binding("end", 20280302, "up_2", quote="2028-03-02", units="date"),
    )
    record = await tool.execute(
        run_id="offline", actor="lead", request=request,
        grounded_texts=["Access starts on 2028-02-27. Access ends on 2028-03-02."],
    )
    assert record.success and record.matches_proposed and record.inputs_grounded
    assert record.provenance_details["start"]["source_token"] == "2028-02-27"
    fabricated = await tool.execute(
        run_id="offline", actor="lead",
        request=computation(
            "days_between(start, end)", 4, "days",
            binding("start", 20280227, units="date"),
            binding("end", 20280302, "up_2", units="date"),
        ),
        grounded_texts=["The batch code is 20280227. Access ends on 2028-03-02."],
    )
    assert fabricated.success and fabricated.matches_proposed
    assert not fabricated.inputs_grounded and fabricated.formula_appropriate is False


async def test_fair_coins_do_not_imply_independent_tosses(run_db):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    request = computation(
        "binomial_pmf(n, k, p)", "0.375", "probability",
        binding("n", 4), binding("k", 2, "up_2"),
        binding("p", "0.5", "const:fair_coin_probability"),
    )
    result = await tool.execute(
        run_id="offline", actor="lead", request=request,
        grounded_texts=["There are 4 fair coin tosses. Find the chance of 2 heads."],
    )
    assert result.success and result.matches_proposed and result.formula_appropriate is False
    assert "independence" in result.provenance_details["__formula__"]["note"]
    independent = await tool.execute(
        run_id="offline", actor="lead",
        request=computation(
            "binomial_pmf(n, k, p)", "0.375", "probability",
            binding("n", 4), binding("k", 2, "up_2"),
            binding("p", "0.5", "const:fair_coin_probability"),
        ),
        grounded_texts=["There are 4 fair independent coin tosses. Find the chance of 2 heads."],
    )
    assert independent.inputs_grounded and independent.matches_proposed


@pytest.mark.parametrize(("text", "name", "quote", "expected"), [
    ("No surcharge.", "surcharge", "No surcharge.", True),
    ("No commission.", "commission_rate", "No commission.", True),
    ("No surcharge.", "shipping_fee", "No surcharge.", False),
    ("No surcharge.", "surcharge_budget", "No surcharge.", False),
    ("No surcharge unless rushed.", "surcharge", "No surcharge", False),
    ("No rebate information is available.", "rebate", "No rebate", False),
    ("Is there no rebate?", "rebate", "no rebate", False),
    ("No commission.", "commission", None, False),
])
async def test_zero_requires_an_explicit_matching_exclusion(run_db, text, name, quote, expected):
    db, _ = run_db
    rec = await DeterministicArithmeticTool(db).execute(
        run_id="offline", actor="lead", grounded_texts=[text],
        request=computation(name, 0, "USD", binding(name, 0, quote=quote)),
    )
    assert rec.matches_proposed and rec.inputs_grounded == expected


async def test_legacy_percent_complement_requires_a_sourced_fraction(run_db):
    db, _ = run_db
    tool = DeterministicArithmeticTool(db)
    for source, ok in [("The rebate is 8%.", True), ("The fee is $0.08.", False)]:
        record = await tool.execute(
            run_id="offline", actor="lead",
            grounded_texts=["The price is $50.", source],
            request=computation("50*(1-0.08)", 46, "USD",
                                binding("price", 50), binding("rebate", "0.08", "up_2")),
        )
        assert record.matches_proposed and record.inputs_grounded == ok


async def test_failed_local_alias_rebinds_only_after_source_repair(run_db):
    db, settings = run_db
    ledger = ClaimLedger(db)
    agent = AgentSpec(run_id="offline", role="Estimator", instructions="Calculate an invoice.")
    task = TaskSpec(run_id="offline", title="Invoice", description="Apply the rates",
                    assigned_agent_id=agent.agent_id)
    plan = ExecutionPlan(run_id="offline", goal="Invoice", agents=[agent], tasks=[task])
    await db.save_execution_plan(plan)
    prompt = "One carton costs $16. Wrapping costs $3. Delivery costs $7."
    subtotal = "The wrapped carton costs $19."
    total = "The delivered carton costs $26."
    request = computation("carton+wrap", 19, "USD", binding("carton", 16),
                          binding("wrap", 3, "up_2"))
    _, claims, records = await ledger.record_specialist_outputs(
        run_id="offline", agent_id=agent.agent_id, task_id=task.task_id, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim(subtotal, computation("carton+wrap", 19, "USD", binding("carton", 16),
                                        binding("wrap", 3, "up_3")), alias="wrapped"),
            claim(total, computation("wrapped+delivery", 26, "USD",
                                     binding("wrapped", 19, "wrapped"),
                                     binding("delivery", 7, "up_3"))),
        ],
    )
    assert not any(r.inputs_grounded for r in records)
    assert records[1].provenance_details["wrapped"]["code"] == "untrusted_reference"
    assert claims[1].premise_refs[0].claim_id == claims[0].claim_id

    def script(model_prompt, schema):
        if schema is RevisionOutput:
            assert "wrap=3" in model_prompt
            return {"updated_findings": subtotal, "revisions": [{
                "target_claim_id": claims[0].claim_id, "expected_version": 1,
                "new_statement": subtotal, "computation": request.model_dump(),
                "revision_notes": "Use the wrapping clause.",
            }, {
                "target_claim_id": claims[1].claim_id, "expected_version": 1,
                "new_statement": "The total is $900.", "revision_notes": "Unrequested rewrite.",
            }]}
        assert schema is ReviewerBatchOutput
        cid, version = re.search(r"Claim (clm_\w+) \(version=(\d+)", model_prompt).groups()
        assert int(version) == 2
        if cid == claims[0].claim_id:
            assert subtotal in model_prompt and "computed=19" in model_prompt
        else:
            assert cid == claims[1].claim_id and total in model_prompt
            assert "computed=26" in model_prompt and f"{claims[0].claim_id}@v2" in model_prompt
        return {"verdicts": [{"claim_id": cid, "claim_version": 2, "decision": "approve",
                              "rationale": "The stated amounts sum to the recorded result."}]}

    executor = ModelExecutor(provider=FakeProvider(handler=script), database=db,
                             budget_manager=BudgetManager(db, settings), settings=settings)
    reviewer = ReviewerPool(executor=executor, database=db, settings=settings)
    initial = await reviewer.review_claims(run_id="offline", original_prompt=prompt, plan=plan)
    repair = TargetedRepairEngine(executor=executor, database=db, ledger=ledger,
                                  message_bus=MessageBus(db), reviewer_pool=reviewer,
                                  settings=settings)
    outcome = await repair.resolve_disputes(run_id="offline", original_prompt=prompt, plan=plan,
                                            blocking_claims=initial.blocking_claims,
                                            verdicts=initial.verdicts)
    assert outcome.repair_calls == 1 and outcome.re_review_calls == 2
    current = await db.get_claims_for_run("offline", latest_only=True)
    assert all(c.version == 2 and c.status == ClaimStatus.SUPPORTED for c in current)
    assert current[1].statement == total and current[1].premise_refs[0].claim_version == 2
    trace = await db.get_run_trace("offline")
    assert any(e.event_type == "revision_rejected" for e in trace.events)
    assert all(c.statement != "The total is $900." for c in trace.claims)


async def test_transitive_invalidation_uses_root_version_only_on_direct_edges(run_db):
    from crewforge.core.claim_graph import ClaimDependencyGraph

    db, _ = run_db
    ledger = ClaimLedger(db)
    prompt = "The room is booked."
    _, original, _ = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            ProposedClaim(claim_ref="booking", statement=prompt),
            ProposedClaim(statement="The room is occupied.", premise_claim_ids=["booking"]),
        ],
    )
    root, middle = original
    middle = await ledger.apply_targeted_revision(
        run_id="offline", target_claim_id=middle.claim_id, expected_version=1,
        new_statement="The reservation occupies the room.", author_agent_id="lead",
        original_prompt=prompt,
    )
    _, leaves, _ = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            ProposedClaim(statement="Reserve the annex instead.", category="recommendation",
                          premise_claim_ids=[f"{middle.claim_id}@v2"]),
            ProposedClaim(statement="Silently dropping an obsolete edge is unsafe.",
                          premise_claim_ids=[f"{middle.claim_id}@v1", root.claim_id]),
        ],
    )
    assert len(leaves) == 1 and leaves[0].premise_refs[0].claim_version == 2
    await ledger.apply_targeted_revision(
        run_id="offline", target_claim_id=root.claim_id, expected_version=1,
        new_statement="The reservation has been cancelled.", author_agent_id="lead",
        original_prompt=prompt,
    )
    stale = await ClaimDependencyGraph(db).invalidate_dependents(
        run_id="offline", root_claim_id=root.claim_id, superseded_version=1, reason="Cancelled.",
    )
    assert {c.claim_id for c in stale} == {middle.claim_id, leaves[0].claim_id}
    assert all(c.status == ClaimStatus.STALE for c in stale)


async def test_exact_prompt_citation_cannot_borrow_another_entitys_quote(run_db):
    db, _ = run_db
    ledger = ClaimLedger(db)
    evidence, _, _ = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=(
            "Rowan declined delivery. Linden accepted delivery."
        ), dependency_findings_text="", claim_proposals=[], evidence_proposals=[
            ProposedEvidence(source_type="user_prompt", source_ref="up_1",
                             excerpt="Linden accepted delivery."),
            ProposedEvidence(source_type="user_prompt", source_ref="up_2",
                             excerpt="Linden accepted delivery."),
        ],
    )
    assert [e.verified_in_input for e in evidence] == [False, True]


async def test_evidence_and_arithmetic_share_document_aliases_for_chunked_sources(run_db):
    db, _ = run_db
    for name, excerpt in [("runbook", "Archive the duplicate."), ("runbook", "Keep the original."),
                          ("doc_1", "Restoration takes 8 hours.")]:
        await db.insert_evidence(Evidence(
            run_id="offline", created_by="lead", source_type="external_retrieved",
            verified_in_input=True, source_ref=name, excerpt=excerpt,
        ))
    ledger = ClaimLedger(db)
    for _ in range(2):
        evds, _, records = await ledger.record_specialist_outputs(
            run_id="offline", agent_id="lead", task_id=None, original_prompt="How long to restore?",
            dependency_findings_text="", evidence_proposals=[
                ProposedEvidence(source_type="external_retrieved", source_ref="doc_1",
                                 excerpt="Restoration takes 8 hours."),
                ProposedEvidence(source_type="external_retrieved", source_ref="doc_2",
                                 excerpt="Restoration takes 8 hours."),
            ], claim_proposals=[
                claim("Restore in 8 hours.", computation("duration", 8, "hours",
                                                         binding("duration", 8, "doc_1"))),
            ],
        )
        assert [e.verified_in_input for e in evds] == [True, False]
        assert records[0].inputs_grounded
