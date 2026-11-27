from pathlib import Path

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.claim_graph import ClaimDependencyGraph
from crewforge.core.executor import ModelExecutor
from crewforge.core.ledger import ClaimLedger
from crewforge.core.messaging import MessageBus
from crewforge.core.repair import TargetedRepairEngine, _build_revision_prompt
from crewforge.core.reviewer import ReviewerPool
from crewforge.schemas import (
    ArithmeticRequest,
    ArithmeticVariableBinding,
    ClaimStatus,
    Evidence,
    ExecutionPlan,
    ProposedClaim,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool
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


async def test_changed_premise_keeps_dependent_unverified_and_repair_gets_new_value(run_db):
    db, settings = run_db
    ledger = ClaimLedger(db)
    prompt = "The initial estimate is $75. The corrected amount is $91. The fee is $6."
    _, claims, records = await ledger.record_specialist_outputs(
        run_id="offline", agent_id="lead", task_id=None, original_prompt=prompt,
        dependency_findings_text="", evidence_proposals=[], claim_proposals=[
            claim("The estimate is $75.", computation(
                "estimate", 75, "USD", binding("estimate", 75),
            )),
            claim("With the fee the charge is $81.", computation(
                "estimate + fee", 81, "USD",
                binding("estimate", 75, "clm_1"), binding("fee", 6, "up_3"),
            )),
        ],
    )
    assert records[1].inputs_grounded
    upstream, dependent = claims
    revised = await ledger.apply_targeted_revision(
        run_id="offline", target_claim_id=upstream.claim_id, expected_version=1,
        new_statement="The corrected amount is $91.", author_agent_id="lead",
        original_prompt=prompt, computation_request=computation(
            "amount", 91, "USD", binding("amount", 91, "up_2"),
        ),
    )
    await db.update_claim_verification_state(
        run_id="offline", claim_id=revised.claim_id, version=2,
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.COMPUTATIONALLY_VERIFIED,
    )
    stale = await ClaimDependencyGraph(db).invalidate_dependents(
        run_id="offline", root_claim_id=revised.claim_id, reason="Corrected amount",
        superseded_version=1,
    )
    assert [c.claim_id for c in stale] == [dependent.claim_id]

    provider = FakeProvider()
    executor = ModelExecutor(
        provider=provider, database=db, budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    repair = TargetedRepairEngine(
        executor=executor, database=db, ledger=ledger, message_bus=MessageBus(db),
        reviewer_pool=ReviewerPool(executor=executor, database=db, settings=settings),
        settings=settings,
    )
    calls, verdicts, blocked = await repair._revalidate_stale_dependents(
        run_id="offline", original_prompt=prompt, plan=ExecutionPlan(
            run_id="offline", goal="Estimate charge", agents=[], tasks=[],
        ),
        verdicts_by_key={}, run_deadline_monotonic=None,
    )
    assert calls == 0 and not provider.calls
    assert len(blocked) == 1
    result = blocked[0]
    assert result.version == 2 and result.status == ClaimStatus.REJECTED
    assert result.tool_execution_ids != dependent.tool_execution_ids
    tools = {r.tool_execution_id: r for r in await db.get_tool_executions_for_run("offline")}
    replay = tools[result.tool_execution_ids[0]]
    assert replay.computed_result == "81" and not replay.inputs_grounded
    assert replay.provenance_details["estimate"]["resolved_id"] == f"{upstream.claim_id}@v2"
    assert replay.provenance_details["estimate"]["candidates"] == ["91 USD"]
    revision_prompt = _build_revision_prompt(
        original_prompt=prompt, disputed_claims=[result], evidence_by_id={},
        verdicts_by_claim_key={(result.claim_id, result.version): verdicts},
        dependency_findings_text="", tool_by_id=tools,
    )
    assert "estimate=75" in revision_prompt
    assert "Candidates: 91 USD." in revision_prompt


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
