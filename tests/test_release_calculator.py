"""Development regression for the archived V5.1 wallet case using scripted model responses."""

import asyncio
import json
from decimal import Decimal
from pathlib import Path

import pytest

from crewforge.schemas import PlannerOutput, ReviewerBatchOutput, SpecialistOutput
from crewforge.tools.arithmetic import evaluate_decimal_expression
from tests.test_requirement_recovery import approved, calculation, one_task_plan
from tests.test_solution_reconciliation import run_case

TRACE = Path(".crewforge/v51_paired_eval/attempt/private/network_full_trace.json")


@pytest.mark.parametrize(("expression", "answer"), [
    ("ceil(n / block)", 7), ("floor(n / block)", 6), ("-n // block", -7), ("-n % block", 100),
])
def test_cell_rounding_boundaries(expression, answer):
    assert evaluate_decimal_expression(expression, {"n": "1300", "block": "200"}) == answer


@pytest.mark.skipif(not TRACE.is_file(), reason="local historical V5.1 trace unavailable")
async def test_derived_wallet_calculations_survive_independent_review_and_release(tmp_path):
    archived = json.loads(await asyncio.to_thread(TRACE.read_text))
    prompt = archived["prompt"]
    claims = []

    def operand(name, value, ref, units):
        return name, value, ref, units

    def add(statement, formula, value, unit, *operands):
        tag = f"deliv_{len(claims) + 1}"
        row = calculation(statement, tag, formula, str(value), *(b[:3] for b in operands))
        row["computation"]["units"] = unit
        for binding, raw in zip(row["computation"]["variable_bindings"], operands, strict=True):
            binding["units"] = raw[3]
        claims.append(row)
        return operand("previous", value, f"clm_{len(claims)}", unit)

    def as_name(name, bound):
        return name, *bound[1:]

    invoice = operand("invoice", 1760, "up_1", "credits")
    wallet = operand("wallet", 900, "up_2", "credits")
    buffer = operand("buffer", 340, "up_2", "credits")
    block = operand("size", 200, "up_4", "credits")
    block_price = operand("rate", 15, "up_4", "dollars/block")
    block_fee = operand("fee", 11, "up_4", "dollars per order")
    single_price = operand("rate", "0.082", "up_5", "dollars/credit")
    single_fee = operand("fee", 5, "up_5", "USD/order")
    extra = operand("extra", 100, "up_8", "credits")
    for increase in (0, 100):
        scenario = "At the increased invoice" if increase else "At the original invoice"
        need = 1200 + increase
        total_invoiced = "invoice + extra" if increase else "invoice"
        invoice_inputs = [invoice, extra] if increase else [invoice]
        req = add(f"{scenario}, the needed purchase is {need} credits.",
                  f"{total_invoiced} + buffer - wallet", need, "credits",
                  *invoice_inputs, buffer, wallet)
        blocks = (need + 199) // 200
        mica_units = add(f"{scenario}, Mica requires buying {blocks * 200} credits.",
                         "ceil(need / size) * size", blocks * 200, "credits",
                         as_name("need", req), block)
        m_cost = blocks * 15 + 11
        mica = add(f"{scenario}, Mica costs ${m_cost:.2f}.", "ceil(need / size) * rate + fee",
                   m_cost, "USD", as_name("need", req), block, block_price, block_fee)
        add(f"{scenario}, Mica leaves {340 + blocks * 200 - need} credits.",
            f"wallet + purchase - ({total_invoiced})", 340 + blocks * 200 - need,
            "credits", wallet, as_name("purchase", mica_units), *invoice_inputs)
        flint_units = add(f"{scenario}, Flint requires buying {need} credits.",
                          "need", need, "credits", as_name("need", req))
        f_cost = Decimal(need) * Decimal("0.082") + 5
        flint = add(f"{scenario}, Flint costs ${f_cost:.2f}.", "purchase * rate + fee",
                    f_cost, "USD", as_name("purchase", flint_units), single_price, single_fee)
        add(f"{scenario}, Flint leaves 340 credits.", f"wallet + purchase - ({total_invoiced})",
            340, "credits", wallet, as_name("purchase", flint_units), *invoice_inputs)
        low, high = (mica, flint) if not increase else (flint, mica)
        savings = Decimal(high[1]) - Decimal(low[1])
        add(f"{scenario}, choose {'Flint' if increase else 'Mica'} to save ${savings:.2f}.",
            "high - low", savings, "USD", as_name("high", high), as_name("low", low))
        claims[-1].update(category="recommendation", premise_claim_ids=[mica[2], flint[2]])

    def script(review_prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan(["Original invoice", "Increased invoice"], [
                ("req_1" if i < 8 else "req_2", row["statement"],
                 "recommendation" if i % 8 == 7 else "calculation")
                for i, row in enumerate(claims)
            ])
        if schema is SpecialistOutput:
            return {"findings": "Compare quoted costs and remaining credits in each case.",
                    "claims": claims}
        assert schema is ReviewerBatchOutput
        return approved(review_prompt, {
            row["statement"]: (row["requirement_refs"], "Expected archived case arithmetic.")
            for row in claims
        }, {})

    result, trace = await run_case(tmp_path, prompt, script)
    assert result.verification.answer_complete, result.answer or result.error
    assert all(row["statement"] in result.answer for row in claims)
    assert len(trace.tool_executions) == 16
    assert all(tool.inputs_grounded and tool.matches_proposed for tool in trace.tool_executions)
    assert result.usage.model_calls == 3


def test_quote_resolution_preserves_identity_and_denominator():
    from crewforge.schemas import ArithmeticRequest
    from crewforge.tools.arithmetic import DeterministicArithmeticTool

    def check(ref, unit, prompt):
        return DeterministicArithmeticTool.evaluate(
            run_id="unit", actor="test",
            request=ArithmeticRequest(
                expression="fee", variable_bindings=[
                    {"name": "fee", "value": "7", "provenance_ref": ref,
                     "source_quote": "$7 per order", "units": unit},
                ],
            ),
            grounded_texts=[prompt],
        ).provenance_details["fee"]

    prompt = "Dispatch charges $7 per order."
    assert check("", "dollars / order", prompt)["resolved_id"] == "up_1"
    assert check("fake_ref", "USD/order", prompt)["code"] == "unknown_reference"
    assert check("up_1", "USD/block", prompt)["code"] == "unit_conflict"
    assert check("up_1", "EUR/order", prompt)["code"] == "unit_conflict"
    assert check("", "USD/order", prompt + " Courier charges $7 per order.")["grounded"] is False
