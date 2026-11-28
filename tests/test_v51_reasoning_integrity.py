"""Scripted independent semantic review and deterministic release controls."""

import re

import pytest

from crewforge.schemas import (
    CompletenessReviewOutput,
    PlannerOutput,
    ReconciliationOutput,
    ReviewerBatchOutput,
    RevisionOutput,
    SpecialistOutput,
    SynthesisOutput,
)
from tests.test_requirement_recovery import calculation, one_task_plan
from tests.test_solution_reconciliation import run_case


@pytest.mark.parametrize(
    ("orchard", "canal", "first", "second", "mode"),
    [
        ("Aster", "Fen", 47, 39, "valid"),
        ("Fern", "Pike", 55, 43, "wrong_saving"),
        ("Pond", "Ash", 45, 36, "unrelated_value"),
        ("Cedar", "Elm", 42, 27, "wrong_direction"),
    ],
)
async def test_recommendation_release_requires_its_own_reviewed_numbers(
    tmp_path,
    orchard,
    canal,
    first,
    second,
    mode,
):
    difference = first - second
    claimed = 99 if mode == "unrelated_value" else difference
    choice = orchard if mode == "wrong_direction" else canal
    facts = [f"{orchard} quotes ${first}.", f"{canal} quotes ${second}."]
    statement = f"Choose {choice} because it saves ${claimed}."
    prompt = " ".join([*facts, "An unrelated item costs $99.", "Recommend the cheaper quote."])

    def script(text, model):
        if model is PlannerOutput:
            return one_task_plan(["Recommend"], [("req_1", "Recommend", "recommendation")])
        if model is SpecialistOutput:
            recommendations = calculation(
                statement,
                "deliv_1",
                "higher - lower",
                difference + 1 if mode == "wrong_saving" else difference,
                ("higher", first, "clm_1"),
                ("lower", second, "clm_2"),
            )
            recommendations.update(category="recommendation", premise_claim_ids=["clm_1", "clm_2"])
            return {
                "findings": "Compare both quotes.",
                "claims": [
                    calculation(fact, "", "quote", value, ("quote", value, f"up_{i}"))
                    for i, (fact, value) in enumerate(zip(facts, [first, second], strict=True), 1)
                ]
                + [recommendations],
            }
        if model is ReviewerBatchOutput:
            verdicts = []
            for ref, claim in re.findall(
                r"alias=(c\d+),[^\n]+\n  Statement: ([^\n]+)",
                text,
            ):
                bad_direction = claim == statement and mode == "wrong_direction"
                verdicts.append(
                    {
                        "claim_id": ref,
                        "decision": "reject" if bad_direction else "approve",
                        "rationale": "Selected vendor is the higher quote."
                        if bad_direction
                        else "Scripted approval to exercise scoped numeric checks.",
                        "covered_deliverable_refs": ["deliv_1"] if claim == statement else [],
                    }
                )
            return {"verdicts": verdicts}
        if model is RevisionOutput:
            return {"updated_findings": "No revision offered."}
        assert model in {ReconciliationOutput, SynthesisOutput}
        return {}

    result, trace = await run_case(tmp_path, prompt, script)
    assert (statement in result.answer) == (mode == "valid")
    assert result.verification.answer_complete == (mode == "valid")
    assert next(c for c in trace.claims if c.statement == statement).is_material


@pytest.mark.parametrize("proof", ["absent", "mismatch", "grounded", "fabricated"])
async def test_numerical_challenge_never_forwards_unverified_direction(tmp_path, proof):
    prompt = "North quotes $87. South quotes $51. Compare quotes and explain the tradeoff."
    incorrect = "North is cheaper by $36."
    amount = "North quotes $87."
    seen = []

    def script(text, model):
        if model is PlannerOutput:
            return one_task_plan(
                ["Amount", "Tradeoff"],
                [("req_1", "Amount", "calculation"), ("req_2", "Tradeoff", "explanation")],
            )
        if model is SpecialistOutput:
            return {
                "findings": "Start with the quotes.",
                "claims": [
                    calculation(amount, "deliv_1", "north", 87, ("north", 87, "up_1")),
                ],
            }
        if model is ReviewerBatchOutput:
            response = {
                "verdicts": [
                    {
                        "claim_id": "c1",
                        "decision": "approve",
                        "rationale": "Verbatim quoted price.",
                        "covered_deliverable_refs": ["deliv_1"],
                    }
                ],
                "solution_challenge": incorrect,
                "missing_requirement_coverage": ["deliv_2"],
            }
            if proof != "absent":
                response["challenge_calculation"] = {
                    "expression": "north - south",
                    "proposed_result": 99 if proof == "mismatch" else 36,
                    "units": "USD",
                    "variable_bindings": [
                        {"name": "north", "value": 87, "provenance_ref": "up_1"},
                        {
                            "name": "south",
                            "value": 51,
                            "provenance_ref": "up_99" if proof == "fabricated" else "up_2",
                        },
                    ],
                }
            return response
        if model is ReconciliationOutput:
            seen.append(text)
            return {}
        if model is CompletenessReviewOutput:
            assert "Open deliverables" in text
            return {"endorsements": []}
        assert model is SynthesisOutput
        return {}

    result, trace = await run_case(tmp_path, prompt, script)
    assert len(seen) == 1 and incorrect not in seen[0] and incorrect not in result.answer
    assert ("It evaluates to 36 USD" in seen[0]) == (proof == "grounded")
    assert amount in result.answer
    [entry] = next(e for e in trace.events if e.event_type == "review_completed").payload[
        "solution_challenges"
    ]
    assert bool(entry["repair_hypothesis"]) == (proof == "grounded")
