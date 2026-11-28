"""Metamorphic grounding, challenge, repair and release with scripted models."""

import re
from decimal import Decimal

import pytest

from crewforge.schemas import (
    ArithmeticRequest,
    Claim,
    PlannerOutput,
    ReconciliationOutput,
    ReviewerBatchOutput,
    RevisionOutput,
    SpecialistOutput,
    SynthesisOutput,
    ToolExecutionRecord,
)
from crewforge.tools.arithmetic import (
    DeterministicArithmeticTool,
    arithmetic_lineage_issue,
)
from tests.test_requirement_recovery import calculation, one_task_plan
from tests.test_solution_reconciliation import run_case


@pytest.mark.parametrize(
    ("entity", "other", "value", "unit", "reversed_order"),
    [
        ("Mesa", "Dune", 37, "litres", False),
        ("Thorn", "Ash", 19, "crates", True),
    ],
)
@pytest.mark.parametrize("symbol", ["total_volume", "x", "otherStressCase"])
@pytest.mark.parametrize("wrong", [False, True])
async def test_same_value_different_source_requires_semantic_verdict(
    tmp_path,
    entity,
    other,
    value,
    unit,
    reversed_order,
    symbol,
    wrong,
):
    a = f"The morning ledger for {entity} shows {value} {unit}."
    b = f"The evening ledger for {other} shows {value} {unit}."
    rows = [b, a] if reversed_order else [a, b]
    quote = b if wrong else a
    alias = f"up_{rows.index(quote) + 1}"
    goal = f"Report the dawn quantity for {entity}."
    statement = f"{entity}'s dawn quantity is {value} {unit}."
    seen = []

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan([goal], [("req_1", goal, "calculation")])
        if schema is SpecialistOutput:
            proposal = calculation(statement, "deliv_1", symbol, value, (symbol, value, alias))
            proposal["computation"]["units"] = unit
            proposal["computation"]["variable_bindings"][0].update(source_quote=quote, units=unit)
            return {"findings": statement, "claims": [proposal]}
        if schema is ReviewerBatchOutput:
            cid, ver = re.search(r"Claim (clm_\w+) \(version=(\d+)", prompt).groups()
            seen.append((cid, int(ver)))
            assert f"selected={quote!r}" in prompt and "inputs_grounded=True" in prompt
            matches = f"selected={a!r}" in prompt
            return {
                "verdicts": [
                    dict(
                        claim_id=cid,
                        claim_version=int(ver),
                        decision="approve" if matches else "reject",
                        rationale="Morning ledger is the dawn observation."
                        if matches
                        else "This passage is the evening ledger of the other site.",
                        covered_deliverable_refs=["deliv_1"] if matches else [],
                        blocking_severity="none" if matches else "blocking",
                    )
                ]
            }
        if schema is RevisionOutput:
            return {"updated_findings": "No further evidence supplied.", "revisions": []}
        if schema is ReconciliationOutput:
            return {}
        assert schema is SynthesisOutput
        return {
            "accepted_segments": [
                dict(
                    segment_type="accepted_claim",
                    claim_refs=[dict(claim_id=seen[0][0], claim_version=1)],
                )
            ]
        }

    result, trace = await run_case(tmp_path, " ".join([*rows, goal]), script)
    [record] = trace.tool_executions
    detail = record.provenance_details[symbol]
    assert detail["binding_status"] == "pending_semantic_review"
    assert detail["source_token"] == str(value) and detail["ref"] == alias
    assert len(seen) == 1
    assert (statement in result.answer) is not wrong
    assert result.verification.answer_complete is not wrong
    if wrong:
        assert trace.claims[0].status.value == "rejected"
        assert not result.verification.requirement_coverage


@pytest.mark.parametrize(
    ("question", "proposal", "wrong", "objection"),
    [
        (
            "Dock East terms follow. No surcharge. Dock West has no listed fee information. "
            "Report the fee for Dock West.",
            dict(
                expression="west",
                proposed_result=0,
                units="USD",
                variable_bindings=[
                    dict(name="west", value=0, provenance_ref="up_2", source_quote="No surcharge."),
                ],
            ),
            "Dock West charges no fee.",
            "The exclusion refers to East. West is unspecified.",
        ),
        (
            "There are 4 fair coin tosses. Report the probability of exactly 2 heads.",
            dict(
                expression="binomial_pmf(n,k,p)",
                proposed_result="0.375",
                units="ratio",
                variable_bindings=[
                    dict(name="n", value=4, provenance_ref="up_1"),
                    dict(name="k", value=2, provenance_ref="up_2"),
                    dict(name="p", value="0.5", provenance_ref="const:fair_coin_probability"),
                ],
            ),
            "The probability is 0.375.",
            "Fair marginals do not establish independent trials.",
        ),
    ],
)
async def test_correct_arithmetic_cannot_release_wrong_scope_or_formula(
    tmp_path,
    question,
    proposal,
    wrong,
    objection,
):
    seen = []

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan([question], [("req_1", question, "calculation")])
        if schema is SpecialistOutput:
            return {
                "findings": wrong,
                "claims": [
                    dict(
                        statement=wrong,
                        category="computation",
                        requirement_refs=["deliv_1"],
                        computation=proposal,
                    )
                ],
            }
        if schema is ReviewerBatchOutput:
            assert "inputs_grounded=True" in prompt and question in prompt
            cid = re.search(r"Claim (clm_\w+) \(version=1", prompt)[1]
            seen.append(cid)
            return {
                "verdicts": [
                    dict(
                        claim_id=cid,
                        decision="reject",
                        rationale=objection,
                        blocking_severity="blocking",
                    )
                ]
            }
        if schema is RevisionOutput:
            return {"updated_findings": objection}
        if schema is ReconciliationOutput:
            return {"selections": [dict(deliverable_id="deliv_1", claim_refs=[f"{seen[0]}@v1"])]}
        assert schema is SynthesisOutput
        return {
            "accepted_segments": [
                dict(
                    segment_type="accepted_claim",
                    text=wrong,
                    claim_refs=[dict(claim_id=seen[0], claim_version=1)],
                )
            ]
        }

    result, trace = await run_case(tmp_path, question, script)
    assert len(seen) == 1 and trace.tool_executions[0].matches_proposed
    assert wrong not in result.answer and result.verification.requirement_coverage == 0
    event = next(e for e in trace.events if e.event_type == "solution_candidate")
    assert event.payload["proposed_selections"] == 1 and not event.payload["persisted_selections"]


async def test_challenger_resolves_specialist_disagreement_about_denominator(tmp_path):
    rows = "Field staff tagged 50 swifts. They recaptured 20 birds. Of these, 5 bore a blue tag."
    goal = "Report the blue fraction among recaptures and identify the denominator population."
    incorrect = "The blue fraction among recaptures is 0.1."
    correct = "The blue fraction among recaptures is 0.25."
    audit = "They recaptured 20 birds."
    seen = []
    repaired = []

    def ratio(fixed):
        name, value, ref = ("cohort", 20, "up_2") if fixed else ("cohort", 50, "up_1")
        p = calculation(
            correct if fixed else incorrect,
            "deliv_1",
            "blue/cohort",
            "0.25" if fixed else "0.1",
            ("blue", 5, "up_3"),
            (name, value, ref),
        )
        p["computation"]["units"] = "ratio"
        return p

    def script(prompt, schema):
        if schema is PlannerOutput:
            plan = one_task_plan(
                [goal],
                [
                    ("req_1", "Calculate the blue fraction", "calculation"),
                    ("req_1", "Identify the denominator population", "explanation"),
                ],
            )
            plan["agents"].append(dict(plan["agents"][0], agent_ref="a2", role="Field auditor"))
            plan["tasks"].append(dict(plan["tasks"][0], task_ref="t2", assigned_agent_ref="a2"))
            plan["deliverables"][1]["task_ref"] = "t2"
            return plan
        if schema is SpecialistOutput:
            if "Assigned task (t1)" in prompt:
                return {
                    "findings": "Use all tagged birds as the denominator.",
                    "claims": [ratio(False)],
                }
            return {
                "findings": "Only the recaptured cohort is eligible.",
                "claims": [
                    dict(
                        statement=audit,
                        category="given_fact",
                        evidence_indices=[0],
                        requirement_refs=["deliv_2"],
                    )
                ],
                "evidence_proposals": [
                    dict(source_type="user_prompt", source_ref="up_2", excerpt=audit)
                ],
            }
        if schema is ReviewerBatchOutput:
            verdicts = []
            for cid, ver, stmt in re.findall(
                r"Claim (clm_\w+) \(version=(\d+),[^\n]+\n  Statement: ([^\n]+)",
                prompt,
            ):
                seen.append(stmt)
                if stmt == incorrect:
                    assert "cohort=50 [up_1" in prompt and audit in prompt
                    v = dict(
                        decision="needs_revision",
                        rationale="Wrong denominator population.",
                        required_correction="Use the field auditor's 20 recaptures from up_2.",
                    )
                elif stmt == correct:
                    assert "cohort=20 [up_2" in prompt and "computed=0.25" in prompt
                    v = dict(
                        decision="approve",
                        rationale="Five of twenty recaptures.",
                        covered_deliverable_refs=["deliv_1"],
                    )
                else:
                    assert stmt == audit and audit in rows
                    v = dict(
                        decision="approve",
                        rationale="Field note names the captured cohort.",
                        covered_deliverable_refs=["deliv_2"],
                    )
                verdicts.append(dict(claim_id=cid, claim_version=int(ver), **v))
            return {"verdicts": verdicts}
        assert schema is RevisionOutput
        assert "Use the field auditor's 20 recaptures" in prompt
        cid = re.search(r"Target claim: claim_id=(clm_\w+)", prompt)[1]
        repaired.append(cid)
        return {
            "updated_findings": correct,
            "revisions": [
                dict(
                    target_claim_id=cid,
                    expected_version=1,
                    new_statement=correct,
                    computation=ratio(True)["computation"],
                    revision_notes="Applied the cohort restriction.",
                )
            ],
        }

    result, trace = await run_case(tmp_path, f"{rows} {goal}", script)
    assert len(repaired) == 1 and set(seen) == {incorrect, correct, audit}
    assert result.verification.answer_complete and incorrect not in result.answer
    assert correct in result.answer and audit in result.answer
    assert len(trace.tool_executions) == 2 and result.usage.model_calls == 6
    assert len({c.author_agent_id for c in trace.claims}) == 2


def test_operand_cycle_cannot_masquerade_as_a_shared_ancestor():
    def rec(tid, sources):
        return ToolExecutionRecord(
            run_id="local",
            tool_execution_id=tid,
            actor="lead",
            expression="x",
            computed_result="1",
            inputs_grounded=True,
            success=True,
            provenance_details={
                str(i): dict(kind="tool_result", resolved_id=t) for i, t in enumerate(sources)
            },
        )

    records = {
        tid: rec(tid, sources)
        for tid, sources in (
            ("root", []),
            ("left", ["root"]),
            ("right", ["root"]),
            ("leaf", ["left", "right"]),
        )
    }
    owners = {
        k: Claim(
            claim_id=k,
            statement=k,
            run_id="local",
            status="supported",
            author_agent_id="lead",
            tool_execution_ids=[k],
        )
        for k in records
    }
    assert not arithmetic_lineage_issue(records["leaf"], records, owners, owners)
    records["root"] = rec("root", ["leaf"])
    assert "cycle" in arithmetic_lineage_issue(records["leaf"], records, owners, owners)
    records["root"] = rec("root", []).model_copy(update={"run_id": "foreign"})
    assert "another run" in arithmetic_lineage_issue(records["leaf"], records, owners, owners)


def test_missing_value_does_not_become_zero_by_renaming():
    for name in ("credit", "enterprise_discount", "zero"):
        request = ArithmeticRequest(
            expression=name,
            proposed_result=0,
            variable_bindings=[
                dict(
                    name=name,
                    value=Decimal(0),
                    provenance_ref="up_1",
                    source_quote="Rebate information was withheld.",
                )
            ],
        )
        result = DeterministicArithmeticTool.evaluate(
            run_id="local",
            actor="lead",
            request=request,
            grounded_texts=["Rebate information was withheld."],
        )
        assert result.matches_proposed and not result.inputs_grounded
        assert result.provenance_details[name]["binding_status"] == "invalid"


@pytest.mark.parametrize("kind", ["empty", "off_target", "parent_tag"])
async def test_lead_candidates_account_for_filtered_proposals_and_evidence(tmp_path, kind):
    fact = "Amber shards fluoresce under the lamp."
    goal = "Identify what the lamp reveals."
    checked = []

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan([goal], [("req_1", goal, "explanation")])
        if schema is SpecialistOutput:
            return {"findings": "The lamp observations are available."}
        if schema is ReconciliationOutput:
            return {
                "evidence_proposals": [
                    dict(source_type="user_prompt", source_ref="up_2", excerpt=goal),
                    dict(source_type="user_prompt", source_ref="up_1", excerpt=fact),
                ],
                "claims": []
                if kind == "empty"
                else [
                    dict(
                        statement=fact,
                        category="given_fact",
                        evidence_indices=[1],
                        requirement_refs=["deliv_99" if kind == "off_target" else "req_1"],
                    )
                ],
            }
        if schema is ReviewerBatchOutput:
            assert f"Statement: {fact}" in prompt and "verified=True" in prompt
            checked.append(True)
            return {
                "verdicts": [
                    dict(
                        claim_id="c1",
                        decision="approve",
                        rationale="The observed fluorescence answers the question.",
                        covered_deliverable_refs=["deliv_1"],
                    )
                ]
            }
        assert schema is SynthesisOutput
        return {}

    result, trace = await run_case(tmp_path, f"{fact} {goal}", script)
    event = next(e for e in trace.events if e.event_type == "claims_recorded" and e.actor == "lead")
    assert event.payload["proposed_evidence"] == 2
    assert event.payload["proposed_claims"] == int(kind != "empty")
    assert event.payload["filtered_positions"] == ([1] if kind == "off_target" else [])
    assert len(checked) == int(kind == "parent_tag")
    if kind == "parent_tag":
        assert [e.excerpt for e in trace.evidence] == [fact]
        assert trace.claims[0].evidence_ids == [trace.evidence[0].evidence_id]
        assert result.verification.answer_complete and fact in result.answer
    else:
        assert not trace.evidence and not trace.claims and not result.verification.answer_complete


def test_explicit_rate_denominator_requires_the_sourced_unit_or_a_conversion():
    def evaluate(unit):
        return DeterministicArithmeticTool.evaluate(
            run_id="local", actor="lead", grounded_texts=["Coolant costs $7 per hour."],
            request=ArithmeticRequest(expression="r", proposed_result=7, variable_bindings=[
                dict(name="r", value=7, provenance_ref="up_1", units=unit)
            ]),
        )
    assert evaluate("USD/hours").inputs_grounded
    conflict = evaluate("USD/min")
    assert conflict.matches_proposed and conflict.provenance_details["r"]["code"] == "unit_conflict"
