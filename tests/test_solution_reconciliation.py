"""Scripted solution fixtures. The production runner owns all IDs, storage, review and release."""

from dataclasses import dataclass

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.schemas import (
    PlannerOutput,
    ReconciliationOutput,
    ReviewerBatchOutput,
    SpecialistOutput,
    SynthesisOutput,
)
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider
from tests.test_requirement_recovery import approved, calculation, one_task_plan


async def run_case(tmp_path, prompt, script, documents=None):
    settings = CrewForgeSettings(
        _env_file=None, allow_paid_runs=False, max_retries=0, max_model_calls_per_run=16,
    )
    provider = FakeProvider(handler=script, usage_reported=False)
    db = Database(tmp_path / "case.db")
    await db.initialize()
    try:
        runner = NetworkRunner(
            provider=provider, database=db, budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        result = await runner.run(prompt, reference_documents=documents)
        trace = await db.get_run_trace(result.run_id)
        assert trace is not None
        return result, trace
    finally:
        await db.close()


def check_solution(result, trace, statements, call_count, record_property):
    assert result.verification, result.error
    assert result.verification.answer_complete, result.answer
    assert result.verification.requirement_coverage == 1
    assert result.verification.content_released_safely
    for statement in statements:
        assert statement in result.answer
        claim = next(c for c in trace.claims if c.statement == statement)
        assert any(v.claim_id == claim.claim_id and v.claim_version == claim.version
                   and v.decision.value == "approve" and v.reviewer_id != claim.author_agent_id
                   for v in trace.review_verdicts)
    # Actual consumption is unavailable with a script. Record the engine's own
    # character-based input estimates (including system prompts) for reproducibility.
    record_property("estimated_input_tokens", sum(
        e.payload["estimated_input_tokens"] for e in trace.events
        if e.event_type == "budget_reserved"
    ))
    record_property("calls", result.usage.model_calls)
    assert result.usage.model_calls == call_count <= 16


async def test_numerical_solution_reconciles_two_specialists_once(tmp_path, record_property):
    prompt = (
        "A gallery receives 73 prints. Of these, 9 need protective backing. "
        "Linen charges $4 per backed print. No rebate. Slate quotes $51 in total. "
        "Calculate each bill, quantify the savings and recommend the cheaper offer."
    )
    linen = "Linen's backing bill is $36 after its zero rebate."
    slate = "Slate quotes $51 in total."
    savings = "Linen saves $15 compared with Slate."
    choice = "Choose Linen because it has the lower total bill."
    seen = {}
    bills = calculation(linen, "deliv_1", "backed_prints * rate * (1 - rebate_rate)", 36,
                        ("backed_prints", 9, "up_2"), ("rate", 4, "up_3"),
                        ("rebate_rate", 0, "up_4"))
    bindings = bills["computation"]["variable_bindings"]
    bindings[0].update(units="prints", source_quote="9 need protective backing")
    bindings[2].update(units="ratio", source_quote="No rebate.")

    def script(model_prompt, schema):
        if schema is PlannerOutput:
            plan = one_task_plan(["Calculate each bill", "Quantify savings and recommend"], [
                ("req_1", "Calculate Linen's bill", "calculation"),
                ("req_1", "Report Slate's bill", "explanation"),
                ("req_2", "Calculate the savings", "calculation"),
                ("req_2", "Recommend the cheaper offer", "recommendation"),
            ])
            plan["agents"].append(dict(plan["agents"][0], agent_ref="a2", role="Quote analyst"))
            plan["tasks"].append(dict(plan["tasks"][0], task_ref="t2", assigned_agent_ref="a2"))
            for idx in (1, 3):
                plan["deliverables"][idx]["task_ref"] = "t2"
            return plan
        if schema is ReconciliationOutput:
            assert "Compare the confirmed totals before choosing the supplier." in model_prompt
            linen_id, v1 = seen[linen]
            slate_id, v2 = seen[slate]
            linen_ref, slate_ref = f"{linen_id}@v{v1}", f"{slate_id}@v{v2}"
            assert f"[{linen_ref}]" in model_prompt and "result 36 USD [tlx_" in model_prompt
            assert linen in model_prompt and slate in model_prompt
            return {"findings": choice, "claims": [
                bills,  # A repeated prerequisite is reused at its reviewed version.
                calculation(savings, "deliv_3", "offer - bill", 15,
                            ("offer", 51, slate_ref), ("bill", 36, "clm_1"), alias="saving"),
                {"statement": choice, "category": "recommendation",
                 "premise_claim_ids": ["clm_2", linen_ref, slate_ref],
                 "requirement_refs": ["deliv_4"]},
            ]}
        if schema is SpecialistOutput:
            if "Assigned task (t1)" in model_prompt:
                return {"findings": linen, "claims": [bills]}
            return {"findings": slate, "evidence_proposals": [
                {"source_type": "user_prompt", "source_ref": "up_5", "excerpt": slate},
            ], "claims": [{"statement": slate, "category": "given_fact",
                           "requirement_refs": ["deliv_2"], "evidence_indices": [0]}]}
        assert schema is ReviewerBatchOutput
        if choice in model_prompt:
            assert "computed=15" in model_prompt and linen in model_prompt and slate in model_prompt
        result = approved(model_prompt, {
            linen: (["deliv_1"], "Nine prints at four dollars with no rebate gives 36."),
            slate: (["deliv_2"], "The Slate quote is explicitly $51."),
            savings: (["deliv_3"], "Subtract the reviewed bill from the flat quote."),
            choice: (["deliv_4"], "The reviewed saving favors Linen."),
        }, seen)
        if choice not in seen:
            result["solution_challenge"] = (
                "Compare the confirmed totals before choosing the supplier."
            )
            result["missing_requirement_coverage"] = ["deliv_3", "deliv_4"]
        return result

    result, trace = await run_case(tmp_path, prompt, script)
    check_solution(result, trace, [linen, slate, savings, choice], 6, record_property)
    assert len(trace.tool_executions) == 2
    assert any(e.event_type == "calculations_reused" for e in trace.events)
    assert all(t.inputs_grounded and t.matches_proposed for t in trace.tool_executions)
    operations = [e.payload["operation"] for e in trace.events if e.event_type == "budget_reserved"]
    assert operations.count("requirement_recovery") == 1
    assert all(c.version == 1 for c in trace.claims)
    decision = next(c for c in trace.claims if c.statement == choice)
    assert decision.author_agent_id == "lead" and len(decision.premise_refs) == 3


@dataclass(frozen=True)
class ReasoningCase:
    question: str
    source: str
    conclusion: str
    rationale: str
    category: str = "logical_inference"
    document: bool = False


@pytest.mark.parametrize("case", [
    ReasoningCase(
        "Does a temporary badge authorize night entry under the supplied rule?",
        "Temporary badges authorize daytime access only.",
        "A temporary badge alone does not authorize night entry.",
        "The rule limits this badge to daytime.", document=True,
    ),
    ReasoningCase(
        "Can the difference in germination be attributed to the lamp?",
        "Only the lamp trays received fertilizer, and only those trays germinated.",
        "The lamp's effect cannot be isolated because fertilizer varied with the lamp.",
        "Both treatments differ together, so the observed contrast cannot identify either effect.",
    ),
    ReasoningCase(
        "Exactly one route can be chosen. Maximize payload within the stated limit and prove it.",
        "The limit is six hours. Iris takes five hours with eight crates. "
        "Moss takes seven hours with ten crates. Reed takes four hours with five crates.",
        "Choose Iris. Moss exceeds the limit and the only other feasible route, Reed, "
        "carries fewer crates.",
        "Enumerating all three routes leaves Iris with the largest feasible payload.",
        category="recommendation",
    ),
    ReasoningCase(
        "Does a silent bell imply a closed hatch?",
        "An open hatch always rings the bell. The bell is silent.",
        "The hatch is closed because an open hatch would ring the bell.",
        "Modus tollens rules out an open hatch in the stated model.",
    ),
    ReasoningCase(
        "Which reading did the operator intend?",
        "The unmarked selector can mean inlet or outlet. No operator selection was recorded.",
        "Either inlet or outlet is possible because the operator selection was not recorded.",
        "Both interpretations remain consistent with the record.", category="uncertainty",
    ),
], ids=["document", "causal", "optimization", "logic", "ambiguity"])
async def test_reviewed_solution_from_primary_premise(tmp_path, record_property, case):
    seen = {}
    prompt = case.question if case.document else f"{case.source} {case.question}"
    documents = [ReferenceDocument(doc_id="permit", source_ref="policy", content=case.source)
                 ] if case.document else []
    tag_kind = "recommendation" if case.category == "recommendation" else "explanation"

    def script(model_prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan([case.question], [("req_1", case.question, tag_kind)])
        if schema is SpecialistOutput:
            assert case.source in model_prompt
            return {
                "findings": case.conclusion,
                "evidence_proposals": [{
                    "source_type": "external_retrieved" if case.document else "user_prompt",
                    "source_ref": "doc_1" if case.document else "user_prompt",
                    "excerpt": case.source,
                }],
                "claims": [
                    {"statement": case.source, "claim_ref": "premise", "evidence_indices": [0],
                     "category": "external_factual_assertion" if case.document else "given_fact"},
                    {"statement": case.conclusion, "category": case.category,
                     "requirement_refs": ["deliv_1"], "premise_claim_ids": ["premise"]},
                ],
            }
        if schema is ReviewerBatchOutput:
            assert case.source in model_prompt and "verified=True" in model_prompt
            return approved(model_prompt, {
                case.source: ([], "The primary evidence states this premise."),
                case.conclusion: (["deliv_1"], case.rationale),
            }, seen)
        assert schema is SynthesisOutput
        return {}

    result, trace = await run_case(tmp_path, prompt, script, documents)
    check_solution(result, trace, [case.conclusion], 3, record_property)
    assert len(trace.agents) == 1 and not trace.tool_executions
    assert all(e.verified_in_input for e in trace.evidence)
    premise, conclusion = trace.claims
    assert conclusion.premise_refs[0].claim_id == premise.claim_id
    assert conclusion.category.value == case.category


async def test_two_missing_comparisons_reuse_the_same_reviewed_total(tmp_path, record_property):
    """Same script can run against the earlier per-assignment recovery implementation."""
    prompt = (
        "Rental is $41. Handling is $12. The cap is $60. The alternative is $68. "
        "State the total and alternative, and calculate cap headroom and savings."
    )
    total = "Rental plus handling is $53."
    alternative = "The alternative is $68."
    headroom = "The cap headroom is $7."
    savings = "The rental saves $15 against the alternative."
    seen = {}
    subtotal = calculation(total, "deliv_1", "rent+handling", 53,
                           ("rent", 41, "up_1"), ("handling", 12, "up_2"))

    def script(model_prompt, schema):
        if schema is PlannerOutput:
            plan = one_task_plan(["Price and headroom", "Alternative and savings"], [
                ("req_1", "Calculate total rental", "calculation"),
                ("req_1", "Calculate budget headroom", "calculation"),
                ("req_2", "Report the alternative price", "explanation"),
                ("req_2", "Calculate rental savings", "comparison"),
            ])
            plan["agents"].append(dict(plan["agents"][0], agent_ref="a2", role="Comparator"))
            plan["tasks"].append(dict(plan["tasks"][0], task_ref="t2", assigned_agent_ref="a2"))
            for item in plan["deliverables"][2:]:
                item["task_ref"] = "t2"
            return plan
        if schema is ReconciliationOutput:
            checklist = model_prompt.split("Assigned atomic deliverables", 1)[1].split(
                "Relevant acceptance criteria", 1,
            )[0]
            cid, version = seen[total]
            ref = f"{cid}@v{version}"
            assert ref in model_prompt
            proposals = [subtotal]  # Specialist volunteers an unchanged, unrelated calculation.
            if "deliv_2" in checklist:
                proposals.append(calculation(headroom, "deliv_2", "cap-total", 7,
                                             ("cap", 60, "up_3"), ("total", 53, ref)))
            if "deliv_4" in checklist:
                proposals.append(calculation(savings, "deliv_4", "offer-total", 15,
                                             ("offer", 68, "up_4"), ("total", 53, ref)))
            return {"findings": "Completed the requested comparisons.", "claims": proposals}
        if schema is SpecialistOutput:
            if "Assigned task (t1)" in model_prompt:
                return {"findings": total, "claims": [subtotal]}
            return {"findings": alternative, "evidence_proposals": [
                {"source_type": "user_prompt", "source_ref": "up_4", "excerpt": alternative},
            ], "claims": [{"statement": alternative, "category": "given_fact",
                           "evidence_indices": [0], "requirement_refs": ["deliv_3"]}]}
        if schema is ReviewerBatchOutput:
            return approved(model_prompt, {
                total: (["deliv_1"], "41 + 12 = 53."),
                alternative: (["deliv_3"], "The other price is supplied."),
                headroom: (["deliv_2"], "60 - 53 = 7."),
                savings: (["deliv_4"], "68 - 53 = 15."),
            }, seen)
        assert schema is SynthesisOutput
        return {}

    result, trace = await run_case(tmp_path, prompt, script)
    counts = [t.expression for t in trace.tool_executions]
    record_property("repeated_base_calculations", counts.count("rent+handling") - 1)
    record_property("arithmetics", len(counts))
    check_solution(result, trace, [total, alternative, headroom, savings], 6, record_property)
    assert counts.count("rent+handling") == 1 and len(counts) == 3


async def test_lead_selects_the_more_informative_reviewed_decision(tmp_path, record_property):
    source = "Only North has a lift. South has stairs only."
    terse = "Use North for step-free access."
    explained = "Use North for its lift. South offers only stairs."
    seen = {}

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan(["Recommend a step-free route"], [
                ("req_1", "Recommend a step-free route", "recommendation"),
            ])
        if schema is SpecialistOutput:
            return {"findings": explained, "evidence_proposals": [
                {"source_type": "user_prompt", "source_ref": "user_prompt", "excerpt": source},
            ], "claims": [
                {"statement": source, "category": "given_fact", "evidence_indices": [0],
                 "claim_ref": "access"},
                *[{"statement": answer, "category": "recommendation",
                   "requirement_refs": ["deliv_1"], "premise_claim_ids": ["access"]}
                  for answer in (terse, explained)],
            ]}
        if schema is ReviewerBatchOutput:
            return approved(prompt, {
                source: ([], "The operator describes both entrances."),
                terse: (["deliv_1"], "North provides step-free access."),
                explained: (["deliv_1"], "The recommendation correctly explains both options."),
            }, seen)
        assert schema is SynthesisOutput
        cid, version = seen[explained]
        assert "Reviewed coverage for release selection" in prompt and cid in prompt
        return {"recommendations": [{
            "segment_type": "recommendation", "claim_refs": [
                {"claim_id": cid, "claim_version": version},
            ],
        }]}

    result, trace = await run_case(tmp_path, f"{source} Recommend a step-free route.", script)
    check_solution(result, trace, [explained], 4, record_property)
    assert terse not in result.answer
