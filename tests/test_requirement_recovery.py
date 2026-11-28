"""Offline production-path regressions for requirement recovery and release."""

import re

import pytest

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.orchestrator import NetworkRunner
from crewforge.schemas import (
    PlannerOutput,
    ReviewerBatchOutput,
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
)
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider

PROMPT = (
    "Juniper's crate costs $34 and packing costs $8. Cedar quotes $53. "
    "If the seal passes inspection, Juniper credits $5. "
    "If the seal fails inspection, Juniper adds a $7 cleanup fee. "
    "Inspection has not taken place. Calculate Juniper's base total and its savings over Cedar, "
    "recommend a supplier for the base case, and report both inspection totals and the actual "
    "inspection outcome."
)
BASE = "Juniper's base total is $42."
CEDAR = "Cedar quotes $53."
PASS = "If the seal passes inspection, Juniper's total is $37."
FAIL = "If the seal fails inspection, Juniper's total is $49."
SAVINGS = "Juniper's base offering saves $11 versus Cedar."
RECOMMEND = "Choose Juniper for the base case."
UNKNOWN = "The actual inspection outcome is unknown because the inspection has not taken place."


def one_task_plan(requirements, obligations):
    return {
        "goal": "Answer the user's request using supplied evidence",
        "user_requirements": requirements,
        "deliverables": [
            {
                "deliverable_id": f"deliv_{i}", "requirement_ref": parent, "task_ref": "t1",
                "text": text, "kind": kind,
            }
            for i, (parent, text, kind) in enumerate(obligations, start=1)
        ],
        "acceptance_criteria": ["Independently verify each deliverable"],
        "agents": [{"agent_ref": "a1", "role": "Analyst", "instructions": "Verify inputs"}],
        "tasks": [{
            "task_ref": "t1", "title": "Analyze", "description": "Work from the supplied inputs.",
            "assigned_agent_ref": "a1", "expected_deliverables": ["Answer each assigned item"],
        }],
    }


def calculation(statement, tag, expression, result, *operands, alias=None):
    return {
        "statement": statement, "category": "computation", "claim_ref": alias,
        "requirement_refs": [tag],
        "computation": {
            "expression": expression, "proposed_result": result, "units": "USD",
            "variable_bindings": [
                {"name": name, "value": value, "provenance_ref": ref}
                for name, value, ref in operands
            ],
        },
    }


def approved(prompt, claims, seen):
    verdicts = []
    for cid, version, statement in re.findall(
        r"Claim (clm_\w+) \(version=(\d+),[^\n]*\n  Statement: ([^\n]+)", prompt,
    ):
        assert statement in claims
        seen[statement] = (cid, int(version))
        verdicts.append({
            "claim_id": cid, "claim_version": int(version), "decision": "approve",
            "rationale": claims[statement][1],
            "covered_deliverable_refs": claims[statement][0],
        })
    assert verdicts
    return {"verdicts": verdicts}


@pytest.mark.parametrize("max_calls", [16, 4])
async def test_targeted_recovery_qualification_and_synthesis_omissions(tmp_path, max_calls):
    db = Database(tmp_path / "network.db")
    await db.initialize()
    settings = CrewForgeSettings(
        _env_file=None, allow_paid_runs=False, max_model_calls_per_run=max_calls,
        max_retries=0, enable_peer_messaging=False,
    )
    recovered = []
    reviewed = {}

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan(
                ["Calculate base total and savings", "Recommend a base supplier",
                 "Report both inspection scenarios and the actual outcome"],
                [
                    ("req_1", "Calculate the Juniper base total", "calculation"),
                    ("req_1", "Compare base prices and calculate savings", "comparison"),
                    ("req_2", "Recommend the supplier for the base case", "recommendation"),
                    ("req_3", "Calculate the total if the seal passes", "conditional"),
                    ("req_3", "Calculate the total if the seal fails", "conditional"),
                    ("req_3", "State whether the inspection outcome is known", "explanation"),
                ],
            )
        if schema is SpecialistOutput and "TARGETED REQUIREMENT RECOVERY" in prompt:
            assigned = prompt.split("Assigned atomic deliverables", 1)[1].split(
                "Relevant acceptance criteria", 1,
            )[0]
            assert set(re.findall(r"deliv_(\d+) /", assigned)) == {"2", "5"}
            assert BASE in reviewed and PASS in reviewed
            base_ref = f"{reviewed[BASE][0]}@v{reviewed[BASE][1]}"
            recovered.append(assigned)
            return {"findings": "Computed the missing savings and failure total.", "claims": [
                calculation(SAVINGS, "deliv_2", "cedar - juniper", 11,
                            ("cedar", 53, "up_2"), ("juniper", 42, base_ref)),
                calculation(FAIL, "deliv_5", "base + cleanup", 49,
                            ("base", 42, base_ref), ("cleanup", 7, "up_4")),
            ]}
        if schema is SpecialistOutput:
            assert "deliv_6" in prompt
            return {
                "findings": "The base and pass totals are known. Inspection is pending.",
                "evidence_proposals": [
                    {"source_type": "user_prompt", "source_ref": "up_2", "excerpt": CEDAR},
                    {
                        "source_type": "user_prompt", "source_ref": "up_5",
                        "excerpt": "Inspection has not taken place.",
                    },
                ],
                "claims": [
                    calculation(BASE, "deliv_1", "crate + packing", 42,
                                ("crate", 34, "up_1"), ("packing", 8, "up_1"), alias="base"),
                    {
                        "statement": CEDAR, "category": "given_fact", "claim_ref": "cedar",
                        "evidence_indices": [0],
                    },
                    calculation(PASS, "deliv_4", "base - credit", 37,
                                ("base", 42, "base"), ("credit", 5, "up_3")),
                    {
                        "statement": RECOMMEND, "category": "recommendation",
                        "premise_claim_ids": ["base", "cedar"], "requirement_refs": ["deliv_3"],
                    },
                    {
                        "statement": UNKNOWN, "category": "uncertainty",
                        "evidence_indices": [1], "requirement_refs": ["deliv_6"],
                    },
                ],
            }
        if schema is ReviewerBatchOutput:
            return approved(prompt, {
                BASE: (["deliv_1"], "The crate and packing total $42."),
                CEDAR: ([], "The supplied Cedar quote is $53."),
                PASS: (["deliv_4"], "Under the pass condition deduct the stated credit."),
                FAIL: (["deliv_5"], "Under the fail condition add the stated cleanup fee."),
                SAVINGS: (["deliv_2"], "Subtract the Juniper base total from Cedar's quote."),
                RECOMMEND: (["deliv_3"], "The two verified base quotes favor Juniper."),
                UNKNOWN: (["deliv_6"], "The input explicitly says inspection is pending."),
            }, reviewed)
        assert schema is SynthesisOutput
        assert BASE in reviewed
        cid, version = reviewed[BASE]
        return {"accepted_segments": [{
            "segment_type": "accepted_claim", "text": BASE,
            "claim_refs": [{"claim_id": cid, "claim_version": version}],
        }]}

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=script), database=db,
            budget_manager=BudgetManager(db, settings), settings=settings,
        )
        result = await runner.run(PROMPT)
        assert result.verification is not None
        assert result.verification.execution_completed
        assert result.verification.content_released_safely
        assert RECOMMEND in result.answer and PASS in result.answer and UNKNOWN in result.answer
        assert CEDAR not in result.answer  # Supporting claim is not a requested deliverable.
        assert "**Conditional / Qualified Statements:**" in result.answer
        assert "**Proposed Actions (Advisory):**" in result.answer
        assert "`crate + packing` = 42 USD" in result.answer and "[up_1]" in result.answer
        trace = await db.get_run_trace(result.run_id)
        assert trace is not None and trace.plan is not None
        assert len(trace.plan.deliverables) == 6
        assert any(v.covered_deliverable_refs == ["deliv_3"] for v in trace.review_verdicts)
        funded = max_calls == 16
        assert bool(recovered) == funded
        assert result.status == (RunStatus.COMPLETED if funded else RunStatus.PARTIAL)
        assert result.verification.answer_complete is funded
        assert (FAIL in result.answer and SAVINGS in result.answer) == funded
        assert result.usage.model_calls == (6 if funded else 4)
        assert any(e.event_type == "deliverables_missing" for e in trace.events)
        if not funded:
            assert "partially_answered" in result.answer
            assert any(
                e.payload.get("stage") == "requirement_recovery" for e in trace.events
            )
    finally:
        await db.close()


async def test_feasible_claim_does_not_count_as_optimality_proof(tmp_path):
    db = Database(tmp_path / "portfolio.db")
    await db.initialize()
    settings = CrewForgeSettings(_env_file=None, allow_paid_runs=False, max_retries=0)
    feasible = "The Willow project fits the stated budget."
    seen = {}
    recovered = []

    def script(prompt, schema):
        if schema is PlannerOutput:
            return one_task_plan(["Report a feasible project and prove optimality"], [
                ("req_1", "Report a project that fits the budget", "explanation"),
                ("req_1", "Prove optimality across feasible alternatives", "explanation"),
            ])
        if schema is SpecialistOutput and "TARGETED REQUIREMENT RECOVERY" in prompt:
            recovered.append(prompt)
            return {"findings": "No complete optimality argument.", "claims": []}
        if schema is SpecialistOutput:
            return {"findings": feasible, "claims": [{
                "statement": feasible, "category": "logical_inference",
                "requirement_refs": ["deliv_1", "deliv_2"],
            }]}
        if schema is ReviewerBatchOutput:
            reply = approved(
                prompt, {feasible: (["deliv_1"], "Feasibility alone is established.")}, seen,
            )
            reply["missing_requirement_coverage"] = ["deliv_2"]
            return reply
        assert schema is SynthesisOutput
        return {}

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=script), database=db,
            budget_manager=BudgetManager(db, settings), settings=settings,
        )
        result = await runner.run(
            "Willow costs $21 and yields 37 impact points. Oak costs $24 and yields 40 points. "
            "The budget is $25. State a feasible project and prove which option maximizes impact."
        )
        assert result.status == RunStatus.PARTIAL
        assert result.verification.execution_completed
        assert result.verification.content_released_safely
        assert not result.verification.answer_complete
        assert result.verification.requirement_coverage == 0.5
        assert len(recovered) == 1 and "Prove optimality" in recovered[0]
        assert feasible in result.answer and "Outstanding" in result.answer
    finally:
        await db.close()
