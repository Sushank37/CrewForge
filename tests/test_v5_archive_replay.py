"""Local diagnostic acceptance. Frozen response reconstruction, explicitly scripted new reviews."""

import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest
from scripts.replay_calculation_graph import original_request, replay

from crewforge.schemas import (
    PlannerOutput,
    ReconciliationOutput,
    ReviewerBatchOutput,
    RevisionOutput,
    RunTrace,
    SpecialistOutput,
    SynthesisOutput,
)
from crewforge.storage.database import Database
from tests.test_solution_reconciliation import run_case

ARCHIVE = Path(".crewforge/optimized_live_diagnostic/attempt/full_trace.json")

# Independently worked from the public question, in dollars and cents.
BASE_A = 10000 * 135 + 8000 * 95
BASE_B = 18000 * 98
HIGH_A = 10000 * 135 + 14000 * 95
HIGH_B = 24000 * 98


def alder_total(shipping):
    return 480000 + shipping - shipping // 20 + 2400 * 70


def birch_total(shipping):
    return 820000 + shipping + 2400 * 20


# These are test-only, human-audited archived claim identities. No grading key is used.
CENTS = {
    "4a4656b11397": HIGH_A,
    "2a35454070a9": HIGH_A // 20,
    "24783a37ef01": alder_total(HIGH_A),
    "64f2e9225952": HIGH_B,
    "6fd96db0595a": 0,
    "8162069461f6": birch_total(HIGH_B),
    "802d16361a78": BASE_A,
    "ddb614c78d22": BASE_A // 20,
    "88b1956a6fcc": alder_total(BASE_A),
    "017ee1781f79": 2700000 - alder_total(BASE_A),
    "8e261eb60c1e": BASE_B,
    "4706afa9009f": 0,
    "15a95506c520": birch_total(BASE_B),
    "e8b6cdb5382e": 2700000 - birch_total(BASE_B),
    "c492aa435f67": alder_total(BASE_A) - birch_total(BASE_B),
    "1eddec19c097": birch_total(HIGH_B) - alder_total(HIGH_A),
    "39fc224cd7db": alder_total(HIGH_A) - 2700000,
    "b4f72e841600": birch_total(HIGH_B) - 2700000,
}
# Compound items in the historical plan require the following findings together.
GROUPS = {
    2: ["ddb614c78d22", "88b1956a6fcc", "017ee1781f79"],
    3: ["8e261eb60c1e", "4706afa9009f", "15a95506c520", "e8b6cdb5382e"],
    5: ["4a4656b11397", "2a35454070a9", "24783a37ef01"],
    6: ["64f2e9225952", "8162069461f6"],
    7: ["1eddec19c097", "39fc224cd7db", "b4f72e841600"],
    8: ["88b1956a6fcc", "15a95506c520"],
    9: ["24783a37ef01", "8162069461f6"],
}


@pytest.mark.skipif(not ARCHIVE.exists(), reason="local diagnostic export is not versioned")
async def test_all_original_arithmetic_requests(tmp_path):
    archive = RunTrace.model_validate_json(await asyncio.to_thread(ARCHIVE.read_text))
    db = Database(tmp_path / "exact.db")
    await db.initialize()
    try:
        rows = await replay(archive, db)
    finally:
        await db.close()
    assert len(rows) == 23 and all(r["matches"] for r in rows)
    assert sum(r["grounded"] for r in rows) == 15
    failures = [code for r in rows for code in r["failures"].values()]
    assert failures.count("value_not_found") == 1
    assert failures.count("quote_not_found") == 2
    assert failures.count("unit_conflict") == 3


@pytest.mark.skipif(not ARCHIVE.exists(), reason="local diagnostic export is not versioned")
async def test_archived_solution_through_runner(tmp_path, record_property):
    archive = RunTrace.model_validate_json(await asyncio.to_thread(ARCHIVE.read_text))
    plan = archive.plan
    tools = {t.tool_execution_id: t for t in archive.tool_executions}
    original = {c.claim_id: c for c in archive.claims if c.version == 1}
    corrected = {c.claim_id: c for c in archive.claims if c.version == 2}
    evidence = {e.evidence_id: e for e in archive.evidence}
    task_refs = {t.task_id: t.task_ref for t in plan.tasks}
    agent_refs = {a.agent_id: a.agent_ref for a in plan.agents}
    owners = {tid: c.claim_id for c in original.values() for tid in c.tool_execution_ids}
    by_text = {c.statement: c for c in reversed(archive.claims)}
    latest_refs, candidate_texts, script_errors = {}, {}, []

    def propose(task, prompt):
        members = [c for c in original.values() if c.task_id == task.task_id]
        # Recover model-local aliases from each recorded consumer, never from numeric equality.
        aliases = {c.claim_id: f"clm_{i}" for i, c in enumerate(members, 1)}
        for c in members:
            for tid in c.tool_execution_ids:
                for d in tools[tid].provenance_details.values():
                    if d["kind"] == "tool_result":
                        aliases[owners[d["resolved_id"]]] = d["ref"]
        for cid, c in original.items():
            if cid in aliases:
                continue
            match = re.search(r"(clm_\w+) \(v1, [^\n]+?\): " + re.escape(c.statement), prompt)
            if match:
                aliases[cid] = match[1]
        claims, proposals = [], []
        for c in members:
            request = None
            if c.tool_execution_ids:
                request = original_request(tools[c.tool_execution_ids[0]])
                for b in request.variable_bindings:
                    b.provenance_ref = aliases.get(b.provenance_ref, b.provenance_ref)
            indices = []
            for eid in c.evidence_ids:
                e = evidence[eid]
                entry = e.model_dump(include={"source_type", "source_ref", "excerpt"})
                if entry not in proposals:
                    proposals.append(entry)
                indices.append(proposals.index(entry))
            claims.append(
                dict(
                    statement=c.statement,
                    category=c.category,
                    claim_ref=aliases[c.claim_id],
                    computation=request,
                    requirement_refs=c.requirement_refs,
                    premise_claim_ids=[aliases[pid] for pid in c.depends_on_claim_ids],
                    evidence_indices=indices,
                )
            )
        return dict(findings=task.findings, claims=claims, evidence_proposals=proposals)

    def script(prompt, schema):
        if schema is PlannerOutput:
            return {
                **plan.model_dump(exclude={"tasks", "agents"}),
                "agents": [a.model_dump() for a in plan.agents],
                "tasks": [
                    dict(
                        t.model_dump(),
                        assigned_agent_ref=agent_refs[t.assigned_agent_id],
                        depends_on=[task_refs[dep] for dep in t.depends_on],
                        expected_deliverables=t.expected_outputs,
                    )
                    for t in plan.tasks
                ],
            }
        if schema is SpecialistOutput:
            task = next(t for t in archive.tasks if f"Assigned task ({t.task_ref})" in prompt)
            return propose(task, prompt)
        if schema is RevisionOutput:
            targets = re.findall(
                r"Target claim: claim_id=(clm_\w+), expected_version=(\d+).*?"
                r"\nCurrent statement: ([^\n]+)",
                prompt,
            )
            revisions = []
            for cid, version, statement in targets:
                old = by_text[statement]
                assert old.claim_id == "clm_802d16361a78" and "8000" in prompt
                fixed = corrected[old.claim_id]
                revisions.append(
                    dict(
                        target_claim_id=cid,
                        expected_version=version,
                        new_statement=fixed.statement,
                        computation=original_request(tools[fixed.tool_execution_ids[0]]),
                        revision_notes="Derive the additional volume from both cited quantities.",
                    )
                )
            return {"updated_findings": "Used the sourced tier difference.", "revisions": revisions}
        if schema is ReconciliationOutput:
            selections = []
            for number, members in GROUPS.items():
                tag = f"deliv_{number}"
                refs = [latest_refs[m][0] for m in members]
                assert all(f"[{ref}]" in prompt for ref in refs)
                candidate_texts[" ".join(latest_refs[m][1] for m in members)] = tag
                selections.append({"deliverable_id": tag, "claim_refs": refs})
            return {"selections": selections}
        if schema is SynthesisOutput:
            return {}  # Exercise release of every explicitly endorsed answer from the ledger.
        assert schema is ReviewerBatchOutput
        verdicts = []
        for block in re.split(r"\n(?=Claim clm_)", prompt):
            match = re.match(
                r"Claim (clm_\w+) \(version=(\d+),[^\n]+\n  Statement: ([^\n]+)", block
            )
            if not match:
                continue
            cid, version, statement = match.groups()
            covers = []
            if statement in candidate_texts:
                covers = [candidate_texts[statement]]
                assert covers[0] in block and "supported" in block
                why = "The exact reviewed members together supply the requested case and workings."
            else:
                old = by_text[statement]
                key = old.claim_id.removeprefix("clm_")
                if key in CENTS:
                    expected = Decimal(CENTS[key]) / 100
                    assert f"computed={expected}," in block
                    assert "inputs_grounded=True" in block
                    # Match the inspected formula and each primary source selection.
                    rec = tools[old.tool_execution_ids[0]]
                    assert f"expr=`{rec.expression}`" in block
                    for d in rec.provenance_details.values():
                        if d["kind"] == "prompt_span":
                            assert d["ref"] in block and d["source_quote"] in block
                    why = "Cited terms and tier, rebate or comparison agree with the audited case."
                    if key == "802d16361a78":
                        assert old.version == 2
                        covers = ["deliv_1"]
                else:
                    assert key == "6ba9ad0b5c86" and "Recommend Birch" in statement
                    assert birch_total(BASE_B) < alder_total(BASE_A)
                    why = "The two reviewed base totals imply the stated saving and choice."
                    covers = ["deliv_4"]
                latest_refs[key] = (f"{cid}@v{version}", statement)
            verdicts.append(
                dict(
                    claim_id=cid,
                    claim_version=int(version),
                    decision="approve",
                    rationale=why,
                    covered_deliverable_refs=covers,
                )
            )
        assert verdicts
        return {"verdicts": verdicts}

    def checked_script(prompt, schema):
        try:
            return script(prompt, schema)
        except Exception as exc:
            script_errors.append((schema.__name__, repr(exc)))
            raise

    result, trace = await run_case(tmp_path, archive.prompt, checked_script)
    assert not script_errors, str(script_errors)[:2400]
    assert result.verification and result.verification.answer_complete, result.answer
    assert (
        result.verification.content_released_safely
        and result.verification.requirement_coverage == 1
    )
    assert len(latest_refs) == len(original) == 19
    narrative = result.answer.split("**Calculations and Sources**")[0]
    assert all(narrative.count(latest_refs[key][1]) == 1
               for members in GROUPS.values() for key in members)
    assert "Recommend Birch" in result.answer and "$205 lower" in result.answer
    latest = {c.claim_id: c for c in trace.claims}
    assert all(c.status.value == "supported" for c in latest.values())
    recomputed = {
        tid for event in trace.events for tid in event.payload.get("recomputed_tool_ids", [])
    }
    assert all(tid in recomputed for c in latest.values() for tid in c.tool_execution_ids)
    # Every original edge remains pinned to its fresh source in the recovered graph.
    for old in original.values():
        ref = latest_refs[old.claim_id[4:]][0]
        new = latest[ref.split("@")[0]]
        assert {f"{p.claim_id}@v{p.claim_version}" for p in new.premise_refs} == {
            latest_refs[dep[4:]][0] for dep in old.depends_on_claim_ids
        }
    ops = [e.payload["operation"] for e in trace.events if e.event_type == "budget_reserved"]
    assert ops.count("revision") == 1 and len(ops) <= 16
    record_property(
        "archive_replay",
        json.dumps(
            {
                "operations": ops,
                "recomputed": len(recomputed),
                "executions": len(trace.tool_executions),
                "coverage": result.verification.requirement_coverage,
            }
        ),
    )
