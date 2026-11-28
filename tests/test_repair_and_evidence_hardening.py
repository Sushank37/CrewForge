"""Offline regression tests for minor-severity repair routing, atomic claims, and evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.core.budget import BudgetManager
from crewforge.core.evidence_verifier import verify_evidence_proposal
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.schemas import ProposedEvidence, RunStatus
from crewforge.storage.database import Database
from tests.fake_provider import ChecklistFixtureProvider as FakeProvider


def test_evidence_verification_document_scoping_and_sentence_negation() -> None:
    docs_by_ref = {
        "doc_1": "No expedited waivers apply here. Standard cap is 12000 USD.",
        "doc_2": "Compute cost is 4500 USD and storage cost is 2300 USD.",
    }

    # 1. Valid excerpt following unrelated negative sentence in doc_1 MUST pass
    ok_res = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_retrieved",
            source_ref="doc_1",
            excerpt="Standard cap is 12000 USD.",
        ),
        original_prompt="Check budget.",
        retrieved_sources_by_ref=docs_by_ref,
    )
    assert ok_res.verified_in_input is True

    # 2. Cross-document misattribution (doc_2 quote attributed to doc_1) MUST fail
    misattrib = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_retrieved",
            source_ref="doc_1",
            excerpt="Compute cost is 4500 USD",
        ),
        original_prompt="Check budget.",
        retrieved_sources_by_ref=docs_by_ref,
    )
    assert misattrib.verified_in_input is False

    # 3. Noncontiguous quotation MUST fail
    noncontig = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_retrieved",
            source_ref="doc_2",
            excerpt="Compute cost is 2300 USD.",
        ),
        original_prompt="Check budget.",
        retrieved_sources_by_ref=docs_by_ref,
    )
    assert noncontig.verified_in_input is False

    # 4. Omitted same-sentence negation MUST still fail
    omitted_neg = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_retrieved",
            source_ref="doc_1",
            excerpt="expedited waivers apply here.",
        ),
        original_prompt="Check budget.",
        retrieved_sources_by_ref=docs_by_ref,
    )
    assert omitted_neg.verified_in_input is False

    # 5. Fabricated / altered number MUST fail
    altered = verify_evidence_proposal(
        proposal=ProposedEvidence(
            source_type="external_retrieved",
            source_ref="doc_1",
            excerpt="Standard cap is 15000 USD.",
        ),
        original_prompt="Check budget.",
        retrieved_sources_by_ref=docs_by_ref,
    )
    assert altered.verified_in_input is False


async def test_minor_needs_revision_triggers_repair_downstream_revalidation_and_multi_tool(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "minor_repair.db")
    await db.initialize()

    review_pass = 0

    def _handler(prompt: str, response_model: Any) -> Any:
        nonlocal review_pass
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Compute subtotal and remaining balance across dependent tasks",
                "user_requirements": [
                    "Compute subtotal expenditure",
                    "Compute remaining balance",
                ],
                "acceptance_criteria": ["Both calculations verified separately"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Cost Analyst",
                        "specialty": "Costs",
                        "instructions": "Compute cost subtotal.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Balance Analyst",
                        "specialty": "Balance",
                        "instructions": "Compute remaining balance from subtotal.",
                        "capabilities": ["reasoning"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute subtotal",
                        "description": "Sum compute and storage costs.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Subtotal 6800 USD"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Compute remaining balance",
                        "description": "Subtract subtotal from 12000 USD cap.",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Remaining 5200 USD"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if schema_name in {"WorkerOutput", "SpecialistOutput"}:
            if "Compute cost subtotal" in prompt or "Sum compute and storage" in prompt:
                return {
                    "findings": "Initial subtotal draft.",
                    "evidence": [
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": "Compute 4500 USD and Storage 2300 USD.",
                        }
                    ],
                    "claims": [
                        {
                            "statement": "Subtotal expenditure is 6800 USD.",
                            "category": "computation",
                            "evidence_indices": [0],
                            "requirement_refs": ["req_1"],
                            "computation": {
                                "expression": "c + s",
                                "variable_bindings": [
                                    {"name": "c", "value": "4500", "provenance_ref": "doc_2"},
                                    {"name": "s", "value": "2300", "provenance_ref": "doc_2"},
                                ],
                                "proposed_result": "6800",
                                "units": "USD",
                            },
                        }
                    ],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                    "outgoing_messages": [],
                }
            m_up = re.search(r"(clm_[A-Za-z0-9_]+)", prompt)
            up_cid = m_up.group(1) if m_up else "clm_missing"
            return {
                "findings": "Remaining balance is 5200 USD.",
                "evidence": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_1",
                        "excerpt": "Standard cap is 12000 USD.",
                    }
                ],
                "claims": [
                    {
                        "statement": "Remaining balance is 5200 USD.",
                        "category": "computation",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_2"],
                        "premise_claim_ids": [up_cid],
                        "computation": {
                            "expression": "cap - sub",
                            "variable_bindings": [
                                {"name": "cap", "value": "12000", "provenance_ref": "doc_1"},
                                {"name": "sub", "value": "6800", "provenance_ref": up_cid},
                            ],
                            "proposed_result": "5200",
                            "units": "USD",
                        },
                    }
                ],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
            }
        if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
            review_pass += 1
            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
            verdicts = []
            for idx, cid in enumerate(cids):
                # On first pass, reject the first material claim with MINOR needs_revision!
                if review_pass == 1 and idx == 0:
                    verdicts.append(
                        {
                            "claim_id": cid,
                            "decision": "needs_revision",
                            "rationale": "Clarify exact line items in statement.",
                            "evidence_refs": [],
                            "required_correction": "State Compute + Storage explicitly.",
                            "blocking_severity": "minor",
                            "blocking_issues": ["Needs line-item specificity"],
                        }
                    )
                else:
                    verdicts.append(
                        {
                            "claim_id": cid,
                            "decision": "approve",
                            "rationale": "Verified deterministically.",
                            "evidence_refs": [],
                            "required_correction": None,
                            "blocking_severity": "none",
                            "blocking_issues": [],
                        }
                    )
            return {"verdicts": verdicts}
        if schema_name == "RevisionOutput":
            m = re.search(r"claim_id=(clm_[A-Za-z0-9_]+), expected_version=(\d+)", prompt)
            cid = m.group(1) if m else "clm_missing"
            ver = int(m.group(2)) if m else 1
            return {
                "updated_findings": "Subtotal expenditure (Compute + Storage) is 6800 USD.",
                "evidence_proposals": [],
                "revisions": [
                    {
                        "target_claim_id": cid,
                        "expected_version": ver,
                        "new_statement": "Subtotal expenditure (Compute + Storage) is 6800 USD.",
                        "category": "computation",
                        "evidence_indices": [],
                        "premise_claim_ids": [],
                        "computation": {
                            "expression": "c + s",
                            "variable_bindings": [
                                {"name": "c", "value": "4500", "provenance_ref": "doc_2"},
                                {"name": "s", "value": "2300", "provenance_ref": "doc_2"},
                            ],
                            "proposed_result": "6800",
                            "units": "USD",
                        },
                        "assumptions": [],
                        "revision_notes": "Added line-item specificity.",
                        "is_unresolved": False,
                    }
                ],
            }
        if schema_name == "SynthesisOutput":
            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
            refs = [{"claim_id": cid, "claim_version": 2} for cid in cids]
            return {
                "direct_summary": "Subtotal is 6800 USD and remaining balance is 5200 USD.",
                "accepted_segments": [
                    {
                        "segment_type": "accepted_claim",
                        "text": "Subtotal expenditure (Compute + Storage) is 6800 USD.",
                        "claim_refs": refs[:1],
                        "requirement_ref": "req_1",
                    },
                    {
                        "segment_type": "accepted_claim",
                        "text": "Remaining balance is 5200 USD.",
                        "claim_refs": refs[1:2] or refs[:1],
                        "requirement_ref": "req_2",
                    },
                ],
                "conditional_segments": [],
                "recommendations": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": cids,
                "assumptions": [],
                "missing_information": [],
            }
        raise AssertionError(f"Unexpected schema: {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-repair-model",
        openai_api_key="sk-offline",
        db_path=tmp_path / "minor_repair.db",
        budget_authority_path=tmp_path / "minor_repair_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        max_revision_rounds=1,
    )
    runner = NetworkRunner(
        provider=FakeProvider(handler=_handler, provider_name="fake", model_name="fake-repair"),
        database=db,
        budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    docs = [
        ReferenceDocument(
            doc_id="doc_1",
            title="Cap",
            content="No expedited waivers apply here. Standard cap is 12000 USD.",
            source_ref="doc_1",
        ),
        ReferenceDocument(
            doc_id="doc_2",
            title="Items",
            content="Compute 4500 USD and Storage 2300 USD.",
            source_ref="doc_2",
        ),
    ]
    res = await runner.run(
        "Compute subtotal expenditure and remaining balance.",
        reference_documents=docs,
    )
    trace = await db.get_run_trace(res.run_id)
    await db.close()

    assert res.status == RunStatus.COMPLETED, (
        f"status={res.status} claims={[(c.claim_id, c.version, c.status) for c in trace.claims]} "
        f"events={[(e.event_type, e.message) for e in trace.events]}"
    )
    assert res.verification is not None
    assert res.verification.honesty_gate_passed is True
    assert "6800" in res.answer
    assert "5200" in res.answer
    # Both claims repaired/revalidated to v2 and backed by separate tool records
    assert len(trace.tool_executions) >= 2
    assert any(c.version == 2 for c in trace.claims)


async def test_invalid_claim_remains_blocked_when_repair_produces_bad_provenance(
    tmp_path: Path,
) -> None:
    """Negative test: invalid claim rejected with minor needs_revision stays blocked if bad."""
    db = Database(tmp_path / "neg_repair.db")
    await db.initialize()

    def _bad_handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Compute value",
                "user_requirements": ["Compute value"],
                "acceptance_criteria": ["Grounded value"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "specialty": "Math",
                        "instructions": "Compute.",
                        "capabilities": ["reasoning"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute",
                        "description": "Compute.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Value"],
                    }
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if schema_name in {"WorkerOutput", "SpecialistOutput", "ReconciliationOutput"}:
            return {
                "findings": "Draft 999.",
                "evidence": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_1",
                        "excerpt": "Standard cap is 12000 USD.",
                    }
                ],
                "claims": [
                    {
                        "statement": "Fabricated value is 999 USD.",
                        "category": "computation",
                        "evidence_indices": [0],
                        "requirement_refs": ["req_1"],
                    }
                ],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
            }
        if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
            cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
            return {
                "verdicts": [
                    {
                        "claim_id": cid,
                        "decision": "reject",
                        "rationale": "Ungrounded fabricated number.",
                        "evidence_refs": [],
                        "required_correction": None,
                        "blocking_severity": "blocking",
                        "blocking_issues": ["Fabricated value"],
                    }
                    for cid in cids
                ]
            }
        if schema_name == "RevisionOutput":
            return {"evidence_proposals": [], "revisions": []}
        if schema_name == "SynthesisOutput":
            return {
                "direct_summary": "No verified claims available.",
                "accepted_segments": [],
                "conditional_segments": [],
                "recommendations": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": [],
                "assumptions": [],
                "missing_information": ["Claim rejected."],
            }
        raise AssertionError(f"Unexpected schema: {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-neg",
        openai_api_key="sk-offline",
        db_path=tmp_path / "neg_repair.db",
        budget_authority_path=tmp_path / "neg_repair_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
    )
    runner = NetworkRunner(
        provider=FakeProvider(handler=_bad_handler, provider_name="fake", model_name="fake-neg"),
        database=db,
        budget_manager=BudgetManager(db, settings),
        settings=settings,
    )
    res = await runner.run(
        "Compute value.",
        reference_documents=[
            ReferenceDocument(
                doc_id="doc_1",
                title="Cap",
                content="Standard cap is 12000 USD.",
                source_ref="doc_1",
            )
        ],
    )
    trace = await db.get_run_trace(res.run_id)
    await db.close()

    assert res.verification is not None
    assert res.verification.honesty_gate_passed is False
    assert trace.honesty_gate is not None
    assert trace.honesty_gate.release_mode == "blocked"
