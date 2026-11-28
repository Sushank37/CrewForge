"""Offline V2.3 regression tests for claim persistence, numeric provenance, and repair."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.core.budget import BudgetManager
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.schemas import Claim, ClaimStatus, Evidence, RunStatus, VerificationStatus
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog
from tests.fake_provider import ChecklistFixtureProvider as FakeProvider


def test_v23_numeric_provenance_context_and_negative_controls() -> None:
    """Verify multiline/comma/decimal numeric context matching and negative rejections."""
    doc1 = Evidence(
        run_id="run_v23_ctx",
        source_type="external_retrieved",
        source_ref="doc_1",
        excerpt=(
            "Vendor A annual licensing cost:\n$50,000.00\n"
            "Vendor B annual licensing cost: $60,000.00\n"
            "Baseline volume is 140,000 expected peak monthly transactions.\n"
            "Stress-test ceiling is 150,000 maximum transactions."
        ),
        verified_in_input=True,
        created_by="lead",
    )
    stale_v1 = Claim(
        claim_id="clm_stale_1",
        run_id="run_v23_ctx",
        statement="Legacy cap is 50000 USD.",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.PROVENANCE_VALIDATED,
        version=1,
        author_agent_id="agt_1",
    )
    fresh_v2 = Claim(
        claim_id="clm_stale_1",
        run_id="run_v23_ctx",
        statement="Updated cap is 60000 USD.",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.PROVENANCE_VALIDATED,
        version=2,
        author_agent_id="agt_1",
    )
    catalog = build_provenance_catalog(
        original_prompt="Evaluate vendor cost and transaction capacity.",
        verified_evidence=[doc1],
        all_claims=[stale_v1, fresh_v2],
        run_id="run_v23_ctx",
    )

    # Positive: cost=50000 across newline + decimal punctuation
    ok_cost, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"cost": Decimal("50000")},
        input_provenance_refs={"cost": "doc_1"},
        catalog=catalog,
        declared_units="USD",
    )
    assert ok_cost is True

    # Positive: transactions=140000 across >20 chars phrase alongside stress_volume=150000
    ok_tx, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={
            "transactions": Decimal("140000"),
            "stress_volume": Decimal("150000"),
        },
        input_provenance_refs={"transactions": "doc_1", "stress_volume": "doc_1"},
        catalog=catalog,
    )
    assert ok_tx is True

    # Negative 1: wrong value
    bad_val, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"cost": Decimal("55000")},
        input_provenance_refs={"cost": "doc_1"},
        catalog=catalog,
    )
    assert bad_val is False

    # Negative 2: fabricated ref
    bad_ref, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"cost": Decimal("50000")},
        input_provenance_refs={"cost": "doc_99"},
        catalog=catalog,
    )
    assert bad_ref is False

    # Negative 3: stale claim version ref
    bad_stale, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"cap": Decimal("50000")},
        input_provenance_refs={"cap": "clm_stale_1@v1"},
        catalog=catalog,
    )
    assert bad_stale is False


async def test_v23_three_document_procurement_six_claims_per_specialist_and_honesty_gate(
    tmp_path: Path,
) -> None:
    """Reproduce the 3-document procurement case with 6 claims from a single specialist."""
    db = Database(tmp_path / "v23_procure.db")
    await db.initialize()

    docs = [
        ReferenceDocument(
            doc_id="doc_1",
            title="Budget Policy",
            content="Maximum approved annual software budget cap is $60,000 USD.",
            source_ref="doc_1",
        ),
        ReferenceDocument(
            doc_id="doc_2",
            title="Vendor Quotes",
            content=(
                "Vendor A base annual license cost:\n$50,000\n"
                "Vendor A with optional premium support annual cost: $58,000.\n"
                "Vendor B annual license cost: $65,000.\n"
                "Vendor C annual license cost: $48,000.\n"
                "Expected workload is 140,000 monthly transactions; "
                "peak stress-test maximum is 150,000 transactions."
            ),
            source_ref="doc_2",
        ),
        ReferenceDocument(
            doc_id="doc_3",
            title="Compliance and Capacity Matrix",
            content=(
                "SOC2 Type II compliance is mandatory. "
                "Vendor A is SOC2 Type II compliant and supports 150,000 maximum transactions. "
                "Vendor B is SOC2 Type II compliant. "
                "Vendor C is NOT SOC2 compliant."
            ),
            source_ref="doc_3",
        ),
    ]

    def _handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Evaluate vendor costs, headroom, compliance, and recommend a vendor.",
                "user_requirements": [
                    "Evaluate all vendor costs and budget headroom",
                    "Assess compliance and transaction capacity",
                    "Recommend the compliant vendor within budget",
                ],
                "deliverables": [
                    {
                        "deliverable_id": f"deliv_{idx}", "requirement_ref": f"req_{requirement}",
                        "task_ref": f"t{task}", "text": text, "kind": kind,
                    }
                    for idx, (requirement, task, text, kind) in enumerate([
                        (1, 1, "Report Vendor A quoted base cost", "explanation"),
                        (1, 1, "Calculate Vendor A base headroom", "calculation"),
                        (1, 1, "Report Vendor A quoted cost with support", "explanation"),
                        (1, 1, "Calculate Vendor A headroom with support", "calculation"),
                        (1, 1, "Report Vendor B quoted cost", "explanation"),
                        (1, 1, "Report Vendor C quoted cost", "explanation"),
                        (2, 2, "Report Vendor A compliance and capacity", "explanation"),
                        (2, 2, "Report Vendor C compliance", "explanation"),
                        (3, 2, "Recommend a compliant vendor", "recommendation"),
                    ], start=1)
                ],
                "acceptance_criteria": [
                    "All vendor costs and headroom calculations grounded and verified",
                    "Compliance and 150,000 maximum transactions verified",
                ],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Cost Analyst",
                        "specialty": "Vendor pricing",
                        "instructions": "Extract all vendor costs and compute budget headroom.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Compliance Analyst",
                        "specialty": "Compliance & capacity",
                        "instructions": "Assess SOC2 compliance, capacity, and recommend vendor.",
                        "capabilities": ["reasoning"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Analyze vendor costs and budget headroom",
                        "description": "Record all vendor costs and compute headroom.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["6 grounded vendor cost/headroom claims"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Assess compliance, capacity, and recommendation",
                        "description": "Evaluate compliance and recommend compliant vendor.",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Compliance assessment and recommendation"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }

        if schema_name == "SpecialistOutput":
            if "Record all vendor costs and compute headroom." in prompt:
                # 6 claims from a single specialist — previously truncated at [:4]!
                return {
                    "findings": (
                        "Vendor A costs $50,000 ($10,000 headroom) or $58,000 with support "
                        "($2,000 headroom). Vendor B costs $65,000. Vendor C costs $48,000."
                    ),
                    "evidence_proposals": [
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_1",
                            "excerpt": (
                                "Maximum approved annual software budget cap is $60,000 USD."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Vendor A base annual license cost:\n$50,000\n"
                                "Vendor A with optional premium support annual cost: $58,000."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Vendor B annual license cost: $65,000.\n"
                                "Vendor C annual license cost: $48,000."
                            ),
                        },
                    ],
                    "claims": [
                        {
                            "claim_ref": "c_a_base",
                            "statement": "Vendor A base annual license cost is $50,000.",
                            "category": "given_fact",
                            "evidence_indices": [1],
                            "requirement_refs": ["deliv_1"],
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_a_headroom_base",
                            "statement": "Vendor A base budget headroom is $10,000 USD.",
                            "category": "computation",
                            "evidence_indices": [0, 1],
                            "premise_claim_ids": ["c_a_base"],
                            "requirement_refs": ["deliv_2"],
                            "computation": {
                                "expression": "cap - cost",
                                "variables": {"cap": "60000", "cost": "50000"},
                                "input_provenance_refs": {"cap": "doc_1", "cost": "doc_2"},
                                "proposed_result": "10000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_a_support",
                            "statement": (
                                "Vendor A with optional premium support annual cost is $58,000."
                            ),
                            "category": "given_fact",
                            "evidence_indices": [1],
                            "requirement_refs": ["deliv_3"],
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_a_headroom_sup",
                            "statement": "Vendor A with support budget headroom is $2,000 USD.",
                            "category": "computation",
                            "evidence_indices": [0, 1],
                            "premise_claim_ids": ["c_a_support"],
                            "requirement_refs": ["deliv_4"],
                            "computation": {
                                "expression": "cap - cost",
                                "variables": {"cap": "60000", "cost": "58000"},
                                "input_provenance_refs": {"cap": "doc_1", "cost": "doc_2"},
                                "proposed_result": "2000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_b_cost",
                            "statement": "Vendor B annual license cost is $65,000.",
                            "category": "given_fact",
                            "evidence_indices": [2],
                            "requirement_refs": ["deliv_5"],
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_c_cost",
                            "statement": "Vendor C annual license cost is $48,000.",
                            "category": "given_fact",
                            "evidence_indices": [2],
                            "requirement_refs": ["deliv_6"],
                            "confidence": 0.99,
                        },
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }

            # Compliance & recommendation task
            cids = re.findall(r"clm_[a-z0-9]+", prompt)
            return {
                "findings": (
                    "Vendor A is SOC2 compliant and supports 150,000 maximum transactions "
                    "(above 140,000 expected); Vendor C is non-compliant. Recommend Vendor A."
                ),
                "evidence_proposals": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_2",
                        "excerpt": (
                            "Expected workload is 140,000 monthly transactions; "
                            "peak stress-test maximum is 150,000 transactions."
                        ),
                    },
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_3",
                        "excerpt": (
                            "Vendor A is SOC2 Type II compliant and supports 150,000 "
                            "maximum transactions. Vendor B is SOC2 Type II compliant. "
                            "Vendor C is NOT SOC2 compliant."
                        ),
                    },
                ],
                "claims": [
                    {
                        "claim_ref": "c_capacity",
                        "statement": (
                            "Vendor A is SOC2 Type II compliant and supports 150,000 maximum "
                            "transactions, providing 10,000 transaction headroom over 140,000."
                        ),
                        "category": "computation",
                        "evidence_indices": [0, 1],
                        "requirement_refs": ["deliv_7"],
                        "computation": {
                            "expression": "stress_volume - transactions",
                            "variables": {"stress_volume": "150000", "transactions": "140000"},
                            "input_provenance_refs": {
                                "stress_volume": "doc_2",
                                "transactions": "doc_2",
                            },
                            "proposed_result": "10000",
                            "units": "transactions",
                        },
                        "confidence": 0.99,
                    },
                    {
                        "claim_ref": "c_noncomp",
                        "statement": "Vendor C is NOT SOC2 compliant.",
                        "category": "given_fact",
                        "evidence_indices": [1],
                        "requirement_refs": ["deliv_8"],
                        "confidence": 0.99,
                    },
                    {
                        "claim_ref": "c_rec",
                        "statement": "Recommend Vendor A as the compliant option within budget.",
                        "category": "recommendation",
                        "evidence_indices": [1],
                        "premise_claim_ids": cids[:2] if cids else [],
                        "requirement_refs": ["deliv_9"],
                        "confidence": 0.98,
                    },
                ],
                "outgoing_messages": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
            }

        if schema_name == "ReviewerBatchOutput":
            aliases = re.findall(r"alias=(c\d+)", prompt)
            return {
                "verdicts": [
                    {
                        "claim_id": alias,
                        "decision": "approve",
                        "rationale": "Grounded in cited doc and arithmetic.",
                        "blocking_severity": "none",
                        "evidence_refs": [],
                    }
                    for alias in aliases
                ]
            }

        if schema_name == "SynthesisOutput":
            claims_found = re.findall(r"(clm_[a-z0-9]+) \(v(\d+)", prompt)
            accepted = [
                {
                    "segment_type": "accepted_claim",
                    "text": "",
                    "claim_refs": [{"claim_id": cid, "claim_version": int(ver)}],
                    "is_factual_assertion": True,
                }
                for cid, ver in claims_found
            ]
            return {
                "user_requirements": [
                    "Evaluate all vendor costs and budget headroom",
                    "Assess compliance and transaction capacity",
                    "Recommend the compliant vendor within budget",
                ],
                "accepted_segments": accepted,
                "conditional_conclusions": [],
                "recommendations": [],
                "assumptions": [],
                "unresolved_issues": [],
                "missing_information": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": [cid for cid, _ in claims_found],
            }

        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-v23-procure",
        openai_api_key="sk-offline",
        db_path=tmp_path / "v23_procure.db",
        budget_authority_path=tmp_path / "v23_procure_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        max_workers=3,
        max_tasks=4,
        max_model_calls_per_run=16,
        max_revision_rounds=2,
        enable_peer_messaging=False,
        separate_review_dimensions=False,
        enable_deterministic_synthesis_fast_path=True,
    )
    provider = FakeProvider(handler=_handler)
    budget = BudgetManager(db, settings)
    runner = NetworkRunner(database=db, provider=provider, budget_manager=budget, settings=settings)

    result = await runner.run(
        "Evaluate Vendors A, B, and C against budget and SOC2 compliance.",
        reference_documents=docs,
    )
    assert result.status == RunStatus.COMPLETED
    assert result.verification.honesty_gate_passed is True

    trace = await db.get_run_trace(result.run_id)
    assert trace is not None
    # Specialist 1 emitted 6 claims and Specialist 2 emitted 3 claims = 9 claims total.
    assert len(trace.claims) == 9
    material_supported = [
        c for c in trace.claims if c.is_material and c.status == ClaimStatus.SUPPORTED
    ]
    assert len(material_supported) == 9

    assert "$10,000" in result.answer
    assert "$2,000" in result.answer
    assert "$65,000" in result.answer
    assert "$48,000" in result.answer
    assert "150,000" in result.answer
    assert "Recommend Vendor A" in result.answer


async def test_v23_dependency_aware_repair_and_negative_downstream_rejection(
    tmp_path: Path,
) -> None:
    """Upstream claims are repaired before dependents; bad downstream stays blocked."""
    db = Database(tmp_path / "v23_dep_repair.db")
    await db.initialize()

    review_calls: list[list[str]] = []

    def _handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Compute base cost and dependent recommendation",
                "user_requirements": ["Compute base cost", "Provide downstream conclusion"],
                "acceptance_criteria": ["Upstream repaired before downstream"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Cost Analyst",
                        "specialty": "Costs",
                        "instructions": "Compute cost.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Decision Analyst",
                        "specialty": "Decisions",
                        "instructions": "Conclude from cost.",
                        "capabilities": ["reasoning"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute cost",
                        "description": "Compute cost.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Cost claim"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Conclude",
                        "description": "Conclude from cost.",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Downstream conclusion"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }

        if schema_name == "SpecialistOutput":
            if "Conclude from cost." not in prompt:
                return {
                    "findings": "Vendor A cost is $50,000.",
                    "evidence_proposals": [
                        {
                            "source_type": "user_prompt",
                            "source_ref": "up_1",
                            "excerpt": "Vendor A cost is $50,000 and budget is $60,000.",
                        }
                    ],
                    "claims": [
                        {
                            "claim_ref": "c_up",
                            "statement": "Vendor A headroom is $10,000 USD.",
                            "category": "computation",
                            "evidence_indices": [0],
                            "requirement_refs": ["req_1"],
                            "computation": {
                                "expression": "cap - cost",
                                "variables": {"cap": "60000", "cost": "50000"},
                                "input_provenance_refs": {"cap": "up_1", "cost": "up_1"},
                                "proposed_result": "10000",
                                "units": "USD",
                            },
                            "confidence": 0.95,
                        }
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }
            up_ids = re.findall(r"clm_[a-z0-9]+", prompt)
            return {
                "findings": "Vendor A is within budget.",
                "evidence_proposals": [],
                "claims": [
                    {
                        "claim_ref": "c_down",
                        "statement": "Vendor A is within budget with $99,999 surplus.",
                        "category": "logical_inference",
                        "evidence_indices": [],
                        "premise_claim_ids": up_ids[:1],
                        "requirement_refs": ["req_2"],
                        "confidence": 0.9,
                    }
                ],
                "outgoing_messages": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
            }

        if schema_name == "ReviewerBatchOutput":
            cids = re.findall(r"Claim (clm_[a-z0-9]+) \(version=(\d+), alias=(c\d+)", prompt)
            review_calls.append([f"{cid}@v{ver}" for cid, ver, _ in cids])
            verdicts = []
            for cid, ver, alias in cids:
                if int(ver) == 1 and "headroom is $10,000" in prompt and alias == "c1":
                    # Reject upstream v1 so it enters repair before downstream is revalidated
                    verdicts.append(
                        {
                            "claim_id": alias,
                            "decision": "needs_revision",
                            "rationale": "Clarify annual license basis.",
                            "required_correction": "State annual license headroom.",
                            "blocking_severity": "minor",
                            "evidence_refs": [],
                        }
                    )
                elif "99,999" in prompt and cid in prompt:
                    # Reject fabricated downstream number
                    verdicts.append(
                        {
                            "claim_id": alias,
                            "decision": "reject",
                            "rationale": "Fabricated surplus $99,999 contradicts $10,000.",
                            "required_correction": "Use verified $10,000 headroom.",
                            "blocking_severity": "blocking",
                            "evidence_refs": [],
                        }
                    )
                else:
                    verdicts.append(
                        {
                            "claim_id": alias,
                            "decision": "approve",
                            "rationale": "Verified.",
                            "blocking_severity": "none",
                            "evidence_refs": [],
                        }
                    )
            return {"verdicts": verdicts}

        if schema_name == "RevisionOutput":
            if "99,999" in prompt:
                return {
                    "updated_findings": "Unable to substantiate $99,999 surplus.",
                    "evidence_proposals": [],
                    "revisions": [],
                }
            m = re.search(r"claim_id=(clm_[A-Za-z0-9_]+), expected_version=(\d+)", prompt)
            cid = m.group(1) if m else "clm_missing"
            ver = int(m.group(2)) if m else 1
            return {
                "updated_findings": "Vendor A annual license headroom is $10,000 USD.",
                "evidence_proposals": [],
                "revisions": [
                    {
                        "target_claim_id": cid,
                        "expected_version": ver,
                        "new_statement": "Vendor A annual license headroom is $10,000 USD.",
                        "revision_notes": "Preserved valid calculation; clarified annual basis.",
                        "computation": None,
                    }
                ],
            }

        if schema_name == "SynthesisOutput":
            return {
                "user_requirements": ["Compute base cost", "Provide downstream conclusion"],
                "accepted_segments": [],
                "conditional_conclusions": [],
                "recommendations": [],
                "assumptions": [],
                "unresolved_issues": ["Downstream conclusion rejected due to fabricated surplus."],
                "missing_information": [],
                "new_factual_assertions": [],
                "referenced_claim_ids": [],
            }

        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-v23-dep",
        openai_api_key="sk-offline",
        db_path=tmp_path / "v23_dep_repair.db",
        budget_authority_path=tmp_path / "v23_dep_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        max_workers=2,
        max_tasks=3,
        max_model_calls_per_run=16,
        max_revision_rounds=2,
        enable_peer_messaging=False,
        separate_review_dimensions=False,
        enable_deterministic_synthesis_fast_path=False,
    )
    provider = FakeProvider(handler=_handler)
    budget = BudgetManager(db, settings)
    runner = NetworkRunner(database=db, provider=provider, budget_manager=budget, settings=settings)

    result = await runner.run("Vendor A cost is $50,000 and budget is $60,000.")
    trace = await db.get_run_trace(result.run_id)
    assert trace is not None

    # Upstream claim repaired to v2 and SUPPORTED, preserving valid tool_execution_ids!
    up_v2 = [c for c in trace.claims if c.version == 2 and "annual license headroom" in c.statement]
    assert len(up_v2) == 1
    assert up_v2[0].status == ClaimStatus.SUPPORTED
    assert len(up_v2[0].tool_execution_ids) == 1

    # Downstream fabricated claim stays REJECTED and is withheld from final answer
    assert "$99,999" not in result.answer
    rejected = [c for c in trace.claims if c.status == ClaimStatus.REJECTED]
    assert any("99,999" in c.statement for c in rejected)


def _load_exact_cloud_complex_002() -> dict[str, Any]:
    import json

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "cloud_complex_002.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_v23_exact_cloud_complex_002_provenance_and_swapped_controls() -> None:
    """Reproduce exact cost=50000 and transactions=140000 cases plus swapped-quantity controls."""
    data = _load_exact_cloud_complex_002()

    doc1 = Evidence(
        run_id="run_exact_002",
        source_type="external_retrieved",
        source_ref="doc_1",
        excerpt=data["reference_documents"][0]["content"],
        verified_in_input=True,
        created_by="lead",
    )
    clm_a100 = Claim(
        claim_id="clm_a100",
        run_id="run_exact_002",
        statement="Vendor A's first-year cost at 100,000 transactions is $50,000.",
        status=ClaimStatus.SUPPORTED,
        verification_status=VerificationStatus.PROVENANCE_VALIDATED,
        version=1,
        author_agent_id="agt_1",
    )
    evd_vendors = Evidence(
        run_id="run_exact_002",
        source_type="external_retrieved",
        source_ref="doc_2",
        excerpt=(
            "Vendor Alpha cost is $50,000, Vendor Beta cost is $56,000. "
            "Setup fee is $10,000 and license cost is $50,000."
        ),
        verified_in_input=True,
        created_by="lead",
    )
    cat = build_provenance_catalog(
        original_prompt=data["prompt"],
        verified_evidence=[doc1, evd_vendors],
        all_claims=[clm_a100],
        run_id="run_exact_002",
    )

    ok_cost, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"cost": Decimal("50000")},
        input_provenance_refs={"cost": "clm_a100"},
        catalog=cat,
        declared_units="USD",
    )
    assert ok_cost is True

    ok_tx, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"transactions": Decimal("140000")},
        input_provenance_refs={"transactions": "doc_1"},
        catalog=cat,
    )
    assert ok_tx is True

    # Unambiguous occurrences still require independent semantic review.
    swap_vendor, vendor = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"alpha": Decimal("56000")},
        input_provenance_refs={"alpha": "doc_2"},
        catalog=cat,
    )
    assert swap_vendor and vendor["alpha"]["binding_status"] == "pending_semantic_review"

    swap_fee, _ = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"setup_fee": Decimal("50000")},
        input_provenance_refs={"setup_fee": "doc_2"},
        catalog=cat,
    )
    assert swap_fee is False

    swap_forecast, forecast = DeterministicArithmeticTool.check_inputs_grounded(
        variables={"stress": Decimal("100000")},
        input_provenance_refs={"stress": "doc_1"},
        catalog=cat,
    )
    assert swap_forecast and forecast["stress"]["binding_status"] == "pending_semantic_review"
    assert "base case" in forecast["stress"]["source_context"]


async def test_v23_exact_cloud_complex_002_end_to_end(tmp_path: Path) -> None:
    """Offline end-to-end regression using exact prompt and 3 docs from cloud_complex_test_002."""
    data = _load_exact_cloud_complex_002()
    docs = [
        ReferenceDocument(
            doc_id=f"doc_{idx}",
            title=item["title"],
            content=item["content"],
            source_ref=f"doc_{idx}",
        )
        for idx, item in enumerate(data["reference_documents"], start=1)
    ]

    db = Database(tmp_path / "v23_exact_002.db")
    await db.initialize()

    def _handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "Evaluate all vendor costs, compliance, recommendation, and capacity.",
                "user_requirements": [
                    "Calculate first-year costs for all three vendors at both forecasts",
                    "Evaluate every mandatory compliance requirement and disqualified vendors",
                    "Recommend the best eligible vendor",
                    "Calculate remaining budget under both forecasts and max transactions",
                ],
                "deliverables": [
                    {
                        "deliverable_id": f"deliv_{idx}", "requirement_ref": f"req_{requirement}",
                        "task_ref": f"t{task}", "text": text, "kind": kind,
                    }
                    for idx, (requirement, task, text, kind) in enumerate([
                        (1, 1, "Calculate Vendor A base forecast cost", "calculation"),
                        (1, 1, "Calculate Vendor A stress forecast cost", "calculation"),
                        (1, 1, "Calculate Vendor B base forecast cost", "calculation"),
                        (1, 1, "Calculate Vendor B stress forecast cost", "calculation"),
                        (1, 1, "Calculate Vendor C base forecast cost", "calculation"),
                        (1, 1, "Calculate Vendor C stress forecast cost", "calculation"),
                        (2, 2, "Explain Vendor A compliance", "explanation"),
                        (2, 2, "Explain why Vendors B and C are disqualified", "explanation"),
                        (3, 2, "Recommend a vendor", "recommendation"),
                        (4, 3, "Calculate base forecast headroom", "calculation"),
                        (4, 3, "Calculate stress forecast headroom", "calculation"),
                        (4, 3, "Calculate maximum affordable transactions", "calculation"),
                    ], start=1)
                ],
                "acceptance_criteria": [
                    "All 6 forecast costs, compliance, balances, and max transactions verified",
                ],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Vendor Cost Analyst",
                        "specialty": "Cost modeling",
                        "instructions": "Calculate all six vendor forecast costs.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Compliance Analyst",
                        "specialty": "Compliance",
                        "instructions": "Evaluate mandatory compliance and recommend vendor.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a3",
                        "role": "Capacity Analyst",
                        "specialty": "Budget & capacity",
                        "instructions": "Compute remaining budgets and maximum transactions.",
                        "capabilities": ["reasoning"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Calculate all six vendor forecast costs",
                        "description": "Task1: calculate all 6 first-year forecast costs.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["6 grounded vendor cost claims"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Evaluate mandatory compliance and recommend",
                        "description": "Task2: evaluate compliance and recommend best eligible.",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Compliance and recommendation claims"],
                    },
                    {
                        "task_ref": "t3",
                        "title": "Calculate remaining budgets and max transactions",
                        "description": "Task3: compute remaining budgets and max transactions.",
                        "assigned_agent_ref": "a3",
                        "depends_on": ["t1", "t2"],
                        "expected_deliverables": ["Remaining budgets and 150,000 max tx"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }

        if schema_name == "SpecialistOutput":
            if "Task1: calculate all 6" in prompt:
                return {
                    "findings": (
                        "A: $50,000 / $58,000; B: $56,000 / $60,800; C: $45,000 / $52,200."
                    ),
                    "evidence_proposals": [
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_1",
                            "excerpt": (
                                "Forecast: 100,000 transactions in the base\n"
                                "            case and 140,000 in the stress case."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Vendor A: $30,000 fixed annual fee plus\n"
                                "            $0.20 per transaction."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Vendor B: $36,000 fixed annual fee plus\n"
                                "            $0.12 per transaction."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Vendor C: $27,000 fixed annual fee plus\n"
                                "            $0.18 per transaction."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_3",
                            "excerpt": (
                                "Vendor B: Mandatory first-year onboarding\n"
                                "            fee of $8,000 in addition to all quoted fees."
                            ),
                        },
                    ],
                    "claims": [
                        {
                            "claim_ref": "a_100",
                            "statement": (
                                "Vendor A's first-year cost at 100,000 transactions is $50,000."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 1],
                            "requirement_refs": ["deliv_1"],
                            "computation": {
                                "expression": "fixed + rate * transactions",
                                "variables": {
                                    "fixed": "30000",
                                    "rate": "0.20",
                                    "transactions": "100000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "50000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "a_140",
                            "statement": (
                                "Vendor A's first-year cost at 140,000 transactions is $58,000."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 1],
                            "requirement_refs": ["deliv_2"],
                            "computation": {
                                "expression": "fixed + rate * transactions",
                                "variables": {
                                    "fixed": "30000",
                                    "rate": "0.20",
                                    "transactions": "140000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "58000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "b_100",
                            "statement": (
                                "Vendor B's first-year cost at 100,000 transactions is $56,000."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 2, 4],
                            "requirement_refs": ["deliv_3"],
                            "computation": {
                                "expression": "fixed + onboarding + rate * transactions",
                                "variables": {
                                    "fixed": "36000",
                                    "onboarding": "8000",
                                    "rate": "0.12",
                                    "transactions": "100000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "onboarding": "doc_3",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "56000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "b_140",
                            "statement": (
                                "Vendor B's first-year cost at 140,000 transactions is $60,800."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 2, 4],
                            "requirement_refs": ["deliv_4"],
                            "computation": {
                                "expression": "fixed + onboarding + rate * transactions",
                                "variables": {
                                    "fixed": "36000",
                                    "onboarding": "8000",
                                    "rate": "0.12",
                                    "transactions": "140000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "onboarding": "doc_3",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "60800",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_100",
                            "statement": (
                                "Vendor C's first-year cost at 100,000 transactions is $45,000."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 3],
                            "requirement_refs": ["deliv_5"],
                            "computation": {
                                "expression": "fixed + rate * transactions",
                                "variables": {
                                    "fixed": "27000",
                                    "rate": "0.18",
                                    "transactions": "100000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "45000",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "c_140",
                            "statement": (
                                "Vendor C's first-year cost at 140,000 transactions is $52,200."
                            ),
                            "category": "computation",
                            "evidence_indices": [0, 3],
                            "requirement_refs": ["deliv_6"],
                            "computation": {
                                "expression": "fixed + rate * transactions",
                                "variables": {
                                    "fixed": "27000",
                                    "rate": "0.18",
                                    "transactions": "140000",
                                },
                                "input_provenance_refs": {
                                    "fixed": "doc_2",
                                    "rate": "doc_2",
                                    "transactions": "doc_1",
                                },
                                "proposed_result": "52200",
                                "units": "USD",
                            },
                            "confidence": 0.99,
                        },
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }

            if "Task2: evaluate compliance" in prompt:
                cids = re.findall(r"clm_[a-z0-9]+", prompt)
                return {
                    "findings": (
                        "Vendor A meets all mandatory requirements. Vendor B is disqualified "
                        "($60,800 exceeds $60,000 stress budget). Vendor C is disqualified "
                        "(EU data residency delayed to week 10). Recommend Vendor A."
                    ),
                    "evidence_proposals": [
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_1",
                            "excerpt": (
                                "Go-live must occur within 8 weeks.\n"
                                "            EU data residency and MFA must be available\n"
                                "            at go-live."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_2",
                            "excerpt": (
                                "Go-live in 6 weeks.\n"
                                "            EU residency and MFA available at go-live."
                            ),
                        },
                        {
                            "source_type": "external_retrieved",
                            "source_ref": "doc_3",
                            "excerpt": (
                                "Vendor C: Revised go-live in 7 weeks.\n"
                                "            However, EU data residency will only become\n"
                                "            available in week 10."
                            ),
                        },
                    ],
                    "claims": [
                        {
                            "claim_ref": "comp_a",
                            "statement": (
                                "Vendor A satisfies all mandatory requirements: budget <= $60,000 "
                                "in both forecasts, 6-week go-live, EU residency and MFA at "
                                "go-live (doc_1, doc_2, doc_3)."
                            ),
                            "category": "given_fact",
                            "evidence_indices": [0, 1],
                            "requirement_refs": ["deliv_7"],
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "comp_bc",
                            "statement": (
                                "Vendor B is disqualified because its $60,800 stress-case cost "
                                "exceeds the $60,000 cap, and Vendor C is disqualified because "
                                "EU data residency is delayed until week 10 (doc_1, doc_3)."
                            ),
                            "category": "given_fact",
                            "evidence_indices": [0, 2],
                            "requirement_refs": ["deliv_8"],
                            "confidence": 0.99,
                        },
                        {
                            "claim_ref": "rec_a",
                            "statement": "Recommend Vendor A as the lowest-cost eligible vendor.",
                            "category": "recommendation",
                            "evidence_indices": [0, 1],
                            "premise_claim_ids": cids[:2] if cids else [],
                            "requirement_refs": ["deliv_9"],
                            "confidence": 0.99,
                        },
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }

            # Task 3: remaining budgets (using cost=50000 against clm_a100!) & max 150,000 tx
            a100_match = re.search(
                r"(clm_[a-z0-9]+) \(v\d+, computation\): Vendor A's first-year cost at 100,000",
                prompt,
            )
            a140_match = re.search(
                r"(clm_[a-z0-9]+) \(v\d+, computation\): Vendor A's first-year cost at 140,000",
                prompt,
            )
            a100_cid = a100_match.group(1) if a100_match else "doc_2"
            a140_cid = a140_match.group(1) if a140_match else "doc_2"
            return {
                "findings": (
                    "Vendor A remaining budget: $10,000 (base) and $2,000 (stress); "
                    "max 150,000 transactions."
                ),
                "evidence_proposals": [
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_1",
                        "excerpt": "First-year spending must not exceed $60,000,",
                    },
                    {
                        "source_type": "external_retrieved",
                        "source_ref": "doc_2",
                        "excerpt": (
                            "Vendor A: $30,000 fixed annual fee plus\n"
                            "            $0.20 per transaction."
                        ),
                    },
                ],
                "claims": [
                    {
                        "claim_ref": "rem_base",
                        "statement": (
                            "Vendor A has $10,000 remaining budget at 100,000 transactions "
                            "(doc_1, doc_2)."
                        ),
                        "category": "computation",
                        "evidence_indices": [0],
                        "premise_claim_ids": [a100_cid] if a100_cid.startswith("clm_") else [],
                        "requirement_refs": ["deliv_10"],
                        "computation": {
                            "expression": "cap - cost",
                            "variables": {"cap": "60000", "cost": "50000"},
                            "input_provenance_refs": {"cap": "doc_1", "cost": a100_cid},
                            "proposed_result": "10000",
                            "units": "USD",
                        },
                        "confidence": 0.99,
                    },
                    {
                        "claim_ref": "rem_stress",
                        "statement": (
                            "Vendor A has $2,000 remaining budget at 140,000 transactions "
                            "(doc_1, doc_2)."
                        ),
                        "category": "computation",
                        "evidence_indices": [0],
                        "premise_claim_ids": [a140_cid] if a140_cid.startswith("clm_") else [],
                        "requirement_refs": ["deliv_11"],
                        "computation": {
                            "expression": "cap - cost",
                            "variables": {"cap": "60000", "cost": "58000"},
                            "input_provenance_refs": {"cap": "doc_1", "cost": a140_cid},
                            "proposed_result": "2000",
                            "units": "USD",
                        },
                        "confidence": 0.99,
                    },
                    {
                        "claim_ref": "max_tx",
                        "statement": (
                            "Vendor A supports a maximum of 150,000 transactions within the "
                            "$60,000 budget (doc_1, doc_2)."
                        ),
                        "category": "computation",
                        "evidence_indices": [0, 1],
                        "requirement_refs": ["deliv_12"],
                        "computation": {
                            "expression": "(cap - fixed) / rate",
                            "variables": {"cap": "60000", "fixed": "30000", "rate": "0.20"},
                            "input_provenance_refs": {
                                "cap": "doc_1",
                                "fixed": "doc_2",
                                "rate": "doc_2",
                            },
                            "proposed_result": "150000",
                            "units": "transactions",
                        },
                        "confidence": 0.99,
                    },
                ],
                "outgoing_messages": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
            }

        if schema_name == "ReviewerBatchOutput":
            aliases = re.findall(r"alias=(c\d+)", prompt)
            return {
                "verdicts": [
                    {
                        "claim_id": alias,
                        "decision": "approve",
                        "rationale": "Grounded in cited docs and arithmetic.",
                        "blocking_severity": "none",
                        "evidence_refs": [],
                    }
                    for alias in aliases
                ]
            }

        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-exact-002",
        openai_api_key="sk-offline",
        db_path=tmp_path / "v23_exact_002.db",
        budget_authority_path=tmp_path / "v23_exact_002_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        max_workers=3,
        max_tasks=4,
        max_model_calls_per_run=16,
        max_revision_rounds=2,
        enable_peer_messaging=False,
        separate_review_dimensions=False,
        enable_deterministic_synthesis_fast_path=True,
    )
    provider = FakeProvider(handler=_handler)
    budget = BudgetManager(db, settings)
    runner = NetworkRunner(database=db, provider=provider, budget_manager=budget, settings=settings)

    result = await runner.run(data["prompt"], reference_documents=docs)
    assert result.status == RunStatus.COMPLETED
    assert result.verification.honesty_gate_passed is True
    assert result.verification.execution_completed is True
    assert result.verification.content_released_safely is True
    assert result.verification.material_claim_count == 12
    assert result.verification.supported_claim_count == 12

    for expected in (
        "$50,000",
        "$58,000",
        "$56,000",
        "$60,800",
        "$45,000",
        "$52,200",
        "$10,000",
        "$2,000",
        "150,000",
        "Recommend Vendor A",
        "6-week go-live",
        "EU residency and MFA at go-live",
        "Vendor B is disqualified",
        "exceeds the $60,000 cap",
        "Vendor C is disqualified",
        "EU data residency is delayed until week 10",
    ):
        assert expected in result.answer

    trace = await db.get_run_trace(result.run_id)
    assert trace is not None
    verified_refs = {e.source_ref for e in trace.evidence if e.verified_in_input}
    assert {"doc_1", "doc_2", "doc_3"}.issubset(verified_refs)
    for claim in trace.claims:
        if claim.status == ClaimStatus.SUPPORTED and claim.is_material:
            assert claim.evidence_ids or claim.tool_execution_ids


async def test_v23_multilevel_dependency_repair_chain(tmp_path: Path) -> None:
    """Verify 3-level L1 -> L2 -> L3 repair/revalidation order and premise version refresh."""
    db = Database(tmp_path / "v23_chain.db")
    await db.initialize()

    review_sequence: list[list[str]] = []

    def _handler(prompt: str, response_model: Any) -> Any:
        schema_name = response_model.__name__ if response_model is not None else ""
        if schema_name in {"ExecutionPlan", "PlannerOutput"}:
            return {
                "goal": "3-level dependency chain",
                "user_requirements": ["L1 cost", "L2 headroom", "L3 recommendation"],
                "acceptance_criteria": ["L1 repaired before L2 and L3 revalidated"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst 1",
                        "specialty": "L1",
                        "instructions": "Emit L1.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Analyst 2",
                        "specialty": "L2",
                        "instructions": "Emit L2.",
                        "capabilities": ["reasoning"],
                    },
                    {
                        "agent_ref": "a3",
                        "role": "Analyst 3",
                        "specialty": "L3",
                        "instructions": "Emit L3.",
                        "capabilities": ["reasoning"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Level 1",
                        "description": "Stage1",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["L1"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Level 2",
                        "description": "Stage2",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["L2"],
                    },
                    {
                        "task_ref": "t3",
                        "title": "Level 3",
                        "description": "Stage3",
                        "assigned_agent_ref": "a3",
                        "depends_on": ["t2"],
                        "expected_deliverables": ["L3"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }

        if schema_name == "SpecialistOutput":
            if "Stage1" in prompt:
                return {
                    "findings": "L1 cost is $50,000.",
                    "evidence_proposals": [
                        {
                            "source_type": "user_prompt",
                            "source_ref": "up_1",
                            "excerpt": "Budget is $60,000 and cost is $50,000.",
                        }
                    ],
                    "claims": [
                        {
                            "claim_ref": "l1",
                            "statement": "L1 cost is $50,000.",
                            "category": "given_fact",
                            "evidence_indices": [0],
                            "requirement_refs": ["req_1"],
                            "confidence": 0.95,
                        }
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }
            cids = re.findall(r"clm_[a-z0-9]+", prompt)
            if "Stage2" in prompt:
                return {
                    "findings": "L2 headroom is $10,000.",
                    "evidence_proposals": [],
                    "claims": [
                        {
                            "claim_ref": "l2",
                            "statement": "L2 headroom is $10,000 USD.",
                            "category": "logical_inference",
                            "evidence_indices": [],
                            "premise_claim_ids": cids[:1],
                            "requirement_refs": ["req_2"],
                            "confidence": 0.95,
                        }
                    ],
                    "outgoing_messages": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                }
            return {
                "findings": "L3 recommend Vendor A.",
                "evidence_proposals": [],
                "claims": [
                    {
                        "claim_ref": "l3",
                        "statement": "L3 recommend Vendor A.",
                        "category": "recommendation",
                        "evidence_indices": [],
                        "premise_claim_ids": cids[-1:],
                        "requirement_refs": ["req_3"],
                        "confidence": 0.95,
                    }
                ],
                "outgoing_messages": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
            }

        if schema_name == "ReviewerBatchOutput":
            items = re.findall(r"Claim (clm_[a-z0-9]+) \(version=(\d+), alias=(c\d+)", prompt)
            review_sequence.append([f"{cid}@v{ver}" for cid, ver, _ in items])
            verdicts = []
            for cid, ver, alias in items:
                if int(ver) == 1 and f"Claim {cid} (version=1, alias={alias}" in prompt and (
                    "L1 cost is $50,000." in prompt.split(f"Claim {cid}")[1].split("Claim ")[0]
                ):
                    verdicts.append(
                        {
                            "claim_id": alias,
                            "decision": "needs_revision",
                            "rationale": "Clarify annual term.",
                            "required_correction": "State first-year cost.",
                            "blocking_severity": "minor",
                            "evidence_refs": [],
                        }
                    )
                else:
                    verdicts.append(
                        {
                            "claim_id": alias,
                            "decision": "approve",
                            "rationale": "Verified.",
                            "blocking_severity": "none",
                            "evidence_refs": [],
                        }
                    )
            return {"verdicts": verdicts}

        if schema_name == "RevisionOutput":
            m = re.search(r"claim_id=(clm_[A-Za-z0-9_]+), expected_version=(\d+)", prompt)
            cid = m.group(1) if m else "clm_missing"
            ver = int(m.group(2)) if m else 1
            return {
                "updated_findings": "L1 first-year cost is $50,000.",
                "evidence_proposals": [],
                "revisions": [
                    {
                        "target_claim_id": cid,
                        "expected_version": ver,
                        "new_statement": "L1 first-year cost is $50,000.",
                        "revision_notes": "Clarified first-year cost.",
                        "computation": None,
                    }
                ],
            }

        raise AssertionError(f"Unexpected schema {schema_name}")

    settings = CrewForgeSettings(
        _env_file=None,
        provider=ProviderType.OPENAI,
        model="fake-v23-chain",
        openai_api_key="sk-offline",
        db_path=tmp_path / "v23_chain.db",
        budget_authority_path=tmp_path / "v23_chain_budget.db",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        max_workers=3,
        max_tasks=4,
        max_model_calls_per_run=16,
        max_revision_rounds=2,
        enable_peer_messaging=False,
        separate_review_dimensions=False,
        enable_deterministic_synthesis_fast_path=True,
    )
    provider = FakeProvider(handler=_handler)
    budget = BudgetManager(db, settings)
    runner = NetworkRunner(database=db, provider=provider, budget_manager=budget, settings=settings)

    result = await runner.run("Budget is $60,000 and cost is $50,000.")
    assert result.status == RunStatus.COMPLETED
    assert result.verification.honesty_gate_passed is True

    trace = await db.get_run_trace(result.run_id)
    assert trace is not None
    latest = {c.claim_id: c for c in trace.claims}
    for c in latest.values():
        assert c.version == 2
        assert c.status == ClaimStatus.SUPPORTED
        # Every dependent's premise_refs must point to v2 (never superseded v1)
        for pref in c.premise_refs:
            assert pref.claim_version == 2
