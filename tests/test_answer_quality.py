"""Offline regression tests for answer quality, revalidation, coverage, and baseline labeling."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.core.baseline import BaselineRunner
from crewforge.core.budget import BudgetManager
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.orchestrator import NetworkRunner
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    ExecutionPlan,
    PlannerOutput,
    ReleaseSegment,
    ReviewDecision,
    ReviewerBatchOutput,
    ReviewVerdict,
    RevisionOutput,
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
    TaskSpec,
    VerificationStatus,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import build_provenance_catalog
from tests.fake_provider import FakeProvider

_CLAIM_HEADER_RE = re.compile(r"Claim (clm_[a-f0-9]+) \(version=(\d+),")
_SUPPORTED_LINE_RE = re.compile(r"- (clm_[a-f0-9]+) \(version=(\d+),.*?status=supported")


def test_dependent_claim_revalidated_after_upstream_revision(tmp_path: Path) -> None:
    """Verify stale dependent claims are refreshed and revalidated once upstream premises pass."""
    db_path = tmp_path / "reval.db"
    settings = CrewForgeSettings(
        db_path=db_path,
        budget_authority_path=tmp_path / "reval_budget.db",
        provider="ollama",
        model="qwen3:4b",
        allow_paid_runs=False,
        max_revision_rounds=1,
        max_model_calls_per_run=16,
        max_retries=0,
        enable_deterministic_synthesis_fast_path=True,
    )

    review_round = 0

    def handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        nonlocal review_round
        if response_model is PlannerOutput:
            return {
                "goal": "Compute total cost and verify budget compliance",
                "user_requirements": [
                    "Compute total cost",
                    "State if under $600,000 budget",
                ],
                "acceptance_criteria": ["Both total cost and budget status verified"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Cost Specialist",
                        "instructions": "Compute total cost.",
                        "capabilities": ["calculation"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Budget Reviewer",
                        "instructions": "Check if total is under budget.",
                        "capabilities": ["analysis"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute total cost",
                        "description": "Add 50400 and 489600.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Total cost"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Check under budget",
                        "description": "Compare total cost against 600000.",
                        "assigned_agent_ref": "a2",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Under-budget conclusion"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if response_model is SpecialistOutput:
            if "Compare total cost against 600000" not in prompt:
                # Emit wrong proposed_result=500000 so v1 is rejected and triggers revision!
                return {
                    "findings": "Total cost is 500000.",
                    "claims": [
                        {
                            "statement": "Total annual cost is 500000 USD.",
                            "category": "computation",
                            "evidence_indices": [0],
                            "computation": {
                                "expression": "a + b",
                                "variables": {"a": "50400", "b": "489600"},
                                "input_provenance_refs": {"a": "user_prompt", "b": "user_prompt"},
                                "proposed_result": "500000",
                                "units": "USD",
                            },
                            "requirement_refs": ["Compute total cost"],
                            "confidence": 0.8,
                        }
                    ],
                    "evidence_proposals": [
                        {
                            "source_type": "calculation",
                            "source_ref": "arithmetic_tool",
                            "excerpt": "a + b",
                        }
                    ],
                    "assumptions": [],
                    "uncertainties": [],
                    "outgoing_messages": [],
                }
            # t2 depends on t1's claim
            m_dep = re.search(r"(clm_[a-zA-Z0-9_]+)", prompt)
            dep_id = m_dep.group(1) if m_dep else None
            return {
                "findings": "The plan stays under budget.",
                "claims": [
                    {
                        "statement": "The plan stays under budget.",
                        "category": "logical_inference",
                        "evidence_indices": [0],
                        "depends_on_claim_ids": [dep_id] if dep_id else [],
                        "premise_refs": (
                            [{"claim_id": dep_id, "claim_version": 1}] if dep_id else []
                        ),
                        "requirement_refs": ["State if under $600,000 budget"],
                        "confidence": 0.9,
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "dependency_finding",
                        "source_ref": dep_id or "t1",
                        "excerpt": "Total cost is 500000.",
                    }
                ],
                "assumptions": [],
                "uncertainties": [],
                "outgoing_messages": [],
            }
        if response_model is RevisionOutput:
            m_cid = re.search(r"claim_id=(clm_[a-f0-9]+),\s*expected_version=(\d+)", prompt)
            cid = m_cid.group(1) if m_cid else "clm_unknown"
            cver = int(m_cid.group(2)) if m_cid else 1
            return {
                "updated_findings": "Corrected total annual cost to 540000 USD.",
                "revisions": [
                    {
                        "target_claim_id": cid,
                        "expected_version": cver,
                        "new_statement": "Total annual cost is 540000 USD.",
                        "category": "computation",
                        "evidence_indices": [0],
                        "computation": {
                            "expression": "a + b",
                            "variables": {"a": "50400", "b": "489600"},
                            "input_provenance_refs": {"a": "user_prompt", "b": "user_prompt"},
                            "proposed_result": "540000",
                            "units": "USD",
                        },
                        "revision_notes": "Corrected sum to 540000 USD.",
                        "confidence": 0.99,
                    }
                ],
                "evidence_proposals": [
                    {
                        "source_type": "calculation",
                        "source_ref": "arithmetic_tool",
                        "excerpt": "50400 + 489600 = 540000",
                    }
                ],
                "uncertainties": [],
                "outgoing_messages": [],
            }
        if response_model is ReviewerBatchOutput:
            review_round += 1
            verdicts = []
            for cid, cver in _CLAIM_HEADER_RE.findall(prompt):
                verdicts.append(
                    {
                        "claim_id": cid,
                        "claim_version": int(cver),
                        "review_dimension": "combined_evidence_consistency",
                        "decision": "approve",
                        "rationale": "Verified against upstream calculation and prompt.",
                        "evidence_refs": [],
                        "blocking_severity": "none",
                    }
                )
            return {"verdicts": verdicts, "missing_requirement_coverage": []}

        # Synthesis: map each supported claim verbatim from the synthesis prompt
        supported_items = re.findall(
            r"- (clm_[a-f0-9]+) \(version=(\d+),[^\n]*status=supported[^\n]*\)\s*\n?\s*([^\n]+)",
            prompt,
        )
        return {
            "user_requirements": ["Compute total cost", "State if under $600,000 budget"],
            "accepted_segments": [
                {
                    "segment_type": "accepted_claim",
                    "text": stmt.strip().lstrip(":").strip(),
                    "claim_refs": [{"claim_id": cid, "claim_version": int(cver)}],
                    "requirement_ref": (
                        "Compute total cost"
                        if "540000" in stmt
                        else "State if under $600,000 budget"
                    ),
                    "is_factual_assertion": True,
                }
                for cid, cver, stmt in supported_items
            ],
            "conditional_conclusions": [],
            "recommendations": [],
            "assumptions": [],
            "unresolved_issues": [],
            "missing_information": [],
            "new_factual_assertions": [],
            "referenced_claim_ids": [cid for cid, _, _ in supported_items],
        }

    async def _run() -> None:
        db = Database(db_path)
        await db.initialize()
        try:
            runner = NetworkRunner(
                provider=FakeProvider(handler=handler, max_retries=0),
                database=db,
                budget_manager=BudgetManager(db, settings),
                settings=settings,
            )
            res = await runner.run("Cloud 50400, staffing 489600, budget 600000.")
            assert res.status == RunStatus.COMPLETED, (
                res.error,
                res.answer,
                res.verification.honesty_gate_reasons if res.verification else None,
            )
            assert res.verification.stale_claim_count == 0
            assert res.verification.supported_claim_count == 2
            assert "540000 USD" in res.answer
            assert "stays under budget" in res.answer
            assert res.answer.count("540000 USD") == 1
            assert "[clm_" not in res.answer
        finally:
            await db.close()

    asyncio.run(_run())


def test_stale_dependent_revalidation_creates_new_version_preserving_v1(tmp_path: Path) -> None:
    """_revalidate_stale_dependents must create v2 and preserve v1 premise_refs & STALE status."""
    from crewforge.core.ledger import ClaimLedger
    from crewforge.core.messaging import MessageBus
    from crewforge.core.repair import TargetedRepairEngine
    from crewforge.core.reviewer import ReviewerPool
    from crewforge.schemas import AgentSpec

    db_path = tmp_path / "reval_v2.db"
    settings = CrewForgeSettings(
        db_path=db_path,
        budget_authority_path=tmp_path / "reval_v2_budget.db",
        provider="ollama",
        model="qwen3:4b",
        allow_paid_runs=False,
        max_model_calls_per_run=16,
        max_retries=0,
    )

    def handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        if response_model is ReviewerBatchOutput:
            verdicts = []
            for cid, cver in _CLAIM_HEADER_RE.findall(prompt):
                verdicts.append(
                    {
                        "claim_id": cid,
                        "claim_version": int(cver),
                        "review_dimension": "combined_evidence_consistency",
                        "decision": "approve",
                        "rationale": "Upstream premise v2 is supported.",
                        "evidence_refs": [],
                        "blocking_severity": "none",
                    }
                )
            return {"verdicts": verdicts, "missing_requirement_coverage": []}
        return {}

    async def _run() -> None:
        db = Database(db_path)
        await db.initialize()
        try:
            await db.create_run("run_rv2", "network", "Check budget", settings.to_run_config())
            plan = ExecutionPlan(
                run_id="run_rv2",
                goal="Budget check",
                user_requirements=["Check budget"],
                agents=[AgentSpec(agent_id="a1", role="Analyst", instructions="Analyze")],
                tasks=[
                    TaskSpec(
                        task_id="t1",
                        task_ref="t1",
                        title="Check",
                        description="Check",
                        assigned_agent_id="a1",
                    )
                ],
            )
            await db.save_execution_plan(plan)
            await db.insert_claim(
                Claim(
                    claim_id="clm_deadbeef01",
                    run_id="run_rv2",
                    version=1,
                    statement="Total is 500000 USD.",
                    category=ClaimCategory.COMPUTATION,
                    author_agent_id="a1",
                    task_id="t1",
                    status=ClaimStatus.REJECTED,
                    verification_status=VerificationStatus.CONTRADICTED,
                )
            )
            await db.insert_claim(
                Claim(
                    claim_id="clm_deadbeef01",
                    run_id="run_rv2",
                    version=2,
                    statement="Total is 540000 USD.",
                    category=ClaimCategory.COMPUTATION,
                    author_agent_id="a1",
                    task_id="t1",
                    status=ClaimStatus.SUPPORTED,
                    verification_status=VerificationStatus.COMPUTATIONALLY_VERIFIED,
                )
            )
            await db.insert_evidence(
                Evidence(
                    evidence_id="evd_dep_1",
                    run_id="run_rv2",
                    source_type="dependency_finding",
                    source_ref="clm_deadbeef01",
                    excerpt="Total is 540000 USD.",
                    created_by="a1",
                    verified_in_input=True,
                )
            )
            await db.insert_claim(
                Claim(
                    claim_id="clm_deadbeef02",
                    run_id="run_rv2",
                    version=1,
                    statement="The plan stays under budget.",
                    category=ClaimCategory.INFERENCE,
                    author_agent_id="a1",
                    task_id="t1",
                    evidence_ids=["evd_dep_1"],
                    depends_on_claim_ids=["clm_deadbeef01"],
                    premise_refs=[ClaimVersionRef(claim_id="clm_deadbeef01", claim_version=1)],
                    status=ClaimStatus.STALE,
                    verification_status=VerificationStatus.STALE_PENDING_REVIEW,
                )
            )
            bm = BudgetManager(db, settings)
            from crewforge.core.executor import ModelExecutor

            ex = ModelExecutor(
                provider=FakeProvider(handler=handler, max_retries=0),
                database=db,
                budget_manager=bm,
                settings=settings,
            )
            ledger = ClaimLedger(db)
            reviewer = ReviewerPool(executor=ex, database=db, settings=settings)
            repair = TargetedRepairEngine(
                executor=ex,
                database=db,
                ledger=ledger,
                message_bus=MessageBus(db),
                reviewer_pool=reviewer,
                settings=settings,
            )
            calls, _verdicts, blocking = await repair._revalidate_stale_dependents(
                run_id="run_rv2",
                original_prompt="Check budget",
                plan=plan,
                verdicts_by_key={},
                run_deadline_monotonic=None,
            )
            assert calls == 1
            assert blocking == []
            versions = [
                c
                for c in await db.get_claims_for_run("run_rv2", latest_only=False)
                if c.claim_id == "clm_deadbeef02"
            ]
            assert [c.version for c in versions] == [1, 2]
            assert versions[0].status == ClaimStatus.STALE
            assert versions[0].premise_refs[0].claim_version == 1
            assert versions[1].status == ClaimStatus.SUPPORTED
            assert versions[1].premise_refs[0].claim_version == 2
        finally:
            await db.close()

    asyncio.run(_run())


def test_requirement_coverage_marks_partial_when_linked_claim_is_stale(tmp_path: Path) -> None:
    """A requirement with both a supported claim and a stale claim must score partially_answered."""

    async def _run() -> None:
        db = Database(tmp_path / "cov.db")
        await db.initialize()
        gate = HonestyGate(db)
        plan = ExecutionPlan(
            run_id="run_cov",
            goal="Cost and budget check",
            user_requirements=["Compute total cost and state whether under budget"],
            tasks=[
                TaskSpec(
                    task_id="t1",
                    task_ref="t1",
                    title="Cost",
                    description="Cost",
                    assigned_agent_id="a1",
                )
            ],
        )
        c_ok = Claim(
            claim_id="clm_ok",
            run_id="run_cov",
            version=1,
            statement="The migration plan is feasible.",
            category=ClaimCategory.INFERENCE,
            author_agent_id="a1",
            status=ClaimStatus.SUPPORTED,
            verification_status=VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
            requirement_refs=["Compute total cost and state whether under budget"],
        )
        c_stale = Claim(
            claim_id="clm_stale",
            run_id="run_cov",
            version=1,
            statement="The plan stays under budget.",
            category=ClaimCategory.RECOMMENDATION,
            author_agent_id="a1",
            status=ClaimStatus.STALE,
            verification_status=VerificationStatus.STALE_PENDING_REVIEW,
            requirement_refs=["Compute total cost and state whether under budget"],
        )
        synth = SynthesisOutput(
            user_requirements=["Compute total cost and state whether under budget"],
            accepted_segments=[
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text="The migration plan is feasible.",
                    claim_refs=[ClaimVersionRef(claim_id="clm_ok", claim_version=1)],
                    is_factual_assertion=True,
                )
            ],
            referenced_claim_ids=["clm_ok"],
        )
        ev = gate.evaluate(
            run_id="run_cov",
            original_prompt="Total 540000 vs 600000",
            plan=plan,
            synthesis=synth,
            all_claims=[c_ok, c_stale],
            evidence_list=[],
            tool_executions=[],
            verdicts=[
                ReviewVerdict(
                    run_id="run_cov",
                    claim_id="clm_ok",
                    claim_version=1,
                    reviewer_id="reviewer_combined",
                    review_dimension="combined_evidence_consistency",
                    decision=ReviewDecision.APPROVE,
                    rationale="Verified calculation.",
                )
            ],
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=False,
        )
        assert ev.verification.requirement_coverage == 0.5
        assert ev.verification.requirements_answered == 0
        assert ev.verification.requirements_partial == 1
        assert ev.verification.requirement_statuses[0].status == "partially_answered"
        await db.close()

    asyncio.run(_run())


def test_baseline_runner_marks_verification_unexecuted(tmp_path: Path) -> None:
    """BaselineRunner must mark release_checks_executed=False and honesty_gate_passed=False."""
    db_path = tmp_path / "base.db"
    settings = CrewForgeSettings(
        db_path=db_path,
        budget_authority_path=tmp_path / "base_budget.db",
        provider="ollama",
        model="qwen3:4b",
        allow_paid_runs=False,
    )

    def handler(_prompt: str, _rm: type[BaseModel] | None) -> Any:
        return {
            "answer": "The result is 42.",
            "missing_information": [],
            "assumptions": [],
            "confidence_notes": "Single-agent response.",
            "computations": [],
            "needs_tool_followup": False,
        }

    async def _run() -> None:
        db = Database(db_path)
        await db.initialize()
        try:
            runner = BaselineRunner(
                provider=FakeProvider(handler=handler, max_retries=0),
                database=db,
                budget_manager=BudgetManager(db, settings),
                settings=settings,
            )
            res = await runner.run("What is 21 * 2?")
            assert res.status == RunStatus.COMPLETED, (
                res.error,
                res.answer,
                res.verification.honesty_gate_reasons if res.verification else None,
            )
            assert res.verification.release_checks_executed is False
            assert res.verification.honesty_gate_passed is False
            assert res.verification.content_released_safely is False
        finally:
            await db.close()

    asyncio.run(_run())


def test_multi_document_chunk_aggregation_in_provenance_catalog() -> None:
    """Multiple multi-chunk documents must each bind doc_k to their full aggregated text."""
    ev_list = [
        Evidence(
            evidence_id="evd_1a",
            run_id="run_docs",
            source_type="external_retrieved",
            source_ref="spec_a.md",
            excerpt="Section 1: Rate is 450.",
            created_by="retriever",
            verified_in_input=True,
        ),
        Evidence(
            evidence_id="evd_1b",
            run_id="run_docs",
            source_type="external_retrieved",
            source_ref="spec_a.md",
            excerpt="Section 2: Shift is 8 hours.",
            created_by="retriever",
            verified_in_input=True,
        ),
        Evidence(
            evidence_id="evd_2a",
            run_id="run_docs",
            source_type="external_retrieved",
            source_ref="spec_b.md",
            excerpt="Part 1: Backlog is 620.",
            created_by="retriever",
            verified_in_input=True,
        ),
        Evidence(
            evidence_id="evd_2b",
            run_id="run_docs",
            source_type="external_retrieved",
            source_ref="spec_b.md",
            excerpt="Part 2: Reserve is 100.",
            created_by="retriever",
            verified_in_input=True,
        ),
    ]
    cat = build_provenance_catalog(original_prompt="Check docs.", verified_evidence=ev_list)
    assert "450" in cat.targets_by_key["spec_a.md"].text
    assert "8 hours" in cat.targets_by_key["spec_a.md"].text
    assert "620" in cat.targets_by_key["spec_b.md"].text
    assert "100" in cat.targets_by_key["spec_b.md"].text
