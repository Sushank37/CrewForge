"""Offline tests for DAG scheduler, specialist execution, ledger, messaging, and orchestrator."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.honesty_gate import HonestyGate
from crewforge.core.ledger import ClaimLedger
from crewforge.core.messaging import MessageBus
from crewforge.core.orchestrator import NetworkRunner
from crewforge.providers.base import ProviderError
from crewforge.schemas import (
    AgentMessage,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    EvidenceReliability,
    FollowUpOutput,
    PlannerOutput,
    ProposedClaim,
    ReleaseSegment,
    ReviewerBatchOutput,
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
    TaskStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database
from tests.fake_provider import FakeProvider

_CLAIM_ID_VER_RE = re.compile(r"Claim (clm_[a-f0-9]+) \(version=(\d+),")
_TARGET_CLAIM_RE = re.compile(r"claim_id=(clm_[a-f0-9]+),\s*expected_version=(\d+)")
_SUPPORTED_LINE_RE = re.compile(r"- (clm_[a-f0-9]+) \(version=(\d+),.*?status=supported")


def _settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "max_concurrency": 3,
        "max_retries": 0,
        "max_model_calls_per_run": 16,
        "run_deadline_seconds": 30.0,
        "max_communication_rounds": 1,
        "max_revision_rounds": 1,
    }
    defaults.update(overrides)
    return CrewForgeSettings(**defaults)  # type: ignore[arg-type]


async def test_true_concurrent_execution_dependency_ordering_and_messaging(
    tmp_path: Path,
) -> None:
    """Verify overlapping parallel tasks, topological ordering, and real peer message delivery."""
    db_path = tmp_path / "network_concurrent.db"
    settings = _settings(db_path, max_concurrency=3)
    db = Database(db_path)
    await db.initialize()

    active_workers = 0
    max_observed_concurrency = 0
    execution_log: list[tuple[str, float]] = []
    followup_prompts: list[str] = []
    synthesis_prompts: list[str] = []

    class InstrumentedProvider(FakeProvider):
        async def generate_structured(
            self,
            prompt: str,
            response_model: type[BaseModel],
            *,
            system_prompt: str | None = None,
            max_output_tokens: int | None = None,
            temperature: float | None = None,
            timeout_seconds: float | None = None,
            max_retries: int | None = None,
        ) -> Any:
            nonlocal active_workers, max_observed_concurrency
            if response_model is SpecialistOutput:
                active_workers += 1
                max_observed_concurrency = max(max_observed_concurrency, active_workers)
                await asyncio.sleep(0.06)
                active_workers -= 1
            return await super().generate_structured(
                prompt,
                response_model,  # type: ignore[arg-type]
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

    def handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        now = time.monotonic()
        if response_model is PlannerOutput:
            return {
                "goal": "Solve multi-part problem",
                "user_requirements": ["Analyze alpha and beta independently, then combine"],
                "acceptance_criteria": ["Both branches analyzed"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Branch A Analyst",
                        "instructions": "Work on A.",
                        "capabilities": ["analysis"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Branch B Analyst",
                        "instructions": "Work on B.",
                        "capabilities": ["analysis"],
                    },
                    {
                        "agent_ref": "a3",
                        "role": "Integrator",
                        "instructions": "Integrate A and B.",
                        "capabilities": ["synthesis"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Task A",
                        "description": "Independent task A",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Finding A"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Task B",
                        "description": "Independent task B",
                        "assigned_agent_ref": "a2",
                        "depends_on": [],
                        "expected_deliverables": ["Finding B"],
                    },
                    {
                        "task_ref": "t3",
                        "title": "Task C",
                        "description": "Dependent integration task",
                        "assigned_agent_ref": "a3",
                        "depends_on": ["t1", "t2"],
                        "expected_deliverables": ["Combined finding"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if response_model is SpecialistOutput:
            if "Assigned task (t1)" in prompt:
                execution_log.append(("t1", now))
                return {
                    "findings": "Finding A: exact match from prompt token.",
                    "claims": [
                        {
                            "statement": "Prompt includes keyword alpha.",
                            "category": "given_fact",
                            "evidence_indices": [0],
                            "assumptions": [],
                            "confidence": 0.9,
                        },
                        {
                            "statement": "External citation unretrieved.",
                            "category": "external_factual_assertion",
                            "evidence_indices": [1],
                            "assumptions": [],
                            "confidence": 0.5,
                        },
                    ],
                    "evidence_proposals": [
                        {
                            "source_type": "user_prompt",
                            "source_ref": "prompt",
                            "excerpt": "alpha",
                        },
                        {
                            "source_type": "external_unverified",
                            "source_ref": "https://example.invalid",
                            "excerpt": "external claim",
                        },
                    ],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                    "outgoing_messages": [
                        {
                            "recipient_ref": "a2",
                            "message_type": "clarification",
                            "content": "Note from a1: verify constraint X=42.",
                            "related_claim_indices": [0],
                        },
                        {
                            "recipient_ref": "a3",
                            "message_type": "finding",
                            "content": "Handoff from a1 to a3: include A.",
                            "related_claim_indices": [0],
                        },
                    ],
                    "task_completed": True,
                }
            if "Assigned task (t2)" in prompt:
                execution_log.append(("t2", now))
                return {
                    "findings": "Finding B completed.",
                    "claims": [
                        {
                            "statement": "Branch beta preliminary analysis completed.",
                            "category": "logical_inference",
                            "evidence_indices": [],
                            "assumptions": ["Evaluated independently"],
                            "confidence": 0.7,
                        }
                    ],
                    "evidence_proposals": [],
                    "assumptions": [],
                    "uncertainties": [],
                    "missing_information": [],
                    "outgoing_messages": [],
                    "task_completed": True,
                }
            execution_log.append(("t3", now))
            assert "Finding A" in prompt
            assert "Finding B" in prompt
            assert "Handoff from a1 to a3: include A." in prompt
            return {
                "findings": "Integrated finding C incorporating A, B, and handoff message.",
                "claims": [],
                "evidence_proposals": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [
                    {
                        "recipient_ref": "lead",
                        "message_type": "finding",
                        "content": "Lead notice: integration C finished cleanly.",
                        "related_claim_indices": [],
                    }
                ],
                "task_completed": True,
            }
        if response_model is FollowUpOutput:
            followup_prompts.append(prompt)
            assert "Note from a1: verify constraint X=42." in prompt
            targets = _TARGET_CLAIM_RE.findall(prompt)
            assert targets, "FollowUp prompt must expose target claim_id and expected_version"
            cid, cver = targets[0]
            return {
                "updated_findings": "Finding B updated after incorporating X=42 from a1.",
                "revised_claims": [
                    {
                        "target_claim_id": cid,
                        "expected_version": int(cver),
                        "new_statement": "Branch beta revised analysis incorporating X=42.",
                        "revision_notes": "Updated from peer message.",
                        "assumptions": ["X=42 verified by peer a1"],
                        "confidence": 0.85,
                    }
                ],
                "new_claims": [],
                "evidence_proposals": [],
                "uncertainties": [],
                "outgoing_messages": [],
            }
        if response_model is ReviewerBatchOutput:
            verdicts = []
            for cid, cver in _CLAIM_ID_VER_RE.findall(prompt):
                verdicts.append(
                    {
                        "claim_id": cid,
                        "claim_version": int(cver),
                        "review_dimension": "combined_evidence_consistency",
                        "decision": "approve",
                        "rationale": "Supported by prompt and consistent.",
                        "evidence_refs": [],
                        "blocking_severity": "none",
                    }
                )
            return {"verdicts": verdicts, "missing_requirement_coverage": []}

        synthesis_prompts.append(prompt)
        assert "Lead notice: integration C finished cleanly." in prompt
        supported = _SUPPORTED_LINE_RE.findall(prompt)
        segments = []
        for cid, cver in supported:
            segments.append(
                {
                    "segment_type": "accepted_claim",
                    "text": "Analysis of alpha and beta completed with verified keyword alpha.",
                    "claim_refs": [{"claim_id": cid, "claim_version": int(cver)}],
                    "is_factual_assertion": True,
                }
            )
        return {
            "user_requirements": ["Analyze alpha and beta independently, then combine"],
            "accepted_segments": segments,
            "conditional_conclusions": [],
            "recommendations": [],
            "assumptions": [],
            "unresolved_issues": [],
            "missing_information": [],
            "new_factual_assertions": [],
            "referenced_claim_ids": ["clm_forged_cross_run", *[c for c, _ in supported]],
            "confidence_notes": "All branches completed.",
        }

    try:
        provider = InstrumentedProvider(handler=handler, max_retries=0)
        budget = BudgetManager(db, settings)
        runner = NetworkRunner(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
        )

        result = await runner.run("Analyze alpha and beta concurrently.")
        assert result.status == RunStatus.PARTIAL  # forged claim_id triggers gate reason
        assert max_observed_concurrency >= 2

        order = [name for name, _ in execution_log]
        assert order.index("t3") > order.index("t1")
        assert order.index("t3") > order.index("t2")
        assert len(followup_prompts) == 1
        assert len(synthesis_prompts) == 1

        trace = await db.get_run_trace(result.run_id)
        assert trace is not None
        assert "clm_forged_cross_run" not in result.referenced_claim_ids
        assert trace.honesty_gate is not None
        assert any("clm_forged_cross_run" in r for r in trace.honesty_gate.reasons)

        evd_by_type = {e.source_type: e for e in trace.evidence}
        assert evd_by_type["user_prompt"].verified_in_input is True
        assert evd_by_type["user_prompt"].reliability == EvidenceReliability.MEDIUM
        assert evd_by_type["external_unverified"].verified_in_input is False
        assert evd_by_type["external_unverified"].reliability == EvidenceReliability.UNVERIFIED

        # External unretrieved claim stays UNRESOLVED; given_fact & revised inference SUPPORTED.
        ext_claims = [c for c in trace.claims if c.category == ClaimCategory.EXTERNAL_FACT]
        assert len(ext_claims) == 1
        assert ext_claims[0].status == ClaimStatus.UNRESOLVED

        versions = [(c.claim_id, c.version) for c in trace.claims]
        assert any(ver == 2 for _, ver in versions)
        assert any(ver == 1 for _, ver in versions)

        # Peer messages + Lead inbox message all consumed!
        assert len(trace.messages) == 3
        assert all(m.delivery_status == "consumed" for m in trace.messages)
    finally:
        await db.close()


async def test_failed_task_blocks_dependents_without_blocking_independent_branch(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "partial_failure.db"
    settings = _settings(db_path, max_concurrency=2, max_communication_rounds=0)
    db = Database(db_path)
    await db.initialize()

    def handler(prompt: str, response_model: type[BaseModel] | None) -> Any:
        if response_model is PlannerOutput:
            return {
                "goal": "Partial failure handling",
                "user_requirements": ["Run two branches"],
                "acceptance_criteria": ["Report partial status on failure"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Failing Worker",
                        "instructions": "Fail on t1.",
                        "capabilities": ["analysis"],
                    },
                    {
                        "agent_ref": "a2",
                        "role": "Independent Worker",
                        "instructions": "Succeed on t3.",
                        "capabilities": ["analysis"],
                    },
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Failing root",
                        "description": "Will fail",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Out 1"],
                    },
                    {
                        "task_ref": "t2",
                        "title": "Blocked dependent",
                        "description": "Depends on t1",
                        "assigned_agent_ref": "a1",
                        "depends_on": ["t1"],
                        "expected_deliverables": ["Out 2"],
                    },
                    {
                        "task_ref": "t3",
                        "title": "Independent branch",
                        "description": "Should still complete",
                        "assigned_agent_ref": "a2",
                        "depends_on": [],
                        "expected_deliverables": ["Out 3"],
                    },
                ],
                "assumptions": [],
                "missing_information": [],
            }
        if response_model is SpecialistOutput:
            if "Assigned task (t1)" in prompt:
                raise ProviderError(
                    "simulated failure",
                    code="api_error",
                    provider="fake",
                    model="fake-model",
                    retryable=False,
                    billing_outcome="unbilled",
                )
            return {
                "findings": "Independent branch t3 succeeded.",
                "claims": [],
                "evidence_proposals": [],
                "assumptions": [],
                "uncertainties": [],
                "missing_information": [],
                "outgoing_messages": [],
                "task_completed": True,
            }
        if response_model is ReviewerBatchOutput:
            return {"verdicts": [], "missing_requirement_coverage": []}
        return {
            "user_requirements": ["Run two branches"],
            "accepted_segments": [],
            "conditional_conclusions": [
                {
                    "segment_type": "conditional_conclusion",
                    "text": "Only the independent branch finished; dependent work was blocked.",
                    "claim_refs": [],
                    "is_factual_assertion": False,
                }
            ],
            "recommendations": [],
            "assumptions": [],
            "unresolved_issues": ["t1 failed and t2 was blocked."],
            "missing_information": [],
            "new_factual_assertions": [],
            "referenced_claim_ids": [],
            "confidence_notes": "t1 failed and t2 was blocked.",
        }

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        res = await runner.run("Test partial DAG execution")
        assert res.status == RunStatus.PARTIAL
        assert len(res.incomplete_tasks) == 2

        trace = await db.get_run_trace(res.run_id)
        assert trace is not None
        by_ref = {t.task_ref: t.status for t in trace.tasks}
        assert by_ref["t1"] == TaskStatus.FAILED
        assert by_ref["t2"] == TaskStatus.BLOCKED
        assert by_ref["t3"] == TaskStatus.COMPLETED
    finally:
        await db.close()


async def test_run_deadline_and_budget_exhaustion_produce_partial_results(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "limits.db"
    settings = _settings(db_path, max_model_calls_per_run=2, max_communication_rounds=0)
    db = Database(db_path)
    await db.initialize()

    def handler(_prompt: str, response_model: type[BaseModel] | None) -> Any:
        if response_model is PlannerOutput:
            return {
                "goal": "Test budget cap",
                "user_requirements": ["Req"],
                "acceptance_criteria": ["Crit"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "Analyst",
                        "instructions": "Work.",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Blocked by call reserve",
                        "description": "Needs call",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Out"],
                    }
                ],
                "assumptions": [],
                "missing_information": [],
            }
        return {
            "user_requirements": ["Req"],
            "accepted_segments": [],
            "conditional_conclusions": [],
            "recommendations": [],
            "assumptions": [],
            "unresolved_issues": ["Task t1 could not run due to call reserve."],
            "missing_information": ["Task t1 could not run due to call reserve."],
            "new_factual_assertions": [],
            "referenced_claim_ids": [],
            "confidence_notes": "Partial result.",
        }

    try:
        runner = NetworkRunner(
            provider=FakeProvider(handler=handler, max_retries=0),
            database=db,
            budget_manager=BudgetManager(db, settings),
            settings=settings,
        )
        res = await runner.run("Test budget cap reservation for synthesis")
        assert res.status == RunStatus.PARTIAL
        assert res.usage.model_calls == 2
        assert res.verification is not None
        assert res.verification.review_completeness is None  # 0 material claims -> None, not 100%!
    finally:
        await db.close()

    db_path_2 = tmp_path / "deadline.db"
    deadline_settings = _settings(
        db_path_2,
        run_deadline_seconds=0.05,
        max_model_calls_per_run=10,
        max_communication_rounds=0,
    )
    db2 = Database(db_path_2)
    await db2.initialize()
    try:

        class SlowWorkerProvider(FakeProvider):
            async def generate_structured(
                self,
                prompt: str,
                response_model: type[BaseModel],
                **kwargs: Any,
            ) -> Any:
                if response_model is SpecialistOutput:
                    await asyncio.sleep(0.25)
                return await super().generate_structured(prompt, response_model, **kwargs)

        runner2 = NetworkRunner(
            provider=SlowWorkerProvider(handler=handler, max_retries=0),
            database=db2,
            budget_manager=BudgetManager(db2, deadline_settings),
            settings=deadline_settings,
        )
        res2 = await runner2.run("Trigger run deadline")
        assert res2.status in {RunStatus.PARTIAL, RunStatus.FAILED}
        trace2 = await db2.get_run_trace(res2.run_id)
        assert trace2 is not None
        assert trace2.tasks[0].status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
        assert trace2.honesty_gate is not None
    finally:
        await db2.close()


async def test_cross_run_isolation_and_ledger_immutability(tmp_path: Path) -> None:
    db_path = tmp_path / "isolation.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()

    try:
        await db.create_run("run_A", "network", "Prompt A", settings.to_run_config())
        await db.create_run("run_B", "network", "Prompt B", settings.to_run_config())

        from crewforge.core.planner import validate_and_materialize_plan

        raw = PlannerOutput.model_validate(
            {
                "goal": "G",
                "user_requirements": ["R"],
                "acceptance_criteria": ["C"],
                "agents": [
                    {
                        "agent_ref": "a1",
                        "role": "R1",
                        "instructions": "I1",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "T1",
                        "description": "D1",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["O1"],
                    }
                ],
            }
        )
        plan_a = validate_and_materialize_plan(run_id="run_A", raw_plan=raw, settings=settings)
        plan_b = validate_and_materialize_plan(run_id="run_B", raw_plan=raw, settings=settings)
        await db.save_execution_plan(plan_a)
        await db.save_execution_plan(plan_b)

        ledger = ClaimLedger(db)
        bus = MessageBus(db)

        _, claims_a, _ = await ledger.record_specialist_outputs(
            run_id="run_A",
            agent_id=plan_a.agents[0].agent_id,
            task_id=plan_a.tasks[0].task_id,
            original_prompt="Prompt A",
            dependency_findings_text="",
            evidence_proposals=[],
            claim_proposals=[
                ProposedClaim(
                    statement="Initial claim v1",
                    evidence_indices=[],
                    assumptions=[],
                    confidence=0.5,
                )
            ],
        )
        v1 = claims_a[0]
        v2 = await ledger.revise_claim(
            prior_claim=v1,
            new_statement="Revised claim v2",
            author_agent_id=plan_a.agents[0].agent_id,
        )
        all_versions = await ledger.get_claims("run_A", latest_only=False)
        assert [(c.version, c.statement) for c in all_versions] == [
            (1, "Initial claim v1"),
            (2, "Revised claim v2"),
        ]
        assert v2.parent_claim_id == v1.claim_id

        with pytest.raises(CrossRunReferenceError):
            await db.insert_message(
                AgentMessage(
                    run_id="run_B",
                    sender_id=plan_b.agents[0].agent_id,
                    recipient_id=None,
                    content="Cross run leak attempt",
                    related_claim_ids=[v1.claim_id],
                ),
                [],
            )

        with pytest.raises(CrossRunReferenceError):
            await bus.read_inbox(run_id="run_B", recipient_id=plan_a.agents[0].agent_id)

        gate = HonestyGate(db)
        gate_res = gate.evaluate(
            run_id="run_A",
            original_prompt="Prompt A",
            plan=plan_a,
            synthesis=SynthesisOutput(
                accepted_segments=[
                    ReleaseSegment(
                        segment_type="accepted_claim",
                        text="Cross-run claim",
                        claim_refs=[ClaimVersionRef(claim_id="clm_other_run", claim_version=1)],
                    )
                ],
                referenced_claim_ids=[v1.claim_id, "clm_other_run"],
            ),
            all_claims=all_versions,
            evidence_list=[],
            tool_executions=[],
            verdicts=[],
            incomplete_tasks=[],
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            is_partial_run=False,
        )
        assert gate_res.gate_record is not None
        assert gate_res.gate_record.passed is False
        assert len(gate_res.gate_record.rejected_claim_refs) >= 1
    finally:
        await db.close()
