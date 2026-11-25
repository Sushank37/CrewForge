"""Offline tests for DAG scheduler, specialist execution, ledger, messaging, and orchestrator."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.ledger import ClaimLedger
from crewforge.core.messaging import MessageBus
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.synthesis import validate_synthesis_output
from crewforge.providers.base import ProviderError
from crewforge.schemas import (
    AgentMessage,
    ClaimStatus,
    EvidenceReliability,
    FollowUpOutput,
    PlannerOutput,
    ProposedClaim,
    RunStatus,
    SpecialistOutput,
    SynthesisOutput,
    TaskStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database
from tests.fake_provider import FakeProvider


def _settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "max_concurrency": 3,
        "max_retries": 0,
        "max_model_calls_per_run": 16,
        "run_deadline_seconds": 30.0,
        "max_communication_rounds": 1,
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
                "user_requirements": ["Analyze A and B independently, then combine in C"],
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
                            "evidence_indices": [0, 1],
                            "assumptions": [],
                            "confidence": 0.8,
                        }
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
                            "statement": "Branch B preliminary claim.",
                            "evidence_indices": [],
                            "assumptions": [],
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
            # Verify dependent task received both dependency findings AND inbound message from a1!
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
                "outgoing_messages": [],
                "task_completed": True,
            }
        if response_model is FollowUpOutput:
            followup_prompts.append(prompt)
            # Confirm specialist a2 actually received a1's peer message in follow-up context!
            assert "Note from a1: verify constraint X=42." in prompt
            return {
                "updated_findings": "Finding B updated after incorporating X=42 from a1.",
                "revised_claims": [
                    {
                        "statement": "Branch B revised claim with constraint X=42.",
                        "evidence_indices": [],
                        "assumptions": [],
                        "confidence": 0.85,
                    }
                ],
                "evidence_proposals": [],
                "uncertainties": [],
                "outgoing_messages": [],
            }
        return {
            "answer": "Final synthesized answer incorporating A, B (revised), and C.",
            "referenced_claim_ids": ["clm_forged_cross_run"],
            "unverified_assertions": [],
            "assumptions": [],
            "missing_information": [],
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
        assert result.status == RunStatus.COMPLETED
        assert max_observed_concurrency >= 2

        order = [name for name, _ in execution_log]
        assert order.index("t3") > order.index("t1")
        assert order.index("t3") > order.index("t2")
        assert len(followup_prompts) == 1

        trace = await db.get_run_trace(result.run_id)
        assert trace is not None
        # Forged claim ID was stripped by synthesis validation.
        assert "clm_forged_cross_run" not in result.referenced_claim_ids
        assert "invalid/out-of-run claim reference" in (result.confidence_notes or "")

        # Evidence validation: user_prompt match -> MEDIUM/verified_in_input=True;
        # external -> UNVERIFIED.
        evd_by_type = {e.source_type: e for e in trace.evidence}
        assert evd_by_type["user_prompt"].verified_in_input is True
        assert evd_by_type["user_prompt"].reliability == EvidenceReliability.MEDIUM
        assert evd_by_type["external_unverified"].verified_in_input is False
        assert evd_by_type["external_unverified"].reliability == EvidenceReliability.UNVERIFIED

        # Claims remain ASSERTED (never SUPPORTED purely on model assertion),
        # and v1+v2 history is preserved.
        assert all(c.status in {ClaimStatus.ASSERTED, ClaimStatus.UNRESOLVED} for c in trace.claims)
        versions = [(c.claim_id, c.version) for c in trace.claims]
        assert any(ver == 2 for _, ver in versions)
        assert any(ver == 1 for _, ver in versions)

        # Peer messages marked consumed after follow-up & dependent task delivery.
        assert len(trace.messages) == 2
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
        return {
            "answer": "Partial answer from t3 only.",
            "referenced_claim_ids": [],
            "unverified_assertions": [],
            "assumptions": [],
            "missing_information": [],
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
    # Allow only 2 calls total: 1 for planner, 0 tasks allowed (1 reserved for synthesis!).
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
            "answer": "Synthesized partial summary within call limit.",
            "referenced_claim_ids": [],
            "unverified_assertions": [],
            "assumptions": [],
            "missing_information": ["Task t1 could not run due to call reserve."],
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
        assert res.usage.model_calls == 2  # 1 planner + 1 synthesis!
    finally:
        await db.close()

    # Now test run_deadline_seconds expiration during worker execution.
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

        _, claims_a = await ledger.record_specialist_outputs(
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

        # Reject cross-run message reference to run_A's claim from run_B.
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

        # Reject cross-run agent ID.
        with pytest.raises(CrossRunReferenceError):
            await bus.read_inbox(run_id="run_B", recipient_id=plan_a.agents[0].agent_id)

        # Synthesis validation strips invalid claim references.
        syn = validate_synthesis_output(
            raw=SynthesisOutput(
                answer="Ans",
                referenced_claim_ids=[v1.claim_id, "clm_other_run"],
            ),
            valid_claim_ids={v1.claim_id},
            extra_assumptions=[],
            extra_missing=[],
            uncertainties=[],
            incomplete_tasks=[],
        )
        assert syn.referenced_claim_ids == [v1.claim_id]
        assert syn.invalid_claim_refs_removed == ["clm_other_run"]
    finally:
        await db.close()
