"""Evidence-constrained Lead synthesis and Honesty Gate integration."""

from __future__ import annotations

from crewforge.core.executor import ModelExecutor
from crewforge.core.honesty_gate import HonestyGate, HonestyGateResult
from crewforge.core.messaging import MessageBus
from crewforge.prompts.synthesis import (
    build_synthesis_system_prompt,
    build_synthesis_user_prompt,
)
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    ReleaseSegment,
    ReviewVerdict,
    SynthesisOutput,
    TaskSpec,
    TaskStatus,
    ToolExecutionRecord,
)
from crewforge.storage.database import Database


def build_partial_synthesis_output(
    *,
    plan: ExecutionPlan,
    claims: list[Claim],
    incomplete_tasks: list[TaskSpec],
    reason: str,
) -> SynthesisOutput:
    """Build a constrained partial SynthesisOutput strictly from reviewed claims.

    Never releases raw unverified worker findings as verified factual prose.
    """
    accepted: list[ReleaseSegment] = []
    recommendations: list[ReleaseSegment] = []
    unresolved: list[str] = [f"Partial execution notice: {reason}"]

    for claim in claims:
        ref = ClaimVersionRef(claim_id=claim.claim_id, claim_version=claim.version)
        if claim.status == ClaimStatus.SUPPORTED and claim.is_material:
            accepted.append(
                ReleaseSegment(
                    segment_type="accepted_claim",
                    text=claim.statement,
                    claim_refs=[ref],
                    is_factual_assertion=True,
                )
            )
        elif claim.category == ClaimCategory.RECOMMENDATION:
            recommendations.append(
                ReleaseSegment(
                    segment_type="recommendation",
                    text=claim.statement,
                    claim_refs=[ref],
                    is_factual_assertion=False,
                )
            )

    for task in incomplete_tasks:
        unresolved.append(
            f"Incomplete task {task.task_ref or task.task_id} ({task.title}): {task.status.value}."
        )

    return SynthesisOutput(
        user_requirements=list(plan.user_requirements),
        accepted_segments=accepted,
        conditional_conclusions=[],
        recommendations=recommendations,
        assumptions=list(plan.assumptions),
        unresolved_issues=unresolved,
        missing_information=list(plan.missing_information),
        new_factual_assertions=[],
        referenced_claim_ids=[c.claim_id for c in claims if c.status == ClaimStatus.SUPPORTED],
    )


class LeadSynthesizer:
    """Synthesizes reviewed claims with Lead inbox consumption and Honesty Gate enforcement."""

    def __init__(
        self,
        *,
        executor: ModelExecutor,
        database: Database,
        message_bus: MessageBus,
    ) -> None:
        self._executor = executor
        self._db = database
        self._bus = message_bus
        self._gate = HonestyGate(database)

    async def synthesize_and_gate(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        tasks: list[TaskSpec],
        all_claims: list[Claim],
        evidence_list: list[Evidence],
        tool_executions: list[ToolExecutionRecord],
        verdicts: list[ReviewVerdict],
        extra_assumptions: list[str],
        extra_missing: list[str],
        uncertainties: list[str],
        is_partial_run: bool,
        run_deadline_monotonic: float | None = None,
    ) -> HonestyGateResult:
        """Consume Lead inbox, run Lead synthesis, and enforce deterministic Honesty Gate."""
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        incomplete_tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]

        latest_by_id: dict[str, Claim] = {}
        for claim in all_claims:
            prev = latest_by_id.get(claim.claim_id)
            if prev is None or claim.version > prev.version:
                latest_by_id[claim.claim_id] = claim
        latest_claims = list(latest_by_id.values())

        # Deliver Lead's inbox messages before building synthesis prompt.
        lead_inbox = await self._bus.read_inbox(
            run_id=run_id,
            recipient_id="lead",
            mark_delivered=True,
        )

        system_prompt = build_synthesis_system_prompt()
        user_prompt = build_synthesis_user_prompt(
            original_prompt=original_prompt,
            plan=plan,
            completed_tasks=completed_tasks,
            incomplete_tasks=incomplete_tasks,
            claims=latest_claims,
            tool_executions=tool_executions,
            lead_inbox=lead_inbox,
        )

        res = await self._executor.execute_structured(
            run_id=run_id,
            actor="lead",
            operation="synthesis",
            prompt=user_prompt,
            response_model=SynthesisOutput,
            system_prompt=system_prompt,
            run_deadline_monotonic=run_deadline_monotonic,
        )
        # Only mark Lead messages consumed once synthesis actually processed them.
        if lead_inbox:
            await self._bus.mark_consumed(
                run_id=run_id,
                recipient_id="lead",
                messages=lead_inbox,
            )

        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="synthesis_completed",
                actor="lead",
                message="Lead synthesis completed; passing output to deterministic Honesty Gate.",
                payload={"lead_messages_consumed": len(lead_inbox)},
            )
        )

        gate_result = self._gate.evaluate(
            run_id=run_id,
            original_prompt=original_prompt,
            plan=plan,
            synthesis=res.parsed,
            all_claims=all_claims,
            evidence_list=evidence_list,
            tool_executions=tool_executions,
            verdicts=verdicts,
            incomplete_tasks=incomplete_tasks,
            extra_assumptions=extra_assumptions,
            extra_missing=extra_missing,
            uncertainties=uncertainties,
            is_partial_run=is_partial_run,
        )
        assert gate_result.gate_record is not None
        await self._gate.persist_decision(gate_result.gate_record)
        return gate_result

    async def gate_partial_fallback(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        tasks: list[TaskSpec],
        all_claims: list[Claim],
        evidence_list: list[Evidence],
        tool_executions: list[ToolExecutionRecord],
        verdicts: list[ReviewVerdict],
        extra_assumptions: list[str],
        extra_missing: list[str],
        uncertainties: list[str],
        reason: str,
    ) -> HonestyGateResult:
        """Apply the exact same Honesty Gate to partial fallback synthesis."""
        incomplete_tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]
        latest_by_id: dict[str, Claim] = {}
        for claim in all_claims:
            prev = latest_by_id.get(claim.claim_id)
            if prev is None or claim.version > prev.version:
                latest_by_id[claim.claim_id] = claim

        partial_output = build_partial_synthesis_output(
            plan=plan,
            claims=list(latest_by_id.values()),
            incomplete_tasks=incomplete_tasks,
            reason=reason,
        )
        gate_result = self._gate.evaluate(
            run_id=run_id,
            original_prompt=original_prompt,
            plan=plan,
            synthesis=partial_output,
            all_claims=all_claims,
            evidence_list=evidence_list,
            tool_executions=tool_executions,
            verdicts=verdicts,
            incomplete_tasks=incomplete_tasks,
            extra_assumptions=extra_assumptions,
            extra_missing=extra_missing,
            uncertainties=uncertainties,
            is_partial_run=True,
        )
        assert gate_result.gate_record is not None
        await self._gate.persist_decision(gate_result.gate_record)
        return gate_result
