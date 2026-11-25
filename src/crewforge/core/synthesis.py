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


def build_deterministic_fast_path_output(
    *,
    plan: ExecutionPlan,
    claims: list[Claim],
) -> SynthesisOutput | None:
    """Build deterministic claim-locked SynthesisOutput when fast-path criteria are satisfied.

    Returns None when model synthesis is needed (e.g., no supported claims or uncovered reqs).
    """
    supported = [c for c in claims if c.is_material and c.status == ClaimStatus.SUPPORTED]
    if not supported:
        return None

    reqs = [r.strip() for r in plan.user_requirements if r.strip()]
    covered_reqs: set[str] = set()
    accepted: list[ReleaseSegment] = []
    for claim in supported:
        covered_reqs.update(claim.requirement_refs)
        accepted.append(
            ReleaseSegment(
                segment_type="accepted_claim",
                text=claim.statement,
                claim_refs=[ClaimVersionRef(claim_id=claim.claim_id, claim_version=claim.version)],
                requirement_ref=claim.requirement_refs[0] if claim.requirement_refs else None,
                is_factual_assertion=True,
            )
        )

    recommendations = [
        ReleaseSegment(
            segment_type="recommendation",
            text=c.statement,
            claim_refs=[ClaimVersionRef(claim_id=c.claim_id, claim_version=c.version)],
            requirement_ref=c.requirement_refs[0] if c.requirement_refs else None,
            is_factual_assertion=False,
        )
        for c in claims
        if c.category == ClaimCategory.RECOMMENDATION and c.status != ClaimStatus.REJECTED
    ]
    for rec in recommendations:
        if rec.requirement_ref:
            covered_reqs.add(rec.requirement_ref)

    all_covered = True
    for idx, req_text in enumerate(reqs, start=1):
        if req_text not in covered_reqs and f"req_{idx}" not in covered_reqs:
            all_covered = False
            break
    if reqs and not all_covered:
        return None

    return SynthesisOutput(
        user_requirements=list(plan.user_requirements),
        accepted_segments=accepted,
        conditional_conclusions=[],
        recommendations=recommendations,
        assumptions=list(plan.assumptions),
        unresolved_issues=[],
        missing_information=list(plan.missing_information),
        new_factual_assertions=[],
        referenced_claim_ids=[c.claim_id for c in supported],
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
        allow_deterministic_fast_path: bool = False,
        run_deadline_monotonic: float | None = None,
    ) -> tuple[HonestyGateResult, str]:
        """Consume Lead inbox, synthesize (or use deterministic fast path), and run Gate."""
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        incomplete_tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]

        latest_by_id: dict[str, Claim] = {}
        for claim in all_claims:
            prev = latest_by_id.get(claim.claim_id)
            if prev is None or claim.version > prev.version:
                latest_by_id[claim.claim_id] = claim
        latest_claims = list(latest_by_id.values())

        # Deliver Lead's inbox messages before building synthesis prompt or checking fast path.
        lead_inbox = await self._bus.read_inbox(
            run_id=run_id,
            recipient_id="lead",
            mark_delivered=True,
        )
        important_inbox = [
            m for m in lead_inbox if m.message_type in {"question", "warning", "clarification"}
        ]

        if (
            allow_deterministic_fast_path
            and not is_partial_run
            and not incomplete_tasks
            and not important_inbox
        ):
            fast_output = build_deterministic_fast_path_output(plan=plan, claims=latest_claims)
            if fast_output is not None:
                if lead_inbox:
                    await self._bus.mark_consumed(
                        run_id=run_id,
                        recipient_id="lead",
                        messages=lead_inbox,
                    )
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="synthesis_fast_path",
                        actor="lead",
                        message=(
                            "Used deterministic claim-locked synthesis fast path without extra "
                            "model call."
                        ),
                        payload={"lead_messages_consumed": len(lead_inbox)},
                    )
                )
                gate_result = self._gate.evaluate(
                    run_id=run_id,
                    original_prompt=original_prompt,
                    plan=plan,
                    synthesis=fast_output,
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
                return gate_result, "deterministic_fast_path"

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
        return gate_result, "model_synthesis"

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
