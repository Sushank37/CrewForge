"""Targeted dispute resolution and claim-version repair engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from crewforge.config import CrewForgeSettings, sanitize_error_message
from crewforge.core.claim_graph import ClaimDependencyGraph
from crewforge.core.executor import ModelExecutor
from crewforge.core.ledger import ClaimLedger, ClaimRevisionError
from crewforge.core.messaging import MessageBus
from crewforge.core.reviewer import ReviewerPool
from crewforge.prompts.worker import build_worker_system_prompt
from crewforge.schemas import (
    AgentSpec,
    Claim,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    ReviewVerdict,
    RevisionOutput,
)
from crewforge.storage.database import CrossRunReferenceError, Database


@dataclass
class RepairRoundOutcome:
    """Summary of targeted specialist repairs and independent re-reviews."""

    revised_claims: list[Claim] = field(default_factory=list)
    invalidated_dependents: list[Claim] = field(default_factory=list)
    repair_calls: int = 0
    re_review_calls: int = 0
    skipped_reason: str | None = None


def _build_revision_prompt(
    *,
    original_prompt: str,
    disputed_claims: list[Claim],
    verdicts_by_claim_key: dict[tuple[str, int], list[ReviewVerdict]],
    evidence_by_id: dict[str, Evidence],
    dependency_findings_text: str,
) -> str:
    blocks: list[str] = []
    for claim in disputed_claims:
        verdicts = verdicts_by_claim_key.get((claim.claim_id, claim.version), [])
        obj_lines = [
            f"  - [{v.reviewer_id}/{v.review_dimension}] {v.decision.value}: {v.rationale}"
            + (f" | Required: {v.required_correction}" if v.required_correction else "")
            for v in verdicts
        ]
        evd_lines = [
            f"  - {eid}: {evidence_by_id[eid].excerpt}"
            for eid in claim.evidence_ids
            if eid in evidence_by_id
        ]
        blocks.append(
            f"Target claim: claim_id={claim.claim_id}, expected_version={claim.version}, "
            f"category={claim.category.value}\n"
            f"Current statement: {claim.statement}\n"
            f"Reviewer objections:\n" + "\n".join(obj_lines) + "\n"
            "Current evidence:\n" + ("\n".join(evd_lines) if evd_lines else "  - None")
        )

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"Relevant dependency context:\n{dependency_findings_text or 'None.'}\n\n"
        "You must address the following blocking reviewer objections. Return one "
        "`TargetedClaimRevision` inside `revisions` for EACH target claim below (do not return "
        "an empty `revisions` list), specifying its exact `target_claim_id` and "
        "`expected_version`. For numeric claims, supply a valid `computation` "
        "(`ArithmeticRequest`) using plain numeric strings without currency symbols.\n\n"
        + "\n\n".join(blocks)
    )


class TargetedRepairEngine:
    """Routes blocking review objections back to originating specialists for targeted repair."""

    def __init__(
        self,
        *,
        executor: ModelExecutor,
        database: Database,
        ledger: ClaimLedger,
        message_bus: MessageBus,
        reviewer_pool: ReviewerPool,
        settings: CrewForgeSettings,
    ) -> None:
        self._executor = executor
        self._db = database
        self._ledger = ledger
        self._bus = message_bus
        self._reviewer = reviewer_pool
        self._settings = settings
        self._graph = ClaimDependencyGraph(database)

    async def resolve_disputes(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        blocking_claims: list[Claim],
        verdicts: list[ReviewVerdict],
        run_deadline_monotonic: float | None = None,
    ) -> RepairRoundOutcome:
        """Execute up to `max_revision_rounds` of targeted specialist repair + re-review."""
        result = RepairRoundOutcome()
        if not blocking_claims or self._settings.max_revision_rounds <= 0:
            return result

        trace = await self._db.get_run_trace(run_id)
        if trace is None:
            return result

        agents_by_id: dict[str, AgentSpec] = {a.agent_id: a for a in plan.agents}
        all_agent_ids = list(agents_by_id.keys())
        evidence_by_id = {e.evidence_id: e for e in trace.evidence}
        tasks_by_id = {t.task_id: t for t in trace.tasks}

        verdicts_by_key: dict[tuple[str, int], list[ReviewVerdict]] = {}
        for v in verdicts:
            verdicts_by_key.setdefault((v.claim_id, v.claim_version), []).append(v)

        # Prioritize claims tied to user requirements first.
        prioritized = sorted(
            blocking_claims,
            key=lambda c: (0 if c.requirement_refs else 1, c.created_at),
        )

        # Group by originating specialist so unaffected specialists are never rerun.
        by_author: dict[str, list[Claim]] = {}
        for claim in prioritized:
            if claim.author_agent_id in agents_by_id:
                by_author.setdefault(claim.author_agent_id, []).append(claim)

        for _round in range(self._settings.max_revision_rounds):
            if not by_author:
                break

            revised_this_round: list[Claim] = []
            for author_id, claims_for_author in by_author.items():
                if (
                    run_deadline_monotonic is not None
                    and (run_deadline_monotonic - time.monotonic()) < 25.0
                ):
                    result.skipped_reason = "Insufficient deadline remaining for targeted repair."
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="optional_stage_skipped",
                            actor="repair_engine",
                            message=result.skipped_reason,
                        )
                    )
                    return result

                summary = await self._db.get_budget_summary(run_id)
                remaining_calls = self._settings.max_model_calls_per_run - int(
                    summary["run_committed_calls"]
                )
                # Need at least 1 call for repair + 1 for re-review + 1 for synthesis = 3.
                if remaining_calls < 3:
                    result.skipped_reason = (
                        "Remaining call budget reserved for re-review and synthesis."
                    )
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="optional_stage_skipped",
                            actor="repair_engine",
                            message=result.skipped_reason,
                        )
                    )
                    return result

                agent = agents_by_id[author_id]
                # Send structured review messages from objecting reviewers via MessageBus.
                sent_review_msgs = []
                for disputed in claims_for_author:
                    c_verdicts = verdicts_by_key.get((disputed.claim_id, disputed.version), [])
                    blocking_v = [
                        v for v in c_verdicts if v.blocking_severity == "blocking"
                    ] or c_verdicts
                    sender_id = blocking_v[0].reviewer_id if blocking_v else "reviewer_combined"
                    msg_content = "; ".join(
                        f"{v.rationale}"
                        + (f" Correction: {v.required_correction}" if v.required_correction else "")
                        for v in blocking_v
                    )
                    sent_msg = await self._bus.send_message(
                        run_id=run_id,
                        sender_id=sender_id,
                        recipient_id=author_id,
                        all_agent_ids=all_agent_ids,
                        message_type="clarification",
                        content=(
                            f"Revision requested for {disputed.claim_id} v{disputed.version}: "
                            f"{msg_content}"
                        ),
                        related_claim_ids=[disputed.claim_id],
                    )
                    sent_review_msgs.append(sent_msg)

                inbound = await self._bus.read_inbox(
                    run_id=run_id,
                    recipient_id=author_id,
                    mark_delivered=True,
                )

                dep_texts: list[str] = []
                for c in claims_for_author:
                    if c.task_id and c.task_id in tasks_by_id:
                        task = tasks_by_id[c.task_id]
                        for dep_id in task.depends_on:
                            dep_task = tasks_by_id.get(dep_id)
                            if dep_task and dep_task.findings:
                                dep_texts.append(dep_task.findings)
                dep_blob = "\n".join(dict.fromkeys(dep_texts))

                revision_prompt = _build_revision_prompt(
                    original_prompt=original_prompt,
                    disputed_claims=claims_for_author,
                    verdicts_by_claim_key=verdicts_by_key,
                    evidence_by_id=evidence_by_id,
                    dependency_findings_text=dep_blob,
                )
                try:
                    res = await self._executor.execute_structured(
                        run_id=run_id,
                        actor=author_id,
                        operation="revision",
                        prompt=revision_prompt,
                        response_model=RevisionOutput,
                        system_prompt=build_worker_system_prompt(agent),
                        run_deadline_monotonic=run_deadline_monotonic,
                    )
                    result.repair_calls += 1
                except Exception as exc:
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="repair_failed",
                            actor=author_id,
                            message=sanitize_error_message(
                                f"Targeted repair failed for {agent.role}: {exc}"
                            ),
                        )
                    )
                    continue

                await self._bus.mark_consumed(
                    run_id=run_id,
                    recipient_id=author_id,
                    messages=inbound,
                )

                if not res.parsed.revisions:
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="revision_empty",
                            actor=author_id,
                            message=f"Specialist {agent.role} returned no targeted revisions.",
                        )
                    )

                new_evd, _, _ = await self._ledger.record_specialist_outputs(
                    run_id=run_id,
                    agent_id=author_id,
                    task_id=claims_for_author[0].task_id or plan.tasks[0].task_id,
                    original_prompt=original_prompt,
                    dependency_findings_text=dep_blob,
                    evidence_proposals=res.parsed.evidence_proposals,
                    claim_proposals=[],
                )
                for e in new_evd:
                    evidence_by_id[e.evidence_id] = e

                for rev in res.parsed.revisions:
                    evd_ids = [
                        new_evd[e_idx].evidence_id
                        for e_idx in rev.evidence_indices
                        if 0 <= e_idx < len(new_evd)
                    ]
                    try:
                        revised_claim = await self._ledger.apply_targeted_revision(
                            run_id=run_id,
                            target_claim_id=rev.target_claim_id,
                            expected_version=rev.expected_version,
                            new_statement=rev.new_statement,
                            author_agent_id=author_id,
                            original_prompt=original_prompt,
                            dependency_findings_text=dep_blob,
                            category=rev.category,
                            additional_evidence_ids=evd_ids,
                            premise_claim_ids=rev.premise_claim_ids or None,
                            computation_request=rev.computation,
                            assumptions=rev.assumptions or None,
                            revision_notes=rev.revision_notes,
                            is_unresolved=rev.is_unresolved,
                            confidence=rev.confidence,
                            require_same_author=True,
                        )
                    except (ClaimRevisionError, CrossRunReferenceError) as exc:
                        await self._db.append_event(
                            ExecutionEvent(
                                run_id=run_id,
                                event_type="revision_rejected",
                                actor=author_id,
                                message=sanitize_error_message(str(exc)),
                            )
                        )
                        continue

                    result.revised_claims.append(revised_claim)
                    revised_this_round.append(revised_claim)

                    # Transitive invalidation of downstream claims referencing prior version.
                    stale = await self._graph.invalidate_dependents(
                        run_id=run_id,
                        root_claim_id=revised_claim.claim_id,
                        reason="Upstream premise revised.",
                        superseded_version=rev.expected_version,
                    )
                    result.invalidated_dependents.extend(stale)

            if not revised_this_round:
                break

            # Independently re-review revised claims (+ any stale dependents if present).
            to_rereview = list(revised_this_round)
            review_outcome = await self._reviewer.review_claims(
                run_id=run_id,
                original_prompt=original_prompt,
                plan=plan,
                target_claims=to_rereview,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            result.re_review_calls += review_outcome.reviewer_calls
            for v in review_outcome.verdicts:
                verdicts_by_key.setdefault((v.claim_id, v.claim_version), []).append(v)

            # Prepare any remaining blocking claims for potential next round.
            by_author = {}
            for claim in review_outcome.blocking_claims:
                if claim.author_agent_id in agents_by_id:
                    by_author.setdefault(claim.author_agent_id, []).append(claim)

        # Ensure any unrevised rejected claims or un-rereviewed stale claims stay non-approved.
        latest_all = await self._db.get_claims_for_run(run_id, latest_only=True)
        for claim in latest_all:
            if claim.status == ClaimStatus.ASSERTED and claim.is_material:
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=ClaimStatus.UNRESOLVED,
                    verification_status=claim.verification_status,
                )

        return result
