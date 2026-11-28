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
    ArithmeticRequest,
    ArithmeticVariableBinding,
    Claim,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    ReviewVerdict,
    RevisionOutput,
    ToolExecutionRecord,
    VerificationStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database
from crewforge.tools.arithmetic import (
    ProvenanceCatalog,
    build_provenance_catalog,
    provenance_failures,
)


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
    provenance_catalog: ProvenanceCatalog | None = None,
    tool_by_id: dict[str, ToolExecutionRecord] | None = None,
    claims_by_id: dict[str, Claim] | None = None,
) -> str:
    blocks: list[str] = []
    tools_map = tool_by_id or {}
    claims_map = claims_by_id or {}
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
        tool_lines: list[str] = []
        for tid in claim.tool_execution_ids:
            rec = tools_map.get(tid)
            if rec is None:
                continue
            var_parts: list[str] = []
            for v_name, v_val in rec.variables.items():
                v_ref = rec.input_provenance_refs.get(v_name, "missing")
                detail = rec.provenance_details.get(v_name, {})
                resolved_id = detail.get("resolved_id")
                src_snippet = ""
                if isinstance(resolved_id, str):
                    if resolved_id in evidence_by_id:
                        src_snippet = f" -> \"{evidence_by_id[resolved_id].excerpt}\""
                    elif resolved_id in claims_map:
                        src_snippet = f" -> \"{claims_map[resolved_id].statement}\""
                var_parts.append(f"{v_name}={v_val} [{v_ref}{src_snippet}]")
            bindings_str = "; ".join(var_parts) if var_parts else "none"
            tool_lines.append(
                f"  - {rec.tool_execution_id}: expr=`{rec.expression}`, "
                f"computed={rec.computed_result} {rec.units}, "
                f"inputs_grounded={rec.inputs_grounded}, success={rec.success}, "
                f"inputs=[{bindings_str}]"
                + (f" | Tool error: {rec.error}" if rec.error else "")
            )
            tool_lines.extend(f"    - {failure}" for failure in provenance_failures(rec))
        blocks.append(
            f"Target claim: claim_id={claim.claim_id}, expected_version={claim.version}, "
            f"category={claim.category.value}\n"
            f"Current statement: {claim.statement}\n"
            f"Reviewer objections:\n" + "\n".join(obj_lines) + "\n"
            "Current evidence:\n" + ("\n".join(evd_lines) if evd_lines else "  - None") + "\n"
            "Existing deterministic calculations:\n"
            + ("\n".join(tool_lines) if tool_lines else "  - None")
        )

    prov_lines = provenance_catalog.prompt_summary_lines() if provenance_catalog else []
    prov_section = (
        "Valid provenance references and source passages:\n" + "\n".join(prov_lines) + "\n\n"
        if prov_lines
        else ""
    )

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"Relevant dependency context:\n{dependency_findings_text or 'None.'}\n\n"
        f"{prov_section}"
        "You must address the following blocking reviewer objections. Return one "
        "`TargetedClaimRevision` inside `revisions` for EACH target claim below (do not return "
        "an empty `revisions` list), specifying its exact `target_claim_id` and "
        "`expected_version`. If a claim already has a valid, grounded ToolExecution and the "
        "objection only concerns citations or wording, set `computation` to null to preserve the "
        "existing verified calculation unless recalculation is necessary. When supplying a new\n"
        "`computation` (`ArithmeticRequest`), use plain numeric strings without currency "
        "symbols. Use each failed operand's diagnostic and a verbatim source_quote when the "
        "alias contains multiple quantities. Derive any subtotal, difference, complement, or "
        "rate inside the formula or cite the current verified prior calculation. Include operand "
        "units on bindings and the output unit on the computation. Recompute stale calculations "
        "from current source values.\n\n"
        + "\n\n".join(blocks)
    )


def _rebind_calculation(
    claim: Claim,
    tools: dict[str, ToolExecutionRecord],
    owners: dict[str, Claim],
    latest: dict[str, Claim],
) -> ArithmeticRequest | None:
    if len(claim.tool_execution_ids) != 1:
        return None
    record = tools.get(claim.tool_execution_ids[0])
    if record is None:
        return None

    invalid = record.provenance_details.get("__claim_premises__", {}).get("values", [])
    bound_refs = {
        d.get("ref") for d in record.provenance_details.values()
        if d.get("source_tool_execution_id") in owners
    }
    if any(ref not in bound_refs and ref not in claim.depends_on_claim_ids for ref in invalid):
        return None  # An unresolvable declared premise needs an explicit specialist correction.

    rebound = False
    bindings = []
    for name, value in record.variables.items():
        ref = record.input_provenance_refs.get(name, "")
        detail = record.provenance_details.get(name, {})
        quote = detail.get("source_quote")
        resolved_id = detail.get("resolved_id", "")
        owner = owners.get(detail.get("source_tool_execution_id") or resolved_id)
        if detail.get("kind") == "supported_claim":
            premise_id = resolved_id.partition("@v")[0]
        else:
            premise_id = owner.claim_id if owner else ""
        if premise := latest.get(premise_id):
            ref = f"{premise.claim_id}@v{premise.version}"
            quote = None
            rebound = True

        units = detail.get("units", "")
        if premise_id and not units:
            units = detail.get("source_units", "")
            if units == "%" and detail.get("value") != detail.get("source_value"):
                units = "ratio"
        bindings.append(ArithmeticVariableBinding(
            name=name, value=value, provenance_ref=ref, source_quote=quote, units=units,
        ))

    if not rebound:
        return None
    # Reuse existing operand values and the proposed result as checks. A changed
    # premise must fail grounding and return to targeted repair with its new value.
    return ArithmeticRequest(
        expression=record.expression,
        variable_bindings=bindings,
        proposed_result=record.proposed_result,
        units=record.units,
        assumptions=record.assumptions,
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
        reserve_gap_calls: int = 0,
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
        pending_blocking: list[Claim] = sorted(
            blocking_claims,
            key=lambda c: (0 if c.requirement_refs else 1, c.created_at),
        )
        repair_attempts: dict[str, int] = {}
        max_waves = max(
            self._settings.max_revision_rounds * 2,
            self._settings.max_revision_rounds + 2,
        )

        for _wave in range(max_waves):
            if not pending_blocking:
                break

            # Dependency-aware ordering: repair upstream claims before claims that depend on them.
            pending_ids = {c.claim_id for c in pending_blocking}
            ready_claims: list[Claim] = []
            deferred_claims: list[Claim] = []
            for claim in pending_blocking:
                dep_ids = {p.claim_id for p in claim.premise_refs} | set(claim.depends_on_claim_ids)
                if dep_ids & (pending_ids - {claim.claim_id}):
                    deferred_claims.append(claim)
                else:
                    ready_claims.append(claim)
            if not ready_claims:
                ready_claims = pending_blocking
                deferred_claims = []

            # Group by originating specialist so unaffected specialists are never rerun.
            by_author: dict[str, list[Claim]] = {}
            for claim in ready_claims:
                if claim.author_agent_id in agents_by_id:
                    by_author.setdefault(claim.author_agent_id, []).append(claim)
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
                    break

                summary = await self._db.get_budget_summary(run_id)
                remaining_calls = self._settings.max_model_calls_per_run - int(
                    summary["run_committed_calls"]
                )
                # Need at least 1 call for repair + 1 for re-review + 1 for synthesis = 3.
                if remaining_calls < 3 + reserve_gap_calls:
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
                    break

                # Refresh ledger and provenance catalog before each repair batch.
                fresh_trace = await self._db.get_run_trace(run_id)
                if fresh_trace is None:
                    continue
                evidence_by_id = {e.evidence_id: e for e in fresh_trace.evidence}
                tasks_by_id = {t.task_id: t for t in fresh_trace.tasks}
                tool_by_id = {t.tool_execution_id: t for t in fresh_trace.tool_executions}
                fresh_latest: dict[str, Claim] = {}
                for fc in fresh_trace.claims:
                    prev = fresh_latest.get(fc.claim_id)
                    if prev is None or fc.version > prev.version:
                        fresh_latest[fc.claim_id] = fc
                claims_for_author = [
                    current for c in claims_for_author
                    if (current := fresh_latest.get(c.claim_id)) and current.status in {
                        ClaimStatus.REJECTED, ClaimStatus.STALE, ClaimStatus.ASSERTED,
                    }
                ]
                if not claims_for_author:
                    continue
                for c in claims_for_author:
                    repair_attempts[c.claim_id] = repair_attempts.get(c.claim_id, 0) + 1

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

                repair_catalog = build_provenance_catalog(
                    original_prompt=original_prompt,
                    verified_evidence=[e for e in fresh_trace.evidence if e.verified_in_input],
                    all_claims=fresh_trace.claims,
                    tool_records=fresh_trace.tool_executions,
                    assumptions=plan.assumptions,
                    run_id=run_id,
                )
                revision_prompt = _build_revision_prompt(
                    original_prompt=original_prompt,
                    disputed_claims=claims_for_author,
                    verdicts_by_claim_key=verdicts_by_key,
                    evidence_by_id=evidence_by_id,
                    dependency_findings_text=dep_blob,
                    provenance_catalog=repair_catalog,
                    tool_by_id=tool_by_id,
                    claims_by_id=fresh_latest,
                )
                try:
                    res = await self._executor.execute_structured(
                        run_id=run_id,
                        actor=author_id,
                        operation="revision",
                        prompt=revision_prompt,
                        response_model=RevisionOutput,
                        reserve_calls_after=2 + reserve_gap_calls,
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

                allowed_targets = {(c.claim_id, c.version) for c in claims_for_author}
                for rev in res.parsed.revisions:
                    if (rev.target_claim_id, rev.expected_version) not in allowed_targets:
                        await self._db.append_event(ExecutionEvent(
                            run_id=run_id, actor=author_id, event_type="revision_rejected",
                            message="Revision targeted a claim outside this repair batch.",
                        ))
                        continue
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
                            plan_assumptions=plan.assumptions,
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

            # Independently re-review revised claims, then revalidate any stale dependents
            # whose upstream premises are now all supported.
            to_rereview = list(revised_this_round)
            review_outcome = await self._reviewer.review_claims(
                run_id=run_id,
                original_prompt=original_prompt,
                plan=plan,
                target_claims=to_rereview, reserve_calls_after=1 + reserve_gap_calls,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            result.re_review_calls += review_outcome.reviewer_calls
            for v in review_outcome.verdicts:
                verdicts_by_key.setdefault((v.claim_id, v.claim_version), []).append(v)

            dep_calls, dep_verdicts, blocking_from_deps = await self._revalidate_stale_dependents(
                run_id=run_id,
                original_prompt=original_prompt,
                plan=plan,
                verdicts_by_key=verdicts_by_key, reserve_calls_after=1 + reserve_gap_calls,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            result.re_review_calls += dep_calls

            # Prepare any remaining blocking or deferred claims for potential next wave.
            next_pending_map: dict[str, Claim] = {}
            latest_after_wave = {
                c.claim_id: c
                for c in await self._db.get_claims_for_run(run_id, latest_only=True)
            }
            for claim in [*review_outcome.blocking_claims, *blocking_from_deps, *deferred_claims]:
                latest_c = latest_after_wave.get(claim.claim_id, claim)
                if (
                    latest_c.status in {ClaimStatus.REJECTED, ClaimStatus.ASSERTED}
                    and repair_attempts.get(latest_c.claim_id, 0)
                    < self._settings.max_revision_rounds
                ):
                    next_pending_map[latest_c.claim_id] = latest_c
            pending_blocking = list(next_pending_map.values())

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

    async def _revalidate_stale_dependents(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        verdicts_by_key: dict[tuple[str, int], list[ReviewVerdict]],
        reserve_calls_after: int = 1,
        run_deadline_monotonic: float | None,
    ) -> tuple[int, list[ReviewVerdict], list[Claim]]:
        """Re-submit stale dependent claims once all upstream premises are supported."""
        total_calls = 0
        all_verdicts: list[ReviewVerdict] = []
        all_blocking: list[Claim] = []
        # Each wave promotes at least one new version or stops. The review reservation
        # bounds depth while permitting long unchanged-value dependency chains.
        attempted: set[tuple[str, tuple[int, ...]]] = set()
        for _pass in range(self._settings.max_model_calls_per_run):
            summary = await self._db.get_budget_summary(run_id)
            remaining = self._settings.max_model_calls_per_run - summary["run_committed_calls"]
            if remaining <= reserve_calls_after:
                break
            latest_all = await self._db.get_claims_for_run(run_id, latest_only=True)
            latest_by_id = {c.claim_id: c for c in latest_all}
            ready: list[Claim] = []
            trace = await self._db.get_run_trace(run_id)
            if trace is None:
                break
            tools = {t.tool_execution_id: t for t in trace.tool_executions}
            owners = {tid: c for c in trace.claims for tid in c.tool_execution_ids}

            for claim in latest_all:
                if not (
                    claim.status == ClaimStatus.STALE
                    or claim.verification_status == VerificationStatus.STALE_PENDING_REVIEW
                ):
                    continue
                up_ids = sorted(
                    {p.claim_id for p in claim.premise_refs} | set(claim.depends_on_claim_ids)
                )
                if not up_ids:
                    continue
                all_supported = all(
                    (up := latest_by_id.get(uid)) is not None
                    and up.status == ClaimStatus.SUPPORTED
                    and up.verification_status != VerificationStatus.STALE_PENDING_REVIEW
                    for uid in up_ids
                )
                if not all_supported:
                    continue
                key = (claim.claim_id, tuple(latest_by_id[uid].version for uid in up_ids))
                if key in attempted:
                    continue
                attempted.add(key)

                new_claim = await self._ledger.apply_targeted_revision(
                    run_id=run_id,
                    target_claim_id=claim.claim_id,
                    expected_version=claim.version,
                    new_statement=claim.statement,
                    author_agent_id=claim.author_agent_id,
                    original_prompt=original_prompt,
                    premise_claim_ids=up_ids,
                    computation_request=_rebind_calculation(claim, tools, owners, latest_by_id),
                    plan_assumptions=plan.assumptions,
                    revision_notes="Revalidate dependent claim using current premise versions.",
                )
                ready.append(new_claim)

            if not ready:
                break

            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="claims_revalidation_started",
                    actor="claim_graph",
                    message=f"Queued {len(ready)} stale dependent claim(s) for revalidation.",
                    payload={"claim_ids": [c.claim_id for c in ready]},
                )
            )
            sub_outcome = await self._reviewer.review_claims(
                run_id=run_id,
                original_prompt=original_prompt,
                plan=plan,
                target_claims=ready, reserve_calls_after=reserve_calls_after,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            total_calls += sub_outcome.reviewer_calls
            all_verdicts.extend(sub_outcome.verdicts)
            all_blocking.extend(sub_outcome.blocking_claims)
            for v in sub_outcome.verdicts:
                verdicts_by_key.setdefault((v.claim_id, v.claim_version), []).append(v)

        return total_calls, all_verdicts, all_blocking
