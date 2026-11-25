"""Independent reviewer pool combining deterministic checks and model-based review."""

from __future__ import annotations

from dataclasses import dataclass, field

from crewforge.config import CrewForgeSettings
from crewforge.core.claim_graph import ClaimDependencyGraph
from crewforge.core.executor import ModelExecutor
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    ReviewDecision,
    ReviewerBatchOutput,
    ReviewerIdentity,
    ReviewVerdict,
    ToolExecutionRecord,
    VerificationStatus,
)
from crewforge.storage.database import Database

REVIEWER_EVIDENCE_ID = "reviewer_evidence"
REVIEWER_NUMBERS_ID = "reviewer_numbers"
REVIEWER_CONSISTENCY_ID = "reviewer_consistency"
REVIEWER_COMBINED_ID = "reviewer_combined"


@dataclass
class ReviewRoundOutcome:
    """Summary of deterministic + model-based review verdicts for a batch of claims."""

    verdicts: list[ReviewVerdict] = field(default_factory=list)
    updated_claims: dict[str, Claim] = field(default_factory=dict)
    blocking_claims: list[Claim] = field(default_factory=list)
    reviewer_calls: int = 0


def _build_review_system_prompt(dimension: str) -> str:
    return (
        f"You are an independent CrewForge Reviewer ({dimension}). "
        "Critically evaluate each claim against the original prompt, user requirements, "
        "provided evidence excerpts, deterministic tool records, and stated assumptions.\n\n"
        "Strict rules:\n"
        "- Do NOT rubber-stamp claims. Reject or request revision whenever evidence is missing, "
        "irrelevant, misquoted, numerically inconsistent, or contradictory.\n"
        "- Verify each claim by exact `claim_id` and `claim_version`.\n"
        "- Return one verdict entry in `verdicts` for EVERY claim listed below.\n"
        "- For numeric claims, verify whether expression and units match the user question "
        "(even when arithmetic succeeded).\n"
        "- For inferences, check that premises support conclusion and assumptions are explicit.\n"
        "- Use `approve`, `reject`, `needs_revision`, or `unresolved` with `blocking_severity` "
        "set to `blocking` when the claim cannot safely be released as stated.\n"
        "- Keep each `rationale` and `required_correction` concise (under 150 characters) "
        "so all verdicts fit cleanly in JSON."
    )


def _build_review_user_prompt(
    *,
    original_prompt: str,
    plan: ExecutionPlan,
    claims: list[Claim],
    evidence_by_id: dict[str, Evidence],
    tool_by_id: dict[str, ToolExecutionRecord],
    prior_claims_by_id: dict[str, Claim],
) -> str:
    reqs = "\n".join(f"- {r}" for r in plan.user_requirements) or "- None."
    claim_blocks: list[str] = []
    for claim in claims:
        evd_lines: list[str] = []
        for eid in claim.evidence_ids:
            evd = evidence_by_id.get(eid)
            if evd is not None:
                evd_lines.append(
                    f"  - {evd.evidence_id} [{evd.source_type}, "
                    f"verified={evd.verified_in_input}, reliability={evd.reliability.value}]: "
                    f"{evd.excerpt}"
                )
        tool_lines: list[str] = []
        for tid in claim.tool_execution_ids:
            rec = tool_by_id.get(tid)
            if rec is not None:
                tool_lines.append(
                    f"  - {rec.tool_execution_id}: expr=`{rec.expression}`, "
                    f"computed={rec.computed_result}, proposed={rec.proposed_result}, "
                    f"units='{rec.units}', matches={rec.matches_proposed}, "
                    f"inputs_grounded={rec.inputs_grounded}, success={rec.success}"
                )
        premise_lines = []
        for p in claim.premise_refs:
            p_status = (
                prior_claims_by_id[p.claim_id].status.value
                if p.claim_id in prior_claims_by_id
                else "unknown"
            )
            premise_lines.append(f"  - {p.claim_id} v{p.claim_version}: {p_status}")
        claim_blocks.append(
            f"Claim {claim.claim_id} (version={claim.version}, category={claim.category.value}, "
            f"author={claim.author_agent_id}):\n"
            f"  Statement: {claim.statement}\n"
            f"  Assumptions: {claim.assumptions or ['None']}\n"
            f"  Evidence:\n" + ("\n".join(evd_lines) if evd_lines else "  - None") + "\n"
            "  ToolExecutions:\n" + ("\n".join(tool_lines) if tool_lines else "  - None") + "\n"
            "  Premises:\n" + ("\n".join(premise_lines) if premise_lines else "  - None")
        )

    return (
        f"Original user prompt:\n{original_prompt}\n\n"
        f"User requirements:\n{reqs}\n\n"
        "Claims to audit independently:\n\n" + "\n\n".join(claim_blocks)
    )


class ReviewerPool:
    """Runs deterministic verification first, then independent batched model review."""

    def __init__(
        self,
        *,
        executor: ModelExecutor,
        database: Database,
        settings: CrewForgeSettings,
    ) -> None:
        self._executor = executor
        self._db = database
        self._settings = settings
        self._graph = ClaimDependencyGraph(database)

    async def register_reviewers(self, run_id: str) -> None:
        """Register standard logical reviewer identities for the run."""
        identities = [
            ReviewerIdentity(
                reviewer_id=REVIEWER_EVIDENCE_ID,
                run_id=run_id,
                role="Evidence Auditor",
                dimensions=["evidence"],
            ),
            ReviewerIdentity(
                reviewer_id=REVIEWER_NUMBERS_ID,
                run_id=run_id,
                role="Number Checker",
                dimensions=["numbers"],
            ),
            ReviewerIdentity(
                reviewer_id=REVIEWER_CONSISTENCY_ID,
                run_id=run_id,
                role="Consistency Reviewer",
                dimensions=["consistency"],
            ),
            ReviewerIdentity(
                reviewer_id=REVIEWER_COMBINED_ID,
                run_id=run_id,
                role="Independent Evidence & Consistency Reviewer",
                dimensions=["combined_evidence_consistency"],
            ),
        ]
        for ident in identities:
            await self._db.register_reviewer(ident)

    async def _deterministic_precheck(
        self,
        *,
        run_id: str,
        claim: Claim,
        evidence_by_id: dict[str, Evidence],
        tool_by_id: dict[str, ToolExecutionRecord],
        latest_by_id: dict[str, Claim],
    ) -> tuple[ReviewVerdict | None, ClaimStatus | None, VerificationStatus | None]:
        """Apply zero-cost deterministic checks prior to LLM review."""
        if not claim.is_material:
            return None, None, None

        # Check premise integrity first.
        for pref in claim.premise_refs:
            premise = latest_by_id.get(pref.claim_id)
            if premise is None or premise.status in {
                ClaimStatus.REJECTED,
                ClaimStatus.STALE,
                ClaimStatus.UNRESOLVED,
            }:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_CONSISTENCY_ID,
                    review_dimension="consistency",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        f"Depends on premise '{pref.claim_id}' whose current state is "
                        f"'{premise.status.value if premise else 'missing'}'."
                    ),
                    required_correction="Revise or re-ground conclusion on verified premises.",
                    blocking_severity="blocking",
                    blocking_issues=["Unverified or rejected premise dependency"],
                )
                return verdict, ClaimStatus.STALE, VerificationStatus.STALE_PENDING_REVIEW

        if claim.category == ClaimCategory.COMPUTATION:
            records = [tool_by_id[tid] for tid in claim.tool_execution_ids if tid in tool_by_id]
            if not records:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_NUMBERS_ID,
                    review_dimension="numbers",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        "Numeric computation claim lacks a deterministic arithmetic tool record."
                    ),
                    required_correction=(
                        "Supply a valid ArithmeticRequest expression and grounded variables."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Missing deterministic tool execution"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED

            bad_rec = next((r for r in records if not r.success), None)
            if bad_rec is not None:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_NUMBERS_ID,
                    review_dimension="numbers",
                    decision=ReviewDecision.REJECT,
                    rationale=f"Deterministic arithmetic tool failed: {bad_rec.error}",
                    required_correction="Provide a valid, safe arithmetic expression.",
                    blocking_severity="blocking",
                    blocking_issues=["Arithmetic evaluation error"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.CONTRADICTED

            mismatch = next((r for r in records if r.matches_proposed is False), None)
            if mismatch is not None:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_NUMBERS_ID,
                    review_dimension="numbers",
                    decision=ReviewDecision.REJECT,
                    rationale=(
                        f"Proposed result '{mismatch.proposed_result}' does not match "
                        f"deterministic computation '{mismatch.computed_result}'."
                    ),
                    required_correction=(
                        f"Correct the calculated value to {mismatch.computed_result}"
                        + (f" {mismatch.units}" if mismatch.units else "")
                        + "."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Arithmetic mismatch"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.CONTRADICTED

            ungrounded = next((r for r in records if not r.inputs_grounded), None)
            if ungrounded is not None:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_NUMBERS_ID,
                    review_dimension="numbers",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        "Arithmetic expression evaluated, but one or more input variables lack "
                        "grounded provenance in the user request or verified dependency findings."
                    ),
                    required_correction=(
                        "Bind variables strictly to grounded input numbers and provenance refs."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Ungrounded arithmetic inputs"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED

        if claim.category == ClaimCategory.EXTERNAL_FACT:
            verified_evds = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id and evidence_by_id[eid].verified_in_input
            ]
            if not verified_evds:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_EVIDENCE_ID,
                    review_dimension="evidence",
                    decision=ReviewDecision.UNRESOLVED,
                    rationale=(
                        "External factual assertion cannot be verified because external "
                        "retrieval is unavailable; citations are unretrieved."
                    ),
                    evidence_refs=list(claim.evidence_ids),
                    blocking_severity="minor",
                    blocking_issues=["Unretrieved external citation"],
                )
                return verdict, ClaimStatus.UNRESOLVED, VerificationStatus.UNSUPPORTED

        if claim.category == ClaimCategory.GIVEN_FACT:
            verified_evds = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id
                and evidence_by_id[eid].verified_in_input
                and evidence_by_id[eid].source_type in {"user_prompt", "dependency_finding"}
            ]
            if not verified_evds:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_EVIDENCE_ID,
                    review_dimension="evidence",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        "Given-fact claim lacks verbatim verified provenance from prompt or deps."
                    ),
                    evidence_refs=list(claim.evidence_ids),
                    required_correction=(
                        "Attach an accurate excerpt from user_prompt or dependency_finding."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Missing verified provenance"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED

        return None, None, None

    @staticmethod
    def _resolve_approved_status(
        claim: Claim,
        evidence_by_id: dict[str, Evidence],
        tool_by_id: dict[str, ToolExecutionRecord],
    ) -> tuple[ClaimStatus, VerificationStatus]:
        if claim.category == ClaimCategory.EXTERNAL_FACT:
            # Never promote unretrieved external facts to SUPPORTED.
            verified_evds = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id and evidence_by_id[eid].verified_in_input
            ]
            if not verified_evds:
                return ClaimStatus.UNRESOLVED, VerificationStatus.UNSUPPORTED
            return ClaimStatus.SUPPORTED, VerificationStatus.PROVENANCE_VALIDATED

        if claim.category == ClaimCategory.COMPUTATION:
            records = [tool_by_id[tid] for tid in claim.tool_execution_ids if tid in tool_by_id]
            if any(
                r.success and r.inputs_grounded and r.matches_proposed is not False for r in records
            ):
                return ClaimStatus.SUPPORTED, VerificationStatus.COMPUTATIONALLY_VERIFIED
            return ClaimStatus.UNRESOLVED, VerificationStatus.UNSUPPORTED

        if claim.category == ClaimCategory.GIVEN_FACT:
            verified_evds = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id and evidence_by_id[eid].verified_in_input
            ]
            if verified_evds:
                return ClaimStatus.SUPPORTED, VerificationStatus.PROVENANCE_VALIDATED
            return ClaimStatus.UNRESOLVED, VerificationStatus.UNSUPPORTED

        return ClaimStatus.SUPPORTED, VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE

    async def review_claims(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        target_claims: list[Claim] | None = None,
        run_deadline_monotonic: float | None = None,
    ) -> ReviewRoundOutcome:
        """Review latest material claims deterministically and via independent model review."""
        await self.register_reviewers(run_id)
        trace = await self._db.get_run_trace(run_id)
        if trace is None:
            return ReviewRoundOutcome()

        latest_all = await self._db.get_claims_for_run(run_id, latest_only=True)
        latest_by_id = {c.claim_id: c for c in latest_all}
        evidence_by_id = {e.evidence_id: e for e in trace.evidence}
        tool_by_id = {t.tool_execution_id: t for t in trace.tool_executions}

        claims_to_review = (
            [latest_by_id[c.claim_id] for c in target_claims if c.claim_id in latest_by_id]
            if target_claims is not None
            else latest_all
        )
        material_claims = [c for c in claims_to_review if c.is_material]

        outcome = ReviewRoundOutcome(updated_claims=dict(latest_by_id))
        if not material_claims:
            return outcome

        pending_model_review: list[Claim] = []

        for claim in material_claims:
            verdict, next_status, next_ver_status = await self._deterministic_precheck(
                run_id=run_id,
                claim=claim,
                evidence_by_id=evidence_by_id,
                tool_by_id=tool_by_id,
                latest_by_id=latest_by_id,
            )
            if verdict is not None and next_status is not None and next_ver_status is not None:
                await self._db.insert_review_verdict(verdict)
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=next_status,
                    verification_status=next_ver_status,
                )
                updated = claim.model_copy(
                    update={"status": next_status, "verification_status": next_ver_status}
                )
                outcome.verdicts.append(verdict)
                outcome.updated_claims[claim.claim_id] = updated
                latest_by_id[claim.claim_id] = updated
                if verdict.blocking_severity == "blocking" and next_status == ClaimStatus.REJECTED:
                    outcome.blocking_claims.append(updated)
                    await self._graph.invalidate_dependents(
                        run_id=run_id,
                        root_claim_id=claim.claim_id,
                        reason=verdict.rationale,
                        superseded_version=claim.version,
                    )
            else:
                pending_model_review.append(claim)

        if not pending_model_review:
            return outcome

        # Determine reviewer passes (1 combined pass in compact mode, or 2 if budget allows).
        summary = await self._db.get_budget_summary(run_id)
        remaining_calls = self._settings.max_model_calls_per_run - int(
            summary["run_committed_calls"]
        )
        # Keep at least 1 call reserved for Lead synthesis.
        if remaining_calls <= 1:
            for claim in pending_model_review:
                await self._mark_unresolved_missing_review(
                    run_id=run_id,
                    claim=claim,
                    reason="Review skipped: remaining call budget reserved for synthesis.",
                    outcome=outcome,
                )
            return outcome

        passes: list[tuple[str, str]]
        if self._settings.separate_review_dimensions and remaining_calls >= 3:
            passes = [
                (REVIEWER_EVIDENCE_ID, "evidence"),
                (REVIEWER_CONSISTENCY_ID, "consistency"),
            ]
        else:
            passes = [(REVIEWER_COMBINED_ID, "combined_evidence_consistency")]

        verdicts_by_claim: dict[tuple[str, int], list[ReviewVerdict]] = {
            (c.claim_id, c.version): [] for c in pending_model_review
        }

        for reviewer_id, dimension in passes:
            eligible_claims = [c for c in pending_model_review if c.author_agent_id != reviewer_id]
            if not eligible_claims:
                continue

            review_prompt = _build_review_user_prompt(
                original_prompt=original_prompt,
                plan=plan,
                claims=eligible_claims,
                evidence_by_id=evidence_by_id,
                tool_by_id=tool_by_id,
                prior_claims_by_id=latest_by_id,
            )
            try:
                res = await self._executor.execute_structured(
                    run_id=run_id,
                    actor=reviewer_id,
                    operation="review",
                    prompt=review_prompt,
                    response_model=ReviewerBatchOutput,
                    system_prompt=_build_review_system_prompt(dimension),
                    run_deadline_monotonic=run_deadline_monotonic,
                )
                outcome.reviewer_calls += 1
            except Exception as exc:
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="review_failed",
                        actor=reviewer_id,
                        message=f"Reviewer {reviewer_id} failed: {exc}",
                    )
                )
                continue

            valid_targets = {(c.claim_id, c.version): c for c in eligible_claims}
            for item in res.parsed.verdicts:
                key = (item.claim_id, item.claim_version)
                if key not in valid_targets:
                    continue
                safe_evd_refs = [eid for eid in item.evidence_refs if eid in evidence_by_id]
                severity = item.blocking_severity
                if (
                    item.decision in {ReviewDecision.REJECT, ReviewDecision.NEEDS_REVISION}
                    and severity == "none"
                ):
                    severity = "blocking"
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=item.claim_id,
                    claim_version=item.claim_version,
                    reviewer_id=reviewer_id,
                    review_dimension=dimension,
                    decision=item.decision,
                    rationale=item.rationale,
                    evidence_refs=safe_evd_refs,
                    required_correction=item.required_correction,
                    blocking_severity=severity,
                    blocking_issues=[item.required_correction] if item.required_correction else [],
                )
                await self._db.insert_review_verdict(verdict)
                verdicts_by_claim[key].append(verdict)
                outcome.verdicts.append(verdict)

        # Finalize claim verification states from collected independent verdicts.
        for claim in pending_model_review:
            c_verdicts = verdicts_by_claim.get((claim.claim_id, claim.version), [])
            if not c_verdicts:
                await self._mark_unresolved_missing_review(
                    run_id=run_id,
                    claim=claim,
                    reason="Missing independent reviewer verdict for this claim version.",
                    outcome=outcome,
                )
                continue

            rejections = [
                v
                for v in c_verdicts
                if v.decision in {ReviewDecision.REJECT, ReviewDecision.NEEDS_REVISION}
            ]
            unresolved_v = [v for v in c_verdicts if v.decision == ReviewDecision.UNRESOLVED]
            approvals = [v for v in c_verdicts if v.decision == ReviewDecision.APPROVE]

            if rejections:
                blocking = any(v.blocking_severity == "blocking" for v in rejections)
                next_status = ClaimStatus.REJECTED
                next_ver = (
                    VerificationStatus.CONTRADICTED
                    if any(v.decision == ReviewDecision.REJECT for v in rejections)
                    else VerificationStatus.UNSUPPORTED
                )
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=next_status,
                    verification_status=next_ver,
                )
                updated = claim.model_copy(
                    update={"status": next_status, "verification_status": next_ver}
                )
                outcome.updated_claims[claim.claim_id] = updated
                if blocking:
                    outcome.blocking_claims.append(updated)
                await self._graph.invalidate_dependents(
                    run_id=run_id,
                    root_claim_id=claim.claim_id,
                    reason=rejections[0].rationale,
                    superseded_version=claim.version,
                )
            elif unresolved_v or len(approvals) < len(passes):
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=ClaimStatus.UNRESOLVED,
                    verification_status=VerificationStatus.UNSUPPORTED,
                )
                updated = claim.model_copy(
                    update={
                        "status": ClaimStatus.UNRESOLVED,
                        "verification_status": VerificationStatus.UNSUPPORTED,
                    }
                )
                outcome.updated_claims[claim.claim_id] = updated
            else:
                next_status, next_ver = self._resolve_approved_status(
                    claim, evidence_by_id, tool_by_id
                )
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=next_status,
                    verification_status=next_ver,
                )
                updated = claim.model_copy(
                    update={"status": next_status, "verification_status": next_ver}
                )
                outcome.updated_claims[claim.claim_id] = updated

        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="review_completed",
                actor="reviewer_pool",
                message=(
                    f"Completed review round: {len(outcome.verdicts)} verdict(s), "
                    f"{len(outcome.blocking_claims)} blocking claim(s)."
                ),
                payload={
                    "reviewer_calls": outcome.reviewer_calls,
                    "blocking_claim_ids": [c.claim_id for c in outcome.blocking_claims],
                },
            )
        )
        return outcome

    async def _mark_unresolved_missing_review(
        self,
        *,
        run_id: str,
        claim: Claim,
        reason: str,
        outcome: ReviewRoundOutcome,
    ) -> None:
        await self._db.update_claim_verification_state(
            run_id=run_id,
            claim_id=claim.claim_id,
            version=claim.version,
            status=ClaimStatus.UNRESOLVED,
            verification_status=VerificationStatus.UNSUPPORTED,
            revision_notes=reason,
        )
        outcome.updated_claims[claim.claim_id] = claim.model_copy(
            update={
                "status": ClaimStatus.UNRESOLVED,
                "verification_status": VerificationStatus.UNSUPPORTED,
                "revision_notes": reason,
            }
        )
