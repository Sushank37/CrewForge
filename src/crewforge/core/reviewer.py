"""Independent reviewer pool combining deterministic checks and model-based review."""

from __future__ import annotations

from dataclasses import dataclass, field

from crewforge.config import CrewForgeSettings
from crewforge.core.claim_graph import ClaimDependencyGraph
from crewforge.core.executor import ModelExecutor
from crewforge.core.requirements import claim_addresses_kind, deliverable_lines
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
from crewforge.tools.arithmetic import (
    arithmetic_lineage_issue,
    extract_numbers_from_text,
    provenance_failures,
)

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
        "- For inferences, verify the premises. Treat proposed plan or worker assumptions as "
        "conditional and require primary evidence for empirical values.\n"
        "- For each approved claim, list the exact deliv_N IDs it independently fulfills in "
        "`covered_deliverable_refs`. Review each obligation against the original request. "
        "An accurate but incomplete calculation, comparison, or proof may be approved as a claim "
        "without covering its obligation. A recommendation needs verified premises and a decision "
        "that answers the requested action. Mark an uncertainty as covering an obligation only "
        "when it identifies the missing evidence and the outcome genuinely cannot be determined. "
        "Review calculable conditional cases independently of the unknown actual outcome. "
        "List unfulfilled checklist IDs in `missing_requirement_coverage`.\n"
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
) -> tuple[str, dict[str, tuple[str, int]]]:
    reqs = (
        "\n".join(f"- [req_{idx}] {r}" for idx, r in enumerate(plan.user_requirements, start=1))
        or "- None."
    )
    alias_map: dict[str, tuple[str, int]] = {}
    claim_blocks: list[str] = []
    for idx, claim in enumerate(claims, start=1):
        alias = f"c{idx}"
        alias_map[alias] = (claim.claim_id, claim.version)
        evd_lines: list[str] = []
        seen_evd: set[str] = set()
        for eid in claim.evidence_ids:
            if eid in seen_evd:
                continue
            seen_evd.add(eid)
            evd = evidence_by_id.get(eid)
            if evd is not None:
                evd_lines.append(
                    f"  - {evd.evidence_id} [{evd.source_type}, source_ref={evd.source_ref}, "
                    f"verified={evd.verified_in_input}, reliability={evd.reliability.value}]: "
                    f"{evd.excerpt}"
                )
        tool_lines: list[str] = []
        for tid in claim.tool_execution_ids:
            rec = tool_by_id.get(tid)
            if rec is not None:
                var_bindings: list[str] = []
                for v_name, v_val in rec.variables.items():
                    v_ref = rec.input_provenance_refs.get(v_name, "missing")
                    detail = rec.provenance_details.get(v_name, {})
                    resolved_id = detail.get("resolved_id")
                    src_snippet = ""
                    if isinstance(resolved_id, str):
                        if resolved_id in evidence_by_id:
                            src_snippet = f" -> \"{evidence_by_id[resolved_id].excerpt}\""
                        elif resolved_id in prior_claims_by_id:
                            src_snippet = f" -> \"{prior_claims_by_id[resolved_id].statement}\""
                        elif resolved_id in tool_by_id:
                            t_ref = tool_by_id[resolved_id]
                            src_snippet = f" -> {t_ref.computed_result} {t_ref.units}".strip()
                    var_bindings.append(f"{v_name}={v_val} [{v_ref}{src_snippet}]")
                bindings_str = "; ".join(var_bindings) if var_bindings else "none"
                tool_lines.append(
                    f"  - {rec.tool_execution_id}: expr=`{rec.expression}`, "
                    f"computed={rec.computed_result}, proposed={rec.proposed_result}, "
                    f"units='{rec.units}', matches={rec.matches_proposed}, "
                    f"inputs_grounded={rec.inputs_grounded}, "
                    f"ungrounded_literals={rec.has_ungrounded_literals}, success={rec.success}, "
                    f"inputs=[{bindings_str}]"
                )
        premise_lines = []
        for p in claim.premise_refs:
            p_obj = prior_claims_by_id.get(p.claim_id)
            p_status = (
                p_obj.status.value
                if (p_obj is not None and p_obj.version == p.claim_version)
                else "stale_or_missing"
            )
            premise_lines.append(f"  - {p.claim_id} v{p.claim_version}: {p_status}")
        claim_blocks.append(
            f"Claim {claim.claim_id} (version={claim.version}, alias={alias}, "
            f"category={claim.category.value}, author={claim.author_agent_id}):\n"
            f"  Statement: {claim.statement}\n"
            f"  Requested coverage: {claim.requirement_refs}\n"
            f"  Assumptions: {claim.assumptions or ['None']}\n"
            f"  Evidence:\n" + ("\n".join(evd_lines) if evd_lines else "  - None") + "\n"
            "  ToolExecutions:\n" + ("\n".join(tool_lines) if tool_lines else "  - None") + "\n"
            "  Premises:\n" + ("\n".join(premise_lines) if premise_lines else "  - None")
        )

    prompt = (
        f"Original user prompt:\n{original_prompt}\n\n"
        f"User requirements:\n{reqs}\n\n"
        f"Atomic deliverables:\n{deliverable_lines(plan.deliverables) or 'None listed.'}\n\n"
        "Claims to audit independently (reference alias or claim_id):\n\n"
        + "\n\n".join(claim_blocks)
    )
    return prompt, alias_map


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
        tool_owners: dict[str, Claim] | None = None,
        plan_assumptions: list[str] | None = None,
    ) -> tuple[ReviewVerdict | None, ClaimStatus | None, VerificationStatus | None]:
        """Apply zero-cost deterministic checks prior to LLM review."""
        if not claim.is_material and claim.category not in {
            ClaimCategory.RECOMMENDATION, ClaimCategory.UNCERTAINTY,
        }:
            return None, None, None

        # Check premise integrity and exact version currency first.
        for pref in claim.premise_refs:
            premise = latest_by_id.get(pref.claim_id)
            if (
                premise is None
                or premise.version != pref.claim_version
                or premise.verification_status == VerificationStatus.STALE_PENDING_REVIEW
                or premise.status
                in {
                    ClaimStatus.REJECTED,
                    ClaimStatus.STALE,
                    ClaimStatus.UNRESOLVED,
                }
            ):
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_CONSISTENCY_ID,
                    review_dimension="consistency",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        f"Depends on premise '{pref.claim_id}' v{pref.claim_version} whose "
                        f"current state is '{premise.status.value if premise else 'missing'}'."
                    ),
                    required_correction=(
                        "Revise or re-ground conclusion on current verified premises."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Unverified, stale, or superseded premise dependency"],
                )
                return verdict, ClaimStatus.STALE, VerificationStatus.STALE_PENDING_REVIEW

        if claim.category == ClaimCategory.RECOMMENDATION and not claim.premise_refs:
            return (
                ReviewVerdict(
                    run_id=run_id, claim_id=claim.claim_id, claim_version=claim.version,
                    reviewer_id=REVIEWER_CONSISTENCY_ID, review_dimension="consistency",
                    decision=ReviewDecision.NEEDS_REVISION, blocking_severity="blocking",
                    rationale="A recommendation needs current verified premise claims.",
                ),
                ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED,
            )

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

            ungrounded = next(
                (r for r in records if not r.inputs_grounded or r.has_ungrounded_literals), None
            )
            if ungrounded is not None:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_NUMBERS_ID,
                    review_dimension="numbers",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        "Arithmetic provenance failed. "
                        + (" ".join(provenance_failures(ungrounded)) or
                           "One or more operands lack a valid source.")
                    ),
                    required_correction=(
                        "Rebind each failed operand or derive it from verified inputs."
                    ),
                    blocking_severity="blocking",
                    blocking_issues=["Ungrounded arithmetic inputs or inline constants"],
                )
                return verdict, ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED

            for record in records:
                issue = arithmetic_lineage_issue(
                    record, tool_by_id, latest_by_id, tool_owners or {},
                )
                if issue:
                    verdict = ReviewVerdict(
                        run_id=run_id,
                        claim_id=claim.claim_id,
                        claim_version=claim.version,
                        reviewer_id=REVIEWER_NUMBERS_ID,
                        review_dimension="numbers",
                        decision=ReviewDecision.NEEDS_REVISION,
                        rationale=issue,
                        required_correction="Recompute using the current verified operand sources.",
                        blocking_severity="blocking",
                        blocking_issues=["Stale arithmetic provenance"],
                    )
                    return verdict, ClaimStatus.REJECTED, VerificationStatus.UNSUPPORTED

        if claim.category == ClaimCategory.EXTERNAL_FACT:
            verified_evds = [
                evidence_by_id[eid]
                for eid in claim.evidence_ids
                if eid in evidence_by_id
                and evidence_by_id[eid].verified_in_input
                and evidence_by_id[eid].source_type == "external_retrieved"
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
                and evidence_by_id[eid].source_type
                in {"user_prompt", "external_retrieved"}
            ]
            assumed_numbers = extract_numbers_from_text(
                "\n".join([*(plan_assumptions or []), *claim.assumptions])
            )
            cited_numbers = extract_numbers_from_text("\n".join(e.excerpt for e in verified_evds))
            unsourced_assumptions = (
                extract_numbers_from_text(claim.statement) & assumed_numbers
            ) - cited_numbers
            if not verified_evds or unsourced_assumptions:
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    claim_version=claim.version,
                    reviewer_id=REVIEWER_EVIDENCE_ID,
                    review_dimension="evidence",
                    decision=ReviewDecision.NEEDS_REVISION,
                    rationale=(
                        "Given-fact claim lacks verified provenance from the prompt "
                        "or retrieved documents."
                        + (
                            " The claim uses numbers from an assumption absent from the citation."
                            if unsourced_assumptions else ""
                        )
                    ),
                    evidence_refs=list(claim.evidence_ids),
                    required_correction=(
                        "Attach an accurate excerpt from the user prompt or a retrieved document."
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
                if eid in evidence_by_id
                and evidence_by_id[eid].verified_in_input
                and evidence_by_id[eid].source_type == "external_retrieved"
            ]
            if not verified_evds:
                return ClaimStatus.UNRESOLVED, VerificationStatus.UNSUPPORTED
            return ClaimStatus.SUPPORTED, VerificationStatus.PROVENANCE_VALIDATED

        if claim.category == ClaimCategory.COMPUTATION:
            records = [tool_by_id[tid] for tid in claim.tool_execution_ids if tid in tool_by_id]
            if any(
                r.success
                and r.inputs_grounded
                and not r.has_ungrounded_literals
                and r.matches_proposed is not False
                for r in records
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
        tool_owners = {tid: c for c in trace.claims for tid in c.tool_execution_ids}

        claims_to_review = (
            [latest_by_id[c.claim_id] for c in target_claims if c.claim_id in latest_by_id]
            if target_claims is not None
            else latest_all
        )
        material_claims = [
            c for c in claims_to_review if c.is_material
            or c.category == ClaimCategory.RECOMMENDATION
            or (plan.deliverables and c.category == ClaimCategory.UNCERTAINTY)
        ]
        items = {d.deliverable_id: d for d in plan.deliverables}

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
                tool_owners=tool_owners,
                plan_assumptions=plan.assumptions,
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
                needs_repair = (updated.is_material or updated.requirement_refs) and (
                    verdict.blocking_severity == "blocking"
                    or verdict.decision == ReviewDecision.NEEDS_REVISION
                )
                if needs_repair and next_status == ClaimStatus.REJECTED:
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
        seen_pass_keys: set[tuple[str, int, str, str]] = set()
        expected_dims = {dim for _, dim in passes}
        completed_dims: set[str] = set()

        for reviewer_id, dimension in passes:
            eligible_claims = [c for c in pending_model_review if c.author_agent_id != reviewer_id]
            if not eligible_claims:
                continue

            review_prompt, alias_map = _build_review_user_prompt(
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
                    reserve_calls_after=1,
                    system_prompt=_build_review_system_prompt(dimension),
                    run_deadline_monotonic=run_deadline_monotonic,
                )
                outcome.reviewer_calls += 1
                completed_dims.add(dimension)
            except Exception as exc:
                await self._db.append_event(
                    ExecutionEvent(
                        run_id=run_id,
                        event_type="review_failed",
                        actor=reviewer_id,
                        message=f"Reviewer {reviewer_id} failed dimension '{dimension}': {exc}",
                    )
                )
                continue

            valid_targets = {(c.claim_id, c.version): c for c in eligible_claims}
            reported_gaps = set(res.parsed.missing_requirement_coverage)
            for item in res.parsed.verdicts:
                if item.claim_id in alias_map:
                    resolved_cid, resolved_ver = alias_map[item.claim_id]
                else:
                    resolved_cid = item.claim_id
                    if item.claim_version is not None:
                        resolved_ver = item.claim_version
                    else:
                        matched = latest_by_id.get(resolved_cid)
                        resolved_ver = matched.version if matched is not None else -1

                key = (resolved_cid, resolved_ver)
                target_claim = valid_targets.get(key)
                if target_claim is None:
                    continue

                # Enforce uniqueness: at most 1 verdict per (claim, ver, reviewer, dim).
                dedup_key = (resolved_cid, resolved_ver, reviewer_id, dimension)
                if dedup_key in seen_pass_keys:
                    await self._db.append_event(
                        ExecutionEvent(
                            run_id=run_id,
                            event_type="duplicate_verdict_ignored",
                            actor=reviewer_id,
                            message=(
                                f"Ignored duplicate verdict for {resolved_cid} v{resolved_ver} "
                                f"in dimension '{dimension}'."
                            ),
                        )
                    )
                    continue
                seen_pass_keys.add(dedup_key)

                # Validate cited evidence genuinely belongs to the targeted claim and run.
                claim_evidence_set = set(target_claim.evidence_ids)
                safe_evd_refs = [
                    eid
                    for eid in item.evidence_refs
                    if eid in evidence_by_id and eid in claim_evidence_set
                ]
                severity = item.blocking_severity
                if (
                    item.decision in {ReviewDecision.REJECT, ReviewDecision.NEEDS_REVISION}
                    and severity == "none"
                ):
                    severity = "blocking"
                verdict = ReviewVerdict(
                    run_id=run_id,
                    claim_id=resolved_cid,
                    claim_version=resolved_ver,
                    reviewer_id=reviewer_id,
                    review_dimension=dimension,
                    decision=item.decision,
                    rationale=item.rationale,
                    evidence_refs=safe_evd_refs,
                    covered_deliverable_refs=[
                        ref for ref in dict.fromkeys(item.covered_deliverable_refs)
                        if item.decision == ReviewDecision.APPROVE
                        and ref not in reported_gaps
                        and ref in target_claim.requirement_refs and ref in items
                        and claim_addresses_kind(target_claim, items[ref])
                    ],
                    required_correction=item.required_correction,
                    blocking_severity=severity,
                    blocking_issues=[item.required_correction] if item.required_correction else [],
                )
                await self._db.insert_review_verdict(verdict)
                verdicts_by_claim[key].append(verdict)
                outcome.verdicts.append(verdict)

        # Re-read latest claims so any mid-batch stale invalidations cannot be overwritten.
        fresh_claims_list = await self._db.get_claims_for_run(run_id, latest_only=True)
        fresh_latest_by_id = {c.claim_id: c for c in fresh_claims_list}

        # Finalize claim verification states from collected independent verdicts.
        for claim in pending_model_review:
            fresh_claim = fresh_latest_by_id.get(claim.claim_id)
            if fresh_claim is not None and (
                fresh_claim.version != claim.version
                or fresh_claim.status == ClaimStatus.STALE
                or fresh_claim.verification_status == VerificationStatus.STALE_PENDING_REVIEW
            ):
                outcome.updated_claims[claim.claim_id] = fresh_claim
                continue

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
            approved_dims = {
                v.review_dimension for v in c_verdicts if v.decision == ReviewDecision.APPROVE
            }

            # Recheck premise dependency currency prior to final approval.
            stale_premise_reason: str | None = None
            for pref in claim.premise_refs:
                p_fresh = fresh_latest_by_id.get(pref.claim_id)
                if (
                    p_fresh is None
                    or p_fresh.version != pref.claim_version
                    or p_fresh.status in {ClaimStatus.STALE, ClaimStatus.REJECTED}
                    or p_fresh.verification_status == VerificationStatus.STALE_PENDING_REVIEW
                    or (p_fresh.is_material and p_fresh.status != ClaimStatus.SUPPORTED)
                ):
                    stale_premise_reason = (
                        f"Premise '{pref.claim_id}' v{pref.claim_version} is "
                        f"{p_fresh.status.value if p_fresh else 'missing'}."
                    )
                    break

            if rejections:
                has_actionable_revision = any(
                    v.decision == ReviewDecision.NEEDS_REVISION for v in rejections
                )
                blocking = any(v.blocking_severity == "blocking" for v in rejections) or (
                    (claim.is_material or claim.requirement_refs) and has_actionable_revision
                )
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
                fresh_latest_by_id[claim.claim_id] = updated
                if blocking:
                    outcome.blocking_claims.append(updated)
                await self._graph.invalidate_dependents(
                    run_id=run_id,
                    root_claim_id=claim.claim_id,
                    reason=rejections[0].rationale,
                    superseded_version=claim.version,
                )
            elif stale_premise_reason is not None:
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=claim.claim_id,
                    version=claim.version,
                    status=ClaimStatus.STALE,
                    verification_status=VerificationStatus.STALE_PENDING_REVIEW,
                    revision_notes=stale_premise_reason,
                )
                updated = claim.model_copy(
                    update={
                        "status": ClaimStatus.STALE,
                        "verification_status": VerificationStatus.STALE_PENDING_REVIEW,
                    }
                )
                outcome.updated_claims[claim.claim_id] = updated
                fresh_latest_by_id[claim.claim_id] = updated
            elif unresolved_v or completed_dims != expected_dims or approved_dims != expected_dims:
                missing_dims = sorted(expected_dims - approved_dims)
                reason = (
                    "Incomplete or conflicting review "
                    f"(missing/unapproved dimensions: {missing_dims})."
                )
                await self._mark_unresolved_missing_review(
                    run_id=run_id,
                    claim=claim,
                    reason=reason,
                    outcome=outcome,
                )
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
                fresh_latest_by_id[claim.claim_id] = updated

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
