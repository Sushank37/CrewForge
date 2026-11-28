"""Deterministic Python Honesty Gate with claim-locked release and honest coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from typing import Any, Literal

from crewforge.core.requirements import claim_addresses_kind, reviewed_deliverables
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    HonestyGateRecord,
    ReleaseSegment,
    RequirementCoverageItem,
    ReviewDecision,
    ReviewVerdict,
    SynthesisOutput,
    TaskSpec,
    ToolExecutionRecord,
    VerificationStatus,
    VerificationSummary,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import (
    arithmetic_lineage_issue,
    extract_numbers_from_text,
    make_tool_target,
)

_APPROVED_VERIFICATION_STATES = frozenset(
    {
        VerificationStatus.PROVENANCE_VALIDATED,
        VerificationStatus.COMPUTATIONALLY_VERIFIED,
        VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
    }
)

_ASSERTIVE_LAUNDERING_RE = re.compile(
    r"\b(?:proves|proved|guarantees|guaranteed|definitely|certainly|caused|causes|"
    r"always|never fails|100%\s+safe|100%\s+effective)\b",
    re.IGNORECASE,
)


def _canonical_norm(text: str) -> str:
    """Normalize trivial whitespace/trailing punctuation without dropping negation or words."""
    return re.sub(r"\s+", " ", text.strip()).rstrip(".!;:, ").strip().lower()


def _split_clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.;])\s+|\n+", text) if part.strip()]


def _claim_locked_match(segment_text: str, claims: list[Claim]) -> bool:
    """Enforce verbatim claim-locked alignment (zero semantic paraphrasing).

    Returns True iff `segment_text` is empty (caller renders directly from `claims`)
    or every non-empty clause in `segment_text` matches an exact referenced ledger claim
    statement (canonical case/whitespace-normalized exact match).
    """
    if not claims:
        return False
    cleaned = segment_text.strip()
    if not cleaned:
        return True

    claim_norms = {_canonical_norm(c.statement) for c in claims if c.statement.strip()}
    if _canonical_norm(cleaned) in claim_norms:
        return True

    joined_norm = _canonical_norm(" ".join(c.statement.strip() for c in claims))
    if _canonical_norm(cleaned) == joined_norm:
        return True

    clauses = [_canonical_norm(c) for c in _split_clauses(cleaned)]
    return bool(clauses) and all(cl in claim_norms for cl in clauses)


def _content_word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4}


def _detect_auxiliary_laundering(
    text: str,
    *,
    allowed_numbers: set[Any],
    rejected_claims: list[Claim],
) -> str | None:
    """Reject unverified numbers, assertive causal claims, or rejected-claim echoes."""
    nums = extract_numbers_from_text(text)
    extra_nums = [n for n in nums if n not in allowed_numbers]
    if extra_nums:
        return f"Auxiliary field introduces unverified number(s): {extra_nums[:3]}."

    norm_text = _canonical_norm(text)
    if not norm_text:
        return None
    for rc in rejected_claims:
        rc_norm = _canonical_norm(rc.statement)
        if rc_norm and (rc_norm in norm_text or norm_text in rc_norm):
            return f"Auxiliary field echoes rejected claim '{rc.claim_id}'."
        rc_words = _content_word_set(rc.statement)
        txt_words = _content_word_set(text)
        if len(rc_words) >= 4 and len(rc_words & txt_words) / len(rc_words) >= 0.75:
            return f"Auxiliary field closely mirrors rejected claim '{rc.claim_id}'."

    if _ASSERTIVE_LAUNDERING_RE.search(text):
        return "Auxiliary field contains unsupported assertive/causal phrasing."

    return None


@dataclass
class HonestyGateResult:
    """Validated release content, verification summary, and persisted gate record."""

    rendered_answer: str
    conversational_answer: str = ""
    referenced_claim_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    confidence_notes: str = ""
    verification: VerificationSummary = field(default_factory=VerificationSummary)
    gate_record: HonestyGateRecord | None = None


class HonestyGate:
    """Enforces claim-locked deterministic verification and release filtering in Python."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def evaluate(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        synthesis: SynthesisOutput,
        all_claims: list[Claim],
        evidence_list: list[Evidence],
        tool_executions: list[ToolExecutionRecord],
        verdicts: list[ReviewVerdict],
        incomplete_tasks: list[TaskSpec],
        extra_assumptions: list[str],
        extra_missing: list[str],
        uncertainties: list[str],
        is_partial_run: bool,
    ) -> HonestyGateResult:
        """Audit structured synthesis against persisted claims, tools, and verdicts."""
        all_claims = [c for c in all_claims if c.run_id == run_id]
        tool_executions = [t for t in tool_executions if t.run_id == run_id]
        verdicts = [v for v in verdicts if v.run_id == run_id]
        evidence_list = [e for e in evidence_list if e.run_id == run_id]
        latest_by_id: dict[str, Claim] = {}
        for claim in all_claims:
            prev = latest_by_id.get(claim.claim_id)
            if prev is None or claim.version > prev.version:
                latest_by_id[claim.claim_id] = claim

        claims_by_key = {(c.claim_id, c.version): c for c in all_claims}
        evidence_by_id = {e.evidence_id: e for e in evidence_list}
        tool_by_id = {t.tool_execution_id: t for t in tool_executions}
        verdicts_by_key: dict[tuple[str, int], list[ReviewVerdict]] = {}
        for v in verdicts:
            verdicts_by_key.setdefault((v.claim_id, v.claim_version), []).append(v)

        latest_claims = list(latest_by_id.values())
        material_claims = [c for c in latest_claims if c.is_material]
        supported_claims = [c for c in material_claims if c.status == ClaimStatus.SUPPORTED]
        unresolved_claims = [
            c for c in material_claims if c.status in {ClaimStatus.UNRESOLVED, ClaimStatus.ASSERTED}
        ]
        rejected_claims = [c for c in material_claims if c.status == ClaimStatus.REJECTED]
        stale_claims = [c for c in material_claims if c.status == ClaimStatus.STALE]
        revision_count = sum(1 for c in all_claims if c.version > 1)

        evidence_types: dict[str, int] = {}
        for evd in evidence_list:
            evidence_types[evd.source_type] = evidence_types.get(evd.source_type, 0) + 1

        reviewed_material = sum(
            1 for claim in material_claims if verdicts_by_key.get((claim.claim_id, claim.version))
        )
        review_completeness: float | None = (
            round(reviewed_material / len(material_claims), 4) if material_claims else None
        )

        rejected_segments: list[dict[str, Any]] = []
        rejected_claim_refs: list[dict[str, Any]] = []
        gate_reasons: list[str] = []

        approved_accepted: list[tuple[ReleaseSegment, list[Claim]]] = []
        approved_conditional: list[tuple[ReleaseSegment, list[Claim]]] = []
        approved_recommendations: list[tuple[ReleaseSegment, list[Claim]]] = []
        used_claim_ids: list[str] = []

        tool_owners = {tid: c for c in all_claims for tid in c.tool_execution_ids}
        current_calculations = {
            tid: tool_by_id[tid]
            for claim in latest_claims
            if claim.status == ClaimStatus.SUPPORTED
            and claim.verification_status in _APPROVED_VERIFICATION_STATES
            and (reviews := verdicts_by_key.get((claim.claim_id, claim.version)))
            and all(v.decision == ReviewDecision.APPROVE
                    and v.reviewer_id != claim.author_agent_id for v in reviews)
            for tid in claim.tool_execution_ids
            if tool_owners.get(tid) == claim
            and tid in tool_by_id and make_tool_target(tool_by_id[tid]).is_trusted
            and not arithmetic_lineage_issue(
                tool_by_id[tid], tool_by_id, latest_by_id, tool_owners,
                require_reviewed_sources=True,
            )
        }

        def _arithmetic_premises_current(claim: Claim) -> bool:
            """Check versioned premises, including arithmetic behind intermediate claims."""
            pending = [claim]
            graph = TopologicalSorter()
            seen: set[tuple[str, int]] = set()
            while pending:
                current = pending.pop()
                key = (current.claim_id, current.version)
                if key in seen:
                    continue
                seen.add(key)
                if current.category != ClaimCategory.COMPUTATION and any(
                    (tool := tool_by_id.get(tid)) is None or not make_tool_target(tool).is_trusted
                    or arithmetic_lineage_issue(tool, tool_by_id, latest_by_id, tool_owners)
                    for tid in current.tool_execution_ids
                ):
                    return False
                pinned = {ref.claim_id: ref.claim_version for ref in current.premise_refs}
                parents = set(current.depends_on_claim_ids) | pinned.keys()
                graph.add(key[0], *parents)
                for premise_id in parents:
                    premise = latest_by_id.get(premise_id)
                    if (
                        premise is None or premise.version != pinned.get(premise_id, 1)
                        or premise.status in {ClaimStatus.STALE, ClaimStatus.REJECTED}
                        or premise.verification_status == VerificationStatus.STALE_PENDING_REVIEW
                    ):
                        return False
                    if premise.is_material:
                        reviews = verdicts_by_key.get((premise.claim_id, premise.version), [])
                        if premise.status != ClaimStatus.SUPPORTED or not reviews or any(
                            v.reviewer_id == premise.author_agent_id
                            or v.decision != ReviewDecision.APPROVE for v in reviews
                        ):
                            return False
                    if premise.category == ClaimCategory.COMPUTATION and not any(
                        tid in current_calculations for tid in premise.tool_execution_ids
                    ):
                        return False
                    pending.append(premise)
            try:
                graph.prepare()
            except CycleError:
                return False
            return True

        allowed_numbers = extract_numbers_from_text(original_prompt)
        for evd in evidence_list:
            if evd.verified_in_input and evd.source_type in {"user_prompt", "external_retrieved"}:
                allowed_numbers.update(extract_numbers_from_text(evd.excerpt))
        for tid, rec in current_calculations.items():
            if tool_owners[tid].category == ClaimCategory.COMPUTATION:
                allowed_numbers.update(extract_numbers_from_text(rec.computed_result or ""))
        for claim in supported_claims:
            if claim.category != ClaimCategory.RECOMMENDATION and (
                _arithmetic_premises_current(claim)
            ) and (
                claim.category != ClaimCategory.COMPUTATION or any(
                    tid in current_calculations for tid in claim.tool_execution_ids
                )
            ):
                allowed_numbers.update(extract_numbers_from_text(claim.statement))

        covered_by_review = reviewed_deliverables(plan, all_claims, verdicts)
        qualified_claims = {
            (claim.claim_id, claim.version)
            for item in plan.deliverables
            for claim in covered_by_review.get(item.deliverable_id, [])
            if item.kind == "conditional" or claim.category == ClaimCategory.UNCERTAINTY
        }

        def _validate_claim_ref(
            cid: str,
            cver: int,
            *,
            require_supported: bool,
            expected_segment_type: str | None = None,
        ) -> tuple[Claim | None, str | None]:
            if (cid, cver) not in claims_by_key:
                return None, f"Unknown or cross-run claim reference '{cid}' v{cver}."
            latest = latest_by_id.get(cid)
            if latest is None or latest.version != cver:
                ver_label = latest.version if latest else "?"
                return None, f"Stale claim reference '{cid}' v{cver} (latest is v{ver_label})."
            if latest.status == ClaimStatus.STALE or (
                latest.verification_status == VerificationStatus.STALE_PENDING_REVIEW
            ):
                return None, f"Claim '{cid}' v{cver} is stale pending re-review."
            if latest.status == ClaimStatus.REJECTED:
                return None, f"Claim '{cid}' v{cver} was rejected."

            for pref in latest.premise_refs:
                p_latest = latest_by_id.get(pref.claim_id)
                if (
                    p_latest is None
                    or p_latest.version != pref.claim_version
                    or p_latest.status in {ClaimStatus.REJECTED, ClaimStatus.STALE}
                ):
                    return None, (
                        f"Claim '{cid}' depends on unverified/stale premise '{pref.claim_id}'."
                    )
                if (
                    require_supported
                    and (
                        p_latest.category == ClaimCategory.ASSUMPTION
                        or (p_latest.is_material and p_latest.status != ClaimStatus.SUPPORTED)
                    )
                ):
                    return None, (
                        f"Claim '{cid}' depends on unsupported material premise '{pref.claim_id}'."
                    )
                if (
                    expected_segment_type == "recommendation"
                    and p_latest.is_material
                    and p_latest.status != ClaimStatus.SUPPORTED
                ):
                    return None, (
                        f"Recommendation '{cid}' depends on unsupported "
                        f"material premise '{pref.claim_id}'."
                    )

            if not _arithmetic_premises_current(latest):
                return None, f"Claim '{cid}' depends on stale or invalidated arithmetic lineage."

            if expected_segment_type == "recommendation":
                if latest.category != ClaimCategory.RECOMMENDATION:
                    return None, f"Unsupported factual claim '{cid}' cannot be a recommendation."
                if latest.status != ClaimStatus.SUPPORTED or not latest.premise_refs:
                    return None, f"Recommendation '{cid}' lacks review or verified premises."
                if any(
                    not latest_by_id[pref.claim_id].is_material
                    or latest_by_id[pref.claim_id].status != ClaimStatus.SUPPORTED
                    for pref in latest.premise_refs
                ):
                    return None, f"Recommendation '{cid}' relies on an unverified premise."
                rec_nums = extract_numbers_from_text(latest.statement)
                local_numbers = set()
                pending = [latest]
                seen = set()
                while pending:
                    source = pending.pop()
                    if source.claim_id in seen:
                        continue
                    seen.add(source.claim_id)
                    if source is not latest:
                        local_numbers.update(extract_numbers_from_text(source.statement))
                    for eid in source.evidence_ids:
                        evd = evidence_by_id.get(eid)
                        if evd and evd.verified_in_input and evd.source_type in {
                            "user_prompt", "external_retrieved",
                        }:
                            local_numbers.update(extract_numbers_from_text(evd.excerpt))
                    for tid in source.tool_execution_ids:
                        if record := current_calculations.get(tid):
                            local_numbers.update(extract_numbers_from_text(
                                record.computed_result or "",
                            ))
                            for binding in record.provenance_details.values():
                                if binding.get("grounded"):
                                    local_numbers.update(extract_numbers_from_text(
                                        binding.get("value", ""),
                                    ))
                    pending.extend(latest_by_id[p.claim_id] for p in source.premise_refs)
                extra_rec_nums = sorted(rec_nums - local_numbers)
                if extra_rec_nums:
                    return (
                        None,
                        f"Recommendation '{cid}' contains unverified numbers: "
                        f"{extra_rec_nums[:2]}.",
                    )

            if expected_segment_type == "conditional_conclusion":
                if plan.deliverables and (cid, cver) not in qualified_claims:
                    return None, f"Conditional claim '{cid}' lacks reviewed qualification."
                if (
                    latest.category == ClaimCategory.COMPUTATION
                    and not any(tid in current_calculations for tid in latest.tool_execution_ids)
                ):
                    return None, f"Conditional calculation '{cid}' lacks current provenance."
                factual_cats = {
                    ClaimCategory.GIVEN_FACT,
                    ClaimCategory.COMPUTATION,
                    ClaimCategory.EXTERNAL_FACT,
                }
                if (
                    latest.category in factual_cats
                    and latest.status != ClaimStatus.SUPPORTED
                    and not latest.assumptions
                ):
                    return (
                        None,
                        f"Unverified factual claim '{cid}' cannot be released "
                        "without explicit conditional assumptions.",
                    )
                cond_nums = extract_numbers_from_text(latest.statement)
                extra_cond_nums = [n for n in cond_nums if n not in allowed_numbers]
                if extra_cond_nums:
                    return (
                        None,
                        f"Conditional conclusion '{cid}' contains unverified "
                        f"number(s): {extra_cond_nums[:2]}.",
                    )

            if require_supported or expected_segment_type == "recommendation":
                if require_supported and (
                    not latest.is_material or latest.category in {
                        ClaimCategory.ASSUMPTION, ClaimCategory.RECOMMENDATION,
                    }
                ):
                    return None, f"Non-factual claim '{cid}' cannot be released as accepted fact."
                if latest.status != ClaimStatus.SUPPORTED:
                    return (
                        None,
                        f"Claim '{cid}' v{cver} has status '{latest.status.value}', not supported.",
                    )
                if latest.verification_status not in _APPROVED_VERIFICATION_STATES:
                    return None, (
                        f"Claim '{cid}' v{cver} verification_status "
                        f"'{latest.verification_status.value}' is not approved."
                    )
                c_verdicts = verdicts_by_key.get((cid, cver), [])
                if any(
                    v.decision
                    in {
                        ReviewDecision.REJECT,
                        ReviewDecision.NEEDS_REVISION,
                        ReviewDecision.UNRESOLVED,
                    }
                    for v in c_verdicts
                ):
                    return (
                        None,
                        f"Claim '{cid}' v{cver} has unresolved/conflicting reviewer verdicts.",
                    )
                if not any(v.decision == ReviewDecision.APPROVE for v in c_verdicts):
                    return None, f"Claim '{cid}' v{cver} lacks an independent approval verdict."
                if any(v.reviewer_id == latest.author_agent_id for v in c_verdicts):
                    return None, f"Claim '{cid}' v{cver} has an invalid self-review verdict."
                if latest.category == ClaimCategory.COMPUTATION and not any(
                    tid in current_calculations for tid in latest.tool_execution_ids
                ):
                    return (
                        None,
                        f"Numeric claim '{cid}' lacks grounded arithmetic verification.",
                    )
                if latest.category == ClaimCategory.EXTERNAL_FACT and not any(
                    eid in evidence_by_id
                    and evidence_by_id[eid].verified_in_input
                    and evidence_by_id[eid].source_type == "external_retrieved"
                    for eid in latest.evidence_ids
                ):
                    return None, f"External factual claim '{cid}' lacks retrieved evidence."
            return latest, None

        # Scenario claims keep their qualification even if synthesis puts them among facts.
        factual_segments = []
        conditional_segments = list(synthesis.conditional_conclusions)
        for seg in synthesis.accepted_segments:
            if seg.claim_refs and all(
                (ref.claim_id, ref.claim_version) in qualified_claims for ref in seg.claim_refs
            ):
                conditional_segments.append(seg)
            else:
                factual_segments.append(seg)

        # 1. Claim-locked accepted factual segments.
        for seg in factual_segments:
            if not seg.claim_refs:
                reason = "Accepted factual segment rejected: missing claim version reference."
                rejected_segments.append({"text": seg.text, "reason": reason})
                gate_reasons.append(reason)
                continue

            valid_claims: list[Claim] = []
            seg_error: str | None = None
            for ref in seg.claim_refs:
                claim_obj, err = _validate_claim_ref(
                    ref.claim_id,
                    ref.claim_version,
                    require_supported=True,
                )
                if err is not None:
                    seg_error = err
                    rejected_claim_refs.append(
                        {
                            "claim_id": ref.claim_id,
                            "claim_version": ref.claim_version,
                            "reason": err,
                        }
                    )
                    break
                assert claim_obj is not None
                valid_claims.append(claim_obj)

            if seg_error is None and not _claim_locked_match(seg.text, valid_claims):
                seg_error = (
                    "Claim-locked release rejected rewritten factual text; altered claims must "
                    "become a new claim version and pass independent review."
                )

            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                locked_text = " ".join(c.statement.strip() for c in valid_claims)
                locked_seg = seg.model_copy(update={"text": locked_text})
                approved_accepted.append((locked_seg, valid_claims))
                for c in valid_claims:
                    if c.claim_id not in used_claim_ids:
                        used_claim_ids.append(c.claim_id)

        # 2. Claim-locked conditional conclusions (block missing refs, rewrites, rejected, stale).
        for seg in conditional_segments:
            if not seg.claim_refs:
                reason = (
                    "Conditional segment rejected: requires a verified or qualified ledger ref."
                )
                rejected_segments.append({"text": seg.text, "reason": reason})
                gate_reasons.append(reason)
                continue

            valid_claims = []
            seg_error = None
            for ref in seg.claim_refs:
                c_obj, err = _validate_claim_ref(
                    ref.claim_id,
                    ref.claim_version,
                    require_supported=False,
                    expected_segment_type="conditional_conclusion",
                )
                if err is not None:
                    seg_error = err
                    rejected_claim_refs.append(
                        {
                            "claim_id": ref.claim_id,
                            "claim_version": ref.claim_version,
                            "reason": err,
                        }
                    )
                    break
                assert c_obj is not None
                valid_claims.append(c_obj)

            if seg_error is None and not _claim_locked_match(seg.text, valid_claims):
                seg_error = "Conditional segment rejected: text alters the referenced ledger claim."

            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                locked_text = " ".join(c.statement.strip() for c in valid_claims)
                locked_seg = seg.model_copy(update={"text": locked_text})
                approved_conditional.append((locked_seg, valid_claims))
                for c in valid_claims:
                    if c.claim_id not in used_claim_ids:
                        used_claim_ids.append(c.claim_id)

        # 3. Recommendations: require ledger claim refs (advisory framing, no factual laundering).
        for seg in synthesis.recommendations:
            if not seg.claim_refs:
                reason = (
                    "Recommendation rejected: must reference a reviewed or proposed ledger claim."
                )
                rejected_segments.append({"text": seg.text, "reason": reason})
                gate_reasons.append(reason)
                continue

            valid_claims = []
            seg_error = None
            for ref in seg.claim_refs:
                c_obj, err = _validate_claim_ref(
                    ref.claim_id,
                    ref.claim_version,
                    require_supported=False,
                    expected_segment_type="recommendation",
                )
                if err is not None:
                    seg_error = err
                    rejected_claim_refs.append(
                        {
                            "claim_id": ref.claim_id,
                            "claim_version": ref.claim_version,
                            "reason": err,
                        }
                    )
                    break
                assert c_obj is not None
                valid_claims.append(c_obj)

            if seg_error is None and not _claim_locked_match(seg.text, valid_claims):
                seg_error = (
                    "Recommendation rejected: text diverges from referenced ledger claim statement."
                )

            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                locked_text = " ".join(c.statement.strip() for c in valid_claims)
                locked_seg = seg.model_copy(update={"text": locked_text})
                approved_recommendations.append((locked_seg, valid_claims))
                for c in valid_claims:
                    if c.claim_id not in used_claim_ids:
                        used_claim_ids.append(c.claim_id)

        # Fill only unrepresented, independently endorsed deliverables. Every recovered
        # segment must pass the same claim reference, premise and numeric checks.
        for item in plan.deliverables:
            candidates = covered_by_review.get(item.deliverable_id, [])
            if any(claim.claim_id in used_claim_ids for claim in candidates):
                continue
            for claim in candidates:
                if claim.category == ClaimCategory.RECOMMENDATION:
                    lane, destination = "recommendation", approved_recommendations
                elif (claim.claim_id, claim.version) in qualified_claims:
                    lane, destination = "conditional_conclusion", approved_conditional
                else:
                    lane, destination = "accepted_claim", approved_accepted
                _, err = _validate_claim_ref(
                    claim.claim_id, claim.version, require_supported=lane == "accepted_claim",
                    expected_segment_type=None if lane == "accepted_claim" else lane,
                )
                if err is not None:
                    continue
                destination.append((ReleaseSegment(
                    segment_type=lane, requirement_ref=item.requirement_ref,
                    claim_refs=[ClaimVersionRef(
                        claim_id=claim.claim_id, claim_version=claim.version,
                    )],
                ).model_copy(update={"text": claim.statement}), [claim]))
                used_claim_ids.append(claim.claim_id)
                break

        # When endorsement is missing, retain a bounded, task-linked factual finding.
        # Coverage still comes exclusively from reviewed deliverable endorsements.
        shown_requirements: set[str] = set()
        task_refs = {task.task_id: task.task_ref for task in plan.tasks}
        qualified_items = {
            d.deliverable_id for d in plan.deliverables
            if d.kind in {"conditional", "recommendation"}
        }
        for item in plan.deliverables:
            if (
                item.requirement_ref in shown_requirements
                or item.kind in {"conditional", "recommendation"}
                or any(
                    c.claim_id in used_claim_ids
                    for c in covered_by_review.get(item.deliverable_id, [])
                )
            ):
                continue
            candidates = [
                c for c in latest_claims
                if item.deliverable_id in c.requirement_refs
                and task_refs.get(c.task_id) == item.task_ref
                and c.is_material and claim_addresses_kind(c, item)
                and (c.claim_id, c.version) not in qualified_claims
                and not qualified_items.intersection(c.requirement_refs)
            ]
            if any(c.claim_id in used_claim_ids for c in candidates):
                shown_requirements.add(item.requirement_ref)
                continue
            for claim in candidates:
                _, err = _validate_claim_ref(claim.claim_id, claim.version, require_supported=True)
                if err is not None:
                    continue
                approved_accepted.append((ReleaseSegment(
                    segment_type="accepted_claim", text=claim.statement,
                    claim_refs=[ClaimVersionRef(
                        claim_id=claim.claim_id, claim_version=claim.version,
                    )],
                ).model_copy(update={"text": claim.statement}), [claim]))
                used_claim_ids.append(claim.claim_id)
                shown_requirements.add(item.requirement_ref)
                break

        # 4. Block new Lead factual assertions & strip laundering from auxiliary synthesis fields.
        unresolved_items: list[str] = []
        laundering_detected = False

        for new_fact in synthesis.new_factual_assertions:
            cleaned = new_fact.strip()
            if cleaned and cleaned.lower() != "none":
                unresolved_items.append(
                    "Unreviewed synthesis assertion withheld from verified release."
                )
                gate_reasons.append(
                    "Stripped unreviewed Lead factual assertion from verified output."
                )
                laundering_detected = True

        # Reject/strip ungrounded auxiliary assertions across assumptions, missing_info, unresolved.
        clean_assumptions: list[str] = []
        for candidate in [*plan.assumptions, *synthesis.assumptions, *extra_assumptions]:
            item = candidate.strip()
            if not item:
                continue
            err = _detect_auxiliary_laundering(
                item, allowed_numbers=allowed_numbers, rejected_claims=rejected_claims
            )
            if err is not None:
                gate_reasons.append(f"Filtered assumption: {err}")
                laundering_detected = True
                continue
            if item not in clean_assumptions:
                clean_assumptions.append(item)

        clean_missing: list[str] = []
        for candidate in [
            *plan.missing_information,
            *synthesis.missing_information,
            *extra_missing,
        ]:
            item = candidate.strip()
            if not item:
                continue
            err = _detect_auxiliary_laundering(
                item, allowed_numbers=allowed_numbers, rejected_claims=rejected_claims
            )
            if err is not None:
                gate_reasons.append(f"Filtered missing-info item: {err}")
                laundering_detected = True
                continue
            if item not in clean_missing:
                clean_missing.append(item)

        for candidate in synthesis.unresolved_issues:
            item = candidate.strip()
            if not item:
                continue
            err = _detect_auxiliary_laundering(
                item, allowed_numbers=allowed_numbers, rejected_claims=rejected_claims
            )
            if err is not None:
                gate_reasons.append(f"Filtered unresolved-issue prose: {err}")
                laundering_detected = True
                continue
            unresolved_items.append(item)

        # Validate raw referenced_claim_ids for cross-run/stale leaks.
        for raw_cid in synthesis.referenced_claim_ids:
            if raw_cid not in latest_by_id:
                rejected_claim_refs.append(
                    {
                        "claim_id": raw_cid,
                        "claim_version": None,
                        "reason": "Unknown or cross-run claim ID in referenced_claim_ids.",
                    }
                )
                gate_reasons.append(f"Removed unknown/cross-run claim reference '{raw_cid}'.")

        # Surface unresolved / stale notices and withhold rejected claim statements cleanly.
        for claim in unresolved_claims:
            unresolved_items.append(
                f"Unresolved ({claim.claim_id} v{claim.version}): {claim.statement}"
            )
        for claim in stale_claims:
            unresolved_items.append(
                f"Stale pending re-review ({claim.claim_id} v{claim.version}) withheld."
            )
        for claim in rejected_claims:
            unresolved_items.append(
                f"Rejected claim ({claim.claim_id} v{claim.version}) withheld from verified output."
            )

        dedup_unresolved = list(dict.fromkeys(u.strip() for u in unresolved_items if u.strip()))

        # 5. Honest requirement linkage metrics (Task 4).
        req_items: list[RequirementCoverageItem] = []
        answered_count = 0
        partial_count = 0
        unresolved_req_count = 0
        missing_req_count = 0
        req_score = 0.0

        reqs = [r.strip() for r in plan.user_requirements if r.strip()]
        # Index released claim IDs per requirement_ref across audited segments & released claims.
        req_to_accepted: dict[str, list[Claim]] = {}
        req_to_partial: dict[str, list[Claim]] = {}
        for seg, claims in approved_accepted:
            targets = list(seg.requirement_ref and [seg.requirement_ref] or [])
            for c in claims:
                targets.extend(c.requirement_refs)
            for t in dict.fromkeys(targets):
                req_to_accepted.setdefault(t, []).extend(claims)
        for seg, claims in [*approved_conditional, *approved_recommendations]:
            targets = list(seg.requirement_ref and [seg.requirement_ref] or [])
            for c in claims:
                targets.extend(c.requirement_refs)
            for t in dict.fromkeys(targets):
                req_to_partial.setdefault(t, []).extend(claims)

        for idx, req_text in enumerate(reqs, start=1):
            req_id = f"req_{idx}"
            items = [d for d in plan.deliverables if d.requirement_ref == req_id]
            if items:
                fulfilled = {
                    d.deliverable_id: [
                        c for c in covered_by_review.get(d.deliverable_id, [])
                        if c.claim_id in used_claim_ids
                    ]
                    for d in items
                }
                outstanding = [d for d in items if not fulfilled[d.deliverable_id]]
                linked = [
                    c for c in latest_claims
                    if any(d.deliverable_id in c.requirement_refs for d in items)
                ]
                released = {
                    c.claim_id: c for cs in fulfilled.values() for c in cs
                }
                if not outstanding:
                    status = "answered"
                    answered_count += 1
                    qualification = [
                        d.text for d in items
                        if all(
                            c.category == ClaimCategory.UNCERTAINTY
                            for c in fulfilled[d.deliverable_id]
                        )
                    ]
                    rationale = (
                        "Reviewed evidence limits apply to " + ", ".join(qualification)
                        if qualification
                        else "All deliverables have independently reviewed answers."
                    )
                elif released:
                    status = "partially_answered"
                    partial_count += 1
                    rationale = "Outstanding deliverables include " + ", ".join(
                        d.text for d in outstanding
                    )
                elif linked:
                    status = "unresolved"
                    unresolved_req_count += 1
                    rationale = "No reviewed answer was released for " + ", ".join(
                        d.text for d in outstanding
                    )
                else:
                    status = "missing"
                    missing_req_count += 1
                    rationale = "No claim addresses " + ", ".join(d.text for d in outstanding)
                req_score += (len(items) - len(outstanding)) / len(items)
                req_items.append(RequirementCoverageItem(
                    requirement_id=req_id, text=req_text, status=status, rationale=rationale,
                    verified_deliverable_refs=[ref for ref, claims in fulfilled.items() if claims],
                    linked_claim_refs=[
                        ClaimVersionRef(claim_id=c.claim_id, claim_version=c.version)
                        for c in (list(released.values()) or linked)
                    ],
                ))
                continue

            acc_claims = list(
                {
                    c.claim_id: c
                    for c in [*req_to_accepted.get(req_id, []), *req_to_accepted.get(req_text, [])]
                }.values()
            )
            part_claims = list(
                {
                    c.claim_id: c
                    for c in [*req_to_partial.get(req_id, []), *req_to_partial.get(req_text, [])]
                }.values()
            )
            blocked_claims = [
                c
                for c in [*rejected_claims, *stale_claims, *unresolved_claims]
                if req_id in c.requirement_refs or req_text in c.requirement_refs
            ]
            if acc_claims and not blocked_claims:
                answered_count += 1
                req_score += 1.0
                req_items.append(
                    RequirementCoverageItem(
                        requirement_id=req_id,
                        text=req_text,
                        status="answered",
                        linked_claim_refs=[
                            {"claim_id": c.claim_id, "claim_version": c.version} for c in acc_claims
                        ],
                        rationale="Linked to verified accepted claim version(s).",
                    )
                )
            elif acc_claims and blocked_claims:
                partial_count += 1
                req_score += 0.5
                req_items.append(
                    RequirementCoverageItem(
                        requirement_id=req_id,
                        text=req_text,
                        status="partially_answered",
                        linked_claim_refs=[
                            {"claim_id": c.claim_id, "claim_version": c.version}
                            for c in [*acc_claims, *blocked_claims]
                        ],
                        rationale=(
                            "Partially answered: verified claims cover part of this requirement, "
                            "while other linked claims remain unresolved, stale, or rejected."
                        ),
                    )
                )
            elif part_claims:
                partial_count += 1
                req_score += 0.5
                req_items.append(
                    RequirementCoverageItem(
                        requirement_id=req_id,
                        text=req_text,
                        status="partially_answered",
                        linked_claim_refs=[
                            {"claim_id": c.claim_id, "claim_version": c.version}
                            for c in part_claims
                        ],
                        rationale="Addressed conditionally or via reviewed advisory action only.",
                    )
                )
            else:
                if blocked_claims:
                    unresolved_req_count += 1
                    req_items.append(
                        RequirementCoverageItem(
                            requirement_id=req_id,
                            text=req_text,
                            status="unresolved",
                            linked_claim_refs=[
                                {"claim_id": c.claim_id, "claim_version": c.version}
                                for c in blocked_claims
                            ],
                            rationale="Linked claims remain unresolved, stale, or rejected.",
                        )
                    )
                else:
                    missing_req_count += 1
                    req_items.append(
                        RequirementCoverageItem(
                            requirement_id=req_id,
                            text=req_text,
                            status="missing",
                            linked_claim_refs=[],
                            rationale="No reviewed claim version linked to this requirement.",
                        )
                    )

        requirement_coverage: float | None = round(req_score / len(reqs), 4) if reqs else None

        answer_complete = (
            all(item.status == "answered" for item in req_items) if plan.deliverables else None
        )
        if plan.deliverables:
            for item in req_items:
                if item.status != "answered":
                    missing_notice = f"{item.requirement_id} ({item.status}): {item.rationale}"
                    clean_missing.append(missing_notice)
                    dedup_unresolved.append(missing_notice)

        if is_partial_run or incomplete_tasks:
            for item in req_items:
                if item.status == "missing":
                    omitted_notice = f"Omitted requirement ({item.requirement_id}): {item.text}"
                    if omitted_notice not in clean_missing:
                        clean_missing.append(omitted_notice)
                    if omitted_notice not in dedup_unresolved:
                        dedup_unresolved.append(omitted_notice)

        # 6. Application-controlled deterministic rendering from exact ledger claims.
        sections: list[str] = []
        if is_partial_run:
            sections.append(
                "**Partial Result:** Execution did not complete all stages. "
                "Only verified findings and explicitly qualified gaps are shown."
            )

        if approved_recommendations:
            lines = ["**Proposed Actions (Advisory):**"]
            for seg, claims in approved_recommendations:
                refs = ", ".join(f"{c.claim_id}@v{c.version}" for c in claims)
                lines.append(f"- {seg.text} [advisory; {refs}]")
            sections.append("\n".join(lines))

        clean_points: list[str] = []
        if approved_accepted:
            seen_norms = {
                _canonical_norm(member.statement)
                for _, members in approved_conditional for member in members
            }
            for _, members in approved_accepted:
                for member in members:
                    parts = [latest_by_id[p.claim_id].statement for p in member.premise_refs]
                    texts = parts if (
                        member.author_agent_id == "lead" and parts
                        and member.statement == " ".join(parts)
                    ) else [member.statement]
                    for text in texts:
                        norm = _canonical_norm(text)
                        if norm and norm not in seen_norms:
                            seen_norms.add(norm)
                            text_out = text.strip()
                            if not text_out.endswith((".", "!", "?")):
                                text_out += "."
                            clean_points.append(text_out)
            if len(clean_points) == 1:
                sections.append(clean_points[0])
            else:
                sections.append("\n".join(f"- {pt}" for pt in clean_points))

        if approved_conditional:
            lines = ["**Conditional / Qualified Statements:**"]
            for seg, claims in approved_conditional:
                refs = ", ".join(f"{c.claim_id}@v{c.version} ({c.status.value})" for c in claims)
                cond_assumps = [a for c in claims for a in c.assumptions if a]
                assump_suffix = f" (assumes: {', '.join(cond_assumps)})" if cond_assumps else ""
                lines.append(f"- {seg.text}{assump_suffix} [{refs}]")
            sections.append("\n".join(lines))

        if plan.deliverables:
            workings = []
            # Include the calculation behind a released decision even when its premise
            # is not a separate requested deliverable. Walk only current reviewed ancestry.
            working_ids = list(used_claim_ids)
            seen_workings = set(working_ids)
            for cid in working_ids:
                for ref in latest_by_id[cid].premise_refs:
                    premise = latest_by_id.get(ref.claim_id)
                    if (
                        premise and premise.version == ref.claim_version
                        and premise.claim_id not in seen_workings
                        and premise.status == ClaimStatus.SUPPORTED
                    ):
                        _, err = _validate_claim_ref(
                            premise.claim_id, premise.version, require_supported=True,
                        )
                        if err is None:
                            seen_workings.add(premise.claim_id)
                            working_ids.append(premise.claim_id)
            for cid in working_ids:
                claim = latest_by_id[cid]
                citations = list(dict.fromkeys(
                    evidence_by_id[eid].source_ref or eid for eid in claim.evidence_ids
                    if eid in evidence_by_id and evidence_by_id[eid].verified_in_input
                    and evidence_by_id[eid].source_type in {"user_prompt", "external_retrieved"}
                ))
                for tid in claim.tool_execution_ids:
                    rec = current_calculations.get(tid)
                    if rec is None:
                        continue
                    bindings = ", ".join(
                        f"{name}={detail['value']} [{detail['resolved_id']}]"
                        for name, detail in rec.provenance_details.items()
                        if detail.get("grounded") and detail.get("resolved_id")
                        and "value" in detail
                    )
                    workings.append(
                        f"- `{rec.expression}` = {rec.computed_result} {rec.units} "
                        f"({bindings or 'see source references'})."
                    )
                if citations:
                    workings.append(f"- {cid}@v{claim.version} sources: {', '.join(citations)}.")
            if workings:
                sections.append("**Calculations and Sources**\n" + "\n".join(workings))

        if not (approved_accepted or approved_conditional or approved_recommendations):
            sections.append(
                "**Insufficient Verified Conclusions:** No assertions passed "
                "independent verification under current evidence or budget constraints."
            )

        if dedup_unresolved:
            lines = ["**Unresolved / Withheld Items:**"]
            for item in dedup_unresolved:
                lines.append(f"- {item}")
            sections.append("\n".join(lines))

        if incomplete_tasks:
            lines = ["**Incomplete Tasks:**"]
            for task in incomplete_tasks:
                lines.append(f"- {task.task_ref or task.task_id}: {task.status.value}")
            sections.append("\n".join(lines))

        rendered_answer = "\n\n".join(sections)

        # The web conversation uses the same audited text without internal trace annotations.
        conversational: dict[str, str] = {}
        for text in clean_points:
            conversational.setdefault(_canonical_norm(text), text)
        for segment, claims in approved_conditional:
            conditions = list(dict.fromkeys(a for c in claims for a in c.assumptions if a))
            text = segment.text + (f" (assuming {', '.join(conditions)})" if conditions else "")
            conversational.setdefault(_canonical_norm(segment.text), text)
        for segment, _ in approved_recommendations:
            conversational.setdefault(_canonical_norm(segment.text), segment.text)
        user_answer = list(conversational.values())
        if not user_answer:
            user_answer.append("I couldn’t verify an answer from the information available.")
        gaps = [
            d.text for item in req_items if item.status != "answered"
            for d in plan.deliverables
            if d.requirement_ref == item.requirement_id
            and d.deliverable_id not in item.verified_deliverable_refs
        ]
        if gaps:
            user_answer.append("I couldn’t fully verify the following:\n"
                               + "\n".join(f"- {text}" for text in gaps))
        if incomplete_tasks or is_partial_run:
            user_answer.append("I couldn’t finish every check in this run.")
        if (laundering_detected or rejected_segments or rejected_claim_refs) and not gaps:
            user_answer.append("I’ve left out conclusions I couldn’t verify.")


        has_usable_output = bool(
            approved_accepted or approved_conditional or approved_recommendations
        )
        content_released_safely = (
            not rejected_segments
            and not rejected_claim_refs
            and not laundering_detected
            and has_usable_output
        )
        execution_completed = not is_partial_run and not incomplete_tasks
        gate_passed = content_released_safely and not stale_claims and execution_completed

        release_mode: Literal["full", "partial", "blocked"]
        if gate_passed and (not plan.deliverables or answer_complete):
            release_mode = "full"
        elif has_usable_output:
            release_mode = "partial"
        else:
            release_mode = "blocked"

        verification = VerificationSummary(
            material_claim_count=len(material_claims),
            supported_claim_count=len(supported_claims),
            unresolved_claim_count=len(unresolved_claims),
            rejected_claim_count=len(rejected_claims),
            stale_claim_count=len(stale_claims),
            revision_count=revision_count,
            evidence_types=evidence_types,
            review_completeness=review_completeness,
            requirement_coverage=requirement_coverage,
            requirements_answered=answered_count,
            requirements_partial=partial_count,
            requirements_unresolved=unresolved_req_count,
            requirements_missing=missing_req_count,
            requirement_statuses=req_items,
            execution_completed=execution_completed,
            answer_complete=answer_complete,
            release_checks_executed=True,
            content_released_safely=content_released_safely,
            coverage_methodology=(
                "independently_reviewed_deliverables"
                if plan.deliverables else "internal_claim_linkage_not_independent_grade"
            ),
            independent_grade=None,
            honesty_gate_passed=gate_passed,
            honesty_gate_reasons=list(dict.fromkeys(gate_reasons)),
        )

        gate_record = HonestyGateRecord(
            run_id=run_id,
            passed=gate_passed,
            release_mode=release_mode,
            rejected_segments=rejected_segments,
            rejected_claim_refs=rejected_claim_refs,
            reasons=list(dict.fromkeys(gate_reasons)),
            verification_summary=verification,
        )

        review_pct = (
            f"{review_completeness * 100:.0f}%"
            if review_completeness is not None
            else "N/A (0 material claims)"
        )
        req_pct = (
            f"{requirement_coverage * 100:.0f}%" if requirement_coverage is not None else "N/A"
        )
        note_parts = [
            f"Verification: {len(supported_claims)}/{len(material_claims)} material claims "
            f"supported, {len(unresolved_claims)} unresolved, {len(rejected_claims)} rejected, "
            f"{len(stale_claims)} stale, {revision_count} revision(s).",
            f"Execution completed: {execution_completed}; checks safely passed: "
            f"{content_released_safely}; internal requirement linkage: {req_pct} "
            f"({answered_count} answered, {partial_count} partial, "
            f"{unresolved_req_count} unresolved, {missing_req_count} missing — not an "
            f"independent grade); review completeness: {review_pct}.",
            f"Honesty Gate: {release_mode} (passed={gate_passed}).",
        ]
        if gate_reasons:
            note_parts.append("Gate filters: " + "; ".join(dict.fromkeys(gate_reasons)))
        if uncertainties:
            safe_uncertainties = [
                u.strip()
                for u in uncertainties
                if u.strip()
                and _detect_auxiliary_laundering(
                    u.strip(), allowed_numbers=allowed_numbers, rejected_claims=rejected_claims
                )
                is None
            ]
            if safe_uncertainties:
                note_parts.append("Specialist uncertainties: " + "; ".join(safe_uncertainties))
        note_parts.append(
            "Note: automated review and arithmetic verification constrain unsupported releases "
            "but remain fallible and do not guarantee empirical truth."
        )

        return HonestyGateResult(
            rendered_answer=rendered_answer,
            conversational_answer="\n\n".join(user_answer),
            referenced_claim_ids=used_claim_ids,
            assumptions=clean_assumptions,
            missing_information=clean_missing,
            confidence_notes=" ".join(note_parts),
            verification=verification,
            gate_record=gate_record,
        )

    async def persist_decision(self, gate_record: HonestyGateRecord) -> None:
        """Persist Honesty Gate decision and audit event."""
        await self._db.save_honesty_gate_decision(gate_record)
        await self._db.append_event(
            ExecutionEvent(
                run_id=gate_record.run_id,
                event_type="honesty_gate_evaluated",
                actor="honesty_gate",
                message=(
                    f"Honesty Gate release_mode={gate_record.release_mode}, "
                    f"passed={gate_record.passed}."
                ),
                payload={
                    "gate_id": gate_record.gate_id,
                    "release_mode": gate_record.release_mode,
                    "rejected_segments_count": len(gate_record.rejected_segments),
                    "rejected_claim_refs_count": len(gate_record.rejected_claim_refs),
                },
            )
        )
