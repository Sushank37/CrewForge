"""Deterministic Python Honesty Gate and release renderer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    HonestyGateRecord,
    ReleaseSegment,
    ReviewDecision,
    ReviewVerdict,
    SynthesisOutput,
    TaskSpec,
    ToolExecutionRecord,
    VerificationStatus,
    VerificationSummary,
)
from crewforge.storage.database import Database
from crewforge.tools.arithmetic import extract_numbers_from_text

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "will",
        "into",
        "only",
        "than",
        "then",
        "when",
        "what",
        "which",
        "have",
        "has",
        "not",
        "but",
        "can",
        "may",
        "based",
    }
)

_APPROVED_VERIFICATION_STATES = frozenset(
    {
        VerificationStatus.PROVENANCE_VALIDATED,
        VerificationStatus.COMPUTATIONALLY_VERIFIED,
        VerificationStatus.INDEPENDENTLY_REVIEWED_INFERENCE,
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


@dataclass
class HonestyGateResult:
    """Validated release content, verification summary, and persisted gate record."""

    rendered_answer: str
    referenced_claim_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    confidence_notes: str = ""
    verification: VerificationSummary = field(default_factory=VerificationSummary)
    gate_record: HonestyGateRecord | None = None


class HonestyGate:
    """Enforces deterministic claim-version, provenance, and release constraints in Python."""

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

        reviewed_material = 0
        for claim in material_claims:
            c_verdicts = verdicts_by_key.get((claim.claim_id, claim.version), [])
            if c_verdicts:
                reviewed_material += 1

        review_completeness: float | None = (
            round(reviewed_material / len(material_claims), 4) if material_claims else None
        )

        rejected_segments: list[dict[str, Any]] = []
        rejected_claim_refs: list[dict[str, Any]] = []
        gate_reasons: list[str] = []

        approved_accepted: list[ReleaseSegment] = []
        approved_conditional: list[ReleaseSegment] = []
        approved_recommendations: list[ReleaseSegment] = []
        unresolved_items: list[str] = list(synthesis.unresolved_issues)
        used_claim_ids: list[str] = []

        prompt_numbers = extract_numbers_from_text(original_prompt)

        def _validate_claim_ref(
            cid: str,
            cver: int,
            *,
            require_supported: bool,
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
            if require_supported:
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
                if not any(v.decision == ReviewDecision.APPROVE for v in c_verdicts):
                    return None, f"Claim '{cid}' v{cver} lacks an independent approval verdict."
                if any(v.reviewer_id == latest.author_agent_id for v in c_verdicts):
                    return None, f"Claim '{cid}' v{cver} has an invalid self-review verdict."
                if latest.category == ClaimCategory.COMPUTATION:
                    recs = [
                        tool_by_id[tid] for tid in latest.tool_execution_ids if tid in tool_by_id
                    ]
                    if not any(
                        r.success and r.inputs_grounded and r.matches_proposed is not False
                        for r in recs
                    ):
                        return (
                            None,
                            f"Numeric claim '{cid}' lacks grounded arithmetic verification.",
                        )
                if latest.category == ClaimCategory.EXTERNAL_FACT and not any(
                    eid in evidence_by_id and evidence_by_id[eid].verified_in_input
                    for eid in latest.evidence_ids
                ):
                    return None, f"External factual claim '{cid}' lacks retrieved evidence."
            return latest, None

        def _segment_matches_claims(segment_text: str, referenced: list[Claim]) -> str | None:
            seg_tokens = _content_tokens(segment_text)
            claim_tokens: set[str] = set()
            allowed_numbers = set(prompt_numbers)
            for c in referenced:
                claim_tokens.update(_content_tokens(c.statement))
                allowed_numbers.update(extract_numbers_from_text(c.statement))
                for tid in c.tool_execution_ids:
                    if tid in tool_by_id and tool_by_id[tid].computed_result is not None:
                        allowed_numbers.update(
                            extract_numbers_from_text(tool_by_id[tid].computed_result or "")
                        )

            if seg_tokens and claim_tokens and not (seg_tokens & claim_tokens):
                return "Segment text is unrelated to the referenced claim content."

            seg_numbers = extract_numbers_from_text(segment_text)
            extra_nums = [n for n in seg_numbers if n not in allowed_numbers]
            if extra_nums:
                return "Segment introduces unverified numeric values absent from referenced claims."
            return None

        # 1. Audit accepted_segments strictly (empty claim_refs or non-supported claims rejected).
        for seg in synthesis.accepted_segments:
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

            if seg_error is None:
                seg_error = _segment_matches_claims(seg.text, valid_claims)

            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                approved_accepted.append(seg)
                for c in valid_claims:
                    if c.claim_id not in used_claim_ids:
                        used_claim_ids.append(c.claim_id)

        # 2. Audit conditional conclusions & recommendations (block rejected/stale/unretrieved).
        for seg in synthesis.conditional_conclusions:
            seg_error = None
            valid_claims = []
            for ref in seg.claim_refs:
                c_obj, err = _validate_claim_ref(
                    ref.claim_id,
                    ref.claim_version,
                    require_supported=False,
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
            if seg_error is None and valid_claims:
                seg_error = _segment_matches_claims(seg.text, valid_claims)
            # Disallow smuggling an empirical numeric claim into conditional text without any ref.
            if seg_error is None and not seg.claim_refs and seg.is_factual_assertion:
                seg_nums = extract_numbers_from_text(seg.text)
                if any(n not in prompt_numbers for n in seg_nums):
                    seg_error = (
                        "Conditional segment contains ungrounded numeric assertion without ref."
                    )

            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                approved_conditional.append(seg)
                for c in valid_claims:
                    if c.claim_id not in used_claim_ids:
                        used_claim_ids.append(c.claim_id)

        for seg in synthesis.recommendations:
            seg_error = None
            for ref in seg.claim_refs:
                _, err = _validate_claim_ref(
                    ref.claim_id,
                    ref.claim_version,
                    require_supported=False,
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
            if seg_error is not None:
                rejected_segments.append({"text": seg.text, "reason": seg_error})
                gate_reasons.append(seg_error)
            else:
                approved_recommendations.append(seg)

        # 3. Unreviewed Lead factual assertions are never released as established facts.
        for new_fact in synthesis.new_factual_assertions:
            cleaned = new_fact.strip()
            if cleaned:
                msg = f"Unreviewed synthesis assertion (not released as verified fact): {cleaned}"
                unresolved_items.append(msg)
                gate_reasons.append(
                    "Stripped unreviewed Lead factual assertion from verified output."
                )

        # Also validate raw referenced_claim_ids for cross-run/stale leaks.
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

        # Surface unresolved / rejected / stale claims transparently.
        for claim in unresolved_claims:
            unresolved_items.append(
                f"Unresolved ({claim.claim_id} v{claim.version}): {claim.statement}"
            )
        for claim in stale_claims:
            unresolved_items.append(
                f"Stale pending re-review ({claim.claim_id} v{claim.version}): {claim.statement}"
            )
        for claim in rejected_claims:
            unresolved_items.append(
                f"Rejected claim ({claim.claim_id} v{claim.version}) withheld from verified output."
            )

        # Requirement coverage calculation.
        reqs = [r.strip() for r in plan.user_requirements if r.strip()]
        covered_reqs = 0
        if reqs:
            approved_text_blob = " ".join(
                [*(s.text for s in approved_accepted), *(s.text for s in approved_conditional)]
            )
            approved_tokens = _content_tokens(approved_text_blob)
            explicit_req_refs = {
                s.requirement_ref
                for s in [*approved_accepted, *approved_conditional, *approved_recommendations]
                if s.requirement_ref
            }
            for idx, req in enumerate(reqs, start=1):
                req_tokens = _content_tokens(req)
                if (
                    req in explicit_req_refs
                    or f"req_{idx}" in explicit_req_refs
                    or (req_tokens and len(req_tokens & approved_tokens) >= 1)
                ):
                    covered_reqs += 1
        requirement_coverage: float | None = round(covered_reqs / len(reqs), 4) if reqs else None

        merged_assumptions = list(
            dict.fromkeys([*plan.assumptions, *synthesis.assumptions, *extra_assumptions])
        )
        merged_missing = list(
            dict.fromkeys(
                [*plan.missing_information, *synthesis.missing_information, *extra_missing]
            )
        )
        dedup_unresolved = list(dict.fromkeys(u.strip() for u in unresolved_items if u.strip()))

        # Render final readable natural-language answer strictly from gate-approved structures.
        sections: list[str] = []
        if is_partial_run:
            sections.append(
                "**Partial Result:** Execution did not complete all stages before budget or "
                "deadline limits. Only verified findings and explicitly qualified gaps are shown."
            )

        if approved_accepted:
            lines = ["**Verified Findings:**"]
            for seg in approved_accepted:
                refs = ", ".join(f"{r.claim_id}@v{r.claim_version}" for r in seg.claim_refs)
                lines.append(f"- {seg.text} [{refs}]")
            sections.append("\n".join(lines))

        if approved_conditional:
            lines = ["**Conditional Analysis:**"]
            for seg in approved_conditional:
                refs = (
                    " ["
                    + ", ".join(f"{r.claim_id}@v{r.claim_version}" for r in seg.claim_refs)
                    + "]"
                    if seg.claim_refs
                    else ""
                )
                lines.append(f"- {seg.text}{refs}")
            sections.append("\n".join(lines))

        if approved_recommendations:
            lines = ["**Recommendations:**"]
            for seg in approved_recommendations:
                lines.append(f"- {seg.text}")
            sections.append("\n".join(lines))

        if not (approved_accepted or approved_conditional or approved_recommendations):
            sections.append(
                "**Insufficient Verified Conclusions:** No material assertions passed "
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
                lines.append(
                    f"- {task.task_ref or task.task_id} ({task.title}): "
                    f"{task.status.value} ({task.error or 'incomplete'})"
                )
            sections.append("\n".join(lines))

        rendered_answer = "\n\n".join(sections)

        has_usable_output = bool(
            approved_accepted or approved_conditional or approved_recommendations
        )
        gate_passed = (
            not rejected_segments
            and not rejected_claim_refs
            and not stale_claims
            and not incomplete_tasks
            and not is_partial_run
            and has_usable_output
        )
        release_mode: Literal["full", "partial", "blocked"]
        if gate_passed:
            release_mode = "full"
        elif has_usable_output or dedup_unresolved or merged_missing:
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

        # Run-specific verification summary instead of generic boilerplate.
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
            f"Review completeness: {review_pct}; requirement coverage: {req_pct}.",
            f"Honesty Gate: {release_mode} (passed={gate_passed}).",
        ]
        if gate_reasons:
            note_parts.append("Gate filters: " + "; ".join(dict.fromkeys(gate_reasons)))
        if uncertainties:
            note_parts.append("Specialist uncertainties: " + "; ".join(uncertainties))
        note_parts.append(
            "Note: automated review and arithmetic verification constrain unsupported releases "
            "but remain fallible and do not guarantee empirical truth."
        )

        return HonestyGateResult(
            rendered_answer=rendered_answer,
            referenced_claim_ids=used_claim_ids,
            assumptions=merged_assumptions,
            missing_information=merged_missing,
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
