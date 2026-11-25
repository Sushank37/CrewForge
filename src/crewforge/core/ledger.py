"""Versioned claim ledger and evidence provenance validation."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from crewforge.schemas import (
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceReliability,
    ExecutionEvent,
    ProposedClaim,
    ProposedEvidence,
)
from crewforge.storage.database import Database

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def validate_evidence_excerpt(
    *,
    proposal: ProposedEvidence,
    original_prompt: str,
    dependency_findings_text: str = "",
) -> tuple[EvidenceReliability, bool]:
    """Validate an evidence proposal against trusted run inputs.

    Note: Matching an excerpt proves only that the quoted text exists in the
    referenced input; it does NOT verify that the excerpt logically supports a claim.
    External citations and calculations remain UNVERIFIED in Phase 2.
    """
    norm_excerpt = _normalize_text(proposal.excerpt)
    if not norm_excerpt:
        return EvidenceReliability.UNVERIFIED, False

    if proposal.source_type == "user_prompt":
        matched = norm_excerpt in _normalize_text(original_prompt)
        return (
            (EvidenceReliability.MEDIUM, True)
            if matched
            else (EvidenceReliability.UNVERIFIED, False)
        )

    if proposal.source_type == "dependency_finding":
        matched = bool(dependency_findings_text) and (
            norm_excerpt in _normalize_text(dependency_findings_text)
        )
        return (
            (EvidenceReliability.LOW, True) if matched else (EvidenceReliability.UNVERIFIED, False)
        )

    # 'external_unverified' and 'calculation' cannot be verified without Phase 3 tools/reviewers.
    return EvidenceReliability.UNVERIFIED, False


class ClaimLedger:
    """Run-scoped immutable versioned ledger for claims and evidence proposals."""

    def __init__(self, database: Database) -> None:
        self._db = database

    async def record_specialist_outputs(
        self,
        *,
        run_id: str,
        agent_id: str,
        task_id: str,
        original_prompt: str,
        dependency_findings_text: str,
        evidence_proposals: list[ProposedEvidence],
        claim_proposals: list[ProposedClaim],
    ) -> tuple[list[Evidence], list[Claim]]:
        """Validate and persist evidence proposals and initial claim versions."""
        persisted_evidence: list[Evidence] = []
        for proposal in evidence_proposals:
            reliability, verified_in_input = validate_evidence_excerpt(
                proposal=proposal,
                original_prompt=original_prompt,
                dependency_findings_text=dependency_findings_text,
            )
            evd = Evidence(
                run_id=run_id,
                source_type=proposal.source_type,
                source_ref=proposal.source_ref,
                excerpt=proposal.excerpt,
                reliability=reliability,
                verified_in_input=verified_in_input,
                task_id=task_id,
                created_by=agent_id,
            )
            await self._db.insert_evidence(evd)
            persisted_evidence.append(evd)

        persisted_claims: list[Claim] = []
        for prop_claim in claim_proposals:
            linked_evidence_ids = [
                persisted_evidence[idx].evidence_id
                for idx in prop_claim.evidence_indices
                if 0 <= idx < len(persisted_evidence)
            ]
            # Phase 2 never marks claims as SUPPORTED purely from model self-assertion.
            status = ClaimStatus.UNRESOLVED if prop_claim.is_unresolved else ClaimStatus.ASSERTED
            claim = Claim(
                run_id=run_id,
                statement=prop_claim.statement,
                status=status,
                version=1,
                parent_claim_id=None,
                author_agent_id=agent_id,
                task_id=task_id,
                evidence_ids=linked_evidence_ids,
                assumptions=prop_claim.assumptions,
                confidence=prop_claim.confidence,
            )
            await self._db.insert_claim(claim)
            persisted_claims.append(claim)

        if persisted_evidence or persisted_claims:
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="claims_recorded",
                    actor=agent_id,
                    message=(
                        f"Recorded {len(persisted_claims)} claim(s) and "
                        f"{len(persisted_evidence)} evidence record(s)."
                    ),
                    payload={
                        "task_id": task_id,
                        "claim_ids": [c.claim_id for c in persisted_claims],
                        "evidence_ids": [e.evidence_id for e in persisted_evidence],
                    },
                )
            )

        return persisted_evidence, persisted_claims

    async def revise_claim(
        self,
        *,
        prior_claim: Claim,
        new_statement: str,
        author_agent_id: str,
        task_id: str | None = None,
        additional_evidence_ids: list[str] | None = None,
        assumptions: list[str] | None = None,
        revision_notes: str | None = None,
        is_unresolved: bool = False,
        confidence: float | None = None,
    ) -> Claim:
        """Append an immutable version (version + 1) without overwriting the original claim."""
        merged_evidence = list(
            dict.fromkeys([*prior_claim.evidence_ids, *(additional_evidence_ids or [])])
        )
        status = ClaimStatus.UNRESOLVED if is_unresolved else ClaimStatus.ASSERTED
        revised = Claim(
            claim_id=prior_claim.claim_id,
            run_id=prior_claim.run_id,
            statement=new_statement,
            status=status,
            version=prior_claim.version + 1,
            parent_claim_id=prior_claim.claim_id,
            author_agent_id=author_agent_id,
            task_id=task_id or prior_claim.task_id,
            evidence_ids=merged_evidence,
            depends_on_claim_ids=prior_claim.depends_on_claim_ids,
            assumptions=assumptions if assumptions is not None else prior_claim.assumptions,
            revision_notes=revision_notes,
            confidence=confidence if confidence is not None else prior_claim.confidence,
            created_at=prior_claim.created_at,
            updated_at=datetime.now(UTC),
        )
        await self._db.insert_claim(revised)
        await self._db.append_event(
            ExecutionEvent(
                run_id=prior_claim.run_id,
                event_type="claim_revised",
                actor=author_agent_id,
                message=f"Recorded claim {revised.claim_id} version {revised.version}.",
                payload={
                    "claim_id": revised.claim_id,
                    "version": revised.version,
                    "parent_claim_id": revised.parent_claim_id,
                },
            )
        )
        return revised

    async def get_claims(self, run_id: str, *, latest_only: bool = False) -> list[Claim]:
        """Fetch claims for a run (all immutable versions or latest per claim_id)."""
        return await self._db.get_claims_for_run(run_id, latest_only=latest_only)
