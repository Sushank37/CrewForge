"""Deterministic evidence provenance checking and external retrieval interface."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from crewforge.schemas import (
    EvidenceReliability,
    ProposedEvidence,
    ToolExecutionRecord,
)
from crewforge.tools.arithmetic import extract_numbers_from_text, split_prompt_clauses

_WHITESPACE_RE = re.compile(r"\s+")
_NEGATION_WORDS = frozenset(
    {"not", "no", "never", "neither", "nor", "without", "excluding", "ignore", "ignoring"}
)


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for deterministic provenance matching."""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


class RetrievedSource(BaseModel):
    """Source returned by an EvidenceRetriever implementation."""

    source_ref: str
    title: str = ""
    excerpt: str
    retrieved_at: str = ""
    source_location: str = ""
    content_hash: str = ""
    reliability: EvidenceReliability = EvidenceReliability.MEDIUM


class EvidenceRetriever(ABC):
    """Provider-independent interface for future external evidence retrieval."""

    @property
    @abstractmethod
    def retrieval_available(self) -> bool:
        """Whether external retrieval is active in the current runtime."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        run_id: str,
        limit: int = 3,
    ) -> list[RetrievedSource]:
        """Retrieve verifiable external excerpts for a factual query."""


class UnavailableExternalRetriever(EvidenceRetriever):
    """Default Phase 3 retriever: unrestricted internet/URL fetching is disabled."""

    @property
    def retrieval_available(self) -> bool:
        return False

    async def retrieve(
        self,
        query: str,
        *,
        run_id: str,
        limit: int = 3,
    ) -> list[RetrievedSource]:
        return []


@dataclass
class ProvenanceCheckResult:
    """Result of deterministic provenance validation for an Evidence proposal."""

    reliability: EvidenceReliability
    verified_in_input: bool
    notes: str


def _preserves_numbers_and_negation(
    *,
    excerpt: str,
    source_text: str,
) -> tuple[bool, str]:
    """Verify excerpt isn't truncating a surrounding negation or altering numbers."""
    norm_excerpt = normalize_text(excerpt)
    norm_source = normalize_text(source_text)
    idx = norm_source.find(norm_excerpt)
    if idx == -1:
        return False, "Excerpt not found in referenced input."

    # Inspect immediate prefix window (~25 chars) within the same sentence/clause
    # for an omitted leading negation marker (do not cross sentence/doc boundaries).
    prefix_start = max(0, idx - 25)
    prefix = norm_source[prefix_start:idx]
    last_boundary = max(
        prefix.rfind("."),
        prefix.rfind("!"),
        prefix.rfind("?"),
        prefix.rfind(";"),
        prefix.rfind("\n"),
    )
    if last_boundary != -1:
        prefix = prefix[last_boundary + 1 :]
    prefix_tokens = set(re.findall(r"[a-z]+", prefix))
    excerpt_tokens = set(re.findall(r"[a-z]+", norm_excerpt))
    if (prefix_tokens & _NEGATION_WORDS) and not (excerpt_tokens & _NEGATION_WORDS):
        return False, "Excerpt drops a nearby negation qualifier from the source context."

    excerpt_nums = extract_numbers_from_text(excerpt)
    if excerpt_nums:
        source_nums = extract_numbers_from_text(source_text)
        if not excerpt_nums.issubset(source_nums):
            return False, "Excerpt numeric tokens do not match source numbers."

    return True, "Excerpt matched source with preserved numbers and polarity."


def verify_evidence_proposal(
    *,
    proposal: ProposedEvidence,
    original_prompt: str,
    dependency_findings_text: str = "",
    matching_tool_records: list[ToolExecutionRecord] | None = None,
    retrieved_sources_text: str = "",
    retrieved_sources_by_ref: dict[str, str] | None = None,
) -> ProvenanceCheckResult:
    """Deterministically validate provenance without conflating substring match with truth.

    - `user_prompt`: checks verbatim substring, numeric consistency, and surrounding negation.
    - `dependency_finding`: checks verbatim presence in completed dependency outputs.
    - `calculation`: requires an actual successful `ToolExecutionRecord` in the same run.
    - `external_retrieved`: requires verbatim match inside genuinely retrieved external docs.
    - `external_unverified`: always remains UNVERIFIED without an active retriever.
    """
    norm_excerpt = normalize_text(proposal.excerpt)
    if not norm_excerpt:
        return ProvenanceCheckResult(
            reliability=EvidenceReliability.UNVERIFIED,
            verified_in_input=False,
            notes="Empty evidence excerpt.",
        )

    if proposal.source_type == "user_prompt":
        source = original_prompt
        if proposal.source_ref.startswith("up_"):
            clauses = {f"up_{idx}": clause for idx, clause in
                       enumerate(split_prompt_clauses(original_prompt), start=1)}
            source = clauses.get(proposal.source_ref, "")
        matched, note = _preserves_numbers_and_negation(
            excerpt=proposal.excerpt,
            source_text=source,
        )
        return ProvenanceCheckResult(
            reliability=EvidenceReliability.MEDIUM if matched else EvidenceReliability.UNVERIFIED,
            verified_in_input=matched,
            notes=note,
        )

    if proposal.source_type == "dependency_finding":
        if not dependency_findings_text:
            return ProvenanceCheckResult(
                reliability=EvidenceReliability.UNVERIFIED,
                verified_in_input=False,
                notes="No dependency findings available for this task.",
            )
        matched, note = _preserves_numbers_and_negation(
            excerpt=proposal.excerpt,
            source_text=dependency_findings_text,
        )
        return ProvenanceCheckResult(
            reliability=EvidenceReliability.LOW if matched else EvidenceReliability.UNVERIFIED,
            verified_in_input=matched,
            notes=note,
        )

    if proposal.source_type in {"calculation", "computation"}:
        records = matching_tool_records or []
        valid_records = [r for r in records if r.success and r.inputs_grounded]
        if valid_records:
            return ProvenanceCheckResult(
                reliability=EvidenceReliability.HIGH,
                verified_in_input=True,
                notes="Backed by successful grounded deterministic arithmetic tool execution.",
            )
        return ProvenanceCheckResult(
            reliability=EvidenceReliability.UNVERIFIED,
            verified_in_input=False,
            notes="Computation evidence lacks a successful grounded tool execution record.",
        )

    if proposal.source_type == "external_retrieved":
        if retrieved_sources_by_ref is not None:
            ref = (proposal.source_ref or "").strip()
            if ref:
                doc_text = retrieved_sources_by_ref.get(ref)
                if not doc_text:
                    return ProvenanceCheckResult(
                        reliability=EvidenceReliability.UNVERIFIED,
                        verified_in_input=False,
                        notes=f"Referenced external document '{ref}' not found.",
                    )
                matched, note = _preserves_numbers_and_negation(
                    excerpt=proposal.excerpt,
                    source_text=doc_text,
                )
                return ProvenanceCheckResult(
                    reliability=EvidenceReliability.MEDIUM
                    if matched
                    else EvidenceReliability.UNVERIFIED,
                    verified_in_input=matched,
                    notes=note,
                )
            if not retrieved_sources_by_ref:
                return ProvenanceCheckResult(
                    reliability=EvidenceReliability.UNVERIFIED,
                    verified_in_input=False,
                    notes="No retrieved external document matches this citation.",
                )
            last_note = "Excerpt not found in referenced input."
            for doc_text in retrieved_sources_by_ref.values():
                matched, note = _preserves_numbers_and_negation(
                    excerpt=proposal.excerpt,
                    source_text=doc_text,
                )
                if matched:
                    return ProvenanceCheckResult(
                        reliability=EvidenceReliability.MEDIUM,
                        verified_in_input=True,
                        notes=note,
                    )
                last_note = note
            return ProvenanceCheckResult(
                reliability=EvidenceReliability.UNVERIFIED,
                verified_in_input=False,
                notes=last_note,
            )
        if not retrieved_sources_text:
            return ProvenanceCheckResult(
                reliability=EvidenceReliability.UNVERIFIED,
                verified_in_input=False,
                notes="No retrieved external document matches this citation.",
            )
        matched, note = _preserves_numbers_and_negation(
            excerpt=proposal.excerpt,
            source_text=retrieved_sources_text,
        )
        return ProvenanceCheckResult(
            reliability=EvidenceReliability.MEDIUM if matched else EvidenceReliability.UNVERIFIED,
            verified_in_input=matched,
            notes=note,
        )

    # External factual assertions and URL-like citations remain unverified without retrieval.
    return ProvenanceCheckResult(
        reliability=EvidenceReliability.UNVERIFIED,
        verified_in_input=False,
        notes="External citation not retrieved by an active EvidenceRetriever.",
    )
