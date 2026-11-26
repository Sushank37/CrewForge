"""Versioned claim ledger with provenance validation and arithmetic tool integration."""

from __future__ import annotations

from datetime import UTC, datetime

from crewforge.core.evidence_verifier import verify_evidence_proposal
from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ClaimVersionRef,
    Evidence,
    ExecutionEvent,
    ProposedClaim,
    ProposedEvidence,
    ToolExecutionRecord,
    VerificationStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database
from crewforge.tools.arithmetic import DeterministicArithmeticTool, build_provenance_catalog

_NON_MATERIAL_CATEGORIES = frozenset(
    {
        ClaimCategory.ASSUMPTION,
        ClaimCategory.RECOMMENDATION,
        ClaimCategory.UNCERTAINTY,
    }
)


class ClaimRevisionError(ValueError):
    """Raised when a targeted claim revision reference is invalid, stale, or unauthorized."""


class ClaimLedger:
    """Run-scoped immutable versioned ledger for claims, evidence, and tool provenance."""

    def __init__(self, database: Database) -> None:
        self._db = database
        self._arithmetic = DeterministicArithmeticTool(database)

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
    ) -> tuple[list[Evidence], list[Claim], list[ToolExecutionRecord]]:
        """Validate evidence, run arithmetic requests, and persist initial claim versions."""
        existing_all = await self._db.get_claims_for_run(run_id, latest_only=False)
        existing_claims = await self._db.get_claims_for_run(run_id, latest_only=True)
        existing_by_id = {c.claim_id: c for c in existing_claims}
        existing_evidence = await self._db.get_evidence_for_run(run_id)
        prior_tool_records = await self._db.get_tool_executions_for_run(run_id)

        # Resolve variables strictly against verified prompt spans, verified evidence,
        # current supported claims, and prior valid tools (never unverified dependency prose).
        verified_evidence_all = [e for e in existing_evidence if e.verified_in_input]
        plan_assumptions: list[str] = []
        catalog = build_provenance_catalog(
            original_prompt=original_prompt,
            verified_evidence=verified_evidence_all,
            all_claims=existing_all,
            tool_records=prior_tool_records,
            assumptions=plan_assumptions,
        )

        # Execute computation requests attached to claims before validating computation evidence.
        from crewforge.tools.arithmetic import ProvenanceTarget

        claim_tool_records: dict[int, ToolExecutionRecord] = {}
        all_new_tool_records: list[ToolExecutionRecord] = []
        for idx, prop_claim in enumerate(claim_proposals):
            if prop_claim.computation is not None:
                rec = await self._arithmetic.execute(
                    run_id=run_id,
                    actor=agent_id,
                    request=prop_claim.computation,
                    catalog=catalog,
                    task_id=task_id,
                )
                claim_tool_records[idx] = rec
                all_new_tool_records.append(rec)
                if (
                    rec.success
                    and rec.inputs_grounded
                    and not rec.has_ungrounded_literals
                    and rec.matches_proposed is not False
                    and rec.computed_result is not None
                ):
                    t_intermediate = ProvenanceTarget(
                        alias=rec.tool_execution_id,
                        kind="tool_result",
                        target_id=rec.tool_execution_id,
                        text=f"{rec.computed_result} {rec.units}".strip(),
                        units=rec.units,
                        is_trusted=True,
                    )
                    catalog.targets_by_key[rec.tool_execution_id] = t_intermediate
                    seq_alias = f"clm_{idx + 1}"
                    if seq_alias not in catalog.targets_by_key:
                        catalog.targets_by_key[seq_alias] = t_intermediate

        combined_tool_records = [*prior_tool_records, *all_new_tool_records]

        persisted_evidence: list[Evidence] = []
        retrieved_docs = [e for e in verified_evidence_all if e.source_type == "external_retrieved"]
        retrieved_sources_text = "\n".join(e.excerpt for e in retrieved_docs)
        retrieved_refs = {
            *(f"doc_{idx}" for idx, _ in enumerate(verified_evidence_all, start=1)),
            *(e.source_ref for e in retrieved_docs if e.source_ref),
            *(e.evidence_id for e in retrieved_docs),
        }
        for proposal in evidence_proposals:
            effective_source_type = proposal.source_type
            check = verify_evidence_proposal(
                proposal=proposal,
                original_prompt=original_prompt,
                dependency_findings_text=dependency_findings_text,
                matching_tool_records=combined_tool_records,
                retrieved_sources_text=retrieved_sources_text,
            )
            # If a specialist cited an attached/retrieved document using `dependency_finding`
            # or `user_prompt`, re-verify strictly against `external_retrieved` when it matches
            # a known retrieved doc ref or verbatim retrieved text.
            if (
                not check.verified_in_input
                and retrieved_sources_text
                and proposal.source_type in {"user_prompt", "dependency_finding"}
                and (
                    proposal.source_ref in retrieved_refs
                    or proposal.source_type == "dependency_finding"
                    and not dependency_findings_text
                )
            ):
                retrieved_proposal = proposal.model_copy(
                    update={"source_type": "external_retrieved"}
                )
                retrieved_check = verify_evidence_proposal(
                    proposal=retrieved_proposal,
                    original_prompt=original_prompt,
                    dependency_findings_text=dependency_findings_text,
                    matching_tool_records=combined_tool_records,
                    retrieved_sources_text=retrieved_sources_text,
                )
                if retrieved_check.verified_in_input:
                    effective_source_type = "external_retrieved"
                    check = retrieved_check
            evd = Evidence(
                run_id=run_id,
                source_type=effective_source_type,
                source_ref=proposal.source_ref,
                excerpt=proposal.excerpt,
                reliability=check.reliability,
                verified_in_input=check.verified_in_input,
                retrieval_status=(
                    "ok"
                    if effective_source_type == "external_retrieved" and check.verified_in_input
                    else (
                        "unavailable"
                        if effective_source_type == "external_unverified"
                        else "not_applicable"
                    )
                ),
                task_id=task_id,
                created_by=agent_id,
            )
            await self._db.insert_evidence(evd)
            persisted_evidence.append(evd)

        persisted_claims: list[Claim] = []
        local_ref_map: dict[str, Claim] = {}

        # Enforce bounded claim count per specialist call (keep at most 4 claims per batch).
        for idx, prop_claim in enumerate(claim_proposals[:4]):
            linked_evidence_ids = [
                persisted_evidence[e_idx].evidence_id
                for e_idx in prop_claim.evidence_indices
                if 0 <= e_idx < len(persisted_evidence)
            ]

            premise_refs: list[ClaimVersionRef] = []
            dep_claim_ids: list[str] = []
            for raw_pid in prop_claim.premise_claim_ids:
                target = existing_by_id.get(raw_pid) or local_ref_map.get(raw_pid)
                if target is not None:
                    premise_refs.append(
                        ClaimVersionRef(claim_id=target.claim_id, claim_version=target.version)
                    )
                    if target.claim_id not in dep_claim_ids:
                        dep_claim_ids.append(target.claim_id)

            tool_ids: list[str] = []
            if idx in claim_tool_records:
                tool_ids.append(claim_tool_records[idx].tool_execution_id)

            # Workers can NEVER promote their own claims to SUPPORTED.
            status = ClaimStatus.UNRESOLVED if prop_claim.is_unresolved else ClaimStatus.ASSERTED
            is_material = prop_claim.category not in _NON_MATERIAL_CATEGORIES

            claim = Claim(
                run_id=run_id,
                statement=prop_claim.statement,
                category=prop_claim.category,
                status=status,
                verification_status=VerificationStatus.UNSUPPORTED,
                version=1,
                parent_claim_id=None,
                author_agent_id=agent_id,
                task_id=task_id,
                evidence_ids=linked_evidence_ids,
                depends_on_claim_ids=dep_claim_ids,
                premise_refs=premise_refs,
                tool_execution_ids=tool_ids,
                requirement_refs=list(prop_claim.requirement_refs),
                assumptions=list(prop_claim.assumptions),
                confidence=prop_claim.confidence,
                is_material=is_material,
            )
            await self._db.insert_claim(claim)
            persisted_claims.append(claim)
            if prop_claim.claim_ref:
                local_ref_map[prop_claim.claim_ref] = claim
            existing_by_id[claim.claim_id] = claim

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
                        "tool_execution_ids": [t.tool_execution_id for t in all_new_tool_records],
                    },
                )
            )

        return persisted_evidence, persisted_claims, all_new_tool_records

    async def apply_targeted_revision(
        self,
        *,
        run_id: str,
        target_claim_id: str,
        expected_version: int,
        new_statement: str,
        author_agent_id: str,
        original_prompt: str,
        dependency_findings_text: str = "",
        category: ClaimCategory | None = None,
        additional_evidence_ids: list[str] | None = None,
        premise_claim_ids: list[str] | None = None,
        computation_request: object | None = None,
        assumptions: list[str] | None = None,
        revision_notes: str | None = None,
        is_unresolved: bool = False,
        confidence: float | None = None,
        require_same_author: bool = True,
    ) -> Claim:
        """Append an immutable claim version targeting an explicit claim_id and expected_version."""
        latest_claims = await self._db.get_claims_for_run(run_id, latest_only=True)
        latest_by_id = {c.claim_id: c for c in latest_claims}

        prior_claim = latest_by_id.get(target_claim_id)
        if prior_claim is None:
            raise CrossRunReferenceError(
                f"Target claim '{target_claim_id}' does not exist in run '{run_id}'."
            )
        if prior_claim.version != expected_version:
            raise ClaimRevisionError(
                f"Stale revision target for '{target_claim_id}': expected v{expected_version}, "
                f"latest is v{prior_claim.version}."
            )
        if require_same_author and prior_claim.author_agent_id != author_agent_id:
            raise ClaimRevisionError(
                f"Unauthorized revision for '{target_claim_id}': author '{author_agent_id}' "
                f"does not match originating specialist '{prior_claim.author_agent_id}'."
            )

        tool_ids = list(prior_claim.tool_execution_ids)
        if computation_request is not None:
            all_claims_run = await self._db.get_claims_for_run(run_id, latest_only=False)
            verified_evd = [
                e for e in await self._db.get_evidence_for_run(run_id) if e.verified_in_input
            ]
            prior_tools = await self._db.get_tool_executions_for_run(run_id)
            catalog = build_provenance_catalog(
                original_prompt=original_prompt,
                verified_evidence=verified_evd,
                all_claims=all_claims_run,
                tool_records=prior_tools,
                assumptions=assumptions if assumptions is not None else prior_claim.assumptions,
            )
            rec = await self._arithmetic.execute(
                run_id=run_id,
                actor=author_agent_id,
                request=computation_request,  # type: ignore[arg-type]
                catalog=catalog,
                task_id=prior_claim.task_id,
            )
            tool_ids = [rec.tool_execution_id]

        resolved_premises = list(prior_claim.premise_refs)
        resolved_dep_ids = list(prior_claim.depends_on_claim_ids)
        if premise_claim_ids is not None:
            resolved_premises = []
            resolved_dep_ids = []
            for pid in premise_claim_ids:
                p_claim = latest_by_id.get(pid)
                if p_claim is None:
                    raise CrossRunReferenceError(
                        f"Premise claim '{pid}' does not exist in run '{run_id}'."
                    )
                if pid == target_claim_id:
                    raise ClaimRevisionError(f"Claim '{target_claim_id}' cannot depend on itself.")
                resolved_premises.append(
                    ClaimVersionRef(claim_id=p_claim.claim_id, claim_version=p_claim.version)
                )
                if p_claim.claim_id not in resolved_dep_ids:
                    resolved_dep_ids.append(p_claim.claim_id)

        merged_evidence = list(
            dict.fromkeys([*prior_claim.evidence_ids, *(additional_evidence_ids or [])])
        )
        next_category = category or prior_claim.category
        status = ClaimStatus.UNRESOLVED if is_unresolved else ClaimStatus.ASSERTED
        is_material = next_category not in _NON_MATERIAL_CATEGORIES

        # A revised claim NEVER inherits approval from its previous version.
        revised = Claim(
            claim_id=prior_claim.claim_id,
            run_id=prior_claim.run_id,
            statement=new_statement,
            category=next_category,
            status=status,
            verification_status=VerificationStatus.UNSUPPORTED,
            version=prior_claim.version + 1,
            parent_claim_id=prior_claim.claim_id,
            author_agent_id=author_agent_id,
            task_id=prior_claim.task_id,
            evidence_ids=merged_evidence,
            depends_on_claim_ids=resolved_dep_ids,
            premise_refs=resolved_premises,
            tool_execution_ids=tool_ids,
            requirement_refs=list(prior_claim.requirement_refs),
            assumptions=assumptions if assumptions is not None else list(prior_claim.assumptions),
            revision_notes=revision_notes,
            confidence=confidence if confidence is not None else prior_claim.confidence,
            is_material=is_material,
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
        """Backward-compatible wrapper around apply_targeted_revision."""
        return await self.apply_targeted_revision(
            run_id=prior_claim.run_id,
            target_claim_id=prior_claim.claim_id,
            expected_version=prior_claim.version,
            new_statement=new_statement,
            author_agent_id=author_agent_id,
            original_prompt="",
            additional_evidence_ids=additional_evidence_ids,
            assumptions=assumptions,
            revision_notes=revision_notes,
            is_unresolved=is_unresolved,
            confidence=confidence,
            require_same_author=False,
        )

    async def get_claims(self, run_id: str, *, latest_only: bool = False) -> list[Claim]:
        """Fetch claims for a run (all immutable versions or latest per claim_id)."""
        return await self._db.get_claims_for_run(run_id, latest_only=latest_only)
