"""Versioned claim ledger with provenance validation and arithmetic tool integration."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from crewforge.core.claim_graph import ClaimCycleError, ClaimDependencyGraph
from crewforge.core.evidence_verifier import verify_evidence_proposal
from crewforge.schemas import (
    ArithmeticRequest,
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
from crewforge.tools.arithmetic import (
    DeterministicArithmeticTool,
    ProvenanceCatalog,
    build_provenance_catalog,
    make_tool_target,
)

_NON_MATERIAL_CATEGORIES = frozenset(
    {
        ClaimCategory.ASSUMPTION,
        ClaimCategory.RECOMMENDATION,
        ClaimCategory.UNCERTAINTY,
    }
)


class ClaimRevisionError(ValueError):
    """Raised when a targeted claim revision reference is invalid, stale, or unauthorized."""


def _calculation_premises(
    record: ToolExecutionRecord,
    claims_by_id: dict[str, Claim],
    owners: dict[str, Claim],
) -> list[str]:
    """Translate numerical lineage into ledger dependencies the graph can invalidate."""
    dependencies: list[str] = []
    for detail in record.provenance_details.values():
        owner = owners.get(detail.get("source_tool_execution_id", ""))
        if owner is None and detail.get("kind") == "supported_claim":
            claim_id = detail["resolved_id"].partition("@v")[0]
            owner = claims_by_id.get(claim_id)
        if owner is not None and owner.claim_id not in dependencies:
            dependencies.append(owner.claim_id)
    return dependencies


def _same_source_calculation(
    request: ArithmeticRequest, record: ToolExecutionRecord, catalog: ProvenanceCatalog,
) -> bool:
    """Reuse only an identical request whose operand identities are still current."""
    if (
        record.expression != request.expression.strip() or record.units != request.units.strip()
        or request.assumptions != record.assumptions or record.matches_proposed is not True
        or str(request.proposed_result) != record.proposed_result
        or len(request.variable_bindings) != len(record.variables)
        or {k: str(v) for k, v in request.variables.items()} != record.variables
    ):
        return False
    for binding in request.variable_bindings:
        source = catalog.targets_by_key.get(binding.provenance_ref)
        detail = record.provenance_details.get(binding.name, {})
        if (
            source is None or not source.is_trusted or source.target_id != detail.get("resolved_id")
            or source.tool_execution_id != detail.get("source_tool_execution_id")
            or binding.units.strip().lower() != detail.get("units")
            or binding.source_quote != detail.get("source_quote")
        ):
            return False
    return True


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
        task_id: str | None,
        original_prompt: str,
        dependency_findings_text: str,
        evidence_proposals: list[ProposedEvidence],
        claim_proposals: list[ProposedClaim],
        plan_assumptions: list[str] | None = None,
        recovery_tasks: dict[str, str] | None = None,
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
        catalog = build_provenance_catalog(
            original_prompt=original_prompt,
            verified_evidence=verified_evidence_all,
            all_claims=existing_all,
            tool_records=prior_tool_records,
            assumptions=plan_assumptions,
            run_id=run_id,
        )

        # Execute computation requests attached to claims before validating computation evidence.
        aliases_by_position: list[set[str]] = []
        for position, proposal in enumerate(claim_proposals, start=1):
            aliases = {f"clm_{position}"}
            ref = proposal.claim_ref
            if (
                ref and ref.strip() == ref and ref != "user_prompt"
                and ref not in catalog.targets_by_key and ":" not in ref and "@v" not in ref
                and not ref.startswith(("up_", "doc_", "tlx_", "evd_"))
            ):
                aliases.add(ref)
            aliases_by_position.append(aliases)
        counts = Counter(ref for aliases in aliases_by_position for ref in aliases)
        local_aliases = [
            {ref for ref in aliases if counts[ref] == 1
             and (not ref.startswith("clm_") or ref == f"clm_{i}")}
            for i, aliases in enumerate(aliases_by_position, start=1)
        ]

        # Retain original positions when pruning a recovery response. A referenced local
        # prerequisite may be untagged, but an unrelated repeated calculation has no work to do.
        selected_tasks = dict.fromkeys(range(len(claim_proposals)), task_id)
        if recovery_tasks is not None:
            selected_tasks = {}
            needed: dict[str, str] = {}
            for idx in reversed(range(len(claim_proposals))):
                proposal = claim_proposals[idx]
                assigned = next((
                    recovery_tasks[tag] for tag in proposal.requirement_refs
                    if tag in recovery_tasks
                ), None)
                for alias in sorted(local_aliases[idx]):
                    if alias in needed:
                        assigned = assigned or needed[alias]
                if assigned is None:
                    continue
                selected_tasks[idx] = assigned
                refs = list(proposal.premise_claim_ids)
                if proposal.computation:
                    refs.extend(proposal.computation.input_provenance_refs.values())
                needed.update(dict.fromkeys(refs, assigned))

        existing_refs = {**existing_by_id, **{
            f"{c.claim_id}@v{c.version}": c for c in existing_claims
        }}
        premise_available = {
            ref: claim.is_material and claim.status in {ClaimStatus.ASSERTED, ClaimStatus.SUPPORTED}
            for ref, claim in existing_refs.items()
        }
        claim_tool_records: dict[int, ToolExecutionRecord] = {}
        all_new_tool_records: list[ToolExecutionRecord] = []
        reviewed_calculations: dict[tuple[str | None, str], list[Claim]] = {}
        if recovery_tasks:
            for c in existing_claims:
                target = catalog.targets_by_key[c.claim_id]
                if (
                    c.status == ClaimStatus.SUPPORTED and target.is_trusted
                    and target.numeric_value is not None
                ):
                    reviewed_calculations.setdefault((c.task_id, c.statement), []).append(c)
        prior_tools = {r.tool_execution_id: r for r in prior_tool_records}
        reused: dict[int, Claim] = {}
        for idx, prop_claim in enumerate(claim_proposals):
            if idx not in selected_tasks:
                continue
            ready = (
                not prop_claim.is_unresolved and prop_claim.category not in _NON_MATERIAL_CATEGORIES
                and all(premise_available.get(ref) for ref in prop_claim.premise_claim_ids)
            )
            if (
                ready and prop_claim.category == ClaimCategory.COMPUTATION
                and prop_claim.computation and not prop_claim.evidence_indices
                and not prop_claim.premise_claim_ids
            ):
                key = (selected_tasks[idx], prop_claim.statement)
                candidates = reviewed_calculations.get(key, [])
                matches = [
                    c for c in candidates if c.assumptions == prop_claim.assumptions
                    and set(prop_claim.requirement_refs) <= set(c.requirement_refs)
                    and len(c.tool_execution_ids) == 1 and _same_source_calculation(
                        prop_claim.computation, prior_tools[c.tool_execution_ids[0]], catalog,
                    )
                ]
                if len(matches) == 1:
                    reused[idx] = matches[0]
                    for alias in local_aliases[idx]:
                        catalog.targets_by_key[alias] = catalog.targets_by_key[matches[0].claim_id]
                        premise_available[alias] = True
                    continue
            if prop_claim.computation is not None:
                rec = await self._arithmetic.execute(
                    run_id=run_id,
                    actor=agent_id,
                    request=prop_claim.computation,
                    catalog=catalog,
                    task_id=selected_tasks[idx],
                    invalid_premise_refs=[
                        pid for pid in prop_claim.premise_claim_ids
                        if not premise_available.get(pid)
                    ],
                )
                claim_tool_records[idx] = rec
                all_new_tool_records.append(rec)
                target = make_tool_target(rec)
                ready = ready and target.is_trusted
                if prop_claim.category == ClaimCategory.COMPUTATION:
                    # Even a failed intermediate has an identity. Preserve its edge for
                    # revalidation without ever treating the failed result as an operand.
                    target.is_trusted = ready
                    catalog.targets_by_key[rec.tool_execution_id] = target
                    for alias in local_aliases[idx]:
                        catalog.targets_by_key.setdefault(alias, target)
            for alias in local_aliases[idx]:
                premise_available.setdefault(alias, ready)

        combined_tool_records = [
            r for r in [*prior_tool_records, *all_new_tool_records]
            if (target := catalog.targets_by_key.get(r.tool_execution_id)) and target.is_trusted
        ]

        persisted_evidence: list[Evidence] = []
        retrieved_sources_by_ref = {
            ref: source.text for ref, source in catalog.targets_by_key.items()
            if source.is_trusted and source.source_type == "external_retrieved"
        }
        retrieved_sources_text = "\n".join(dict.fromkeys(retrieved_sources_by_ref.values()))
        retrieved_refs = set(retrieved_sources_by_ref.keys())
        for proposal in evidence_proposals:
            effective_source_type = proposal.source_type
            check = verify_evidence_proposal(
                proposal=proposal,
                original_prompt=original_prompt,
                dependency_findings_text=dependency_findings_text,
                matching_tool_records=combined_tool_records,
                retrieved_sources_text=retrieved_sources_text,
                retrieved_sources_by_ref=retrieved_sources_by_ref,
            )
            # If a specialist cited an attached/retrieved document using `dependency_finding`
            # or `user_prompt`, re-verify strictly against `external_retrieved` when it matches
            # a known retrieved doc ref or verbatim retrieved text.
            if (
                not check.verified_in_input
                and retrieved_sources_by_ref
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
                    retrieved_sources_by_ref=retrieved_sources_by_ref,
                )
                if retrieved_check.verified_in_input:
                    effective_source_type = "external_retrieved"
                    check = retrieved_check
            doc_target = (
                catalog.targets_by_key.get(proposal.source_ref)
                if effective_source_type == "external_retrieved" and check.verified_in_input
                else None
            )
            evd = Evidence(
                run_id=run_id,
                source_type=effective_source_type,
                source_ref=proposal.source_ref,
                excerpt=proposal.excerpt,
                source_location=f"excerpt:{doc_target.target_id}" if doc_target else None,
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
        owners = {tid: claim for claim in existing_claims for tid in claim.tool_execution_ids}

        # Persist all valid proposed claims within existing run/call budgets.
        for idx, prop_claim in enumerate(claim_proposals):
            if idx not in selected_tasks:
                continue
            if idx in reused:
                local_ref_map.update(dict.fromkeys(local_aliases[idx], reused[idx]))
                continue
            linked_evidence_ids = [
                persisted_evidence[e_idx].evidence_id
                for e_idx in prop_claim.evidence_indices
                if 0 <= e_idx < len(persisted_evidence)
            ]

            premise_refs: list[ClaimVersionRef] = []
            dep_claim_ids: list[str] = []
            record = claim_tool_records.get(idx)
            calculation_premises = (
                _calculation_premises(record, existing_by_id, owners) if record else []
            )
            for raw_pid in [*prop_claim.premise_claim_ids, *calculation_premises]:
                target = (existing_refs.get(raw_pid) or existing_by_id.get(raw_pid)
                          or local_ref_map.get(raw_pid))
                if target is not None and target.claim_id not in dep_claim_ids:
                    premise_refs.append(
                        ClaimVersionRef(claim_id=target.claim_id, claim_version=target.version)
                    )
                    dep_claim_ids.append(target.claim_id)

            if (record is None or prop_claim.category != ClaimCategory.COMPUTATION) and any(
                ref not in existing_refs and ref not in local_ref_map
                for ref in prop_claim.premise_claim_ids
            ):
                await self._db.append_event(ExecutionEvent(
                    run_id=run_id, actor=agent_id, event_type="claim_proposal_rejected",
                    message="Unknown, ambiguous or obsolete premise reference.",
                    payload={"claim_position": idx + 1, "premises": prop_claim.premise_claim_ids},
                ))
                continue
            tool_ids = [record.tool_execution_id] if record else []

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
                task_id=selected_tasks[idx],
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
            for alias in local_aliases[idx]:
                local_ref_map[alias] = claim
            for tid in tool_ids:
                owners[tid] = claim
            existing_by_id[claim.claim_id] = claim

        if reused:
            await self._db.append_event(ExecutionEvent(
                run_id=run_id, event_type="calculations_reused", actor=agent_id,
                message="Used current reviewed calculations for repeated recovery proposals.",
                payload={"versions": [f"{c.claim_id}@v{c.version}" for c in reused.values()]},
            ))
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
        computation_request: ArithmeticRequest | None = None,
        plan_assumptions: list[str] | None = None,
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
        calculation_premises: list[str] = []
        all_claims_run = None
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
                assumptions=plan_assumptions,
                run_id=run_id,
            )
            rec = await self._arithmetic.execute(
                run_id=run_id,
                actor=author_agent_id,
                request=computation_request,
                catalog=catalog,
                task_id=prior_claim.task_id,
            )
            tool_ids = [rec.tool_execution_id]
            owners = {tid: claim for claim in latest_claims for tid in claim.tool_execution_ids}
            calculation_premises = _calculation_premises(rec, latest_by_id, owners)

        resolved_premises: list[ClaimVersionRef] = []
        for pref in prior_claim.premise_refs:
            p_latest = latest_by_id.get(pref.claim_id)
            if p_latest is not None:
                resolved_premises.append(
                    ClaimVersionRef(claim_id=p_latest.claim_id, claim_version=p_latest.version)
                )
            else:
                resolved_premises.append(pref)
        resolved_dep_ids = list(prior_claim.depends_on_claim_ids)
        if premise_claim_ids is not None:
            resolved_premises = []
            resolved_dep_ids = []
            for pid in premise_claim_ids:
                cid, sep, version = pid.partition("@v")
                p_claim = latest_by_id.get(cid)
                if p_claim and sep and str(p_claim.version) != version:
                    raise ClaimRevisionError(f"Superseded premise reference '{pid}'.")
                if p_claim is None:
                    raise CrossRunReferenceError(
                        f"Premise claim '{pid}' does not exist in run '{run_id}'."
                    )
                if cid == target_claim_id:
                    raise ClaimRevisionError(f"Claim '{target_claim_id}' cannot depend on itself.")
                resolved_premises.append(
                    ClaimVersionRef(claim_id=p_claim.claim_id, claim_version=p_claim.version)
                )
                if p_claim.claim_id not in resolved_dep_ids:
                    resolved_dep_ids.append(p_claim.claim_id)

        for pid in calculation_premises:
            if pid == target_claim_id:
                raise ClaimRevisionError(f"Claim '{target_claim_id}' cannot depend on itself.")
            if pid not in {pref.claim_id for pref in resolved_premises}:
                resolved_premises.append(
                    ClaimVersionRef(claim_id=pid, claim_version=latest_by_id[pid].version)
                )
            if pid not in resolved_dep_ids:
                resolved_dep_ids.append(pid)

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
        if premise_claim_ids is not None or calculation_premises:
            if all_claims_run is None:
                all_claims_run = await self._db.get_claims_for_run(run_id, latest_only=False)
            try:
                ClaimDependencyGraph.validate_graph([*all_claims_run, revised], run_id=run_id)
            except ClaimCycleError as exc:
                raise ClaimRevisionError(str(exc)) from exc
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
