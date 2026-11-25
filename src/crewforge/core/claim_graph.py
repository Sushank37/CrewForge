"""Deterministic claim dependency graph and cascading stale invalidation."""

from __future__ import annotations

from collections import deque

from crewforge.schemas import (
    Claim,
    ClaimStatus,
    ExecutionEvent,
    VerificationStatus,
)
from crewforge.storage.database import CrossRunReferenceError, Database


class ClaimCycleError(ValueError):
    """Raised when claim premise dependencies form a cycle."""


class ClaimDependencyGraph:
    """Validates version-pinned claim dependencies and propagates invalidations transitively."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def validate_graph(claims: list[Claim], *, run_id: str) -> None:
        """Validate same-run ownership, existing premise versions, and acyclicity."""
        by_id_ver: dict[tuple[str, int], Claim] = {}
        latest_ids: set[str] = set()
        for claim in claims:
            if claim.run_id != run_id:
                raise CrossRunReferenceError(
                    f"Claim '{claim.claim_id}' belongs to '{claim.run_id}', not '{run_id}'."
                )
            by_id_ver[(claim.claim_id, claim.version)] = claim
            latest_ids.add(claim.claim_id)

        adjacency: dict[str, list[str]] = {cid: [] for cid in latest_ids}
        in_degree: dict[str, int] = dict.fromkeys(latest_ids, 0)

        # Use latest version per claim_id for cycle detection.
        latest_by_id: dict[str, Claim] = {}
        for claim in claims:
            prev = latest_by_id.get(claim.claim_id)
            if prev is None or claim.version > prev.version:
                latest_by_id[claim.claim_id] = claim

        for claim in latest_by_id.values():
            seen_parents: set[str] = set()
            for pref in claim.premise_refs:
                if (pref.claim_id, pref.claim_version) not in by_id_ver:
                    raise CrossRunReferenceError(
                        f"Claim '{claim.claim_id}' v{claim.version} references missing premise "
                        f"'{pref.claim_id}' v{pref.claim_version}."
                    )
                if pref.claim_id == claim.claim_id:
                    raise ClaimCycleError(f"Claim '{claim.claim_id}' cannot depend on itself.")
                if pref.claim_id not in seen_parents:
                    seen_parents.add(pref.claim_id)
                    adjacency[pref.claim_id].append(claim.claim_id)
                    in_degree[claim.claim_id] += 1

            for dep_id in claim.depends_on_claim_ids:
                if dep_id not in latest_ids:
                    raise CrossRunReferenceError(
                        f"Claim '{claim.claim_id}' references unknown dependency '{dep_id}'."
                    )
                if dep_id == claim.claim_id:
                    raise ClaimCycleError(f"Claim '{claim.claim_id}' cannot depend on itself.")
                if dep_id not in seen_parents:
                    seen_parents.add(dep_id)
                    adjacency[dep_id].append(claim.claim_id)
                    in_degree[claim.claim_id] += 1

        queue: deque[str] = deque(cid for cid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for nxt in adjacency[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if latest_ids and visited != len(latest_ids):
            raise ClaimCycleError("Claim dependency graph contains a cycle.")

    async def invalidate_dependents(
        self,
        *,
        run_id: str,
        root_claim_id: str,
        reason: str,
        superseded_version: int | None = None,
    ) -> list[Claim]:
        """Transitively mark dependent claims as STALE_PENDING_REVIEW.

        Preserves historical claim rows and prior verdicts; ensures previous
        review verdicts cannot authorize stale downstream claims.
        """
        latest_claims = await self._db.get_claims_for_run(run_id, latest_only=True)
        latest_by_id = {c.claim_id: c for c in latest_claims}

        # Build reverse dependency edges: premise_id -> list[dependent_claims].
        dependents_map: dict[str, list[Claim]] = {cid: [] for cid in latest_by_id}
        for claim in latest_claims:
            parent_ids = set(claim.depends_on_claim_ids) | {p.claim_id for p in claim.premise_refs}
            for pid in parent_ids:
                if pid in dependents_map:
                    dependents_map[pid].append(claim)

        queue: deque[str] = deque([root_claim_id])
        visited_roots: set[str] = {root_claim_id}
        invalidated: list[Claim] = []

        while queue:
            cur_id = queue.popleft()
            for dep_claim in dependents_map.get(cur_id, []):
                if dep_claim.claim_id in visited_roots:
                    continue
                # If superseded_version is given, only invalidate claims pinned to <= that version
                # (or unpinned depends_on_claim_ids).
                if superseded_version is not None and dep_claim.premise_refs:
                    pinned_versions = [
                        p.claim_version for p in dep_claim.premise_refs if p.claim_id == cur_id
                    ]
                    if pinned_versions and all(v > superseded_version for v in pinned_versions):
                        continue

                visited_roots.add(dep_claim.claim_id)
                queue.append(dep_claim.claim_id)

                stale_note = f"Stale due to upstream premise '{cur_id}': {reason}"
                await self._db.update_claim_verification_state(
                    run_id=run_id,
                    claim_id=dep_claim.claim_id,
                    version=dep_claim.version,
                    status=ClaimStatus.STALE,
                    verification_status=VerificationStatus.STALE_PENDING_REVIEW,
                    revision_notes=stale_note,
                )
                updated = dep_claim.model_copy(
                    update={
                        "status": ClaimStatus.STALE,
                        "verification_status": VerificationStatus.STALE_PENDING_REVIEW,
                        "revision_notes": stale_note,
                    }
                )
                invalidated.append(updated)

        if invalidated:
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="claims_invalidated",
                    actor="claim_graph",
                    message=(
                        f"Invalidated {len(invalidated)} dependent claim(s) "
                        f"after change to '{root_claim_id}'."
                    ),
                    payload={
                        "root_claim_id": root_claim_id,
                        "invalidated_claim_ids": [c.claim_id for c in invalidated],
                        "reason": reason,
                    },
                )
            )
        return invalidated
