"""Reviewed, version-specific coverage of atomic user deliverables."""

from crewforge.schemas import (
    Claim,
    ClaimCategory,
    ClaimStatus,
    ExecutionPlan,
    RequirementDeliverable,
    ReviewDecision,
    ReviewVerdict,
)


def deliverable_lines(items: list[RequirementDeliverable]) -> str:
    return "\n".join(
        f"- [{item.deliverable_id} / {item.requirement_ref}] "
        f"({item.kind}, task {item.task_ref}) {item.text}"
        for item in items
    )


def claim_addresses_kind(claim: Claim, item: RequirementDeliverable) -> bool:
    if claim.category == ClaimCategory.UNCERTAINTY:
        return True  # Review must explicitly confirm the missing evidence justifies uncertainty.
    if item.kind == "recommendation":
        return claim.category == ClaimCategory.RECOMMENDATION
    if item.kind == "calculation":
        return claim.category == ClaimCategory.COMPUTATION
    return claim.is_material and claim.category != ClaimCategory.RECOMMENDATION


def reviewed_deliverables(
    plan: ExecutionPlan, claims: list[Claim], verdicts: list[ReviewVerdict],
) -> dict[str, list[Claim]]:
    """Index independently approved current claim versions that explicitly fulfill an obligation."""
    items = {d.deliverable_id: d for d in plan.deliverables}
    latest: dict[str, Claim] = {}
    for claim in claims:
        if claim.claim_id not in latest or claim.version > latest[claim.claim_id].version:
            latest[claim.claim_id] = claim
    approved: dict[tuple[str, int], set[str]] = {}
    blocked = set()
    for verdict in verdicts:
        claim = latest.get(verdict.claim_id)
        if (
            claim is None or claim.version != verdict.claim_version
            or verdict.run_id != claim.run_id
        ):
            continue
        key = (claim.claim_id, claim.version)
        if (
            verdict.reviewer_id == claim.author_agent_id
            or verdict.decision != ReviewDecision.APPROVE
        ):
            blocked.add(key)
            continue
        approved.setdefault(key, set()).update(verdict.covered_deliverable_refs)

    covered: dict[str, list[Claim]] = {}
    for claim in latest.values():
        key = (claim.claim_id, claim.version)
        if (
            claim.status != ClaimStatus.SUPPORTED or key in blocked
            or claim.verification_status.value not in {
                "provenance_validated", "computationally_verified",
                "independently_reviewed_inference",
            }
        ):
            continue
        if any(
            (p := latest.get(ref.claim_id)) is None or p.version != ref.claim_version
            or p.status != ClaimStatus.SUPPORTED or not p.is_material
            for ref in claim.premise_refs
        ):
            continue
        for item_id in approved.get(key, set()):
            item = items.get(item_id)
            if (
                item and item_id in claim.requirement_refs and claim_addresses_kind(claim, item)
            ):
                covered.setdefault(item_id, []).append(claim)
    return covered
