"""Lead synthesis and release metadata validation (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from crewforge.core.executor import ModelExecutor
from crewforge.prompts.synthesis import (
    build_synthesis_system_prompt,
    build_synthesis_user_prompt,
)
from crewforge.schemas import (
    Claim,
    ExecutionEvent,
    ExecutionPlan,
    SynthesisOutput,
    TaskSpec,
    TaskStatus,
)
from crewforge.storage.database import Database


@dataclass
class ValidatedSynthesis:
    """Validated final synthesis output with cross-run claim reference checks."""

    answer: str
    referenced_claim_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    confidence_notes: str = ""
    invalid_claim_refs_removed: list[str] = field(default_factory=list)
    used_partial_fallback: bool = False


def validate_synthesis_output(
    *,
    raw: SynthesisOutput,
    valid_claim_ids: set[str],
    extra_assumptions: list[str],
    extra_missing: list[str],
    uncertainties: list[str],
    incomplete_tasks: list[TaskSpec],
) -> ValidatedSynthesis:
    """Validate claim references belong to the current run and append Phase 2 honesty notes."""
    kept_refs: list[str] = []
    removed_refs: list[str] = []
    for cid in raw.referenced_claim_ids:
        if cid in valid_claim_ids and cid not in kept_refs:
            kept_refs.append(cid)
        elif cid not in valid_claim_ids:
            removed_refs.append(cid)

    merged_assumptions = list(dict.fromkeys([*raw.assumptions, *extra_assumptions]))
    merged_missing = list(dict.fromkeys([*raw.missing_information, *extra_missing]))

    notes_parts: list[str] = []
    if raw.confidence_notes:
        notes_parts.append(raw.confidence_notes.strip())
    if raw.unverified_assertions:
        notes_parts.append(
            "Synthesis unverified assertions: " + "; ".join(raw.unverified_assertions)
        )
    if uncertainties:
        notes_parts.append("Specialist uncertainties: " + "; ".join(uncertainties))
    if incomplete_tasks:
        notes_parts.append(
            "Incomplete tasks: "
            + ", ".join(f"{t.task_ref or t.task_id} ({t.status.value})" for t in incomplete_tasks)
        )
    if removed_refs:
        notes_parts.append(f"Stripped {len(removed_refs)} invalid/out-of-run claim reference(s).")
    notes_parts.append(
        "Phase 2 status: claims are recorded as asserted/unresolved and have not undergone "
        "Phase 3 independent review or deterministic verification."
    )

    return ValidatedSynthesis(
        answer=raw.answer.strip(),
        referenced_claim_ids=kept_refs,
        assumptions=merged_assumptions,
        missing_information=merged_missing,
        confidence_notes=" ".join(notes_parts),
        invalid_claim_refs_removed=removed_refs,
        used_partial_fallback=False,
    )


def build_partial_fallback_synthesis(
    *,
    plan: ExecutionPlan,
    completed_tasks: list[TaskSpec],
    incomplete_tasks: list[TaskSpec],
    claims: list[Claim],
    extra_assumptions: list[str],
    extra_missing: list[str],
    uncertainties: list[str],
    reason: str,
) -> ValidatedSynthesis:
    """Assemble an explicit partial result when budget/deadline prevents Lead synthesis call."""
    lines = [f"[Partial Result — Lead synthesis call skipped: {reason}]"]
    if completed_tasks:
        lines.append("\nCompleted specialist findings:")
        for task in completed_tasks:
            lines.append(
                f"- {task.task_ref or task.task_id} ({task.title}): {task.findings or 'No text.'}"
            )
    if incomplete_tasks:
        lines.append("\nIncomplete tasks:")
        for task in incomplete_tasks:
            lines.append(
                f"- {task.task_ref or task.task_id} ({task.title}): "
                f"{task.status.value} ({task.error or 'incomplete'})"
            )

    notes = (
        f"Partial synthesis fallback triggered ({reason}). "
        "Claims remain unverified (Phase 3 reviewer pool not yet implemented)."
    )
    if uncertainties:
        notes += " Uncertainties: " + "; ".join(uncertainties)

    return ValidatedSynthesis(
        answer="\n".join(lines),
        referenced_claim_ids=[c.claim_id for c in claims],
        assumptions=list(dict.fromkeys([*plan.assumptions, *extra_assumptions])),
        missing_information=list(dict.fromkeys([*plan.missing_information, *extra_missing])),
        confidence_notes=notes,
        used_partial_fallback=True,
    )


class LeadSynthesizer:
    """Synthesizes completed specialist findings and claim ledger into a validated answer."""

    def __init__(self, *, executor: ModelExecutor, database: Database) -> None:
        self._executor = executor
        self._db = database

    async def synthesize(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        tasks: list[TaskSpec],
        claims: list[Claim],
        extra_assumptions: list[str],
        extra_missing: list[str],
        uncertainties: list[str],
        run_deadline_monotonic: float | None = None,
    ) -> ValidatedSynthesis:
        """Execute Lead synthesis call and validate referenced claim IDs."""
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        incomplete_tasks = [t for t in tasks if t.status != TaskStatus.COMPLETED]
        valid_claim_ids = {c.claim_id for c in claims}

        system_prompt = build_synthesis_system_prompt()
        user_prompt = build_synthesis_user_prompt(
            original_prompt=original_prompt,
            plan=plan,
            completed_tasks=completed_tasks,
            incomplete_tasks=incomplete_tasks,
            claims=claims,
        )

        res = await self._executor.execute_structured(
            run_id=run_id,
            actor="lead",
            operation="synthesis",
            prompt=user_prompt,
            response_model=SynthesisOutput,
            system_prompt=system_prompt,
            run_deadline_monotonic=run_deadline_monotonic,
        )
        validated = validate_synthesis_output(
            raw=res.parsed,
            valid_claim_ids=valid_claim_ids,
            extra_assumptions=extra_assumptions,
            extra_missing=extra_missing,
            uncertainties=uncertainties,
            incomplete_tasks=incomplete_tasks,
        )
        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="synthesis_completed",
                actor="lead",
                message="Lead synthesis completed and claim references validated.",
                payload={
                    "referenced_claim_ids": validated.referenced_claim_ids,
                    "invalid_claim_refs_removed": validated.invalid_claim_refs_removed,
                },
            )
        )
        return validated
