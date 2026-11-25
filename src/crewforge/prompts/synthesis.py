"""Prompt builders for Lead final synthesis."""

from __future__ import annotations

from crewforge.schemas import Claim, ExecutionPlan, TaskSpec


def build_synthesis_system_prompt() -> str:
    """Build Lead synthesis instructions emphasizing honesty over false certainty."""
    return (
        "You are the Lead Synthesizer of CrewForge. Combine completed specialist findings and "
        "recorded claims into a clear, accurate, well-structured answer to the user's request.\n\n"
        "Honesty & provenance rules:\n"
        "- Reference relevant `claim_id` values strictly from the provided Claim Ledger.\n"
        "- Note explicitly that claims have not yet undergone independent automated review "
        "(Phase 3), and express appropriate uncertainty for material assertions lacking "
        "independent verification.\n"
        "- Never present unverified claims, external citations, or incomplete tasks "
        "as verified facts.\n"
        "- If you introduce any new substantive assertion not present in specialist findings, "
        "list it in `unverified_assertions`.\n"
        "- Clearly surface `assumptions`, `missing_information`, and any failed/incomplete tasks."
    )


def build_synthesis_user_prompt(
    *,
    original_prompt: str,
    plan: ExecutionPlan,
    completed_tasks: list[TaskSpec],
    incomplete_tasks: list[TaskSpec],
    claims: list[Claim],
) -> str:
    """Format bounded findings, claims, and incomplete tasks for Lead synthesis."""
    req_block = "\n".join(f"- {r}" for r in plan.user_requirements) or "- None."
    completed_block = (
        "\n\n".join(
            f"Task [{t.task_ref or t.task_id}] {t.title} (status={t.status.value}):\n"
            f"{t.findings or 'No findings recorded.'}"
            for t in completed_tasks
        )
        or "None."
    )
    incomplete_block = (
        "\n".join(
            f"- [{t.task_ref or t.task_id}] {t.title}: "
            f"status={t.status.value}, error={t.error or 'blocked'}"
            for t in incomplete_tasks
        )
        or "None."
    )
    claims_block = (
        "\n".join(
            f"- {c.claim_id} (v{c.version}, status={c.status.value}, "
            f"evidence={len(c.evidence_ids)}): {c.statement}"
            for c in claims
        )
        or "None."
    )
    assumptions_block = "\n".join(f"- {a}" for a in plan.assumptions) or "- None."
    missing_block = "\n".join(f"- {m}" for m in plan.missing_information) or "- None."

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"User requirements:\n{req_block}\n\n"
        f"Completed specialist findings:\n{completed_block}\n\n"
        f"Current claim ledger:\n{claims_block}\n\n"
        f"Plan assumptions:\n{assumptions_block}\n\n"
        f"Plan missing information:\n{missing_block}\n\n"
        f"Incomplete or failed tasks:\n{incomplete_block}"
    )
