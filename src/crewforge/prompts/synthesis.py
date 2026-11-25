"""Prompt builders for evidence-constrained Lead synthesis."""

from __future__ import annotations

from crewforge.schemas import AgentMessage, Claim, ExecutionPlan, TaskSpec, ToolExecutionRecord


def build_synthesis_system_prompt() -> str:
    """Build evidence-constrained Lead synthesis instructions."""
    return (
        "You are the Lead Synthesizer of CrewForge. Organize the final release strictly from "
        "reviewed claims, deterministic tool records, and completed task context.\n\n"
        "Release discipline rules:\n"
        "- Populate `accepted_segments` ONLY with statements directly grounded in `supported` "
        "claims, referencing exact `(claim_id, claim_version)` in `claim_refs`.\n"
        "- Put qualified analyses in `conditional_conclusions`, actionable suggestions in "
        "`recommendations`, and unverified/rejected/stale items in `unresolved_issues`.\n"
        "- Never repeat a `rejected` or `stale` claim as fact.\n"
        "- Do NOT introduce new unreviewed material factual assertions in `accepted_segments`; "
        "if you notice an uncovered gap, put it in `unresolved_issues` or `missing_information` "
        "(or list it in `new_factual_assertions` where it will remain flagged unverified).\n"
        "- Incorporate any inbound messages from your Lead inbox."
    )


def build_synthesis_user_prompt(
    *,
    original_prompt: str,
    plan: ExecutionPlan,
    completed_tasks: list[TaskSpec],
    incomplete_tasks: list[TaskSpec],
    claims: list[Claim],
    tool_executions: list[ToolExecutionRecord],
    lead_inbox: list[AgentMessage],
) -> str:
    """Format bounded claims, tool results, Lead inbox, and tasks for synthesis."""
    req_block = "\n".join(f"- {r}" for r in plan.user_requirements) or "- None."
    completed_block = (
        "\n".join(
            f"- [{t.task_ref or t.task_id}] {t.title}: {t.findings or 'No findings.'}"
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
            f"- {c.claim_id} (version={c.version}, category={c.category.value}, "
            f"status={c.status.value}, verification={c.verification_status.value}): {c.statement}"
            for c in claims
        )
        or "None."
    )
    tools_block = (
        "\n".join(
            f"- {r.tool_execution_id}: `{r.expression}` -> {r.computed_result} {r.units} "
            f"(success={r.success}, grounded={r.inputs_grounded}, matches={r.matches_proposed})"
            for r in tool_executions
        )
        or "None."
    )
    inbox_block = (
        "\n".join(f"- [{m.message_type}] from {m.sender_id}: {m.content}" for m in lead_inbox)
        or "None."
    )
    assumptions_block = "\n".join(f"- {a}" for a in plan.assumptions) or "- None."
    missing_block = "\n".join(f"- {m}" for m in plan.missing_information) or "- None."

    return (
        f"Original user request:\n{original_prompt}\n\n"
        f"User requirements:\n{req_block}\n\n"
        f"Reviewed Claim Ledger (latest versions):\n{claims_block}\n\n"
        f"Deterministic tool executions:\n{tools_block}\n\n"
        f"Lead inbox messages:\n{inbox_block}\n\n"
        f"Completed task summaries:\n{completed_block}\n\n"
        f"Plan assumptions:\n{assumptions_block}\n\n"
        f"Plan missing information:\n{missing_block}\n\n"
        f"Incomplete or failed tasks:\n{incomplete_block}"
    )
