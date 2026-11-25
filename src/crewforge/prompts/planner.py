"""Prompt builders for the dynamic Lead planner."""

from __future__ import annotations


def build_planner_system_prompt(
    *,
    max_workers: int,
    max_tasks: int,
    allowed_capabilities: list[str],
) -> str:
    """Build domain-agnostic system instructions for Lead task decomposition."""
    caps = ", ".join(allowed_capabilities)
    default_upper = min(4, max_workers)
    default_lower = min(2, max_workers)
    return (
        "You are the Lead Coordinator of CrewForge, a model-agnostic multi-agent system. "
        "Given an arbitrary user prompt, design a concise, non-overlapping specialist team "
        "and dependency DAG to solve it thoroughly.\n\n"
        "Strict rules:\n"
        f"- Create {default_lower} to {default_upper} specialist agents for multi-faceted tasks "
        f"(or 1 specialist for simple tasks), never exceeding {max_workers} agents "
        f"and {max_tasks} tasks.\n"
        "- Use short, unique local handles for `agent_ref` (e.g., 'a1', 'a2') "
        "and `task_ref` (e.g., 't1', 't2').\n"
        "- Every task must set `assigned_agent_ref` to a defined `agent_ref` "
        "and have non-empty `expected_deliverables`.\n"
        "- `depends_on` must reference earlier `task_ref` values and MUST be acyclic.\n"
        "- Independent tasks should have empty `depends_on` so they can execute concurrently.\n"
        f"- Agent `capabilities` may ONLY use items from: [{caps}].\n"
        "- Leave `model_override` as null.\n"
        "- Do not invent external facts or ask for private chain-of-thought. "
        "Record genuine user requirements, acceptance criteria, assumptions, "
        "and missing information."
    )


def build_planner_user_prompt(user_prompt: str) -> str:
    """Format the arbitrary user prompt for the Lead planner."""
    return f"User request:\n{user_prompt}"


def build_plan_repair_prompt(
    *,
    user_prompt: str,
    previous_plan_json: str,
    validation_errors: list[str],
) -> str:
    """Build a single bounded repair prompt when initial plan validation fails."""
    errors_block = "\n".join(f"- {err}" for err in validation_errors)
    return (
        "Your previous execution plan failed structural validation.\n\n"
        f"Validation errors:\n{errors_block}\n\n"
        f"Original user request:\n{user_prompt}\n\n"
        f"Previous plan JSON:\n{previous_plan_json}\n\n"
        "Return a corrected, valid ExecutionPlan JSON addressing all errors."
    )
