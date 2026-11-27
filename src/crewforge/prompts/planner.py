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
    return (
        "You are the Lead Coordinator of CrewForge, a model-agnostic multi-agent system. "
        "Given an arbitrary user prompt, design a lean, non-overlapping specialist team "
        "and dependency DAG to solve it thoroughly.\n\n"
        "Strict rules:\n"
        f"- HARD LIMIT: return AT MOST {max_workers} agent(s) in `agents` and AT MOST "
        f"{max_tasks} task(s) in `tasks`.\n"
        "- Size the team dynamically to match actual task complexity: use 1-2 specialists for "
        "simple or single-domain requests, and recruit larger teams only when genuinely distinct "
        "sub-domains or independent parallel workstreams require them. Never recruit unnecessary "
        "agents merely for visual effect.\n"
        "- A separate Lead synthesis step and independent reviewer step run automatically AFTER "
        "your DAG completes, so do NOT add a final summary, review, or synthesis task.\n"
        "- Use short, unique local handles for `agent_ref` (e.g., 'a1', 'a2') "
        "and `task_ref` (e.g., 't1', 't2').\n"
        "- Every task must set `assigned_agent_ref` to a defined `agent_ref` "
        "and have non-empty `expected_deliverables`.\n"
        "- Decompose multi-step problems into focused atomic tasks (1-3 concrete deliverables "
        "per task) rather than overloading a single task.\n"
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
        "Return a corrected, valid ExecutionPlan JSON addressing ALL errors "
        "(merge or drop excess agents/tasks if a count exceeded its limit)."
    )
