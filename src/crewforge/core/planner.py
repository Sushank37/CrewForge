"""Dynamic Lead planner that decomposes arbitrary prompts into validated ExecutionPlans."""

from __future__ import annotations

from collections import deque
from uuid import uuid4

from crewforge.config import CrewForgeSettings
from crewforge.core.executor import ModelExecutor
from crewforge.prompts.planner import (
    build_plan_repair_prompt,
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from crewforge.providers.base import ProviderResponseError
from crewforge.schemas import (
    AgentSpec,
    ExecutionEvent,
    ExecutionPlan,
    PlannerOutput,
    TaskSpec,
    TaskStatus,
)
from crewforge.storage.database import Database


class PlanValidationError(Exception):
    """Raised when a model-generated execution plan fails structural validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_and_materialize_plan(
    *,
    run_id: str,
    raw_plan: PlannerOutput,
    settings: CrewForgeSettings,
) -> ExecutionPlan:
    """Validate a compact PlannerOutput and assign trusted CrewForge identifiers."""
    errors: list[str] = []
    allowed_caps = set(settings.allowed_capabilities)
    max_workers = settings.effective_max_workers
    max_tasks = settings.effective_max_tasks

    if len(raw_plan.agents) > max_workers:
        errors.append(f"Agent count {len(raw_plan.agents)} exceeds max_workers ({max_workers}).")
    if len(raw_plan.tasks) > max_tasks:
        errors.append(f"Task count {len(raw_plan.tasks)} exceeds max_tasks ({max_tasks}).")
    if not [c for c in raw_plan.acceptance_criteria if c.strip()]:
        errors.append("Plan must define at least one non-empty acceptance criterion.")
    if not [r for r in raw_plan.user_requirements if r.strip()]:
        errors.append("Plan must define at least one non-empty user requirement.")

    agent_refs: dict[str, AgentSpec] = {}
    for idx, p_agent in enumerate(raw_plan.agents, start=1):
        ref = p_agent.agent_ref.strip()
        if not ref:
            errors.append(f"Agent #{idx} has empty agent_ref.")
            continue
        if ref in agent_refs:
            errors.append(f"Duplicate agent_ref '{ref}'.")
            continue
        if p_agent.model_override is not None:
            errors.append(
                f"Agent '{ref}' specified model_override='{p_agent.model_override}', "
                "which is prohibited in same-model runs."
            )
        unsupported_caps = [cap for cap in p_agent.capabilities if cap not in allowed_caps]
        if unsupported_caps:
            errors.append(f"Agent '{ref}' requested unsupported capabilities: {unsupported_caps}.")

        agent_id = f"agt_{run_id[-6:]}_{idx}_{uuid4().hex[:4]}"
        agent_refs[ref] = AgentSpec(
            agent_id=agent_id,
            agent_ref=ref,
            run_id=run_id,
            role=p_agent.role.strip(),
            instructions=p_agent.instructions.strip(),
            capabilities=list(dict.fromkeys(p_agent.capabilities)) or ["reasoning"],
            model_override=None,
            created_by="lead",
        )

    task_ref_to_id: dict[str, str] = {}
    seen_task_refs: set[str] = set()
    for idx, p_task in enumerate(raw_plan.tasks, start=1):
        t_ref = p_task.task_ref.strip()
        if not t_ref:
            errors.append(f"Task #{idx} has empty task_ref.")
            continue
        if t_ref in seen_task_refs:
            errors.append(f"Duplicate task_ref '{t_ref}'.")
            continue
        seen_task_refs.add(t_ref)
        task_ref_to_id[t_ref] = f"tsk_{run_id[-6:]}_{idx}_{uuid4().hex[:4]}"

    adjacency: dict[str, list[str]] = {ref: [] for ref in seen_task_refs}
    in_degree: dict[str, int] = dict.fromkeys(seen_task_refs, 0)
    plan_id = f"pln_{uuid4().hex[:12]}"
    materialized_tasks: list[TaskSpec] = []

    for p_task in raw_plan.tasks:
        t_ref = p_task.task_ref.strip()
        if t_ref not in task_ref_to_id:
            continue
        if p_task.assigned_agent_ref not in agent_refs:
            errors.append(
                f"Task '{t_ref}' references unknown assigned_agent_ref "
                f"'{p_task.assigned_agent_ref}'."
            )
        cleaned_deliverables = [d.strip() for d in p_task.expected_deliverables if d.strip()]
        if not cleaned_deliverables:
            errors.append(f"Task '{t_ref}' must define at least one non-empty deliverable.")

        dep_ids: list[str] = []
        seen_deps: set[str] = set()
        for dep_ref in p_task.depends_on:
            dep_clean = dep_ref.strip()
            if dep_clean == t_ref:
                errors.append(f"Task '{t_ref}' cannot depend on itself.")
                continue
            if dep_clean not in task_ref_to_id:
                errors.append(f"Task '{t_ref}' depends on unknown task_ref '{dep_clean}'.")
                continue
            if dep_clean not in seen_deps:
                seen_deps.add(dep_clean)
                dep_ids.append(task_ref_to_id[dep_clean])
                adjacency[dep_clean].append(t_ref)
                in_degree[t_ref] += 1

        if p_task.assigned_agent_ref in agent_refs:
            materialized_tasks.append(
                TaskSpec(
                    task_id=task_ref_to_id[t_ref],
                    task_ref=t_ref,
                    run_id=run_id,
                    plan_id=plan_id,
                    title=p_task.title.strip(),
                    description=p_task.description.strip(),
                    assigned_agent_id=agent_refs[p_task.assigned_agent_ref].agent_id,
                    depends_on=dep_ids,
                    expected_outputs=cleaned_deliverables,
                    acceptance_criteria=[
                        c.strip() for c in p_task.acceptance_criteria if c.strip()
                    ],
                    status=TaskStatus.PENDING,
                    created_by="lead",
                )
            )

    queue: deque[str] = deque(ref for ref, deg in in_degree.items() if deg == 0)
    visited_count = 0
    while queue:
        current = queue.popleft()
        visited_count += 1
        for nxt in adjacency[current]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if seen_task_refs and visited_count != len(seen_task_refs):
        errors.append("Task dependency graph contains a cycle.")

    requirements = [r.strip() for r in raw_plan.user_requirements if r.strip()]
    if raw_plan.deliverables is not None:
        req_ids = {f"req_{i}" for i in range(1, len(requirements) + 1)}
        assigned_req_ids = {d.requirement_ref for d in raw_plan.deliverables}
        deliverable_ids = {d.deliverable_id for d in raw_plan.deliverables}
        if req_ids != assigned_req_ids:
            errors.append("Deliverables must cover every numbered user requirement.")
        if len(deliverable_ids) != len(raw_plan.deliverables):
            errors.append("Deliverable IDs must be unique.")
        for d in raw_plan.deliverables:
            if d.task_ref not in seen_task_refs or not d.text.strip():
                errors.append(f"Deliverable {d.deliverable_id} needs text and a valid task_ref.")

    if errors:
        raise PlanValidationError(errors)

    return ExecutionPlan(
        plan_id=plan_id,
        run_id=run_id,
        goal=raw_plan.goal.strip(),
        user_requirements=requirements,
        deliverables=raw_plan.deliverables or [],
        acceptance_criteria=[c.strip() for c in raw_plan.acceptance_criteria if c.strip()],
        agents=list(agent_refs.values()),
        tasks=materialized_tasks,
        assumptions=[a.strip() for a in raw_plan.assumptions if a.strip()],
        missing_information=[m.strip() for m in raw_plan.missing_information if m.strip()],
        created_by="lead",
    )


class LeadPlanner:
    """Generates, validates, optionally repairs (at most once), and persists ExecutionPlans."""

    def __init__(
        self,
        *,
        executor: ModelExecutor,
        database: Database,
        settings: CrewForgeSettings,
    ) -> None:
        self._executor = executor
        self._db = database
        self._settings = settings

    async def create_plan(
        self,
        *,
        run_id: str,
        user_prompt: str,
        run_deadline_monotonic: float | None = None,
    ) -> ExecutionPlan:
        """Generate and persist a validated ExecutionPlan for an arbitrary prompt."""
        system_prompt = build_planner_system_prompt(
            max_workers=self._settings.effective_max_workers,
            max_tasks=self._settings.effective_max_tasks,
            allowed_capabilities=self._settings.allowed_capabilities,
        )
        planner_prompt = build_planner_user_prompt(user_prompt)

        previous_raw_json = ""
        validation_errors: list[str] = []

        try:
            res = await self._executor.execute_structured(
                run_id=run_id,
                actor="lead",
                operation="planner",
                prompt=planner_prompt,
                response_model=PlannerOutput,
                system_prompt=system_prompt,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            previous_raw_json = res.raw_text
            plan = validate_and_materialize_plan(
                run_id=run_id,
                raw_plan=res.parsed,
                settings=self._settings,
            )
        except (PlanValidationError, ProviderResponseError) as exc:
            validation_errors = (
                exc.errors if isinstance(exc, PlanValidationError) else [exc.message]
            )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="plan_validation_failed",
                    actor="lead",
                    message="Initial plan failed validation; attempting single budgeted repair.",
                    payload={"errors": validation_errors},
                )
            )
            repair_prompt = build_plan_repair_prompt(
                user_prompt=user_prompt,
                previous_plan_json=previous_raw_json or "{}",
                validation_errors=validation_errors,
            )
            repair_res = await self._executor.execute_structured(
                run_id=run_id,
                actor="lead",
                operation="plan_repair",
                prompt=repair_prompt,
                response_model=PlannerOutput,
                system_prompt=system_prompt,
                run_deadline_monotonic=run_deadline_monotonic,
            )
            plan = validate_and_materialize_plan(
                run_id=run_id,
                raw_plan=repair_res.parsed,
                settings=self._settings,
            )
            await self._db.append_event(
                ExecutionEvent(
                    run_id=run_id,
                    event_type="plan_repaired",
                    actor="lead",
                    message="Plan repair succeeded and passed validation.",
                    payload={"plan_id": plan.plan_id},
                )
            )

        await self._db.save_execution_plan(plan)
        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="plan_created",
                actor="lead",
                message=(
                    f"Persisted execution plan with {len(plan.agents)} specialist(s) "
                    f"and {len(plan.tasks)} task(s)."
                ),
                payload={
                    "plan_id": plan.plan_id,
                    "roles": [a.role for a in plan.agents],
                    "task_refs": [t.task_ref for t in plan.tasks],
                },
            )
        )
        return plan
