"""Deterministic dependency-aware asyncio DAG task scheduler."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from crewforge.config import CrewForgeSettings, sanitize_error_message
from crewforge.core.budget import BudgetError
from crewforge.core.executor import RunDeadlineExceededError
from crewforge.core.worker import SpecialistWorker, WorkerExecutionOutcome
from crewforge.providers.base import ProviderError
from crewforge.schemas import AgentSpec, ExecutionEvent, ExecutionPlan, TaskSpec, TaskStatus
from crewforge.storage.database import Database


@dataclass
class SchedulerOutcome:
    """Result summary produced by running a DAG ExecutionPlan."""

    tasks_by_id: dict[str, TaskSpec]
    worker_outcomes: dict[str, WorkerExecutionOutcome] = field(default_factory=dict)
    deadline_exceeded: bool = False
    budget_exhausted: bool = False
    assumptions: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)


class DAGScheduler:
    """Executes ExecutionPlan tasks in topological dependency order with bounded concurrency."""

    def __init__(
        self,
        *,
        database: Database,
        worker: SpecialistWorker,
        settings: CrewForgeSettings,
    ) -> None:
        self._db = database
        self._worker = worker
        self._settings = settings

    async def run_plan(
        self,
        *,
        run_id: str,
        original_prompt: str,
        plan: ExecutionPlan,
        reserve_synthesis_calls: int = 1,
        run_deadline_monotonic: float | None = None,
    ) -> SchedulerOutcome:
        """Schedule and execute all plan tasks without deadlocking on failures."""
        agents_by_id: dict[str, AgentSpec] = {a.agent_id: a for a in plan.agents}
        tasks_by_id: dict[str, TaskSpec] = {t.task_id: t.model_copy() for t in plan.tasks}
        worker_outcomes: dict[str, WorkerExecutionOutcome] = {}
        active_asyncio_tasks: dict[asyncio.Task[WorkerExecutionOutcome], str] = {}

        assumptions: list[str] = []
        uncertainties: list[str] = []
        missing_info: list[str] = []
        deadline_exceeded = False
        budget_exhausted = False

        async def _mark_blocked_dependents() -> None:
            changed = True
            while changed:
                changed = False
                for task_id, task in tasks_by_id.items():
                    if task.status != TaskStatus.PENDING:
                        continue
                    failed_deps = [
                        dep_id
                        for dep_id in task.depends_on
                        if tasks_by_id[dep_id].status in {TaskStatus.FAILED, TaskStatus.BLOCKED}
                    ]
                    if failed_deps:
                        reason = (
                            f"Blocked due to failed/blocked dependency: {', '.join(failed_deps)}."
                        )
                        tasks_by_id[task_id] = task.model_copy(
                            update={"status": TaskStatus.BLOCKED, "error": reason}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=task_id,
                            to_status=TaskStatus.BLOCKED,
                            reason=reason,
                            error=reason,
                        )
                        await self._db.append_event(
                            ExecutionEvent(
                                run_id=run_id,
                                event_type="task_blocked",
                                actor="scheduler",
                                message=f"Task {task.task_ref or task_id} blocked.",
                                payload={"task_id": task_id, "blocked_by": failed_deps},
                            )
                        )
                        changed = True

        async def _block_remaining_pending(reason: str) -> None:
            for task_id, task in list(tasks_by_id.items()):
                if task.status == TaskStatus.PENDING:
                    tasks_by_id[task_id] = task.model_copy(
                        update={"status": TaskStatus.BLOCKED, "error": reason}
                    )
                    await self._db.transition_task(
                        run_id=run_id,
                        task_id=task_id,
                        to_status=TaskStatus.BLOCKED,
                        reason=reason,
                        error=reason,
                    )

        async def _can_launch_more_calls() -> bool:
            summary = await self._db.get_budget_summary(run_id)
            committed_calls = int(summary["run_committed_calls"])
            in_flight_calls = len(active_asyncio_tasks)
            remaining_before_reserve = (
                self._settings.max_model_calls_per_run
                - committed_calls
                - in_flight_calls
                - reserve_synthesis_calls
            )
            return remaining_before_reserve >= 1

        try:
            while True:
                await _mark_blocked_dependents()

                pending_tasks = [
                    task for task in tasks_by_id.values() if task.status == TaskStatus.PENDING
                ]
                if not pending_tasks and not active_asyncio_tasks:
                    break

                if (
                    run_deadline_monotonic is not None
                    and time.monotonic() >= run_deadline_monotonic
                ):
                    deadline_exceeded = True
                    await _block_remaining_pending("Run deadline exceeded.")
                    if not active_asyncio_tasks:
                        break

                ready_tasks = [
                    task
                    for task in pending_tasks
                    if all(
                        tasks_by_id[dep_id].status == TaskStatus.COMPLETED
                        for dep_id in task.depends_on
                    )
                ]

                if not deadline_exceeded and ready_tasks:
                    for task in ready_tasks:
                        if not await _can_launch_more_calls():
                            budget_exhausted = True
                            await _block_remaining_pending(
                                "Call budget reserved for Lead synthesis."
                            )
                            break

                        agent = agents_by_id[task.assigned_agent_id]
                        tasks_by_id[task.task_id] = task.model_copy(
                            update={"status": TaskStatus.RUNNING}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=task.task_id,
                            to_status=TaskStatus.RUNNING,
                            reason="Dependencies satisfied.",
                        )
                        await self._db.append_event(
                            ExecutionEvent(
                                run_id=run_id,
                                event_type="task_started",
                                actor=agent.agent_id,
                                message=(
                                    f"Started task {task.task_ref or task.task_id}: {task.title}"
                                ),
                                payload={"task_id": task.task_id, "agent_id": agent.agent_id},
                            )
                        )
                        completed_snapshot = {
                            tid: t
                            for tid, t in tasks_by_id.items()
                            if t.status == TaskStatus.COMPLETED
                        }
                        coro = self._worker.execute_task(
                            run_id=run_id,
                            original_prompt=original_prompt,
                            plan=plan,
                            agent=agent,
                            task=tasks_by_id[task.task_id],
                            completed_tasks_by_id=completed_snapshot,
                            run_deadline_monotonic=run_deadline_monotonic,
                        )
                        async_task = asyncio.create_task(coro)
                        active_asyncio_tasks[async_task] = task.task_id

                if not active_asyncio_tasks:
                    # Either all tasks finished or remaining pending tasks were blocked.
                    continue

                wait_timeout: float | None = None
                if run_deadline_monotonic is not None:
                    wait_timeout = max(0.01, run_deadline_monotonic - time.monotonic())

                done, _ = await asyncio.wait(
                    active_asyncio_tasks.keys(),
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done and run_deadline_monotonic is not None:
                    deadline_exceeded = True
                    await _block_remaining_pending("Run deadline exceeded.")
                    for pending_handle in list(active_asyncio_tasks.keys()):
                        pending_handle.cancel()
                    await asyncio.gather(*active_asyncio_tasks.keys(), return_exceptions=True)
                    for pending_handle, tid in list(active_asyncio_tasks.items()):
                        active_asyncio_tasks.pop(pending_handle, None)
                        if tasks_by_id[tid].status == TaskStatus.RUNNING:
                            tasks_by_id[tid] = tasks_by_id[tid].model_copy(
                                update={
                                    "status": TaskStatus.FAILED,
                                    "error": "Task cancelled due to run deadline.",
                                }
                            )
                            await self._db.transition_task(
                                run_id=run_id,
                                task_id=tid,
                                to_status=TaskStatus.FAILED,
                                reason="Run deadline exceeded.",
                                error="Task cancelled due to run deadline.",
                            )
                    break

                for finished_handle in done:
                    tid = active_asyncio_tasks.pop(finished_handle)
                    task = tasks_by_id[tid]
                    try:
                        outcome = finished_handle.result()
                    except asyncio.CancelledError:
                        tasks_by_id[tid] = task.model_copy(
                            update={"status": TaskStatus.FAILED, "error": "Task cancelled."}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=tid,
                            to_status=TaskStatus.FAILED,
                            reason="Task cancelled.",
                            error="Task cancelled.",
                        )
                        raise
                    except RunDeadlineExceededError as exc:
                        deadline_exceeded = True
                        err_msg = sanitize_error_message(str(exc))
                        tasks_by_id[tid] = task.model_copy(
                            update={"status": TaskStatus.FAILED, "error": err_msg}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=tid,
                            to_status=TaskStatus.FAILED,
                            reason="Run deadline exceeded.",
                            error=err_msg,
                        )
                    except BudgetError as exc:
                        budget_exhausted = True
                        err_msg = sanitize_error_message(str(exc))
                        tasks_by_id[tid] = task.model_copy(
                            update={"status": TaskStatus.FAILED, "error": err_msg}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=tid,
                            to_status=TaskStatus.FAILED,
                            reason="Budget exhausted.",
                            error=err_msg,
                        )
                    except (ProviderError, Exception) as exc:
                        err_msg = sanitize_error_message(f"{type(exc).__name__}: {exc}")
                        tasks_by_id[tid] = task.model_copy(
                            update={"status": TaskStatus.FAILED, "error": err_msg}
                        )
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=tid,
                            to_status=TaskStatus.FAILED,
                            reason="Task execution failed.",
                            error=err_msg,
                        )
                        await self._db.append_event(
                            ExecutionEvent(
                                run_id=run_id,
                                event_type="task_failed",
                                actor=task.assigned_agent_id,
                                message=f"Task {task.task_ref or tid} failed: {err_msg}",
                                payload={"task_id": tid},
                            )
                        )
                    else:
                        worker_outcomes[tid] = outcome
                        tasks_by_id[tid] = task.model_copy(
                            update={
                                "status": TaskStatus.COMPLETED,
                                "findings": outcome.findings,
                            }
                        )
                        assumptions.extend(outcome.assumptions)
                        uncertainties.extend(outcome.uncertainties)
                        missing_info.extend(outcome.missing_information)
                        await self._db.transition_task(
                            run_id=run_id,
                            task_id=tid,
                            to_status=TaskStatus.COMPLETED,
                            reason="Task completed.",
                            findings=outcome.findings,
                        )
                        await self._db.append_event(
                            ExecutionEvent(
                                run_id=run_id,
                                event_type="task_completed",
                                actor=task.assigned_agent_id,
                                message=f"Completed task {task.task_ref or tid}: {task.title}",
                                payload={
                                    "task_id": tid,
                                    "claims_count": len(outcome.claims),
                                    "messages_sent": len(outcome.sent_messages),
                                },
                            )
                        )
        except asyncio.CancelledError:
            for handle in list(active_asyncio_tasks.keys()):
                handle.cancel()
            await asyncio.gather(*active_asyncio_tasks.keys(), return_exceptions=True)
            raise

        return SchedulerOutcome(
            tasks_by_id=tasks_by_id,
            worker_outcomes=worker_outcomes,
            deadline_exceeded=deadline_exceeded,
            budget_exhausted=budget_exhausted,
            assumptions=list(dict.fromkeys(assumptions)),
            uncertainties=list(dict.fromkeys(uncertainties)),
            missing_information=list(dict.fromkeys(missing_info)),
        )
