"""Offline tests for dynamic Lead planner validation, repair, and team diversity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from crewforge.config import CrewForgeSettings
from crewforge.core.budget import BudgetManager
from crewforge.core.executor import ModelExecutor
from crewforge.core.planner import (
    LeadPlanner,
    PlanValidationError,
    validate_and_materialize_plan,
)
from crewforge.schemas import PlannerOutput
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider


def _settings(db_path: Path, **overrides: object) -> CrewForgeSettings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "db_path": db_path,
        "max_workers": 4,
        "max_tasks": 6,
    }
    defaults.update(overrides)
    return CrewForgeSettings(**defaults)  # type: ignore[arg-type]


def _valid_raw_plan(**overrides: object) -> PlannerOutput:
    payload: dict[str, Any] = {
        "goal": "Analyze system trade-offs",
        "user_requirements": ["Compare options A and B"],
        "acceptance_criteria": ["Explicit assumptions documented"],
        "agents": [
            {
                "agent_ref": "a1",
                "role": "Architecture Analyst",
                "instructions": "Evaluate option A.",
                "capabilities": ["analysis"],
            },
            {
                "agent_ref": "a2",
                "role": "Risk Reviewer",
                "instructions": "Evaluate risks of option B.",
                "capabilities": ["critique"],
            },
        ],
        "tasks": [
            {
                "task_ref": "t1",
                "title": "Analyze Option A",
                "description": "Assess option A.",
                "assigned_agent_ref": "a1",
                "depends_on": [],
                "expected_deliverables": ["Option A analysis"],
            },
            {
                "task_ref": "t2",
                "title": "Synthesize Option B risks",
                "description": "Assess risks after t1.",
                "assigned_agent_ref": "a2",
                "depends_on": ["t1"],
                "expected_deliverables": ["Risk summary"],
            },
        ],
        "assumptions": ["No live telemetry"],
        "missing_information": ["Traffic volume"],
    }
    payload.update(overrides)
    return PlannerOutput.model_validate(payload)


def test_planner_validation_rejects_duplicates_missing_refs_cycles_and_overrides(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path / "p.db")

    # Duplicate agent_ref
    dup_agents = _valid_raw_plan(
        agents=[
            {
                "agent_ref": "a1",
                "role": "Role 1",
                "instructions": "Inst 1",
                "capabilities": ["analysis"],
            },
            {
                "agent_ref": "a1",
                "role": "Role 2",
                "instructions": "Inst 2",
                "capabilities": ["critique"],
            },
        ]
    )
    with pytest.raises(PlanValidationError, match="Duplicate agent_ref"):
        validate_and_materialize_plan(run_id="run_1", raw_plan=dup_agents, settings=settings)

    # Missing agent ref + unknown dependency
    bad_refs = _valid_raw_plan(
        tasks=[
            {
                "task_ref": "t1",
                "title": "Task 1",
                "description": "Desc",
                "assigned_agent_ref": "missing_agent",
                "depends_on": ["missing_task"],
                "expected_deliverables": ["Output"],
            }
        ]
    )
    with pytest.raises(PlanValidationError) as exc_info:
        validate_and_materialize_plan(run_id="run_1", raw_plan=bad_refs, settings=settings)
    joined = " | ".join(exc_info.value.errors)
    assert "unknown assigned_agent_ref" in joined
    assert "unknown task_ref" in joined

    # Dependency cycle
    cyclic = _valid_raw_plan(
        tasks=[
            {
                "task_ref": "t1",
                "title": "T1",
                "description": "D1",
                "assigned_agent_ref": "a1",
                "depends_on": ["t2"],
                "expected_deliverables": ["O1"],
            },
            {
                "task_ref": "t2",
                "title": "T2",
                "description": "D2",
                "assigned_agent_ref": "a2",
                "depends_on": ["t1"],
                "expected_deliverables": ["O2"],
            },
        ]
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_and_materialize_plan(run_id="run_1", raw_plan=cyclic, settings=settings)

    # Unsupported capability & model_override
    bad_caps = _valid_raw_plan(
        agents=[
            {
                "agent_ref": "a1",
                "role": "Hacker",
                "instructions": "Run code",
                "capabilities": ["arbitrary_code_exec"],
                "model_override": "gpt-5",
            },
            {
                "agent_ref": "a2",
                "role": "Reviewer",
                "instructions": "Review",
                "capabilities": ["critique"],
            },
        ]
    )
    with pytest.raises(PlanValidationError) as cap_err:
        validate_and_materialize_plan(run_id="run_1", raw_plan=bad_caps, settings=settings)
    cap_msg = " | ".join(cap_err.value.errors)
    assert "unsupported capabilities" in cap_msg
    assert "model_override" in cap_msg


async def test_planner_repairs_once_and_generates_dynamic_teams(tmp_path: Path) -> None:
    db_path = tmp_path / "planner.db"
    settings = _settings(db_path)
    db = Database(db_path)
    await db.initialize()

    try:
        call_count = 0

        def handler(prompt: str, _model: type[BaseModel] | None) -> Any:
            nonlocal call_count
            call_count += 1
            if "financial" in prompt.lower():
                if call_count == 1:
                    # First attempt has a cycle -> triggers single repair attempt.
                    return _valid_raw_plan(
                        tasks=[
                            {
                                "task_ref": "t1",
                                "title": "Cashflow",
                                "description": "Compute net cashflow",
                                "assigned_agent_ref": "a1",
                                "depends_on": ["t1"],
                                "expected_deliverables": ["Net cashflow"],
                            }
                        ]
                    )
                return _valid_raw_plan(
                    agents=[
                        {
                            "agent_ref": "a1",
                            "role": "Cash Flow Analyst",
                            "instructions": "Calculate cash flows.",
                            "capabilities": ["calculation"],
                        },
                        {
                            "agent_ref": "a2",
                            "role": "Investment Risk Reviewer",
                            "instructions": "Evaluate risks and missing info.",
                            "capabilities": ["critique"],
                        },
                    ],
                    tasks=[
                        {
                            "task_ref": "t1",
                            "title": "Payback calculation",
                            "description": "Compute simple payback.",
                            "assigned_agent_ref": "a1",
                            "depends_on": [],
                            "expected_deliverables": ["Payback period"],
                        },
                        {
                            "task_ref": "t2",
                            "title": "Investment attractiveness check",
                            "description": "Evaluate missing factors.",
                            "assigned_agent_ref": "a2",
                            "depends_on": ["t1"],
                            "expected_deliverables": ["Risk evaluation"],
                        },
                    ],
                )
            return _valid_raw_plan(
                agents=[
                    {
                        "agent_ref": "a1",
                        "role": "Distributed Systems Architect",
                        "instructions": "Analyze consensus protocols.",
                        "capabilities": ["analysis"],
                    }
                ],
                tasks=[
                    {
                        "task_ref": "t1",
                        "title": "Consensus comparison",
                        "description": "Compare Raft vs Paxos.",
                        "assigned_agent_ref": "a1",
                        "depends_on": [],
                        "expected_deliverables": ["Protocol comparison"],
                    }
                ],
            )

        provider = FakeProvider(handler=handler, max_retries=0)
        budget = BudgetManager(db, settings)
        executor = ModelExecutor(
            provider=provider,
            database=db,
            budget_manager=budget,
            settings=settings,
        )
        planner = LeadPlanner(executor=executor, database=db, settings=settings)

        await db.create_run(
            "run_fin", "network", "Evaluate financial investment", settings.to_run_config()
        )
        fin_plan = await planner.create_plan(
            run_id="run_fin",
            user_prompt="Evaluate financial investment",
        )
        assert [a.role for a in fin_plan.agents] == [
            "Cash Flow Analyst",
            "Investment Risk Reviewer",
        ]
        assert call_count == 2  # initial + 1 repair

        await db.create_run("run_sys", "network", "Compare Raft vs Paxos", settings.to_run_config())
        sys_plan = await planner.create_plan(
            run_id="run_sys",
            user_prompt="Compare Raft vs Paxos",
        )
        assert [a.role for a in sys_plan.agents] == ["Distributed Systems Architect"]
    finally:
        await db.close()
