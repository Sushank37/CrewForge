"""Offline tests for CrewForge Mission Control read-only observability dashboard."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from starlette.testclient import TestClient

from crewforge.config import CrewForgeSettings, ProviderType
from crewforge.dashboard import create_dashboard_app
from crewforge.dashboard_api import build_agent_topology
from crewforge.schemas import (
    AgentMessage,
    AgentSpec,
    Claim,
    ClaimCategory,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ExecutionPlan,
    HonestyGateRecord,
    ReviewDecision,
    ReviewerIdentity,
    ReviewVerdict,
    RunResult,
    RunStatus,
    TaskSpec,
    TaskStatus,
    UsageMetrics,
)
from crewforge.storage.database import Database


async def _seed_synthetic_run(db_path: Path, run_id: str = "run_test123456") -> None:
    """Create a synthetic isolated run in SQLite for dashboard unit testing."""
    db = Database(db_path)
    await db.initialize()
    await db.create_run(
        run_id=run_id,
        mode="network",
        prompt="Test prompt with secret sk-ant-api03-secretkeyvalue123456789",
        config=CrewForgeSettings(provider=ProviderType.OLLAMA, model="qwen3:4b").to_run_config(),
    )
    plan = ExecutionPlan(
        run_id=run_id,
        goal="Solve synthetic test",
        agents=[
            AgentSpec(
                agent_id=f"lead_{run_id}",
                role="lead",
                instructions="Coordinate",
                model_override="qwen3:4b",
            ),
            AgentSpec(
                agent_id=f"spec_{run_id}",
                role="specialist",
                instructions="Compute",
                model_override="qwen3:4b",
            ),
        ],
        tasks=[
            TaskSpec(
                task_id=f"t1_{run_id}",
                title="Compute",
                description="Compute total",
                assigned_agent_id=f"spec_{run_id}",
                depends_on=[],
                status=TaskStatus.COMPLETED,
            )
        ],
    )
    await db.save_execution_plan(plan)
    await db.insert_evidence(
        Evidence(
            evidence_id=f"evd_{run_id}",
            run_id=run_id,
            source_type="user_prompt",
            source_ref="up_1",
            excerpt="450 orders per hour",
            created_by=f"spec_{run_id}",
        )
    )
    await db.insert_claim(
        Claim(
            claim_id=f"clm_{run_id}",
            run_id=run_id,
            version=1,
            statement="Capacity is 3600.",
            category=ClaimCategory.COMPUTATION,
            author_agent_id=f"spec_{run_id}",
            evidence_ids=[f"evd_{run_id}"],
            status=ClaimStatus.ASSERTED,
        )
    )
    await db.insert_claim(
        Claim(
            claim_id=f"clm_{run_id}",
            run_id=run_id,
            version=2,
            statement="Regular-shift capacity is 3,600 orders.",
            category=ClaimCategory.COMPUTATION,
            author_agent_id=f"spec_{run_id}",
            evidence_ids=[f"evd_{run_id}"],
            status=ClaimStatus.SUPPORTED,
        )
    )
    await db.register_reviewer(
        ReviewerIdentity(
            reviewer_id=f"rev_{run_id}",
            run_id=run_id,
            role="reviewer",
            dimensions=["general"],
        )
    )
    await db.insert_review_verdict(
        ReviewVerdict(
            verdict_id=f"ver_{run_id}",
            run_id=run_id,
            claim_id=f"clm_{run_id}",
            claim_version=2,
            reviewer_id=f"rev_{run_id}",
            decision=ReviewDecision.APPROVE,
            rationale="Verified against up_1.",
        )
    )
    await db.insert_message(
        AgentMessage(
            message_id=f"msg_{run_id}",
            run_id=run_id,
            sender_id=f"lead_{run_id}",
            recipient_id=f"spec_{run_id}",
            message_type="task_handoff",
            content="Please compute capacity. Authorization: Bearer supersecrettoken999",
        ),
        delivery_recipients=[f"spec_{run_id}"],
    )
    await db.append_event(
        ExecutionEvent(
            event_id=f"evt_{run_id}",
            run_id=run_id,
            event_type="plan_created",
            actor=f"lead_{run_id}",
            message="Plan created",
            payload={"task_count": 1},
            timestamp=datetime.now(UTC),
        )
    )
    await db.save_honesty_gate_decision(
        HonestyGateRecord(
            run_id=run_id,
            passed=True,
            release_mode="full",
            supported_claim_ids=[f"clm_{run_id}"],
            withheld_claim_ids=[],
            rejected_segment_ids=[],
            unresolved_requirements=[],
            reasons=["All material claims supported."],
        )
    )
    await db.complete_run(
        RunResult(
            run_id=run_id,
            mode="network",
            status=RunStatus.COMPLETED,
            prompt="Test prompt",
            provider="ollama",
            model="qwen3:4b",
            returned_model="qwen3:4b",
            answer="Regular-shift capacity is 3,600 orders.",
            usage=UsageMetrics(model_calls=3, input_tokens=100, output_tokens=50),
        )
    )


def test_dashboard_topology_builder() -> None:
    topo = build_agent_topology(
        {
            "status": "completed",
            "model": "gpt-6-luna",
            "roster": [
                {"agent_id": "lead", "role": "lead", "specialty": "coordinator"},
                {"agent_id": "worker_1", "role": "specialist", "specialty": "finance"},
            ],
            "tasks": [
                {
                    "task_id": "t1",
                    "assigned_agent_id": "worker_1",
                    "depends_on": [],
                    "status": "completed",
                }
            ],
            "peer_messages": [
                {
                    "sender_id": "lead",
                    "recipient_id": "worker_1",
                    "message_type": "handoff",
                }
            ],
            "review_verdicts": [{"reviewer_id": "reviewer_a", "decision": "approve"}],
            "honesty_gate": {"passed": True, "release_mode": "full"},
        }
    )
    node_ids = {n["id"] for n in topo["nodes"]}
    assert node_ids == {"lead", "worker_1", "reviewer_a", "honesty_gate"}
    edge_types = {e["edge_type"] for e in topo["edges"]}
    assert edge_types == {"task_dependency", "peer_message"}


async def test_live_trace_api_polling_and_redaction(tmp_path: Path) -> None:
    db_path = tmp_path / "demo_live.db"
    await _seed_synthetic_run(db_path, run_id="run_live001")

    app = create_dashboard_app(
        db_path=db_path,
        replay_dir=tmp_path / "replays",
        benchmark_path=tmp_path / "bench.json",
    )
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["mode"] == "read_only_observability"

    runs = client.get("/api/live/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_live001"

    detail = client.get("/api/live/runs/run_live001").json()
    assert detail["is_historical_replay"] is False
    assert len(detail["claims"]) == 2
    assert [c["version"] for c in detail["claims"]] == [1, 2]
    assert "supersecrettoken999" not in json.dumps(detail)
    assert "sk-ant-api03-secretkeyvalue123456789" not in json.dumps(detail)

    await _seed_synthetic_run(db_path, run_id="run_live002")
    runs_after = client.get("/api/live/runs").json()["runs"]
    assert {r["run_id"] for r in runs_after} == {"run_live001", "run_live002"}


def test_replay_security_malformed_and_benchmark(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    (replay_dir / "malformed.json").write_text("{not valid json", encoding="utf-8")

    genuine_trace = Path("demo/traces/phase5e_canary_a.json")
    assert genuine_trace.is_file()
    (replay_dir / "phase5e_canary_a.json").write_text(
        genuine_trace.read_text(encoding="utf-8"), encoding="utf-8"
    )

    bench_path = Path("demo/v2_reconciled_scores.json")
    assert bench_path.is_file()

    app = create_dashboard_app(
        db_path=tmp_path / "missing.db",
        replay_dir=replay_dir,
        benchmark_path=bench_path,
    )
    client = TestClient(app)

    files = client.get("/api/replay/files").json()["files"]
    assert [f["filename"] for f in files] == ["phase5e_canary_a.json"]

    loaded = client.get("/api/replay/files/phase5e_canary_a.json")
    assert loaded.status_code == 200
    payload = loaded.json()
    assert payload["is_historical_replay"] is True
    assert payload["run_id"] == "run_08160b7fa1bd"
    seqs = [e["seq"] for e in payload["execution_events"]]
    assert seqs == list(range(1, len(seqs) + 1))

    assert client.get("/api/replay/files/malformed.json").status_code == 400
    assert client.get("/api/replay/files/nonexistent.json").status_code == 404
    assert client.get("/api/replay/files/..%2F..%2F.env").status_code == 400
    assert client.get("/api/live/runs/invalid_id!").status_code == 400

    bench = client.get("/api/benchmark").json()
    assert bench["summary"]["baseline_mean"] == 96.5
    assert bench["summary"]["network_mean"] == 55.5
    assert bench["summary"]["network_wins"] == 1
    assert bench["grading_method"] == "ChatGPT-assisted blind semantic grading"
    assert "Not executed" in bench["david_vs_goliath"]["status"]
