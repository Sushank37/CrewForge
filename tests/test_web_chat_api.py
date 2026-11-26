"""Offline end-to-end tests for CrewForge conversational API, judge settings, and MCP status."""

from __future__ import annotations

import time
from pathlib import Path

from starlette.testclient import TestClient

from crewforge.dashboard import create_dashboard_app
from tests.fake_provider import FakeProvider


def test_judge_settings_never_expose_api_key(tmp_path: Path) -> None:
    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        replay_dir=Path("demo/traces"),
        benchmark_path=Path("demo/v2_reconciled_scores.json"),
    )
    client = TestClient(app)

    initial = client.get("/api/settings").json()
    assert initial["allow_paid_runs"] is False
    assert initial["api_key_configured"] is False
    assert "api_key" not in initial

    updated = client.post(
        "/api/settings",
        json={
            "provider": "openai",
            "model": "gpt-6-luna",
            "api_key": "sk-test-secret-judge-key-987654",
            "allow_paid_runs": True,
            "max_cost_per_run_usd": 0.03,
        },
    )
    assert updated.status_code == 200
    body = updated.json()
    assert body["api_key_configured"] is True
    assert body["api_key_hint"] == "••••7654"
    assert "sk-test-secret-judge-key-987654" not in updated.text
    assert "sk-test-secret-judge-key-987654" not in client.get("/api/settings").text


def test_chat_prompt_to_answer_with_offline_provider_and_attachments(tmp_path: Path) -> None:
    def handler(_prompt: str, schema: type | None) -> dict:
        name = schema.__name__ if schema else ""
        if "Plan" in name:
            return {
                "goal": "Calculate total",
                "user_requirements": ["Compute 450 * 8"],
                "acceptance_criteria": ["Grounded result"],
                "agents": [
                    {
                        "agent_ref": "spec_calc",
                        "role": "Capacity Specialist",
                        "instructions": "Compute capacity",
                        "capabilities": ["analysis"],
                    }
                ],
                "tasks": [
                    {
                        "task_ref": "t1",
                        "title": "Compute shift capacity",
                        "description": "Multiply 450 by 8",
                        "assigned_agent_ref": "spec_calc",
                        "depends_on": [],
                        "expected_deliverables": ["capacity"],
                        "acceptance_criteria": ["grounded"],
                    }
                ],
            }
        if "Specialist" in name:
            return {
                "findings": "Capacity is 3600 orders.",
                "evidence_proposals": [
                    {
                        "source_type": "user_prompt",
                        "source_ref": "up_1",
                        "excerpt": "450 orders/hr for 8 hours",
                    }
                ],
                "claims": [
                    {
                        "statement": "Shift capacity is 3,600 orders.",
                        "category": "computation",
                        "evidence_refs": ["up_1"],
                        "arithmetic": {
                            "expression": "rate * hours",
                            "variables": {"rate": "450", "hours": "8"},
                            "input_provenance_refs": {"rate": "up_1", "hours": "up_1"},
                            "proposed_result": "3600",
                            "units": "orders",
                        },
                        "requirement_refs": ["req_1"],
                        "is_material": True,
                    }
                ],
                "outgoing_messages": [],
                "missing_information": [],
                "assumptions": [],
                "uncertainties": [],
            }
        if "Review" in name:
            return {
                "verdicts": [
                    {
                        "claim_ref": "clm_1",
                        "decision": "approve",
                        "rationale": "Grounded arithmetic.",
                        "blocking_severity": "none",
                        "blocking_issues": [],
                    }
                ],
                "overall_notes": "Approved.",
            }
        return {
            "summary": "Shift capacity is 3,600 orders.",
            "accepted_segments": [],
            "missing_information": [],
            "assumptions": [],
        }

    fake = FakeProvider(handler=handler)

    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
        replay_dir=Path("demo/traces"),
        benchmark_path=Path("demo/v2_reconciled_scores.json"),
        provider_factory=lambda _s: fake,
    )
    with TestClient(app) as client:
        chat = client.post("/api/chats", json={"title": "Test Chat"}).json()
        chat_id = chat["chat_id"]

        sent = client.post(
            f"/api/chats/{chat_id}/messages",
            json={
                "prompt": "Calculate 450 orders/hr for 8 hours.",
                "mode": "network",
                "attachments": [
                    {"name": "spec.md", "content": "Rate: 450 orders/hr, Shift: 8 hours."}
                ],
            },
        )
        assert sent.status_code == 202
        run_id = sent.json()["run_id"]

        detail = None
        for _ in range(60):
            detail = client.get(f"/api/chats/{chat_id}").json()
            asst = [m for m in detail["messages"] if m["role"] == "assistant"][0]
            if asst["status"] != "running":
                break
            time.sleep(0.05)

        assert detail is not None
        asst = [m for m in detail["messages"] if m["role"] == "assistant"][0]
        assert asst["status"] in ("completed", "partial"), asst.get("error")
        assert asst["run_id"] == run_id

        trace = client.get(f"/api/live/runs/{run_id}").json()
        assert trace["run_id"] == run_id
        assert any(a["role"] == "Capacity Specialist" for a in trace["roster"])

        mcp_info = client.get("/api/mcp/status").json()
        assert mcp_info["tools"] == ["solve", "baseline", "trace"]
