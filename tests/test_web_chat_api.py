"""Offline end-to-end tests for CrewForge conversational API, isolation, security, and MCP."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import crewforge.dashboard as dashboard_mod
from crewforge.dashboard import create_dashboard_app
from crewforge.web_api import JudgeSessionManager, JudgeSettingsUpdate
from tests.fake_provider import FakeProvider


def test_judge_isolation_from_team_budget_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove judge settings never read .env / OPENAI_API_KEY and never touch team budget DB."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-team-key-must-never-leak")
    monkeypatch.setenv("CREWFORGE_ALLOW_PAID_RUNS", "true")

    mgr = JudgeSessionManager(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
    )
    rt = mgr.build_runtime_settings()
    assert rt.budget_authority_path.resolve() == (tmp_path / "judge_budget.db").resolve()
    assert rt.openai_api_key is None
    assert rt.allow_paid_runs is False

    # Custom pricing for arbitrary paid model (requires BYOK mode)
    mgr.update_settings(
        JudgeSettingsUpdate(
            billing_mode="byok",
            provider="openai",
            model="custom-judge-model-v1",
            api_key="sk-judge-only-key-4321",
            allow_paid_runs=True,
            custom_input_price_per_1m=1.25,
            custom_output_price_per_1m=5.50,
        )
    )
    rt2 = mgr.build_runtime_settings()
    assert rt2.openai_api_key is not None
    assert rt2.openai_api_key.get_secret_value() == "sk-judge-only-key-4321"
    assert rt2.pricing_json["custom-judge-model-v1"].input_per_1m == 1.25
    assert rt2.pricing_json["custom-judge-model-v1"].output_per_1m == 5.50


def test_groq_free_mode_enforcement_and_host_rebinding_protection(tmp_path: Path) -> None:
    from crewforge.providers.openai_provider import OpenAIProvider

    mgr = JudgeSessionManager(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
    )
    # Free mode rejects paid OpenAI provider
    with pytest.raises(ValueError, match="Free mode"):
        mgr.update_settings(JudgeSettingsUpdate(billing_mode="free", provider="openai"))

    # Groq free-tier preset requires verified checkbox before building runtime settings
    mgr.update_settings(
        JudgeSettingsUpdate(
            billing_mode="free",
            provider="openai_compatible",
            openai_base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            api_key="gsk_test_free_tier_key_0000",
        )
    )
    with pytest.raises(ValueError, match="verified Groq free-tier"):
        mgr.build_runtime_settings()

    mgr.update_settings(JudgeSettingsUpdate(free_tier_verified=True))
    rt = mgr.build_runtime_settings()
    assert rt.allow_paid_runs is False
    assert rt.max_cost_per_run_usd == 0.0
    assert rt.total_budget_ceiling_usd == 0.0
    prov = OpenAIProvider.from_settings(rt)
    assert prov.is_paid_provider is False

    # Host header DNS rebinding protection
    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
        replay_dir=Path("demo/traces"),
        benchmark_path=Path("demo/v2_reconciled_scores.json"),
    )
    with TestClient(app) as client:
        evil = client.get("/api/health", headers={"Host": "evil.attacker.com:8765"})
        assert evil.status_code == 403


def test_local_auth_origin_and_size_limits(tmp_path: Path) -> None:
    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
        replay_dir=Path("demo/traces"),
        benchmark_path=Path("demo/v2_reconciled_scores.json"),
        auth_token="test-local-token",
        require_auth=True,
    )
    with TestClient(app) as client:
        # Unauthenticated /api/settings rejected
        assert client.get("/api/settings").status_code == 401
        assert client.get("/api/health").status_code == 200

        # Valid token succeeds and never leaks raw key
        headers = {"X-CrewForge-Token": "test-local-token"}
        upd = client.post(
            "/api/settings",
            headers=headers,
            json={
                "billing_mode": "byok",
                "api_key": "sk-test-secret-judge-key-987654",
                "provider": "openai",
            },
        )
        assert upd.status_code == 200
        assert "sk-test-secret-judge-key-987654" not in upd.text

        # Cross-origin mutating request rejected (403)
        csrf = client.post(
            "/api/chats",
            headers={**headers, "Origin": "https://evil.example"},
            json={"title": "CSRF"},
        )
        assert csrf.status_code == 403

        # Oversized body (>512 KB) rejected (413)
        huge = client.post(
            "/api/chats",
            headers=headers,
            content=b"x" * 600_000,
        )
        assert huge.status_code == 413


def test_require_react_fails_fast_when_dist_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dashboard_mod, "_FRONTEND_DIST_DIR", tmp_path / "missing_dist")
    with pytest.raises(RuntimeError, match="React production build not found"):
        create_dashboard_app(
            db_path=tmp_path / "judge_session.db",
            require_react=True,
        )


def test_chat_and_shared_budget_isolation_offline(tmp_path: Path) -> None:
    team_budget = Path(".crewforge/budget_authority.db")
    team_bytes_before = team_budget.read_bytes() if team_budget.exists() else None

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
    judge_budget = tmp_path / "judge_budget.db"

    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=judge_budget,
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
        assert any(a["role"] == "Capacity Specialist" for a in trace["roster"])
        assert judge_budget.exists()

    # Team's canonical budget ledger must be completely untouched
    if team_bytes_before is not None:
        assert team_budget.read_bytes() == team_bytes_before
    assert os.environ.get("CREWFORGE_ALLOW_PAID_RUNS") != "true"


def test_groq_bounded_rate_limit_backoff_without_model_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Groq free-tier provider retries 429 up to 2 times without switching models."""
    import asyncio

    from crewforge.providers.base import ProviderRateLimitError
    from crewforge.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        api_key="gsk-test-key",
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        use_chat_completions=True,
        is_free_tier=True,
        provider_label="openai_compatible",
    )
    assert provider._is_groq is True

    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    attempts = 0

    async def flaky_call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ProviderRateLimitError(
                "rate limited",
                provider=provider.provider_name,
                model=provider._model,
            )
        return "ok"

    result, retries, _latency = asyncio.run(
        provider._run_with_rate_limit_backoff(flaky_call, timeout_seconds=5.0, retries_cap=0)
    )
    assert result == "ok"
    assert retries == 0
    assert attempts == 3
    assert sleep_delays == [2.0, 4.0]
    assert provider._model == "openai/gpt-oss-120b"

    # Exceeding the 2-retry cap must raise ProviderRateLimitError rather than switching models
    sleep_delays.clear()
    fail_attempts = 0

    async def always_rate_limited() -> str:
        nonlocal fail_attempts
        fail_attempts += 1
        raise ProviderRateLimitError(
            "still rate limited",
            provider=provider.provider_name,
            model=provider._model,
        )

    with pytest.raises(ProviderRateLimitError):
        asyncio.run(
            provider._run_with_rate_limit_backoff(
                always_rate_limited, timeout_seconds=5.0, retries_cap=0
            )
        )
    assert fail_attempts == 3
    assert sleep_delays == [2.0, 4.0]


def test_build_provider_from_settings_supports_openai_compatible(tmp_path: Path) -> None:
    """Ensure build_provider_from_settings instantiates OpenAIProvider for openai_compatible."""
    from crewforge.core.baseline import build_provider_from_settings
    from crewforge.providers.openai_provider import OpenAIProvider

    mgr = JudgeSessionManager(
        db_path=tmp_path / "judge_session.db",
        budget_db_path=tmp_path / "judge_budget.db",
    )
    mgr.update_settings(
        JudgeSettingsUpdate(
            billing_mode="free",
            free_tier_verified=True,
            provider="openai_compatible",
            model="openai/gpt-oss-120b",
            openai_base_url="https://api.groq.com/openai/v1",
            api_key="gsk-test-groq-key-1234",
        )
    )
    rt = mgr.build_runtime_settings()
    provider = build_provider_from_settings(rt)
    assert isinstance(provider, OpenAIProvider)
    assert provider._is_groq is True
    assert provider._is_free_tier is True
