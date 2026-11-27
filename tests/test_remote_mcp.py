"""Offline verification tests for Remote HTTPS MCP readiness and judge session isolation."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from crewforge.config import CrewForgeSettings
from crewforge.dashboard import create_dashboard_app
from crewforge.web_api import JudgeSessionRegistry
from tests.fake_provider import FakeProvider
from tests.mcp_test_server import _scripted_mcp_handler


def _extract_tool_payload(call_result: Any) -> dict[str, Any]:
    if getattr(call_result, "structuredContent", None) is not None:
        return dict(call_result.structuredContent)
    text_parts = [
        item.text for item in call_result.content if getattr(item, "type", None) == "text"
    ]
    return json.loads("".join(text_parts))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def test_remote_https_host_origin_and_token_leakage_protection(tmp_path: Path) -> None:
    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        auth_token="mcp-secret-token",
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        # Unauthenticated access rejected
        unauth_mcp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert unauth_mcp.status_code == 401
        unauth_settings = await client.get("/api/settings")
        assert unauth_settings.status_code == 401

        # Unknown Host rejected (DNS rebinding protection)
        bad_host = await client.get("/api/health", headers={"Host": "evil.example.com"})
        assert bad_host.status_code == 403

        # Allowed remote Host passes /api/health
        ok_health = await client.get(
            "/api/health", headers={"Host": "judge.example.trycloudflare.com"}
        )
        assert ok_health.status_code == 200
        assert ok_health.json()["remote_mcp_enabled"] is True

        # Remote HTML index must NOT leak the MCP auth token
        remote_index = await client.get("/", headers={"Host": "judge.example.trycloudflare.com"})
        assert remote_index.status_code == 200
        assert "mcp-secret-token" not in remote_index.text

        # Non-HTTPS remote Origin rejected for mutating requests
        bad_origin = await client.post(
            "/api/chats",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Origin": "http://judge.example.trycloudflare.com",
                "Authorization": "Bearer mcp-secret-token",
            },
            json={"title": "Insecure"},
        )
        assert bad_origin.status_code == 403

        # HTTPS remote Origin accepted
        ok_origin = await client.post(
            "/api/chats",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Origin": "https://judge.example.trycloudflare.com",
                "Authorization": "Bearer mcp-secret-token",
                "X-CrewForge-Session": "judge-alpha",
            },
            json={"title": "Secure"},
        )
        assert ok_origin.status_code == 201

        # Invalid session ID rejected
        bad_session = await client.get(
            "/api/settings",
            headers={
                "Authorization": "Bearer mcp-secret-token",
                "X-CrewForge-Session": "../escape",
            },
        )
        assert bad_session.status_code == 400

        # Remote HTTPS MCP status reflects forwarded HTTPS endpoint and session header
        status_resp = await client.get(
            "/api/mcp/status",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "X-Forwarded-Proto": "https",
                "Authorization": "Bearer mcp-secret-token",
                "X-CrewForge-Session": "judge-alpha",
            },
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["http_endpoint"] == "https://judge.example.trycloudflare.com/mcp"
        assert status_data["remote_https"] is True
        assert status_data["session_id"] == "judge-alpha"
        shared_cfg = status_data["shared_http_config"]["mcpServers"]["crewforge-shared"]
        assert shared_cfg["url"] == "https://judge.example.trycloudflare.com/mcp"
        assert shared_cfg["headers"]["Authorization"] == "Bearer mcp-secret-token"
        assert shared_cfg["headers"]["X-CrewForge-Session"] == "judge-alpha"


async def test_judge_session_isolation_settings_chats_and_byok_no_dev_quotas(
    tmp_path: Path,
) -> None:
    registry = JudgeSessionRegistry(base_db_path=tmp_path / "judge_session.db")
    alpha = registry.get_session("judge-alpha")
    beta = registry.get_session("judge-beta")

    from crewforge.web_api import JudgeSettingsUpdate

    alpha_patch = {
        "billing_mode": "byok",
        "provider": "openai",
        "model": "gpt-5.4",
        "api_key": "sk-alpha-secret-9999",
        "allow_paid_runs": True,
        "max_output_tokens": 16384,
        "max_cost_per_run_usd": 2.5,
        "total_budget_ceiling_usd": 20.0,
    }
    alpha.update_settings(JudgeSettingsUpdate.model_validate(alpha_patch))
    alpha_runtime = alpha.build_runtime_settings()
    assert alpha_runtime.openai_api_key is not None
    assert alpha_runtime.openai_api_key.get_secret_value() == "sk-alpha-secret-9999"
    assert alpha_runtime.max_output_tokens == 16384
    assert alpha_runtime.allow_paid_runs is True

    # Beta session has zero visibility into Alpha's credentials or chats
    alpha.create_chat(title="Alpha Confidential Chat")
    beta_public = beta.get_public_settings()
    assert beta_public["api_key_configured"] is False
    assert beta_public["api_key_hint"] is None
    assert beta_public["billing_mode"] == "free"
    assert beta.list_chats() == []

    # App-level isolation across HTTP endpoints with separate MCP auth token
    app = create_dashboard_app(
        db_path=tmp_path / "app_judge.db",
        auth_token="mcp-only-auth-token",
        require_auth=True,
        require_react=False,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        set_alpha = await client.post(
            "/api/settings",
            headers={
                "Authorization": "Bearer mcp-only-auth-token",
                "X-CrewForge-Session": "judge-alpha",
            },
            json={
                "billing_mode": "byok",
                "provider": "openai",
                "model": "gpt-5.4",
                "api_key": "sk-judge-alpha-key",
                "allow_paid_runs": True,
                "max_output_tokens": 16384,
            },
        )
        assert set_alpha.status_code == 200
        assert set_alpha.json()["api_key_configured"] is True

        get_beta = await client.get(
            "/api/settings",
            headers={
                "Authorization": "Bearer mcp-only-auth-token",
                "X-CrewForge-Session": "judge-beta",
            },
        )
        assert get_beta.status_code == 200
        assert get_beta.json()["api_key_configured"] is False
        assert "sk-judge-alpha-key" not in get_beta.text


async def test_remote_mcp_initialize_list_tools_solve_and_cross_session_trace_isolation(
    tmp_path: Path,
) -> None:
    def _fake_provider_factory(settings: CrewForgeSettings) -> FakeProvider:
        return FakeProvider(
            handler=_scripted_mcp_handler,
            provider_name="fake-remote-mcp",
            model_name=settings.model,
        )

    app = create_dashboard_app(
        db_path=tmp_path / "remote_mcp.db",
        auth_token="remote-mcp-bearer-token",
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
        provider_factory=_fake_provider_factory,
    )
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    try:
        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started

        base_url = f"http://127.0.0.1:{port}"
        alpha_headers = {
            "Authorization": "Bearer remote-mcp-bearer-token",
            "Host": "judge.example.trycloudflare.com",
            "X-CrewForge-Session": "judge-alpha",
        }
        beta_headers = {
            "Authorization": "Bearer remote-mcp-bearer-token",
            "Host": "judge.example.trycloudflare.com",
            "X-CrewForge-Session": "judge-beta",
        }

        # Configure Alpha judge's BYOK model & key via API (separate from MCP auth token)
        async with httpx.AsyncClient(headers=alpha_headers) as api_client:
            upd = await api_client.post(
                f"{base_url}/api/settings",
                json={
                    "billing_mode": "byok",
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "api_key": "sk-alpha-byok-key",
                    "allow_paid_runs": True,
                    "max_output_tokens": 16384,
                },
            )
            assert upd.status_code == 200

        # Connect as Alpha over Streamable HTTP MCP
        async with (
            httpx.AsyncClient(headers=alpha_headers, timeout=30.0) as http_client,
            streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            init_result = await session.initialize()
            assert init_result.serverInfo.name == "CrewForge"

            tools_response = await session.list_tools()
            assert sorted(t.name for t in tools_response.tools) == [
                "baseline",
                "solve",
                "trace",
            ]

            solve_res = await session.call_tool(
                "solve",
                {"prompt": "Analyze why schema validation matters in multi-agent systems."},
            )
            assert not solve_res.isError
            solve_data = _extract_tool_payload(solve_res)
            assert solve_data["status"] == "completed"
            assert solve_data["model"] == "gpt-5.4"
            run_id = solve_data["run_id"]

            trace_res = await session.call_tool("trace", {"run_id": run_id})
            assert not trace_res.isError
            trace_data = _extract_tool_payload(trace_res)
            assert trace_data["found"] is True

        # Connect as Beta over Streamable HTTP MCP: must NOT see Alpha's run
        async with (
            httpx.AsyncClient(headers=beta_headers, timeout=30.0) as http_client,
            streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            beta_trace_res = await session.call_tool("trace", {"run_id": run_id})
            assert not beta_trace_res.isError
            beta_trace = _extract_tool_payload(beta_trace_res)
            assert beta_trace["found"] is False

        # Mission Control /api/live/runs/{run_id} also enforces session isolation
        async with httpx.AsyncClient() as check_client:
            alpha_run = await check_client.get(
                f"{base_url}/api/live/runs/{run_id}", headers=alpha_headers
            )
            beta_run = await check_client.get(
                f"{base_url}/api/live/runs/{run_id}", headers=beta_headers
            )
            assert alpha_run.status_code == 200
            assert beta_run.status_code == 404
    finally:
        server.should_exit = True
        await serve_task
