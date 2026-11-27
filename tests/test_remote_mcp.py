"""Offline verification tests for Remote HTTPS MCP hardening and judge session isolation."""

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
from crewforge.web_api import JudgeSessionRegistry, JudgeSettingsUpdate
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


async def test_admin_judge_separation_host_validation_and_zero_token_leakage(
    tmp_path: Path,
) -> None:
    admin_secret = "admin-local-only-secret-token"
    app = create_dashboard_app(
        db_path=tmp_path / "judge_session.db",
        auth_token=admin_secret,
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        # 1. Unauthenticated access rejected
        unauth_mcp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert unauth_mcp.status_code == 401
        unauth_settings = await client.get("/api/settings")
        assert unauth_settings.status_code == 401

        # 2. Malicious Host headers rejected (DNS rebinding protection)
        for bad_host_val in (
            "evil.example.com",
            "judge.example.trycloudflare.com.evil.com",
            "127.0.0.1.evil.com",
        ):
            bad_host = await client.get("/api/health", headers={"Host": bad_host_val})
            assert bad_host.status_code == 403

        # 3. Local admin issues two unique revocable judge tokens on localhost
        issue_a = await client.post(
            "/api/admin/tokens",
            headers={"Authorization": f"Bearer {admin_secret}"},
            json={"label": "Judge Alpha"},
        )
        assert issue_a.status_code == 201
        alpha_info = issue_a.json()
        alpha_token = alpha_info["token"]
        alpha_session = alpha_info["session_id"]
        assert alpha_token.startswith("cfj_")
        assert alpha_session.startswith("judge_")

        issue_b = await client.post(
            "/api/admin/tokens",
            headers={"Authorization": f"Bearer {admin_secret}"},
            json={"label": "Judge Beta"},
        )
        assert issue_b.status_code == 201
        beta_info = issue_b.json()
        beta_token = beta_info["token"]
        beta_session = beta_info["session_id"]
        assert alpha_token != beta_token
        assert alpha_session != beta_session

        # 4. Admin token listing NEVER exposes raw tokens or hashes
        list_resp = await client.get(
            "/api/admin/tokens",
            headers={"Authorization": f"Bearer {admin_secret}"},
        )
        assert list_resp.status_code == 200
        assert alpha_token not in list_resp.text
        assert beta_token not in list_resp.text
        assert "token_hash" not in list_resp.text

        # 5. Local admin token is REJECTED over remote tunnel host or CF-Connecting-IP
        remote_admin = await client.get(
            "/api/settings",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {admin_secret}",
            },
        )
        assert remote_admin.status_code == 403
        forwarded_admin = await client.get(
            "/api/admin/tokens",
            headers={
                "CF-Connecting-IP": "203.0.113.10",
                "Authorization": f"Bearer {admin_secret}",
            },
        )
        assert forwarded_admin.status_code == 403

        # 6. Judge tokens are REJECTED from admin endpoints
        judge_admin = await client.get(
            "/api/admin/tokens",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {alpha_token}",
            },
        )
        assert judge_admin.status_code == 403

        # 7. Judges cannot spoof another session or the default dev session
        spoof_beta = await client.get(
            "/api/settings",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {alpha_token}",
                "X-CrewForge-Session": beta_session,
            },
        )
        assert spoof_beta.status_code == 403
        spoof_default = await client.get(
            "/api/settings",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {alpha_token}",
                "X-CrewForge-Session": "default",
            },
        )
        assert spoof_default.status_code == 403

        # 8. HTML and /api/mcp/status NEVER leak auth tokens or provider credentials
        for host_hdr in ("127.0.0.1", "judge.example.trycloudflare.com"):
            html_resp = await client.get("/", headers={"Host": host_hdr})
            assert html_resp.status_code == 200
            assert admin_secret not in html_resp.text
            assert alpha_token not in html_resp.text

        # Non-HTTPS remote Origin rejected
        bad_origin = await client.post(
            "/api/chats",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Origin": "http://judge.example.trycloudflare.com",
                "Authorization": f"Bearer {alpha_token}",
            },
            json={"title": "Insecure"},
        )
        assert bad_origin.status_code == 403

        # Remote HTTPS MCP status derives session server-side and hides raw token
        status_resp = await client.get(
            "/api/mcp/status",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "X-Forwarded-Proto": "https",
                "Authorization": f"Bearer {alpha_token}",
            },
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["http_endpoint"] == "https://judge.example.trycloudflare.com/mcp"
        assert status_data["remote_https"] is True
        assert status_data["session_id"] == alpha_session
        assert alpha_token not in status_resp.text
        shared_cfg = status_data["shared_http_config"]["mcpServers"]["crewforge-shared"]
        assert shared_cfg["headers"] == {"Authorization": "Bearer <YOUR_MCP_TOKEN>"}

        # 9. Revoking a judge token immediately invalidates access
        revoke_resp = await client.delete(
            f"/api/admin/tokens/{beta_session}",
            headers={"Authorization": f"Bearer {admin_secret}"},
        )
        assert revoke_resp.status_code == 200
        beta_after_revoke = await client.get(
            "/api/settings",
            headers={
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {beta_token}",
            },
        )
        assert beta_after_revoke.status_code == 401


async def test_judge_session_credential_isolation_and_unrestricted_byok(
    tmp_path: Path,
) -> None:
    registry = JudgeSessionRegistry(base_db_path=tmp_path / "judge_session.db")
    tok_a = registry.issue_judge_token(label="Alpha")
    tok_b = registry.issue_judge_token(label="Beta")
    alpha = registry.get_session(tok_a["session_id"])
    beta = registry.get_session(tok_b["session_id"])

    alpha.update_settings(
        JudgeSettingsUpdate.model_validate(
            {
                "billing_mode": "byok",
                "provider": "openai",
                "model": "gpt-5.4",
                "api_key": "sk-alpha-secret-9999",
                "allow_paid_runs": True,
                "max_output_tokens": 16384,
                "max_cost_per_run_usd": 2.5,
                "total_budget_ceiling_usd": 20.0,
            }
        )
    )
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


async def test_standard_remote_mcp_client_ownership_solve_and_trace_isolation(
    tmp_path: Path,
) -> None:
    def _fake_provider_factory(settings: CrewForgeSettings) -> FakeProvider:
        return FakeProvider(
            handler=_scripted_mcp_handler,
            provider_name="fake-remote-mcp",
            model_name=settings.model,
        )

    admin_secret = "local-admin-secret"
    app = create_dashboard_app(
        db_path=tmp_path / "remote_mcp.db",
        auth_token=admin_secret,
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
        provider_factory=_fake_provider_factory,
    )
    registry: JudgeSessionRegistry = app.state.registry
    alpha_token = registry.issue_judge_token(label="Judge Alpha")["token"]
    beta_token = registry.issue_judge_token(label="Judge Beta")["token"]

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
        # Standard remote MCP client headers: ONLY Host + Authorization Bearer
        alpha_headers = {
            "Authorization": f"Bearer {alpha_token}",
            "Host": "judge.example.trycloudflare.com",
        }
        beta_headers = {
            "Authorization": f"Bearer {beta_token}",
            "Host": "judge.example.trycloudflare.com",
        }

        # Configure Alpha judge's BYOK model & key via authenticated /api/settings
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
            assert "sk-alpha-byok-key" not in upd.text

        # Connect as Alpha over standard Streamable HTTP MCP and verify MCP session ownership
        async with (
            httpx.AsyncClient(headers=alpha_headers, timeout=30.0) as http_client,
            streamable_http_client(f"{base_url}/mcp", http_client=http_client) as (
                read_stream,
                write_stream,
                get_mcp_session_id,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            init_result = await session.initialize()
            assert init_result.serverInfo.name == "CrewForge"

            alpha_mcp_sid = get_mcp_session_id()
            assert alpha_mcp_sid

            # Judge Beta attempting to hijack Judge Alpha's mcp-session-id must get 403
            async with httpx.AsyncClient(headers=beta_headers) as hijack_client:
                hijack = await hijack_client.post(
                    f"{base_url}/mcp",
                    headers={
                        "mcp-session-id": alpha_mcp_sid,
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={"jsonrpc": "2.0", "id": 99, "method": "tools/list"},
                )
                assert hijack.status_code == 403

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
            admin_run = await check_client.get(
                f"{base_url}/api/live/runs/{run_id}",
                headers={"Authorization": f"Bearer {admin_secret}"},
            )
            assert alpha_run.status_code == 200
            assert beta_run.status_code == 404
            assert admin_run.status_code == 404
    finally:
        server.should_exit = True
        await serve_task
