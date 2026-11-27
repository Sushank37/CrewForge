"""Offline tests for public ngrok deployment security, endpoint allowlisting, and capacity."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from crewforge.dashboard import create_dashboard_app
from crewforge.web_api import validate_remote_provider_url
from tests.fake_provider import FakeProvider

NGROK_HOST = "stairlike-causatively-cordell.ngrok-free.dev"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _mcp_fake_handler(prompt: str, response_model: Any) -> Any:
    schema_name = response_model.__name__ if response_model is not None else ""
    if schema_name in {"ExecutionPlan", "PlannerOutput"}:
        return {
            "goal": "Verify remote MCP over ngrok",
            "user_requirements": ["Confirm authenticated MCP execution"],
            "acceptance_criteria": ["Verified output"],
            "agents": [
                {
                    "agent_ref": "a1",
                    "role": "Security Analyst",
                    "specialty": "Security",
                    "instructions": "Verify prompt.",
                    "capabilities": ["reasoning"],
                }
            ],
            "tasks": [
                {
                    "task_ref": "t1",
                    "title": "Verify",
                    "description": "Verify",
                    "assigned_agent_ref": "a1",
                    "depends_on": [],
                    "expected_deliverables": ["Result"],
                }
            ],
            "assumptions": [],
            "missing_information": [],
        }
    if schema_name in {"WorkerOutput", "SpecialistOutput"}:
        return {
            "findings": "Verified.",
            "evidence": [
                {
                    "source_type": "user_prompt",
                    "source_ref": "up_1",
                    "excerpt": prompt[:40],
                }
            ],
            "claims": [
                {
                    "statement": "Remote execution verified.",
                    "category": "given_fact",
                    "evidence_indices": [0],
                    "requirement_refs": ["req_1"],
                }
            ],
            "assumptions": [],
            "uncertainties": [],
            "missing_information": [],
            "outgoing_messages": [],
        }
    if schema_name in {"ReviewBatchOutput", "ReviewerBatchOutput"}:
        import re

        cids = list(dict.fromkeys(re.findall(r"(clm_[A-Za-z0-9_]+)", prompt)))
        return {
            "verdicts": [
                {
                    "claim_id": cid,
                    "decision": "approve",
                    "rationale": "Verified.",
                    "evidence_refs": [],
                    "required_correction": None,
                    "blocking_severity": "none",
                    "blocking_issues": [],
                }
                for cid in cids
            ]
        }
    if schema_name == "SynthesisOutput":
        import re

        m = re.search(r"(clm_[A-Za-z0-9_]+)@v(\d+)", prompt)
        cid = m.group(1) if m else "clm_1"
        ver = int(m.group(2)) if m else 1
        return {
            "direct_summary": "Remote execution verified.",
            "accepted_segments": [
                {
                    "segment_type": "accepted_claim",
                    "text": "Remote execution verified.",
                    "claim_refs": [{"claim_id": cid, "claim_version": ver}],
                    "requirement_ref": "req_1",
                }
            ],
            "conditional_segments": [],
            "recommendations": [],
            "new_factual_assertions": [],
            "referenced_claim_ids": [cid],
            "assumptions": [],
            "missing_information": [],
        }
    raise AssertionError(f"Unexpected schema: {schema_name}")


def test_validate_remote_provider_url_blocks_ssrf_and_allows_approved_apis() -> None:
    assert validate_remote_provider_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert (
        validate_remote_provider_url("https://api.groq.com/openai/v1")
        == "https://api.groq.com/openai/v1"
    )

    for bad in (
        "http://api.openai.com/v1",
        "http://127.0.0.1:11434/v1",
        "https://localhost:8080/v1",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.5/v1",
        "https://internal.local/v1",
        "https://metadata.internal/v1",
        "https://user:pass@api.openai.com/v1",
        "https://unapproved.example.com/v1",
    ):
        with pytest.raises(ValueError):
            validate_remote_provider_url(bad)

    # DNS rebinding / private IP resolution blocked
    with pytest.raises(ValueError, match="restricted network"):
        validate_remote_provider_url(
            "https://api.openai.com/v1",
            dns_resolver=lambda _host: ["127.0.0.1"],
        )


async def test_ngrok_remote_auth_ssrf_ollama_adaptive_capacity_and_mcp_security(
    tmp_path: Path,
) -> None:
    sent_codes: dict[str, str] = {}
    admin_secret = "local-dev-admin-secret"
    app = create_dashboard_app(
        db_path=tmp_path / "ngrok_test.db",
        auth_token=admin_secret,
        require_auth=True,
        require_react=False,
        allowed_hosts=[NGROK_HOST],
        provider_factory=lambda s: FakeProvider(
            handler=_mcp_fake_handler, provider_name="fake-ngrok", model_name=s.model
        ),
        email_delivery_fn=lambda e, c: sent_codes.__setitem__(e, c),
    )

    ngrok_headers = {
        "Host": NGROK_HOST,
        "X-Forwarded-Proto": "https",
        "X-Forwarded-For": "203.0.113.42",
        "Origin": f"https://{NGROK_HOST}",
    }

    transport = httpx.ASGITransport(app=app)
    # 1. Malicious Host header rejected (403)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"https://{NGROK_HOST}"
    ) as bad_host_client:
        bad_host = await bad_host_client.get("/api/auth/me", headers={"Host": "evil.attacker.com"})
        assert bad_host.status_code == 403

    # 2. Local admin token MUST be rejected over ngrok (even if Host is rewritten to 127.0.0.1)
    async with httpx.AsyncClient(
        transport=transport, base_url=f"https://{NGROK_HOST}"
    ) as tunnel_admin_attempt:
        r_admin = await tunnel_admin_attempt.get(
            "/api/admin/tokens",
            headers={
                "Host": "127.0.0.1:8765",
                "X-Forwarded-For": "203.0.113.42",
                "Authorization": f"Bearer {admin_secret}",
            },
        )
        assert r_admin.status_code == 403

    # 3. Local dev account CAN use Ollama on direct localhost
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as local:
        local_ok = await local.post(
            "/api/settings",
            headers={"Authorization": f"Bearer {admin_secret}"},
            json={
                "billing_mode": "free",
                "provider": "ollama",
                "model": "qwen3:4b",
                "adaptive_team_capacity": True,
            },
        )
        assert local_ok.status_code == 200
        assert local_ok.json()["provider"] == "ollama"

    # 4. Remote user signup + OTP + isolation + endpoint/Ollama restrictions + adaptive capacity
    async with (
        httpx.AsyncClient(transport=transport, base_url=f"https://{NGROK_HOST}") as user_a,
        httpx.AsyncClient(transport=transport, base_url=f"https://{NGROK_HOST}") as user_b,
    ):
        req_a = await user_a.post(
            "/api/auth/email/request",
            headers=ngrok_headers,
            json={"email": "alice@example.com", "mode": "signup"},
        )
        assert req_a.status_code == 200
        ver_a = await user_a.post(
            "/api/auth/email/verify",
            headers=ngrok_headers,
            json={
                "email": "alice@example.com",
                "code": sent_codes["alice@example.com"],
                "mode": "signup",
            },
        )
        assert ver_a.status_code == 200
        csrf_a = ver_a.json()["csrf_token"]
        a_hdrs = {**ngrok_headers, "X-CSRF-Token": csrf_a}

        # Remote user CANNOT use Ollama
        ollama_block = await user_a.post(
            "/api/settings",
            headers=a_hdrs,
            json={"billing_mode": "free", "provider": "ollama", "model": "qwen3:4b"},
        )
        assert ollama_block.status_code == 400

        # Remote user CANNOT point openai_base_url to localhost, private IP, metadata, or unapproved
        for bad_url in (
            "http://127.0.0.1:8080/v1",
            "https://169.254.169.254/v1",
            "https://10.0.0.1/v1",
            "https://unapproved.example.com/v1",
        ):
            bad_ep = await user_a.post(
                "/api/settings",
                headers=a_hdrs,
                json={
                    "billing_mode": "byok",
                    "provider": "openai_compatible",
                    "model": "custom-model",
                    "openai_base_url": bad_url,
                    "api_key": "sk-test",
                },
            )
            assert bad_ep.status_code == 400

        # Approved HTTPS model API succeeds AND preserves adaptive team capacity
        good_ep = await user_a.post(
            "/api/settings",
            headers=a_hdrs,
            json={
                "billing_mode": "byok",
                "provider": "openai",
                "model": "gpt-5.4",
                "openai_base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
                "max_model_calls_per_run": 20,
                "adaptive_team_capacity": True,
            },
        )
        assert good_ep.status_code == 200
        assert good_ep.json()["adaptive_team_capacity"] is True
        assert good_ep.json()["max_workers"] == 17

        # User A creates a private chat; User B cannot see it
        chat_a = await user_a.post(
            "/api/chats", headers=a_hdrs, json={"title": "Alice Ngrok Chat"}
        )
        assert chat_a.status_code == 201
        chat_a_id = chat_a.json()["chat_id"]

        await user_b.post(
            "/api/auth/email/request",
            headers=ngrok_headers,
            json={"email": "bob@example.com", "mode": "signup"},
        )
        ver_b = await user_b.post(
            "/api/auth/email/verify",
            headers=ngrok_headers,
            json={
                "email": "bob@example.com",
                "code": sent_codes["bob@example.com"],
                "mode": "signup",
            },
        )
        assert ver_b.status_code == 200
        cross_chat = await user_b.get(f"/api/chats/{chat_a_id}", headers=ngrok_headers)
        assert cross_chat.status_code == 404

    # 5. Live localhost HTTP server test for MCP auth + session ownership over ngrok Host
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    serve_task = asyncio.create_task(server.serve())
    try:
        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started

        tok_a = app.state.registry.issue_judge_token(label="Ngrok A")["token"]
        tok_b = app.state.registry.issue_judge_token(label="Ngrok B")["token"]
        base_url = f"http://127.0.0.1:{port}"

        async with httpx.AsyncClient() as client:
            unauth_mcp = await client.post(
                f"{base_url}/mcp",
                headers={"Host": NGROK_HOST, "X-Forwarded-Proto": "https"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            )
            assert unauth_mcp.status_code == 401

            init_a = await client.post(
                f"{base_url}/mcp",
                headers={
                    "Host": NGROK_HOST,
                    "X-Forwarded-Proto": "https",
                    "Authorization": f"Bearer {tok_a}",
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ngrok-test", "version": "1.0"},
                    },
                },
            )
            assert init_a.status_code == 200
            mcp_sid = init_a.headers.get("mcp-session-id")
            assert mcp_sid

            # Token B cannot hijack Token A's mcp-session-id
            hijack = await client.post(
                f"{base_url}/mcp",
                headers={
                    "Host": NGROK_HOST,
                    "X-Forwarded-Proto": "https",
                    "Authorization": f"Bearer {tok_b}",
                    "mcp-session-id": mcp_sid,
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            assert hijack.status_code == 403
    finally:
        server.should_exit = True
        await serve_task
