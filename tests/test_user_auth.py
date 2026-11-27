"""Offline tests for Google/verified-email auth, HttpOnly cookies, CSRF, and user MCP tokens."""

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


async def test_google_and_verified_email_auth_cookies_csrf_and_privacy(tmp_path: Path) -> None:
    sent_codes: dict[str, str] = {}

    def _capture_email(email: str, code: str) -> None:
        sent_codes[email] = code

    def _mock_google_verifier(id_token: str) -> dict[str, Any]:
        if id_token == "valid-google-token-alice":
            return {
                "sub": "google-sub-alice-001",
                "email": "alice@example.com",
                "email_verified": True,
                "name": "Alice Judge",
            }
        if id_token == "unverified-google-token":
            return {
                "sub": "google-sub-unverified",
                "email": "unverified@example.com",
                "email_verified": False,
                "name": "Unverified",
            }
        raise ValueError("Invalid Google token.")

    admin_secret = "local-admin-secret-xyz"
    app = create_dashboard_app(
        db_path=tmp_path / "dev_default.db",
        auth_token=admin_secret,
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
        google_client_id="mock-client-id.apps.googleusercontent.com",
        google_token_verifier=_mock_google_verifier,
        email_delivery_fn=_capture_email,
    )

    # Seed a private development chat in the local "default" workspace
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as admin_client:
        dev_chat = await admin_client.post(
            "/api/chats",
            headers={"Authorization": f"Bearer {admin_secret}"},
            json={"title": "Private Local Dev Chat"},
        )
        assert dev_chat.status_code == 201

    remote_headers = {
        "Host": "judge.example.trycloudflare.com",
        "X-Forwarded-Proto": "https",
        "Origin": "https://judge.example.trycloudflare.com",
    }

    # 1. Google Sign-In on Device 1 (HTTPS tunnel) -> permanent user_id + HttpOnly/Secure cookie
    async with httpx.AsyncClient(
        transport=transport, base_url="https://judge.example.trycloudflare.com"
    ) as dev1:
        bad_google = await dev1.post(
            "/api/auth/google",
            headers=remote_headers,
            json={"id_token": "unverified-google-token"},
        )
        assert bad_google.status_code == 401

        login_resp = await dev1.post(
            "/api/auth/google",
            headers=remote_headers,
            json={"id_token": "valid-google-token-alice"},
        )
        assert login_resp.status_code == 200
        login_data = login_resp.json()
        alice_uid = login_data["user"]["user_id"]
        assert alice_uid.startswith("usr_")
        csrf_1 = login_data["csrf_token"]
        set_cookie_hdr = "; ".join(login_resp.headers.get_list("set-cookie"))
        assert "crewforge_session=" in set_cookie_hdr
        assert "HttpOnly" in set_cookie_hdr
        assert "Secure" in set_cookie_hdr

        # Private dev chats are NOT visible to Alice
        alice_chats = await dev1.get("/api/chats", headers=remote_headers)
        assert alice_chats.status_code == 200
        assert alice_chats.json()["chats"] == []

        # CSRF protection: mutating request without X-CSRF-Token is rejected (403)
        no_csrf = await dev1.post(
            "/api/chats",
            headers=remote_headers,
            json={"title": "CSRF Should Fail"},
        )
        assert no_csrf.status_code == 403

        # Mutating request WITH valid X-CSRF-Token succeeds
        created_chat = await dev1.post(
            "/api/chats",
            headers={**remote_headers, "X-CSRF-Token": csrf_1},
            json={"title": "Alice Persistent Chat"},
        )
        assert created_chat.status_code == 201

        # Remote user cannot access /api/admin/* even with valid cookie + CSRF
        admin_attempt = await dev1.get("/api/admin/tokens", headers=remote_headers)
        assert admin_attempt.status_code == 403

        # Logout revokes browser session
        logout_no_csrf = await dev1.post("/api/auth/logout", headers=remote_headers)
        assert logout_no_csrf.status_code == 403
        logout_ok = await dev1.post(
            "/api/auth/logout",
            headers={**remote_headers, "X-CSRF-Token": csrf_1},
        )
        assert logout_ok.status_code == 200
        after_logout = await dev1.get("/api/chats", headers=remote_headers)
        assert after_logout.status_code == 401

    # 2. Device 2 logs in via Verified Email for alice@example.com -> SAME user_id & chat history!
    async with httpx.AsyncClient(
        transport=transport, base_url="https://judge.example.trycloudflare.com"
    ) as dev2:
        req_code = await dev2.post(
            "/api/auth/email/request",
            headers=remote_headers,
            json={"email": "Alice@Example.com"},
        )
        assert req_code.status_code == 200
        assert "code" not in req_code.json()
        code = sent_codes["alice@example.com"]
        assert len(code) == 6

        # Wrong code rejected
        wrong_verify = await dev2.post(
            "/api/auth/email/verify",
            headers=remote_headers,
            json={"email": "alice@example.com", "code": "000000"},
        )
        assert wrong_verify.status_code == 401

        # Correct code succeeds and resolves to Alice's permanent user_id
        ok_verify = await dev2.post(
            "/api/auth/email/verify",
            headers=remote_headers,
            json={"email": "alice@example.com", "code": code},
        )
        assert ok_verify.status_code == 200
        assert ok_verify.json()["user"]["user_id"] == alice_uid

        # Single-use code cannot be reused (even from a fresh unauthenticated client)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://judge.example.trycloudflare.com"
        ) as fresh_client:
            reuse_verify = await fresh_client.post(
                "/api/auth/email/verify",
                headers=remote_headers,
                json={"email": "alice@example.com", "code": code},
            )
            assert reuse_verify.status_code == 401

        # Chat history created on Device 1 persists on Device 2!
        dev2_chats = await dev2.get("/api/chats", headers=remote_headers)
        assert dev2_chats.status_code == 200
        titles = [c["title"] for c in dev2_chats.json()["chats"]]
        assert titles == ["Alice Persistent Chat"]


async def test_user_mcp_token_rotation_preserves_workspace_and_trace(tmp_path: Path) -> None:
    sent_codes: dict[str, str] = {}

    def _fake_provider_factory(settings: CrewForgeSettings) -> FakeProvider:
        return FakeProvider(
            handler=_scripted_mcp_handler,
            provider_name="fake-user-mcp",
            model_name=settings.model,
        )

    app = create_dashboard_app(
        db_path=tmp_path / "dev_default.db",
        auth_token="local-admin-only",
        require_auth=True,
        require_react=False,
        allowed_hosts=["judge.example.trycloudflare.com"],
        provider_factory=_fake_provider_factory,
        email_delivery_fn=lambda e, c: sent_codes.__setitem__(e, c),
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
        local_browser_hdrs = {
            "Host": "127.0.0.1",
            "Origin": "http://127.0.0.1",
        }

        async with httpx.AsyncClient(headers=local_browser_hdrs) as browser:
            await browser.post(
                f"{base_url}/api/auth/email/request",
                json={"email": "bob@example.com"},
            )
            verify = await browser.post(
                f"{base_url}/api/auth/email/verify",
                json={"email": "bob@example.com", "code": sent_codes["bob@example.com"]},
            )
            assert verify.status_code == 200
            bob_uid = verify.json()["user"]["user_id"]
            csrf = verify.json()["csrf_token"]

            # Issue MCP token #1 for Bob's account
            tok1_resp = await browser.post(
                f"{base_url}/api/mcp/tokens",
                headers={"X-CSRF-Token": csrf},
                json={"label": "Laptop MCP"},
            )
            assert tok1_resp.status_code == 201
            tok1 = tok1_resp.json()["token"]
            assert tok1_resp.json()["user_id"] == bob_uid

            # Run MCP solve with token #1
            mcp_hdrs_1 = {
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {tok1}",
            }
            async with (
                httpx.AsyncClient(headers=mcp_hdrs_1, timeout=30.0) as hc1,
                streamable_http_client(f"{base_url}/mcp", http_client=hc1) as (r1, w1, _),
                ClientSession(r1, w1) as s1,
            ):
                await s1.initialize()
                solve_res = await s1.call_tool(
                    "solve",
                    {"prompt": "Analyze why schema validation matters in multi-agent systems."},
                )
                assert not solve_res.isError
                run_id = _extract_tool_payload(solve_res)["run_id"]

            # Rotate Bob's MCP token (revokes tok1, issues tok2 for the SAME user_id)
            tok2_resp = await browser.post(
                f"{base_url}/api/mcp/tokens",
                headers={"X-CSRF-Token": csrf},
                json={"label": "Rotated MCP", "revoke_previous": True},
            )
            assert tok2_resp.status_code == 201
            tok2 = tok2_resp.json()["token"]
            assert tok2_resp.json()["user_id"] == bob_uid
            assert tok2 != tok1

            # Old token #1 is immediately rejected
            old_check = await httpx.AsyncClient().get(
                f"{base_url}/api/settings",
                headers={
                    "Host": "judge.example.trycloudflare.com",
                    "Authorization": f"Bearer {tok1}",
                },
            )
            assert old_check.status_code == 401

            # New rotated token #2 still sees Bob's existing workspace and trace!
            mcp_hdrs_2 = {
                "Host": "judge.example.trycloudflare.com",
                "Authorization": f"Bearer {tok2}",
            }
            async with (
                httpx.AsyncClient(headers=mcp_hdrs_2, timeout=30.0) as hc2,
                streamable_http_client(f"{base_url}/mcp", http_client=hc2) as (r2, w2, _),
                ClientSession(r2, w2) as s2,
            ):
                await s2.initialize()
                trace_res = await s2.call_tool("trace", {"run_id": run_id})
                assert not trace_res.isError
                assert _extract_tool_payload(trace_res)["found"] is True
    finally:
        server.should_exit = True
        await serve_task
