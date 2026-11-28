"""Verify the release entry point keeps the anonymous and private surfaces separate."""

from starlette.testclient import TestClient

from crewforge.config import CrewForgeSettings
from crewforge.production import create_release_app


def test_release_serves_dashboard_and_anonymous_mcp(monkeypatch, tmp_path):
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text('<script type="module" src="/assets/app.js"></script>')
    assets = frontend / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("void 0;")
    monkeypatch.setattr("crewforge.dashboard._FRONTEND_DIST_DIR", frontend)
    monkeypatch.setenv("RAILWAY_PUBLIC_DOMAIN", "crewforge.example")
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "a" * 40)
    monkeypatch.setenv("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE", "false")
    monkeypatch.setenv("CREWFORGE_DASHBOARD_DATA_DIR", str(tmp_path / "dashboard"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_DB_PATH", str(tmp_path / "public.db"))
    monkeypatch.setenv("CREWFORGE_PUBLIC_MCP_BUDGET_DB_PATH", str(tmp_path / "budget.db"))
    app = create_release_app(settings=CrewForgeSettings(
        _env_file=None, model="test-model", provider="openai", allow_paid_runs=True,
    ))
    headers = {"Accept": "application/json, text/event-stream"}
    with TestClient(app, base_url="https://crewforge.example") as client:
        assert client.get("/health").json()["commit"] == "a" * 40
        assert client.get("/health").json()["public_paid_inference_enabled"] is False
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").text == "void 0;"
        assert client.get("/api/chats").status_code == 401
        assert client.get("/api/settings").status_code == 401
        assert client.post("/mcp/workspace", headers=headers, json={}).status_code == 401
        init = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25", "capabilities": {},
                "clientInfo": {"name": "offline", "version": "1.0"},
            },
        })
        assert init.status_code == 200, init.text
        headers["Mcp-Session-Id"] = init.headers["mcp-session-id"]
        client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        tools = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        assert tools.status_code == 200 and '"run_task"' in tools.text
        assert all(name in tools.text for name in (
            "run_single_agent", "get_task_status", "get_task_result",
            "get_task_trace", "cancel_task",
        ))
        assert "admin" not in tools.text
