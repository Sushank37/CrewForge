"""Integration tests for CrewForge MCP server using the official MCP Python SDK."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from crewforge.config import CrewForgeSettings
from crewforge.mcp_server import create_mcp_server
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider

REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_tool_payload(call_result: Any) -> dict[str, Any]:
    if getattr(call_result, "structuredContent", None) is not None:
        return dict(call_result.structuredContent)
    text_parts = [
        item.text
        for item in call_result.content
        if getattr(item, "type", None) == "text"
    ]
    return json.loads("".join(text_parts))


async def test_mcp_stdio_client_initialization_discovery_baseline_and_trace(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "mcp_stdio.db"
    env = os.environ.copy()
    env["CREWFORGE_DB_PATH"] = str(db_path)
    env["PYTHONPATH"] = str(REPO_ROOT)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.mcp_test_server"],
        env=env,
        cwd=str(REPO_ROOT),
    )

    async with (
        stdio_client(server_params) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        init_result = await session.initialize()
        assert init_result.serverInfo.name == "CrewForge"

        tools_response = await session.list_tools()
        tool_names = sorted(tool.name for tool in tools_response.tools)
        assert tool_names == ["baseline", "trace"]
        assert "solve" not in tool_names

        baseline_res = await session.call_tool(
            "baseline",
            {"prompt": "Summarize key risks of skipping schema validation."},
        )
        assert not baseline_res.isError
        baseline_data = _extract_tool_payload(baseline_res)
        assert baseline_data["status"] == "completed"
        assert baseline_data["provider"] == "fake-mcp"
        assert baseline_data["model"] == "fake-mcp-model"
        run_id = baseline_data["run_id"]
        assert run_id.startswith("run_")

        trace_res = await session.call_tool("trace", {"run_id": run_id})
        assert not trace_res.isError
        trace_data = _extract_tool_payload(trace_res)
        assert trace_data["found"] is True
        assert trace_data["run_id"] == run_id
        assert len(trace_data["events"]) == 4
        assert len(trace_data["usage_records"]) == 1

        missing_res = await session.call_tool("trace", {"run_id": "run_missing_id"})
        missing_data = _extract_tool_payload(missing_res)
        assert missing_data["found"] is False


def test_mcp_streamable_http_app_builds_cleanly(tmp_path: Path) -> None:
    db_path = tmp_path / "mcp_http.db"
    settings = CrewForgeSettings(_env_file=None, db_path=db_path)
    server = create_mcp_server(
        settings=settings,
        database=Database(db_path),
        provider=FakeProvider(),
        host="127.0.0.1",
        port=8000,
    )
    app = server.streamable_http_app()
    assert app is not None
