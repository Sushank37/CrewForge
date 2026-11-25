"""Integration tests for CrewForge MCP server using stdio and localhost Streamable HTTP."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from crewforge.config import CrewForgeSettings
from crewforge.mcp_server import create_mcp_server
from crewforge.storage.database import Database
from tests.fake_provider import FakeProvider
from tests.mcp_test_server import _scripted_mcp_handler

REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_tool_payload(call_result: Any) -> dict[str, Any]:
    if getattr(call_result, "structuredContent", None) is not None:
        return dict(call_result.structuredContent)
    text_parts = [
        item.text for item in call_result.content if getattr(item, "type", None) == "text"
    ]
    return json.loads("".join(text_parts))


async def test_mcp_stdio_client_discovery_solve_baseline_and_trace(
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
        assert tool_names == ["baseline", "solve", "trace"]

        solve_res = await session.call_tool(
            "solve",
            {"prompt": "Analyze why schema validation matters in multi-agent systems."},
        )
        assert not solve_res.isError
        solve_data = _extract_tool_payload(solve_res)
        assert solve_data["mode"] == "network"
        assert solve_data["status"] == "completed"
        assert solve_data["provider"] == "fake-mcp"
        assert solve_data["model"] == "fake-mcp-model"
        assert solve_data["specialist_roles"] == ["Primary Analyst", "Risk Reviewer"]
        assert solve_data["verification"] is not None
        assert solve_data["verification"]["honesty_gate_passed"] is True
        assert solve_data["verification"]["supported_claim_count"] == 2
        assert "events" not in solve_data
        assert "tasks" not in solve_data
        solve_run_id = solve_data["run_id"]

        solve_trace_res = await session.call_tool("trace", {"run_id": solve_run_id})
        assert not solve_trace_res.isError
        solve_trace = _extract_tool_payload(solve_trace_res)
        assert solve_trace["found"] is True
        assert solve_trace["plan"] is not None
        assert len(solve_trace["agents"]) == 2
        assert len(solve_trace["reviewers"]) >= 1
        assert len(solve_trace["tasks"]) == 2
        assert len(solve_trace["claims"]) >= 2
        assert len(solve_trace["review_verdicts"]) >= 2
        assert solve_trace["honesty_gate"] is not None
        assert solve_trace["honesty_gate"]["passed"] is True

        baseline_res = await session.call_tool(
            "baseline",
            {"prompt": "Summarize key risks of skipping schema validation."},
        )
        assert not baseline_res.isError
        baseline_data = _extract_tool_payload(baseline_res)
        assert baseline_data["mode"] == "baseline"
        assert baseline_data["status"] == "completed"
        baseline_run_id = baseline_data["run_id"]

        trace_res = await session.call_tool("trace", {"run_id": baseline_run_id})
        assert not trace_res.isError
        trace_data = _extract_tool_payload(trace_res)
        assert trace_data["found"] is True
        assert trace_data["run_id"] == baseline_run_id
        assert len(trace_data["usage_records"]) == 1

        missing_res = await session.call_tool("trace", {"run_id": "run_missing_id"})
        missing_data = _extract_tool_payload(missing_res)
        assert missing_data["found"] is False


async def test_mcp_streamable_http_live_localhost_client_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "mcp_http.db"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])

    settings = CrewForgeSettings(_env_file=None, db_path=db_path, max_retries=0)
    db = Database(db_path)
    provider = FakeProvider(
        provider_name="fake-http",
        model_name="fake-http-model",
        max_retries=0,
        handler=_scripted_mcp_handler,
    )
    server = create_mcp_server(
        settings=settings,
        database=db,
        provider=provider,
        host="127.0.0.1",
        port=port,
    )
    app = server.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uvi_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(uvi_server.serve())

    try:
        for _ in range(40):
            if uvi_server.started:
                break
            await asyncio.sleep(0.05)
        assert uvi_server.started

        url = f"http://127.0.0.1:{port}/mcp"
        async with (
            streamable_http_client(url) as (read_stream, write_stream, _get_session_id),
            ClientSession(read_stream, write_stream) as session,
        ):
            init_res = await session.initialize()
            assert init_res.serverInfo.name == "CrewForge"

            tools = await session.list_tools()
            assert sorted(t.name for t in tools.tools) == ["baseline", "solve", "trace"]

            solve_res = await session.call_tool(
                "solve",
                {"prompt": "Analyze schema validation over Streamable HTTP."},
            )
            solve_data = _extract_tool_payload(solve_res)
            assert solve_data["status"] == "completed"
            trace_res = await session.call_tool("trace", {"run_id": solve_data["run_id"]})
            trace_data = _extract_tool_payload(trace_res)
            assert trace_data["found"] is True
    finally:
        uvi_server.should_exit = True
        await serve_task
        await db.close()
