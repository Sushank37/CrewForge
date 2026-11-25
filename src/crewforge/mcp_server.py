"""Official MCP Python SDK server exposing CrewForge baseline and trace tools."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from crewforge.config import CrewForgeSettings, configure_logging
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager
from crewforge.providers.base import ModelProvider
from crewforge.storage.database import Database

logger = logging.getLogger("crewforge.mcp")


def create_mcp_server(
    *,
    settings: CrewForgeSettings | None = None,
    database: Database | None = None,
    provider: ModelProvider | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    """Create and configure the CrewForge FastMCP server instance.

    Exposes only Phase 1 tools:
    - baseline(prompt)
    - trace(run_id)
    """
    resolved_settings = settings or CrewForgeSettings()
    configure_logging(resolved_settings.log_level)

    resolved_db = database or Database(resolved_settings.db_path)
    owns_provider = provider is None
    resolved_provider: ModelProvider | None = provider
    initialized = False

    async def _ensure_initialized() -> tuple[Database, ModelProvider, BudgetManager]:
        nonlocal resolved_provider, initialized
        if not initialized:
            await resolved_db.initialize()
            initialized = True
        if resolved_provider is None:
            resolved_provider = build_provider_from_settings(resolved_settings)
        budget_manager = BudgetManager(resolved_db, resolved_settings)
        return resolved_db, resolved_provider, budget_manager

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
        await resolved_db.initialize()
        nonlocal initialized
        initialized = True
        try:
            yield
        finally:
            if owns_provider and resolved_provider is not None:
                await resolved_provider.aclose()
            await resolved_db.close()

    mcp = FastMCP(
        name="CrewForge",
        instructions=(
            "CrewForge Phase 1 MCP server providing single-agent baseline execution "
            "and persistent execution trace retrieval."
        ),
        host=host,
        port=port,
        lifespan=_lifespan,
    )

    @mcp.tool()
    async def baseline(prompt: str) -> dict[str, Any]:
        """Run the single-agent baseline on an arbitrary prompt and return a structured result."""
        db, active_provider, budget_manager = await _ensure_initialized()
        runner = BaselineRunner(
            provider=active_provider,
            database=db,
            budget_manager=budget_manager,
            settings=resolved_settings,
        )
        logger.info(
            "Executing baseline tool (provider=%s, model=%s)",
            active_provider.provider_name,
            active_provider.model_name,
        )
        result = await runner.run(prompt)
        return result.model_dump(mode="json")

    @mcp.tool()
    async def trace(run_id: str) -> dict[str, Any]:
        """Retrieve the persisted execution trace, events, usage, and budget records for a run."""
        db, _, _ = await _ensure_initialized()
        run_trace = await db.get_run_trace(run_id)
        if run_trace is None:
            return {
                "found": False,
                "run_id": run_id,
                "error": f"Run '{run_id}' was not found.",
            }
        data = run_trace.model_dump(mode="json")
        data["found"] = True
        return data

    return mcp


def run_mcp_server(
    *,
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    settings: CrewForgeSettings | None = None,
) -> None:
    """Start the CrewForge MCP server on the requested transport."""
    resolved_settings = settings or CrewForgeSettings()
    configure_logging(resolved_settings.log_level)
    if transport == "streamable-http" and host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Streamable HTTP transport is restricted to localhost addresses.")

    logger.info("Starting CrewForge MCP server (transport=%s)", transport)
    server = create_mcp_server(settings=resolved_settings, host=host, port=port)
    server.run(transport=transport)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `crewforge-mcp`."""
    parser = argparse.ArgumentParser(
        prog="crewforge-mcp",
        description="Run the CrewForge MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Localhost bind address for streamable-http (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Localhost port for streamable-http (default: 8000).",
    )
    args = parser.parse_args(argv)
    try:
        run_mcp_server(transport=args.transport, host=args.host, port=args.port)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Error starting CrewForge MCP server: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
