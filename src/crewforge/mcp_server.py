"""Official MCP Python SDK server exposing CrewForge solve, baseline, and trace tools."""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from crewforge.config import CrewForgeSettings, configure_logging
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
from crewforge.core.executor import ModelExecutor
from crewforge.core.orchestrator import NetworkRunner
from crewforge.providers.base import ModelProvider
from crewforge.storage.database import Database

logger = logging.getLogger("crewforge.mcp")


def _build_transport_security(
    allowed_hosts: Iterable[str] | None,
) -> TransportSecuritySettings | None:
    """Configure FastMCP DNS rebinding protection for localhost plus explicit remote HTTPS hosts."""
    hosts = [h.strip().lower() for h in (allowed_hosts or ()) if h and h.strip()]
    if not hosts:
        return None
    base_hosts = ["127.0.0.1", "localhost", "[::1]", "testserver", *hosts]
    allowed_host_patterns: list[str] = []
    allowed_origin_patterns: list[str] = [
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://localhost",
        "http://localhost:*",
        "http://[::1]",
        "http://[::1]:*",
    ]
    for h in base_hosts:
        if h not in allowed_host_patterns:
            allowed_host_patterns.append(h)
        wildcard = f"{h}:*"
        if wildcard not in allowed_host_patterns:
            allowed_host_patterns.append(wildcard)
    for h in hosts:
        for origin in (f"https://{h}", f"https://{h}:*"):
            if origin not in allowed_origin_patterns:
                allowed_origin_patterns.append(origin)
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_host_patterns,
        allowed_origins=allowed_origin_patterns,
    )


def create_mcp_server(
    *,
    settings: CrewForgeSettings | None = None,
    database: Database | None = None,
    database_factory: Callable[[], Database | Awaitable[Database]] | None = None,
    provider: ModelProvider | None = None,
    settings_factory: Callable[[], CrewForgeSettings] | None = None,
    provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    allowed_hosts: Iterable[str] | None = None,
) -> FastMCP:
    """Create and configure the CrewForge FastMCP server instance.

    Shares one ModelExecutor concurrency semaphore across concurrent MCP tool runs,
    or resolves live isolated judge-session settings when settings_factory is supplied.
    """
    resolved_settings = settings or (
        settings_factory() if settings_factory else CrewForgeSettings()
    )
    if settings_factory is None:
        configure_logging(resolved_settings.log_level)

    resolved_db = database or Database(resolved_settings.effective_db_path)
    owns_provider = provider is None and provider_factory is None
    resolved_provider: ModelProvider | None = provider
    shared_budget: BudgetManager | None = None
    shared_executor: ModelExecutor | None = None
    initialized = False

    async def _ensure_initialized() -> tuple[
        Database, ModelProvider, BudgetManager, ModelExecutor, CrewForgeSettings
    ]:
        nonlocal resolved_provider, shared_budget, shared_executor, initialized
        active_db = resolved_db
        if database_factory is not None:
            candidate_db = database_factory()
            active_db = await candidate_db if inspect.isawaitable(candidate_db) else candidate_db
            await active_db.initialize()
        elif not initialized:
            await active_db.initialize()
            initialized = True
        if settings_factory is not None:
            live_settings = settings_factory()
            live_provider = (
                provider
                if provider is not None
                else (
                    provider_factory(live_settings)
                    if provider_factory is not None
                    else build_provider_from_settings(live_settings)
                )
            )
            live_authority = UnifiedBudgetAuthority(
                settings=live_settings,
                migrate_default_dbs=False,
            )
            live_budget = BudgetManager(active_db, live_settings, authority=live_authority)
            live_executor = ModelExecutor(
                provider=live_provider,
                database=active_db,
                budget_manager=live_budget,
                settings=live_settings,
            )
            return active_db, live_provider, live_budget, live_executor, live_settings

        if resolved_provider is None:
            resolved_provider = build_provider_from_settings(resolved_settings)
        if shared_budget is None:
            shared_budget = BudgetManager(active_db, resolved_settings)
        if shared_executor is None:
            shared_executor = ModelExecutor(
                provider=resolved_provider,
                database=active_db,
                budget_manager=shared_budget,
                settings=resolved_settings,
            )
        return active_db, resolved_provider, shared_budget, shared_executor, resolved_settings

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
            "CrewForge Phase 3 MCP server providing verified dynamic multi-agent problem "
            "solving with deterministic arithmetic, independent review, targeted repair, and "
            "an enforceable Honesty Gate (solve), single-agent baseline comparison (baseline), "
            "and full persisted execution trace inspection (trace)."
        ),
        host=host,
        port=port,
        lifespan=_lifespan,
        transport_security=_build_transport_security(allowed_hosts),
    )

    @mcp.tool()
    async def solve(prompt: str) -> dict[str, Any]:
        """Solve an arbitrary prompt using CrewForge's verified multi-agent network."""
        db, active_provider, budget_manager, executor, active_settings = await _ensure_initialized()
        runner = NetworkRunner(
            provider=active_provider,
            database=db,
            budget_manager=budget_manager,
            settings=active_settings,
            executor=executor,
        )
        logger.info(
            "Executing solve tool (provider=%s, model=%s)",
            active_provider.provider_name,
            active_provider.model_name,
        )
        result = await runner.run(prompt)
        return result.model_dump(mode="json")

    @mcp.tool()
    async def baseline(prompt: str) -> dict[str, Any]:
        """Run the single-agent baseline with equivalent arithmetic tool parity."""
        db, active_provider, budget_manager, executor, active_settings = await _ensure_initialized()
        runner = BaselineRunner(
            provider=active_provider,
            database=db,
            budget_manager=budget_manager,
            settings=active_settings,
            executor=executor,
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
        """Retrieve the persisted execution trace, verdicts, tool records, and gate decisions."""
        db, _, _, _, _ = await _ensure_initialized()
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
