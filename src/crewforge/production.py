"""Host the public MCP and authenticated dashboard on the same HTTPS origin."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from crewforge.config import CrewForgeSettings
from crewforge.core.baseline import build_provider_from_settings
from crewforge.dashboard import create_dashboard_app
from crewforge.providers.base import ModelProvider, ProviderConfigurationError
from crewforge.public_mcp_service import create_public_mcp_app

_PROJECT = Path(os.environ.get("CREWFORGE_PROJECT_DIR", Path(__file__).resolve().parents[2]))


def create_release_app(*, settings: CrewForgeSettings | None = None) -> Starlette:
    """Keep anonymous requests on the public app and private APIs behind dashboard auth."""
    base_settings = settings or CrewForgeSettings()
    paid_enabled = os.environ.get("CREWFORGE_RELEASE_ENABLE_PAID_INFERENCE") == "true"
    public_settings = base_settings.model_copy(update={
        "allow_paid_runs": paid_enabled and base_settings.allow_paid_runs,
    })
    public = create_public_mcp_app(settings=public_settings)

    def dashboard_provider(runtime: CrewForgeSettings) -> ModelProvider:
        if runtime.allow_paid_runs and not paid_enabled:
            raise ProviderConfigurationError(
                "Paid model execution is disabled for this release.",
                provider=runtime.provider.value,
                model=runtime.model,
            )
        return build_provider_from_settings(runtime)

    dashboard_dir = Path(os.environ.get("CREWFORGE_DASHBOARD_DATA_DIR", "/data/dashboard"))
    public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    dashboard = create_dashboard_app(
        db_path=dashboard_dir / "judge_session.db",
        budget_db_path=dashboard_dir / "judge_budget.db",
        replay_dir=_PROJECT / "demo/traces",
        benchmark_path=_PROJECT / "demo/v2_reconciled_scores.json",
        provider_factory=dashboard_provider,
        mcp_path="/mcp/workspace",
        mcp_port=int(os.environ.get("PORT", "8000")),
        require_auth=True,
        require_react=True,
        allowed_hosts=[public_domain] if public_domain else None,
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with public.router.lifespan_context(public), dashboard.router.lifespan_context(
            dashboard
        ):
            yield

    async def release_health(_request: Request) -> JSONResponse:
        commit = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or os.environ.get(
            "CREWFORGE_RELEASE_SHA", ""
        )
        return JSONResponse({
            "status": "ok",
            "service": "crewforge",
            "commit": commit if re.fullmatch(r"[a-f0-9]{40}", commit) else None,
            "paid_inference_enabled": paid_enabled,
            "public_paid_inference_enabled": public_settings.allow_paid_runs,
        })

    return Starlette(
        routes=[
            Route("/health", release_health),
            Route("/mcp", public, methods=["GET", "POST", "DELETE"]),
            Mount("/", dashboard),
        ],
        lifespan=lifespan,
    )


if __name__ == "__main__":
    uvicorn.run(
        create_release_app(),
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
