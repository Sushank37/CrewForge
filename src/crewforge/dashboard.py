"""CrewForge Mission Control — lightweight localhost observability server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from crewforge.dashboard_api import (
    BenchmarkStore,
    DashboardNotFoundError,
    DashboardValidationError,
    ReadOnlyTraceStore,
    ReplayStore,
)

_UI_DIR = Path(__file__).resolve().parent / "dashboard_ui"

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#3b82f6"/>'
    '<text x="16" y="21" text-anchor="middle" fill="#ffffff" '
    'font-family="sans-serif" font-size="14" font-weight="700">CF</text></svg>'
)


def create_dashboard_app(
    *,
    db_path: str | Path = ".crewforge/demo_live.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
) -> Starlette:
    """Create the read-only Starlette app for CrewForge Mission Control."""
    trace_store = ReadOnlyTraceStore(db_path=db_path)
    replay_store = ReplayStore(replay_dir=replay_dir)
    benchmark_store = BenchmarkStore(benchmark_path=benchmark_path, dvg_path=dvg_path)

    async def index(_request: Request) -> Response:
        return FileResponse(_UI_DIR / "index.html")

    async def favicon(_request: Request) -> Response:
        return Response(_FAVICON_SVG, media_type="image/svg+xml")

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "crewforge-mission-control",
                "mode": "read_only_observability",
                "db_path": str(trace_store.db_path),
                "db_exists": trace_store.db_path.is_file(),
                "replay_dir": str(replay_store.replay_dir),
            }
        )

    async def live_runs(_request: Request) -> JSONResponse:
        return JSONResponse({"runs": trace_store.list_runs()})

    async def live_run_detail(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        try:
            payload = await trace_store.get_run_payload(run_id)
            return JSONResponse(payload)
        except DashboardValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except DashboardNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    async def replay_files(_request: Request) -> JSONResponse:
        return JSONResponse({"files": replay_store.list_files()})

    async def replay_file_detail(request: Request) -> JSONResponse:
        filename = request.path_params["filename"]
        try:
            payload = replay_store.load_replay(filename)
            return JSONResponse(payload)
        except DashboardValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except DashboardNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    async def benchmark(_request: Request) -> JSONResponse:
        try:
            return JSONResponse(benchmark_store.load())
        except DashboardValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except DashboardNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)

    routes = [
        Route("/", index),
        Route("/favicon.ico", favicon),
        Route("/api/health", health),
        Route("/api/live/runs", live_runs),
        Route("/api/live/runs/{run_id}", live_run_detail),
        Route("/api/replay/files", replay_files),
        Route("/api/replay/files/{filename:path}", replay_file_detail),
        Route("/api/benchmark", benchmark),
        Mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static"),
    ]
    return Starlette(routes=routes)


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path = ".crewforge/demo_live.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
) -> None:
    """Start the CrewForge Mission Control uvicorn server on localhost."""
    app = create_dashboard_app(
        db_path=db_path,
        replay_dir=replay_dir,
        benchmark_path=benchmark_path,
        dvg_path=dvg_path,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `crewforge-dashboard`."""
    parser = argparse.ArgumentParser(
        prog="crewforge-dashboard",
        description="CrewForge Mission Control read-only observability dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--db-path",
        default=".crewforge/demo_live.db",
        help="Explicitly configured demo SQLite database for live monitoring.",
    )
    parser.add_argument(
        "--replay-dir",
        default="demo/traces",
        help="Directory containing sanitized DemoTraceExport JSON files.",
    )
    parser.add_argument(
        "--benchmark-file",
        default="demo/v2_reconciled_scores.json",
        help="Path to reconciled benchmark summary JSON.",
    )
    parser.add_argument(
        "--dvg-file",
        default=None,
        help="Optional path to finalized David vs Goliath reconciled report JSON.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_dashboard(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        replay_dir=args.replay_dir,
        benchmark_path=args.benchmark_file,
        dvg_path=args.dvg_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
