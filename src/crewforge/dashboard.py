"""CrewForge web server: Conversational UI, Gamified Office, and Mission Control."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from crewforge.config import CrewForgeSettings
from crewforge.dashboard_api import (
    BenchmarkStore,
    DashboardNotFoundError,
    DashboardValidationError,
    ReadOnlyTraceStore,
    ReplayStore,
)
from crewforge.providers.base import ModelProvider
from crewforge.web_api import (
    ChatMessageRequest,
    JudgeSessionManager,
    JudgeSettingsUpdate,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIST_DIR = _ROOT_DIR / "frontend" / "dist"
_LEGACY_UI_DIR = Path(__file__).resolve().parent / "dashboard_ui"

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#6366f1"/>'
    '<text x="16" y="21" text-anchor="middle" fill="#ffffff" '
    'font-family="sans-serif" font-size="14" font-weight="700">CF</text></svg>'
)


def create_dashboard_app(
    *,
    db_path: str | Path = ".crewforge/judge_session.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
    budget_db_path: str | Path | None = None,
    provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
) -> Starlette:
    """Create the Starlette app for CrewForge Chat, Gamified Office, and Mission Control."""
    trace_store = ReadOnlyTraceStore(db_path=db_path)
    replay_store = ReplayStore(replay_dir=replay_dir)
    benchmark_store = BenchmarkStore(benchmark_path=benchmark_path, dvg_path=dvg_path)
    session_mgr = JudgeSessionManager(
        db_path=db_path,
        budget_db_path=budget_db_path,
        provider_factory=provider_factory,
    )

    use_react = (_FRONTEND_DIST_DIR / "index.html").is_file()
    ui_dir = _FRONTEND_DIST_DIR if use_react else _LEGACY_UI_DIR

    async def index(_request: Request) -> Response:
        return FileResponse(ui_dir / "index.html")

    async def favicon(_request: Request) -> Response:
        return Response(_FAVICON_SVG, media_type="image/svg+xml")

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "crewforge-mission-control",
                "mode": "read_only_observability",
                "ui": "react" if use_react else "legacy",
                "db_path": str(trace_store.db_path),
                "db_exists": trace_store.db_path.is_file(),
                "replay_dir": str(replay_store.replay_dir),
            }
        )

    # ---------------- Read-Only Mission Control Endpoints ----------------
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

    # ---------------- Conversational & Judge Settings Endpoints ----------------
    async def get_settings(_request: Request) -> JSONResponse:
        return JSONResponse(session_mgr.get_public_settings())

    async def post_settings(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            patch = JudgeSettingsUpdate.model_validate(body)
            return JSONResponse(session_mgr.update_settings(patch))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def list_chats(_request: Request) -> JSONResponse:
        return JSONResponse({"chats": session_mgr.list_chats()})

    async def create_chat(request: Request) -> JSONResponse:
        title = "New Chat"
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("title"), str):
                title = body["title"]
        except Exception:
            pass
        return JSONResponse(session_mgr.create_chat(title=title), status_code=201)

    async def get_chat(request: Request) -> JSONResponse:
        chat = session_mgr.get_chat(request.path_params["chat_id"])
        if chat is None:
            return JSONResponse({"error": "Chat not found."}, status_code=404)
        return JSONResponse(chat)

    async def delete_chat(request: Request) -> JSONResponse:
        ok = session_mgr.delete_chat(request.path_params["chat_id"])
        if not ok:
            return JSONResponse({"error": "Chat not found."}, status_code=404)
        return JSONResponse({"deleted": True})

    async def post_chat_message(request: Request) -> JSONResponse:
        chat_id = request.path_params["chat_id"]
        try:
            body = await request.json()
            req = ChatMessageRequest.model_validate(body)
            res = await session_mgr.submit_message(chat_id, req)
            return JSONResponse(res, status_code=202)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    async def cancel_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        cancelled = session_mgr.cancel_run(run_id)
        return JSONResponse({"run_id": run_id, "cancelled": cancelled})

    async def mcp_status(_request: Request) -> JSONResponse:
        return JSONResponse(session_mgr.get_mcp_status())

    routes: list[Route | Mount] = [
        Route("/", index),
        Route("/favicon.ico", favicon),
        Route("/api/health", health),
        Route("/api/live/runs", live_runs),
        Route("/api/live/runs/{run_id}", live_run_detail),
        Route("/api/replay/files", replay_files),
        Route("/api/replay/files/{filename:path}", replay_file_detail),
        Route("/api/benchmark", benchmark),
        Route("/api/settings", get_settings, methods=["GET"]),
        Route("/api/settings", post_settings, methods=["POST"]),
        Route("/api/chats", list_chats, methods=["GET"]),
        Route("/api/chats", create_chat, methods=["POST"]),
        Route("/api/chats/{chat_id}", get_chat, methods=["GET"]),
        Route("/api/chats/{chat_id}", delete_chat, methods=["DELETE"]),
        Route("/api/chats/{chat_id}/messages", post_chat_message, methods=["POST"]),
        Route("/api/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/api/mcp/status", mcp_status, methods=["GET"]),
    ]
    if use_react and (_FRONTEND_DIST_DIR / "assets").is_dir():
        routes.append(
            Mount(
                "/assets",
                StaticFiles(directory=str(_FRONTEND_DIST_DIR / "assets")),
                name="assets",
            )
        )
    else:
        routes.append(Mount("/static", StaticFiles(directory=str(_LEGACY_UI_DIR)), name="static"))

    return Starlette(routes=routes)


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path = ".crewforge/judge_session.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
) -> None:
    """Start the CrewForge web server on localhost."""
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
        description="CrewForge conversational UI, gamified office, and Mission Control.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--db-path",
        default=".crewforge/judge_session.db",
        help="Isolated SQLite database for live sessions and observability.",
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
