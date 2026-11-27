"""CrewForge web server: Conversational UI, Gamified Office, MCP, and Mission Control."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from crewforge.auth import (
    AuthRateLimiter,
    EmailDeliveryError,
    EmailDeliveryNotConfiguredError,
    is_email_delivery_configured,
    verify_google_id_token,
)
from crewforge.config import CrewForgeSettings, sanitize_error_message
from crewforge.dashboard_api import (
    BenchmarkStore,
    DashboardNotFoundError,
    DashboardValidationError,
    ReadOnlyTraceStore,
    ReplayStore,
)
from crewforge.mcp_server import create_mcp_server
from crewforge.providers.base import ModelProvider
from crewforge.web_api import (
    ChatMessageRequest,
    JudgeSessionRegistry,
    JudgeSettingsUpdate,
    current_session_id_var,
    validate_session_id,
)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIST_DIR = _ROOT_DIR / "frontend" / "dist"
_LEGACY_UI_DIR = Path(__file__).resolve().parent / "dashboard_ui"

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_ALLOWED_HOST_HEADERS = _LOCAL_HOSTS | {"testserver"}
_MAX_REQUEST_BYTES = 524_288  # 512 KB
_SESSION_COOKIE_NAME = "crewforge_session"
_CSRF_COOKIE_NAME = "crewforge_csrf"
_PUBLIC_AUTH_PATHS = {
    "/api/health",
    "/api/auth/me",
    "/api/auth/google",
    "/api/auth/email/request",
    "/api/auth/email/verify",
    "/api/auth/logout",
}

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#6366f1"/>'
    '<text x="16" y="21" text-anchor="middle" fill="#ffffff" '
    'font-family="sans-serif" font-size="14" font-weight="700">CF</text></svg>'
)


def _parse_allowed_hosts(allowed_hosts: Iterable[str] | None) -> tuple[str, ...]:
    raw_items: list[str] = []
    if allowed_hosts is not None:
        raw_items.extend(str(item) for item in allowed_hosts)
    env_hosts = os.environ.get("CREWFORGE_ALLOWED_HOSTS", "")
    if env_hosts:
        raw_items.extend(env_hosts.split(","))
    cleaned: list[str] = []
    for item in raw_items:
        host = item.strip().lower()
        if not host:
            continue
        parsed = urlparse(host if "://" in host else f"https://{host}").hostname
        if parsed and parsed not in cleaned:
            cleaned.append(parsed)
    return tuple(cleaned)


class _LocalSecurityMiddleware(BaseHTTPMiddleware):
    """Enforce Host/Origin validation, admin/judge separation, MCP ownership, and token auth."""

    def __init__(
        self,
        app: Starlette,
        *,
        admin_token: str,
        registry: JudgeSessionRegistry,
        require_auth: bool = True,
        allowed_hosts: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._admin_token_hash = hashlib.sha256(admin_token.encode("utf-8")).hexdigest()
        self._registry = registry
        self._require_auth = require_auth
        extra = set(_parse_allowed_hosts(allowed_hosts))
        self._allowed_host_headers = _ALLOWED_HOST_HEADERS | extra
        self._allowed_origin_hosts = _LOCAL_HOSTS | extra
        self._remote_hosts = extra - _LOCAL_HOSTS
        self._mcp_lock = threading.Lock()
        self._mcp_session_owners: dict[str, str] = {}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 0. Host header validation (DNS rebinding protection)
        raw_host = request.headers.get("host", "")
        parsed_host = ""
        if raw_host:
            parsed_host = (urlparse(f"http://{raw_host}").hostname or "").lower()
            if parsed_host not in self._allowed_host_headers:
                return JSONResponse({"error": "Invalid Host header."}, status_code=403)

        is_forwarded_remote = bool(
            request.headers.get("cf-connecting-ip")
            or request.headers.get("x-forwarded-host")
        )
        is_local_request = (
            (not parsed_host or parsed_host in _ALLOWED_HOST_HEADERS)
            and not is_forwarded_remote
        )

        # 1. Request size limit
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > _MAX_REQUEST_BYTES:
                    return JSONResponse(
                        {"error": "Request body exceeds 512 KB limit."}, status_code=413
                    )
            except ValueError:
                return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)

        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            body = await request.body()
            if len(body) > _MAX_REQUEST_BYTES:
                return JSONResponse(
                    {"error": "Request body exceeds 512 KB limit."}, status_code=413
                )

            # 2. Origin / Referer validation (CSRF protection)
            origin = request.headers.get("origin") or request.headers.get("referer")
            if origin:
                parsed = urlparse(origin)
                hostname = (parsed.hostname or "").lower()
                if hostname not in self._allowed_origin_hosts:
                    return JSONResponse(
                        {"error": "Cross-origin request rejected."}, status_code=403
                    )
                if hostname in self._remote_hosts and parsed.scheme != "https":
                    return JSONResponse(
                        {"error": "Remote origins must use HTTPS."}, status_code=403
                    )

        # 3. Authentication & server-side session derivation on /api/* and /mcp
        path = request.url.path
        is_public_auth = path in _PUBLIC_AUTH_PATHS
        requires_endpoint_auth = (
            (path.startswith("/api/") and not is_public_auth) or path.startswith("/mcp")
        )
        role = "admin"
        session_id = "default"
        auth_via = "local"
        expected_csrf: str | None = None

        header_tok = request.headers.get("x-crewforge-token")
        auth_hdr = request.headers.get("authorization", "")
        bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else None
        provided = (header_tok or bearer_tok or "").strip()
        cookie_val = (request.cookies.get(_SESSION_COOKIE_NAME) or "").strip()

        if provided:
            provided_hash = hashlib.sha256(provided.encode("utf-8")).hexdigest()
            if secrets.compare_digest(provided_hash, self._admin_token_hash):
                if not is_local_request and requires_endpoint_auth:
                    return JSONResponse(
                        {"error": "Local administrator access is restricted to localhost."},
                        status_code=403,
                    )
                if is_local_request:
                    role = "admin"
                    session_id = "default"
                    auth_via = "bearer"
            else:
                judge_session = self._registry.authenticate_judge_token(provided)
                if judge_session is not None:
                    role = "user"
                    session_id = judge_session
                    auth_via = "bearer"
                elif requires_endpoint_auth:
                    return JSONResponse(
                        {"error": "Unauthorized endpoint access."}, status_code=401
                    )
        elif cookie_val:
            cookie_sess = self._registry.auth_store.authenticate_browser_session(cookie_val)
            if cookie_sess is not None:
                role = "user"
                session_id = cookie_sess["user_id"]
                auth_via = "cookie"
                expected_csrf = cookie_sess["csrf_token"]
            elif requires_endpoint_auth:
                return JSONResponse(
                    {"error": "Unauthorized endpoint access."}, status_code=401
                )
        elif requires_endpoint_auth and (self._require_auth or not is_local_request):
            return JSONResponse(
                {"error": "Unauthorized endpoint access."}, status_code=401
            )

        # Enforce CSRF token on mutating cookie-authenticated requests (including logout)
        if (
            auth_via == "cookie"
            and expected_csrf is not None
            and request.method in ("POST", "PUT", "PATCH", "DELETE")
        ):
            csrf_hdr = (request.headers.get("x-csrf-token") or "").strip()
            if not csrf_hdr or not secrets.compare_digest(csrf_hdr, expected_csrf):
                return JSONResponse(
                    {"error": "Invalid or missing CSRF token."}, status_code=403
                )

        if requires_endpoint_auth:
            # 4. Admin-only route guard
            if path.startswith("/api/admin/") and (role != "admin" or not is_local_request):
                return JSONResponse(
                    {"error": "Administrator privileges required."}, status_code=403
                )

            # 5. Prevent cross-session or default-session spoofing via X-CrewForge-Session
            claimed_session = request.headers.get("x-crewforge-session")
            if claimed_session is not None and claimed_session.strip():
                try:
                    validated_claim = validate_session_id(claimed_session)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
                if validated_claim != session_id:
                    return JSONResponse(
                        {"error": "Cross-session access denied."}, status_code=403
                    )

            # 6. Enforce MCP Streamable HTTP session ownership (mcp-session-id)
            if path.startswith("/mcp"):
                mcp_sid = request.headers.get("mcp-session-id")
                if mcp_sid:
                    with self._mcp_lock:
                        owner = self._mcp_session_owners.get(mcp_sid)
                    if owner is not None and owner != session_id:
                        return JSONResponse(
                            {"error": "MCP session belongs to another user."}, status_code=403
                        )

        request.state.auth_role = role
        request.state.session_id = session_id
        request.state.auth_via = auth_via
        request.state.is_local_request = is_local_request
        request.state.csrf_token = expected_csrf
        token = current_session_id_var.set(session_id)
        try:
            response = await call_next(request)
        finally:
            current_session_id_var.reset(token)

        if path.startswith("/mcp") and response.status_code < 400:
            resp_mcp_sid = response.headers.get("mcp-session-id")
            if resp_mcp_sid:
                with self._mcp_lock:
                    if request.method == "DELETE":
                        self._mcp_session_owners.pop(resp_mcp_sid, None)
                    else:
                        self._mcp_session_owners.setdefault(resp_mcp_sid, session_id)
        return response


def create_dashboard_app(
    *,
    db_path: str | Path = ".crewforge/judge_session.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
    budget_db_path: str | Path | None = None,
    provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
    auth_token: str | None = None,
    require_auth: bool = False,
    require_react: bool = False,
    allowed_hosts: Iterable[str] | None = None,
    google_client_id: str | None = None,
    google_token_verifier: Callable[[str], Any] | None = None,
    email_delivery_fn: Callable[[str, str], Any] | None = None,
    rate_limiter: AuthRateLimiter | None = None,
) -> Starlette:
    """Create the Starlette app for CrewForge Chat, Gamified Office, MCP, and Mission Control."""
    resolved_google_client_id = (
        google_client_id
        or os.environ.get("CREWFORGE_GOOGLE_CLIENT_ID", "").strip()
        or None
    )
    resolved_rate_limiter = rate_limiter or AuthRateLimiter()
    react_index = _FRONTEND_DIST_DIR / "index.html"
    use_react = react_index.is_file()
    if require_react and not use_react:
        raise RuntimeError(
            "React production build not found at frontend/dist/index.html. "
            "Run: cd frontend && npm ci && npm run build"
        )
    ui_dir = _FRONTEND_DIST_DIR if use_react else _LEGACY_UI_DIR

    resolved_admin_token = (
        auth_token
        or os.environ.get("CREWFORGE_ADMIN_TOKEN", "").strip()
        or os.environ.get("CREWFORGE_MCP_AUTH_TOKEN", "").strip()
        or secrets.token_urlsafe(24)
    )
    extra_hosts = _parse_allowed_hosts(allowed_hosts)
    replay_store = ReplayStore(replay_dir=replay_dir)
    benchmark_store = BenchmarkStore(benchmark_path=benchmark_path, dvg_path=dvg_path)
    registry = JudgeSessionRegistry(
        base_db_path=db_path,
        base_budget_db_path=budget_db_path,
        provider_factory=provider_factory,
    )

    # Shared MCP server routing each request to its isolated judge session DB & settings
    mcp_server = create_mcp_server(
        database_factory=registry.get_mcp_database,
        settings_factory=lambda: registry.get_current_session().build_runtime_settings(),
        provider_factory=provider_factory,
        host="127.0.0.1",
        port=8765,
        allowed_hosts=extra_hosts,
    )
    mcp_http_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        await registry.get_mcp_database("default")
        async with mcp_server.session_manager.run():
            try:
                yield
            finally:
                await registry.close_all()

    async def index(_request: Request) -> Response:
        html_path = ui_dir / "index.html"
        if use_react:
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return FileResponse(html_path)

    async def favicon(_request: Request) -> Response:
        return Response(_FAVICON_SVG, media_type="image/svg+xml")

    async def health(_request: Request) -> JSONResponse:
        default_mgr = registry.get_session("default")
        return JSONResponse(
            {
                "status": "ok",
                "service": "crewforge-mission-control",
                "mode": "read_only_observability",
                "ui": "react" if use_react else "legacy",
                "db_path": str(default_mgr.db_path),
                "db_exists": default_mgr.db_path.is_file(),
                "replay_dir": str(replay_store.replay_dir),
                "remote_mcp_enabled": bool(extra_hosts),
            }
        )

    # ---------------- Read-Only Mission Control Endpoints ----------------
    async def live_runs(_request: Request) -> JSONResponse:
        mgr = registry.get_current_session()
        trace_store = ReadOnlyTraceStore(db_path=mgr.db_path)
        return JSONResponse({"runs": trace_store.list_runs()})

    async def live_run_detail(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        mgr = registry.get_current_session()
        trace_store = ReadOnlyTraceStore(db_path=mgr.db_path)
        raw_since = request.query_params.get("since_seq")
        since_seq: int | None = None
        if raw_since is not None and raw_since.strip():
            try:
                since_seq = int(raw_since)
                if since_seq < 0:
                    raise ValueError("since_seq must be non-negative")
            except ValueError:
                return JSONResponse({"error": "Invalid since_seq parameter."}, status_code=400)
        try:
            payload = await trace_store.get_run_payload(run_id, since_seq=since_seq)
            if payload.get("status") == "running":
                wait_state = mgr.get_provider_wait_state(run_id)
                if wait_state:
                    payload["waiting_for_provider"] = True
                    payload["waiting_reason"] = wait_state["waiting_reason"]
                    payload["provider_wait_seconds"] = wait_state["provider_wait_seconds"]
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
        return JSONResponse(registry.get_current_session().get_public_settings())

    async def post_settings(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            patch = JudgeSettingsUpdate.model_validate(body)
            return JSONResponse(registry.get_current_session().update_settings(patch))
        except Exception as exc:
            return JSONResponse({"error": sanitize_error_message(str(exc))}, status_code=400)

    async def list_chats(_request: Request) -> JSONResponse:
        return JSONResponse({"chats": registry.get_current_session().list_chats()})

    async def create_chat(request: Request) -> JSONResponse:
        title = "New Chat"
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("title"), str):
                title = body["title"]
        except Exception:
            pass
        chat = registry.get_current_session().create_chat(title=title)
        return JSONResponse(chat, status_code=201)

    async def get_chat(request: Request) -> JSONResponse:
        chat = registry.get_current_session().get_chat(request.path_params["chat_id"])
        if chat is None:
            return JSONResponse({"error": "Chat not found."}, status_code=404)
        return JSONResponse(chat)

    async def delete_chat(request: Request) -> JSONResponse:
        ok = registry.get_current_session().delete_chat(request.path_params["chat_id"])
        if not ok:
            return JSONResponse({"error": "Chat not found."}, status_code=404)
        return JSONResponse({"deleted": True})

    async def post_chat_message(request: Request) -> JSONResponse:
        chat_id = request.path_params["chat_id"]
        try:
            body = await request.json()
            req = ChatMessageRequest.model_validate(body)
            res = await registry.get_current_session().submit_message(chat_id, req)
            return JSONResponse(res, status_code=202)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"error": sanitize_error_message(str(exc))}, status_code=400)

    async def cancel_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        cancelled = registry.get_current_session().cancel_run(run_id)
        return JSONResponse({"run_id": run_id, "cancelled": cancelled})

    def _is_https_request(request: Request) -> bool:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        return forwarded_proto == "https" or request.url.scheme == "https"

    def _set_auth_cookies(
        response: JSONResponse, request: Request, sess: dict[str, str]
    ) -> JSONResponse:
        secure_cookie = _is_https_request(request)
        response.set_cookie(
            key=_SESSION_COOKIE_NAME,
            value=sess["session_cookie"],
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
            path="/",
            max_age=7 * 24 * 3600,
        )
        response.set_cookie(
            key=_CSRF_COOKIE_NAME,
            value=sess["csrf_token"],
            httponly=False,
            secure=secure_cookie,
            samesite="lax",
            path="/",
            max_age=7 * 24 * 3600,
        )
        return response

    def _client_ip(request: Request) -> str:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if xff:
            return xff
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    async def auth_me(request: Request) -> JSONResponse:
        auth_via = getattr(request.state, "auth_via", "local")
        session_id = getattr(request.state, "session_id", "default")
        role = getattr(request.state, "auth_role", "admin")
        is_local = getattr(request.state, "is_local_request", True)
        auth_required = bool(require_auth or not is_local)
        email_ready = bool(email_delivery_fn is not None or is_email_delivery_configured())
        if auth_via == "local":
            return JSONResponse(
                {
                    "authenticated": False,
                    "auth_required": auth_required,
                    "google_enabled": bool(resolved_google_client_id or google_token_verifier),
                    "google_client_id": resolved_google_client_id,
                    "email_delivery_configured": email_ready,
                }
            )
        user = registry.auth_store.get_user(session_id) if session_id != "default" else None
        return JSONResponse(
            {
                "authenticated": True,
                "auth_required": auth_required,
                "role": role,
                "user_id": session_id,
                "user": user,
                "csrf_token": getattr(request.state, "csrf_token", None),
                "google_enabled": bool(resolved_google_client_id or google_token_verifier),
                "google_client_id": resolved_google_client_id,
                "email_delivery_configured": email_ready,
            }
        )

    async def auth_google(request: Request) -> JSONResponse:
        allowed, retry_after = resolved_rate_limiter.check_and_record(
            "google_auth_ip", _client_ip(request), limit=20, window_seconds=600
        )
        if not allowed:
            return JSONResponse(
                {"error": "Too many authentication attempts. Please try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        try:
            body = await request.json()
            id_token = str(body.get("id_token") or "") if isinstance(body, dict) else ""
            raw_mode = (
                str(body.get("mode") or "login").strip().lower()
                if isinstance(body, dict)
                else "login"
            )
            if raw_mode not in {"login", "signup"}:
                return JSONResponse({"error": "Invalid authentication mode."}, status_code=400)
            claims = await verify_google_id_token(
                id_token,
                client_id=resolved_google_client_id,
                custom_verifier=google_token_verifier,
            )
            user = registry.auth_store.resolve_google_user(
                sub=claims["sub"],
                email=claims["email"],
                email_verified=claims["email_verified"],
                display_name=claims.get("name"),
                mode=raw_mode,
            )
            if user.get("requires_email_verification"):
                registry.auth_store.request_email_verification(
                    user["email"], mode="signup", delivery_fn=email_delivery_fn
                )
                return JSONResponse(
                    {
                        "authenticated": False,
                        "requires_email_verification": True,
                        "email": user["email"],
                        "message": (
                            "Please enter the 6-digit verification code sent to your email "
                            "to activate your new account."
                        ),
                    }
                )
            sess = registry.auth_store.create_browser_session(
                user_id=user["user_id"], auth_provider="google"
            )
            resp = JSONResponse(
                {
                    "authenticated": True,
                    "user": user,
                    "csrf_token": sess["csrf_token"],
                }
            )
            return _set_auth_cookies(resp, request, sess)
        except EmailDeliveryNotConfiguredError:
            return JSONResponse(
                {
                    "error": (
                        "Email verification is temporarily unavailable, so new account "
                        "registrations cannot be completed right now."
                    )
                },
                status_code=503,
            )
        except EmailDeliveryError:
            return JSONResponse(
                {"error": "Unable to send verification email right now. Please try again."},
                status_code=502,
            )
        except Exception as exc:
            return JSONResponse({"error": sanitize_error_message(str(exc))}, status_code=401)

    async def auth_email_request(request: Request) -> JSONResponse:
        client_ip = _client_ip(request)
        allowed_ip, retry_ip = resolved_rate_limiter.check_and_record(
            "email_request_ip", client_ip, limit=5, window_seconds=600
        )
        if not allowed_ip:
            return JSONResponse(
                {"error": "Too many verification code requests. Please try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_ip)},
            )
        try:
            body = await request.json()
            email = str(body.get("email") or "") if isinstance(body, dict) else ""
            raw_mode = (
                str(body.get("mode") or "login").strip().lower()
                if isinstance(body, dict)
                else "login"
            )
            if raw_mode not in {"login", "signup"}:
                return JSONResponse({"error": "Invalid authentication mode."}, status_code=400)
            norm_key = email.strip().lower()
            if norm_key:
                allowed_email, retry_email = resolved_rate_limiter.check_and_record(
                    "email_request_addr", norm_key, limit=3, window_seconds=600
                )
                if not allowed_email:
                    return JSONResponse(
                        {"error": "Too many verification requests for this email address."},
                        status_code=429,
                        headers={"Retry-After": str(retry_email)},
                    )
            res = registry.auth_store.request_email_verification(
                email, mode=raw_mode, delivery_fn=email_delivery_fn
            )
            return JSONResponse(res)
        except EmailDeliveryNotConfiguredError:
            return JSONResponse(
                {
                    "error": (
                        "Email sign-in is temporarily unavailable. "
                        "Please try again later or contact the administrator."
                    )
                },
                status_code=503,
            )
        except EmailDeliveryError:
            return JSONResponse(
                {"error": "Unable to send verification email right now. Please try again."},
                status_code=502,
            )
        except Exception as exc:
            return JSONResponse({"error": sanitize_error_message(str(exc))}, status_code=400)

    async def auth_email_verify(request: Request) -> JSONResponse:
        allowed, retry_after = resolved_rate_limiter.check_and_record(
            "email_verify_ip", _client_ip(request), limit=10, window_seconds=600
        )
        if not allowed:
            return JSONResponse(
                {"error": "Too many verification attempts. Please try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Invalid request body.")
            email = str(body.get("email") or "")
            code = str(body.get("code") or "")
            raw_mode = str(body.get("mode") or "login").strip().lower()
            if raw_mode not in {"login", "signup"}:
                return JSONResponse({"error": "Invalid authentication mode."}, status_code=400)
            user = registry.auth_store.verify_email_code(email, code, mode=raw_mode)
            sess = registry.auth_store.create_browser_session(
                user_id=user["user_id"], auth_provider="email"
            )
            resp = JSONResponse(
                {
                    "authenticated": True,
                    "user": user,
                    "csrf_token": sess["csrf_token"],
                }
            )
            return _set_auth_cookies(resp, request, sess)
        except Exception as exc:
            return JSONResponse({"error": sanitize_error_message(str(exc))}, status_code=401)

    async def auth_logout(request: Request) -> JSONResponse:
        cookie_val = (request.cookies.get(_SESSION_COOKIE_NAME) or "").strip()
        if cookie_val:
            registry.auth_store.revoke_browser_session(cookie_val)
        resp = JSONResponse({"authenticated": False})
        resp.delete_cookie(_SESSION_COOKIE_NAME, path="/")
        resp.delete_cookie(_CSRF_COOKIE_NAME, path="/")
        return resp

    async def user_list_mcp_tokens(_request: Request) -> JSONResponse:
        uid = current_session_id_var.get()
        if uid == "default":
            return JSONResponse({"tokens": []})
        return JSONResponse({"tokens": registry.list_judge_tokens(user_id=uid)})

    async def user_issue_mcp_token(request: Request) -> JSONResponse:
        uid = current_session_id_var.get()
        if uid == "default":
            return JSONResponse(
                {"error": "Sign in as a user or use /api/admin/tokens for judge provisioning."},
                status_code=400,
            )
        label = "MCP Client"
        revoke_previous = False
        try:
            body = await request.json()
            if isinstance(body, dict):
                if isinstance(body.get("label"), str):
                    label = body["label"]
                revoke_previous = bool(body.get("revoke_previous", False))
        except Exception:
            pass
        issued = registry.issue_judge_token(
            label=label, user_id=uid, revoke_previous=revoke_previous
        )
        return JSONResponse(issued, status_code=201)

    async def user_revoke_mcp_token(request: Request) -> JSONResponse:
        uid = current_session_id_var.get()
        token_id = request.path_params["token_id"]
        if uid == "default":
            return JSONResponse({"error": "Token not found."}, status_code=404)
        revoked = registry.revoke_judge_token(token_id, owner_user_id=uid)
        if not revoked:
            return JSONResponse({"error": "Active MCP token not found."}, status_code=404)
        return JSONResponse({"revoked": True, "token_id": token_id})

    async def admin_list_tokens(_request: Request) -> JSONResponse:
        return JSONResponse({"tokens": registry.list_judge_tokens()})

    async def admin_issue_token(request: Request) -> JSONResponse:
        label = "Judge"
        try:
            body = await request.json()
            if isinstance(body, dict) and isinstance(body.get("label"), str):
                label = body["label"]
        except Exception:
            pass
        issued = registry.issue_judge_token(label=label)
        return JSONResponse(issued, status_code=201)

    async def admin_revoke_token(request: Request) -> JSONResponse:
        identifier = request.path_params["identifier"]
        revoked = registry.revoke_judge_token(identifier)
        if not revoked:
            return JSONResponse({"error": "Active judge token not found."}, status_code=404)
        return JSONResponse({"revoked": True, "identifier": identifier})

    async def mcp_status(request: Request) -> JSONResponse:
        mgr = registry.get_current_session()
        session_id = current_session_id_var.get()
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
        req_host = request.headers.get("host", "").strip()
        scheme = "https" if forwarded_proto == "https" else request.url.scheme
        if req_host:
            public_url = f"{scheme}://{req_host}/mcp"
        else:
            public_url = f"{str(request.base_url).rstrip('/')}/mcp"
        internal_port = request.url.port or 8765
        internal_url = (
            f"http://127.0.0.1:{internal_port}/mcp"
            if scheme == "https" or (urlparse(public_url).hostname or "") not in _LOCAL_HOSTS
            else public_url
        )
        header_tok = request.headers.get("x-crewforge-token")
        auth_hdr = request.headers.get("authorization", "")
        bearer_tok = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else None
        caller_tok = (header_tok or bearer_tok or "").strip() or None
        data = await mgr.get_mcp_status(
            mcp_url=public_url,
            auth_token=caller_tok,
            session_id=session_id,
            public_url=public_url,
            internal_url=internal_url,
        )
        return JSONResponse(data)

    routes: list[Route | Mount] = [
        Route("/", index),
        Route("/favicon.ico", favicon),
        Route("/api/health", health),
        Route("/api/auth/me", auth_me, methods=["GET"]),
        Route("/api/auth/google", auth_google, methods=["POST"]),
        Route("/api/auth/email/request", auth_email_request, methods=["POST"]),
        Route("/api/auth/email/verify", auth_email_verify, methods=["POST"]),
        Route("/api/auth/logout", auth_logout, methods=["POST"]),
        Route("/api/mcp/tokens", user_list_mcp_tokens, methods=["GET"]),
        Route("/api/mcp/tokens", user_issue_mcp_token, methods=["POST"]),
        Route("/api/mcp/tokens/{token_id}", user_revoke_mcp_token, methods=["DELETE"]),
        Route("/api/admin/tokens", admin_list_tokens, methods=["GET"]),
        Route("/api/admin/tokens", admin_issue_token, methods=["POST"]),
        Route("/api/admin/tokens/{identifier}", admin_revoke_token, methods=["DELETE"]),
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
        *mcp_http_app.routes,
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

    app = Starlette(routes=routes, lifespan=_lifespan)
    app.state.registry = registry
    app.add_middleware(
        _LocalSecurityMiddleware,
        admin_token=resolved_admin_token,
        registry=registry,
        require_auth=require_auth,
        allowed_hosts=extra_hosts,
    )
    return app


def run_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path = ".crewforge/judge_session.db",
    replay_dir: str | Path = "demo/traces",
    benchmark_path: str | Path = "demo/v2_reconciled_scores.json",
    dvg_path: str | Path | None = None,
    require_react: bool = True,
    allowed_hosts: Iterable[str] | None = None,
    auth_token: str | None = None,
) -> None:
    """Start the CrewForge web server on localhost (optionally fronted by an HTTPS tunnel)."""
    if host not in _LOCAL_HOSTS:
        raise ValueError("CrewForge web server is restricted to localhost bindings.")
    app = create_dashboard_app(
        db_path=db_path,
        replay_dir=replay_dir,
        benchmark_path=benchmark_path,
        dvg_path=dvg_path,
        auth_token=auth_token,
        require_auth=True,
        require_react=require_react,
        allowed_hosts=allowed_hosts,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `crewforge-dashboard`."""
    parser = argparse.ArgumentParser(
        prog="crewforge-dashboard",
        description="CrewForge conversational UI, gamified office, MCP, and Mission Control.",
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
    parser.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help="Allow falling back to legacy dashboard_ui if frontend/dist is not built.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Optional public hostname allowed behind an HTTPS reverse proxy / tunnel.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    run_dashboard(
        host=args.host,
        port=args.port,
        db_path=args.db_path,
        replay_dir=args.replay_dir,
        benchmark_path=args.benchmark_file,
        dvg_path=args.dvg_file,
        require_react=not args.allow_legacy_fallback,
        allowed_hosts=args.allowed_host,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
