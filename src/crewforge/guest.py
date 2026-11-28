"""Browser access to the public job manager with per-session ownership."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from crewforge.public_mcp_service import PublicJobManager

_COOKIE = "crewforge_guest"
_MAX_BODY = 512 * 1024
_DISABLED = "Guest execution is paused while paid inference is disabled."


def _reply(data: dict[str, Any], status: int = 200) -> JSONResponse:
    return JSONResponse(data, status, headers={"Cache-Control": "no-store"})


def _token(request: Request) -> str | None:
    value = request.cookies.get(_COOKIE, "")
    return value if re.fullmatch(r"[A-Za-z0-9_-]{43}", value) else None


def _owner(request: Request) -> str | None:
    token = _token(request)
    return hashlib.sha256(token.encode()).hexdigest() if token else None


def create_guest_app(manager: PublicJobManager, *, public_domain: str) -> Starlette:
    def same_origin(request: Request) -> bool:
        expected = f"https://{public_domain}" if public_domain else (
            f"{request.url.scheme}://{request.url.netloc}"
        )
        return request.headers.get("origin") == expected

    async def session(request: Request) -> JSONResponse:
        if request.headers.get("sec-fetch-site") == "cross-site":
            return _reply({"error": "Same-origin request required."}, 403)
        enabled = manager.settings.allow_paid_runs
        response = _reply({
            "available": enabled,
            "message": None if enabled else _DISABLED,
        })
        if not _token(request):
            response.set_cookie(
                _COOKIE,
                secrets.token_urlsafe(32),
                path="/api/guest",
                httponly=True,
                samesite="strict",
                secure=bool(public_domain) or request.url.scheme == "https",
            )
        return response

    async def read_payload(request: Request) -> dict[str, Any] | JSONResponse:
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return _reply({"error": "Expected JSON."}, 415)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_BODY:
                return _reply({"error": "Request body exceeds 512 KB limit."}, 413)
            body.extend(chunk)
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeError):
            return _reply({"error": "Invalid JSON."}, 400)
        if not isinstance(data, dict):
            return _reply({"error": "Expected a JSON object."}, 400)
        return data

    async def submit(request: Request) -> JSONResponse:
        if not same_origin(request):
            return _reply({"error": "Same-origin request required."}, 403)
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Open the guest office to start a session."}, 401)
        if not manager.settings.allow_paid_runs:
            return _reply({"error": _DISABLED}, 503)
        data = await read_payload(request)
        if isinstance(data, JSONResponse):
            return data
        try:
            if set(data) == {"compare_to"} and isinstance(data["compare_to"], str):
                job = await manager.compare_guest_job(data["compare_to"], owner)
                if job is None:
                    return _reply({"error": "Task not found."}, 404)
            elif set(data) == {"prompt"} and isinstance(data["prompt"], str):
                job = await manager.enqueue_task(
                    prompt=data["prompt"], mode="network", guest_owner=owner,
                )
            else:
                raise ValueError("Provide a task prompt or an original team run.")
        except ValueError as exc:
            return _reply({"error": str(exc)}, 400)
        return _reply(job, 202 if job["accepted"] else 429)

    async def chats(request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Open the guest workspace to start a session."}, 401)
        if request.method == "GET":
            return _reply({"chats": manager.list_guest_chats(owner)})
        if not same_origin(request):
            return _reply({"error": "Same-origin request required."}, 403)
        return _reply(manager.create_guest_chat(owner), 201)

    async def chat(request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Chat not found."}, 404)
        chat_id = request.path_params["chat_id"]
        detail = manager.get_guest_chat(owner, chat_id)
        return _reply(detail) if detail is not None else _reply({"error": "Chat not found."}, 404)

    async def message(request: Request) -> JSONResponse:
        if not same_origin(request):
            return _reply({"error": "Same-origin request required."}, 403)
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Chat not found."}, 404)
        if not manager.settings.allow_paid_runs:
            return _reply({"error": _DISABLED}, 503)
        data = await read_payload(request)
        if isinstance(data, JSONResponse):
            return data
        prompt, request_id, documents = (data.get(x) for x in ("prompt", "request_id", "documents"))
        if (
            not isinstance(prompt, str) or not prompt.strip()
            or not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9-]{36}", request_id)
            or not isinstance(documents, list) or len(documents) > 5
            or any(
                not isinstance(doc, dict) or set(doc) != {"title", "content"}
                or not all(isinstance(v, str) for v in doc.values())
                for doc in documents
            )
        ):
            return _reply({"error": "Provide a message, its identifier and up to five documents."},
                          400)
        try:
            job = await manager.enqueue_guest_message(
                owner=owner, chat_id=request.path_params["chat_id"], prompt=prompt,
                request_id=request_id, documents=documents,
            )
        except ValueError as exc:
            return _reply({"error": str(exc)}, 400)
        if job is None:
            return _reply({"error": "Chat not found."}, 404)
        return _reply(job, 202 if job["accepted"] else 429)

    async def inspect(request: Request) -> JSONResponse:
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Task not found."}, 404)
        task_id = request.path_params["task_id"]
        status = await manager.get_task_status(task_id, guest_owner=owner)
        if not status["found"]:
            return _reply({"error": "Task not found."}, 404)
        trace = None
        result = None
        if status["status"] != "queued":
            trace = await manager.get_task_trace(task_id, guest_owner=owner)
        if status["status"] not in {"queued", "running"}:
            result = await manager.get_task_result(task_id, guest_owner=owner)
        return _reply({"status": status, "trace": trace, "result": result})

    async def cancel(request: Request) -> JSONResponse:
        if not same_origin(request):
            return _reply({"error": "Same-origin request required."}, 403)
        owner = _owner(request)
        if owner is None:
            return _reply({"error": "Task not found."}, 404)
        result = await manager.cancel_task(request.path_params["task_id"], guest_owner=owner)
        return _reply(result, 200 if result["found"] else 404)

    return Starlette(routes=[
        Route("/session", session),
        Route("/chats", chats, methods=["GET", "POST"]),
        Route("/chats/{chat_id}", chat),
        Route("/chats/{chat_id}/messages", message, methods=["POST"]),
        Route("/jobs", submit, methods=["POST"]),
        Route("/jobs/{task_id}", inspect),
        Route("/jobs/{task_id}/cancel", cancel, methods=["POST"]),
    ])
