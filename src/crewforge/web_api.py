"""Backend chat, judge-session settings, and MCP status layer for CrewForge UI."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

from crewforge.auth import AuthStore
from crewforge.config import (
    DEFAULT_MODEL_PRICING,
    BudgetAllocationCategory,
    CrewForgeSettings,
    ModelPricing,
    ProviderType,
    sanitize_error_message,
)
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.providers.base import ModelProvider
from crewforge.providers.groq_quota import run_deadline_var, wait_observer_var
from crewforge.storage.database import Database

_ENV_ISOLATION_LOCK = threading.Lock()
GROQ_FREE_ENDPOINT = "https://api.groq.com/openai/v1"
GROQ_FREE_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
current_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "crewforge_session_id", default="default"
)


def validate_session_id(raw: str | None) -> str:
    """Validate an explicit judge session identifier, defaulting to 'default'."""
    if raw is None or not raw.strip():
        return "default"
    cleaned = raw.strip()
    if cleaned == "default":
        return "default"
    if not _SESSION_ID_RE.match(cleaned):
        raise ValueError(
            "Invalid judge session ID. Use 4-64 alphanumeric, '-' or '_' characters."
        )
    return cleaned


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


class AttachmentInput(BaseModel):
    """Reference document attached in the chat composer."""

    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=200_000)


class ChatMessageRequest(BaseModel):
    """Request payload for submitting a prompt in a chat session."""

    prompt: str = Field(min_length=1, max_length=32_000)
    mode: Literal["network", "baseline"] = "network"
    attachments: list[AttachmentInput] = Field(default_factory=list, max_length=5)


class JudgeSettingsUpdate(BaseModel):
    """Non-secret + transient API key settings submitted from the Settings page."""

    billing_mode: Literal["free", "byok"] | None = None
    free_tier_verified: bool | None = None
    provider: Literal["ollama", "openai", "openai_compatible"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=120)
    api_key: str | None = None
    clear_api_key: bool = False
    openai_base_url: str | None = Field(default=None, max_length=300)
    ollama_base_url: str | None = Field(default=None, max_length=300)
    openai_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    allow_paid_runs: bool | None = None
    max_output_tokens: int | None = Field(default=None, ge=256, le=32768)
    max_workers: int | None = Field(default=None, ge=1, le=32)
    max_tasks: int | None = Field(default=None, ge=1, le=64)
    max_concurrency: int | None = Field(default=None, ge=1, le=32)
    max_model_calls_per_run: int | None = Field(default=None, ge=1, le=128)
    max_cost_per_run_usd: float | None = Field(default=None, ge=0.0, le=5.0)
    total_budget_ceiling_usd: float | None = Field(default=None, ge=0.0, le=100.0)
    custom_input_price_per_1m: float | None = Field(default=None, ge=0.0, le=1000.0)
    custom_output_price_per_1m: float | None = Field(default=None, ge=0.0, le=1000.0)


class JudgeSessionManager:
    """Manages isolated judge-session chats, transient settings, and real runner tasks."""

    def __init__(
        self,
        *,
        db_path: str | Path = ".crewforge/judge_session.db",
        budget_db_path: str | Path | None = None,
        provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.budget_db_path = (
            Path(budget_db_path)
            if budget_db_path is not None
            else self.db_path.with_name("judge_budget.db")
        )
        self._provider_factory = provider_factory
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._provider_wait_state: dict[str, dict[str, Any]] = {}

        # Distributable product defaults: isolated judge session, no team budget or .env fallback
        self._config: dict[str, Any] = {
            "billing_mode": "free",
            "free_tier_verified": False,
            "provider": "ollama",
            "model": "qwen3:4b",
            "openai_base_url": "",
            "ollama_base_url": "http://127.0.0.1:11434",
            "openai_reasoning_effort": "low",
            "allow_paid_runs": False,
            "max_output_tokens": 8192,
            "max_workers": 3,
            "max_tasks": 3,
            "max_concurrency": 3,
            "max_model_calls_per_run": 16,
            "max_cost_per_run_usd": 0.0,
            "total_budget_ceiling_usd": 0.0,
            "custom_input_price_per_1m": 0.0,
            "custom_output_price_per_1m": 0.0,
        }
        # Kept strictly in backend process memory — never persisted, never reads .env
        self._api_key: str | None = None
        self._init_chat_tables()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_chat_tables(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    chat_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    content TEXT NOT NULL,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    attachments_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES conversations(chat_id)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_public_settings(self) -> dict[str, Any]:
        """Return safe non-secret settings (never returns raw API key)."""
        hint = None
        if self._api_key:
            tail = self._api_key[-4:] if len(self._api_key) >= 4 else "****"
            hint = f"••••{tail}"
        return {
            **self._config,
            "api_key_configured": bool(self._api_key),
            "api_key_hint": hint,
            "session_db_path": str(self.db_path),
            "session_budget_db_path": str(self.budget_db_path),
        }

    def update_settings(self, patch: JudgeSettingsUpdate) -> dict[str, Any]:
        """Apply validated judge-session configuration updates in memory."""
        data = patch.model_dump(exclude_none=True)
        raw_key = data.pop("api_key", None)
        clear_key = data.pop("clear_api_key", False)

        candidate = {**self._config, **data}
        if candidate["billing_mode"] == "free":
            if candidate["provider"] == "openai":
                raise ValueError(
                    "Free mode only supports local Ollama or verified Groq free-tier models."
                )
            if candidate["provider"] == "openai_compatible":
                if str(candidate.get("openai_base_url", "")).rstrip("/") != GROQ_FREE_ENDPOINT:
                    raise ValueError(
                        f"Free mode hosted endpoint must be {GROQ_FREE_ENDPOINT}."
                    )
                if candidate["model"] not in GROQ_FREE_MODELS:
                    raise ValueError(
                        f"Free mode hosted model must be one of {GROQ_FREE_MODELS}."
                    )
            candidate["allow_paid_runs"] = False
            candidate["max_cost_per_run_usd"] = 0.0
            candidate["total_budget_ceiling_usd"] = 0.0
            candidate["custom_input_price_per_1m"] = 0.0
            candidate["custom_output_price_per_1m"] = 0.0

        if clear_key:
            self._api_key = None
        elif isinstance(raw_key, str) and raw_key.strip():
            self._api_key = raw_key.strip()
        self._config = candidate
        return self.get_public_settings()

    def build_runtime_settings(self) -> CrewForgeSettings:
        """Build an isolated CrewForgeSettings instance without .env or host env fallback."""
        prov = ProviderType(self._config["provider"])
        is_free_mode = self._config.get("billing_mode", "free") == "free"
        if is_free_mode and prov == ProviderType.OPENAI_COMPATIBLE and not self._config.get(
            "free_tier_verified"
        ):
            raise ValueError(
                "Free mode requires confirming a verified Groq free-tier account in Settings."
            )

        is_cloud = prov in (ProviderType.OPENAI, ProviderType.OPENAI_COMPATIBLE)
        max_out = int(self._config["max_output_tokens"])
        model_name = str(self._config["model"])

        pricing: dict[str, ModelPricing] = dict(DEFAULT_MODEL_PRICING)
        custom_in = self._config.get("custom_input_price_per_1m")
        custom_out = self._config.get("custom_output_price_per_1m")
        if is_free_mode:
            pricing[model_name] = ModelPricing(input_per_1m=0.0, output_per_1m=0.0)
        elif custom_in is not None and custom_out is not None:
            pricing[model_name] = ModelPricing(
                input_per_1m=float(custom_in),
                output_per_1m=float(custom_out),
            )

        kwargs: dict[str, Any] = {
            "provider": prov,
            "model": model_name,
            "openai_api_key": self._api_key,
            "openai_base_url": self._config["openai_base_url"] or None,
            "ollama_base_url": self._config["ollama_base_url"] or "http://127.0.0.1:11434",
            "openai_reasoning_effort": self._config["openai_reasoning_effort"],
            "cloud_reasoning_profile": is_cloud,
            "small_model_profile": not is_cloud,
            "ollama_think": False if not is_cloud else None,
            "allow_paid_runs": bool(self._config["allow_paid_runs"]),
            "max_output_tokens": max_out,
            "planner_max_output_tokens": min(6144, max_out),
            "worker_max_output_tokens": max_out,
            "review_max_output_tokens": min(6144, max_out),
            "revision_max_output_tokens": max_out,
            "synthesis_max_output_tokens": min(6144, max_out),
            "max_workers": int(self._config["max_workers"]),
            "max_tasks": int(self._config["max_tasks"]),
            "max_concurrency": 1 if not is_cloud else int(self._config["max_concurrency"]),
            "max_communication_rounds": 1,
            "max_revision_rounds": 1,
            "max_retries": 0,
            "max_model_calls_per_run": int(self._config["max_model_calls_per_run"]),
            "max_cost_per_run_usd": float(self._config["max_cost_per_run_usd"]),
            "max_cumulative_cost_usd": float(self._config["total_budget_ceiling_usd"]),
            "total_budget_ceiling_usd": float(self._config["total_budget_ceiling_usd"]),
            "protected_reserve_usd": 0.0,
            "budget_allocation_category": BudgetAllocationCategory.GENERAL,
            "db_path": self.db_path,
            "cloud_db_path": self.db_path,
            "budget_authority_path": self.budget_db_path,
            "pricing_json": pricing,
            "request_timeout_seconds": 180.0,
            "run_deadline_seconds": 480.0,
            "enable_http_retrieval": False,
        }
        # Prevent silent fallback to host OPENAI_API_KEY / CREWFORGE_* or team .env
        with _ENV_ISOLATION_LOCK:
            saved_env: dict[str, str] = {}
            for key in list(os.environ.keys()):
                if key == "OPENAI_API_KEY" or key.startswith("CREWFORGE_"):
                    saved_env[key] = os.environ.pop(key)
            try:
                settings = CrewForgeSettings(_env_file=None, **kwargs)
            finally:
                os.environ.update(saved_env)

        if self._api_key is None:
            settings.openai_api_key = None
        return settings

    # ---------------- Conversation Persistence ----------------
    def list_chats(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT chat_id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def create_chat(self, title: str = "New Chat") -> dict[str, Any]:
        now = _utc_iso()
        chat = {
            "chat_id": f"chat_{uuid4().hex[:10]}",
            "title": (title or "New Chat").strip()[:80],
            "created_at": now,
            "updated_at": now,
        }
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO conversations (chat_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (chat["chat_id"], chat["title"], chat["created_at"], chat["updated_at"]),
            )
            conn.commit()
        finally:
            conn.close()
        return chat

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT chat_id, title, created_at, updated_at "
                "FROM conversations WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None:
                return None
            msgs = conn.execute(
                "SELECT message_id, chat_id, role, mode, content, run_id, status, "
                "attachments_json, error, created_at, updated_at "
                "FROM chat_messages WHERE chat_id = ? ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
            out_msgs = []
            for m in msgs:
                d = dict(m)
                d["attachments"] = json.loads(d.pop("attachments_json", "[]"))
                out_msgs.append(d)
            return {**dict(row), "messages": out_msgs}
        finally:
            conn.close()

    def delete_chat(self, chat_id: str) -> bool:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
            cur = conn.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def _update_message(
        self,
        message_id: str,
        *,
        status: str,
        content: str | None = None,
        error: str | None = None,
    ) -> None:
        now = _utc_iso()
        conn = self._connect()
        try:
            if content is not None:
                conn.execute(
                    "UPDATE chat_messages SET status = ?, content = ?, error = ?, updated_at = ? "
                    "WHERE message_id = ?",
                    (status, content, error, now, message_id),
                )
            else:
                conn.execute(
                    "UPDATE chat_messages SET status = ?, error = ?, updated_at = ? "
                    "WHERE message_id = ?",
                    (status, error, now, message_id),
                )
            conn.commit()
        finally:
            conn.close()

    async def submit_message(
        self,
        chat_id: str,
        req: ChatMessageRequest,
    ) -> dict[str, Any]:
        """Persist user prompt, start real CrewForge execution, and return run_id."""
        chat = self.get_chat(chat_id)
        if chat is None:
            raise KeyError(f"Chat '{chat_id}' not found.")
        # Validate runtime settings before creating a running message so invalid config fails fast.
        self.build_runtime_settings()

        now = _utc_iso()
        user_msg_id = f"cmsg_{uuid4().hex[:10]}"
        asst_msg_id = f"cmsg_{uuid4().hex[:10]}"
        run_id = f"run_{uuid4().hex[:12]}"
        att_meta = [{"name": a.name, "size": len(a.content)} for a in req.attachments]

        conn = self._connect()
        try:
            if chat["title"] == "New Chat":
                new_title = req.prompt.strip().splitlines()[0][:54] or "New Chat"
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE chat_id = ?",
                    (new_title, now, chat_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE chat_id = ?",
                    (now, chat_id),
                )
            conn.execute(
                "INSERT INTO chat_messages (message_id, chat_id, role, mode, content, run_id, "
                "status, attachments_json, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_msg_id,
                    chat_id,
                    "user",
                    req.mode,
                    req.prompt.strip(),
                    run_id,
                    "completed",
                    json.dumps(att_meta),
                    None,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO chat_messages (message_id, chat_id, role, mode, content, run_id, "
                "status, attachments_json, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asst_msg_id,
                    chat_id,
                    "assistant",
                    req.mode,
                    "",
                    run_id,
                    "running",
                    "[]",
                    None,
                    now,
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        ref_docs = [
            ReferenceDocument(
                doc_id=f"doc_{idx + 1}",
                source_ref=a.name,
                content=a.content,
            )
            for idx, a in enumerate(req.attachments)
        ]

        task = asyncio.create_task(
            self._execute_run(
                run_id=run_id,
                assistant_msg_id=asst_msg_id,
                prompt=req.prompt.strip(),
                mode=req.mode,
                ref_docs=ref_docs,
            )
        )
        self._tasks[run_id] = task
        return {
            "chat_id": chat_id,
            "user_message_id": user_msg_id,
            "assistant_message_id": asst_msg_id,
            "run_id": run_id,
            "status": "running",
        }

    def get_provider_wait_state(self, run_id: str) -> dict[str, Any] | None:
        """Return active provider quota-wait metadata for a running execution, if any."""
        state = self._provider_wait_state.get(run_id)
        if state is None:
            return None
        if time.monotonic() >= float(state.get("expires_at", 0.0)):
            self._provider_wait_state.pop(run_id, None)
            return None
        return state

    async def _execute_run(
        self,
        *,
        run_id: str,
        assistant_msg_id: str,
        prompt: str,
        mode: Literal["network", "baseline"],
        ref_docs: list[ReferenceDocument],
    ) -> None:
        db = Database(self.db_path)

        def _on_provider_wait(wait_s: float, reason: str) -> None:
            self._provider_wait_state[run_id] = {
                "waiting_for_provider": True,
                "waiting_reason": reason,
                "provider_wait_seconds": round(wait_s, 1),
                "expires_at": time.monotonic() + wait_s,
            }

        deadline_token = None
        wait_token = wait_observer_var.set(_on_provider_wait)
        try:
            settings = self.build_runtime_settings()
            deadline_token = run_deadline_var.set(time.monotonic() + settings.run_deadline_seconds)
            await db.initialize()
            authority = UnifiedBudgetAuthority(
                settings=settings,
                migrate_default_dbs=False,
            )
            budget = BudgetManager(database=db, settings=settings, authority=authority)
            provider = (
                self._provider_factory(settings)
                if self._provider_factory is not None
                else build_provider_from_settings(settings)
            )
            if mode == "baseline":
                runner = BaselineRunner(
                    provider=provider,
                    database=db,
                    budget_manager=budget,
                    settings=settings,
                )
            else:
                runner = NetworkRunner(
                    provider=provider,
                    database=db,
                    budget_manager=budget,
                    settings=settings,
                )
            res = await runner.run(
                prompt,
                run_id=run_id,
                reference_documents=ref_docs or None,
            )
            self._update_message(
                assistant_msg_id,
                status=res.status.value,
                content=res.answer or "",
                error=sanitize_error_message(res.error) if res.error else None,
            )
        except asyncio.CancelledError:
            self._update_message(
                assistant_msg_id,
                status="cancelled",
                content="Execution cancelled by user.",
                error="Cancelled by user.",
            )
            raise
        except Exception as exc:
            self._update_message(
                assistant_msg_id,
                status="failed",
                content="",
                error=sanitize_error_message(str(exc)),
            )
        finally:
            if deadline_token is not None:
                run_deadline_var.reset(deadline_token)
            wait_observer_var.reset(wait_token)
            self._provider_wait_state.pop(run_id, None)
            self._tasks.pop(run_id, None)
            with contextlib.suppress(Exception):
                await db.close()

    def cancel_run(self, run_id: str) -> bool:
        """Cancel an in-flight run task if active."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def get_mcp_status(
        self,
        *,
        mcp_url: str = "http://127.0.0.1:8765/mcp",
        auth_token: str | None = None,
        session_id: str = "default",
        public_url: str | None = None,
        internal_url: str | None = None,
    ) -> dict[str, Any]:
        """Perform an actual MCP Streamable HTTP handshake and tool discovery."""
        http_reachable = False
        discovered_tools: list[str] = []
        handshake_error: str | None = None

        probe_url = internal_url or mcp_url
        exposed_url = public_url or mcp_url
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if session_id and session_id != "default":
            headers["X-CrewForge-Session"] = session_id

        try:
            async with (
                asyncio.timeout(2.0),
                httpx.AsyncClient(headers=headers) as http_client,
                streamable_http_client(probe_url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                tools_resp = await session.list_tools()
                discovered_tools = [t.name for t in tools_resp.tools]
                http_reachable = True
        except Exception as exc:
            handshake_error = sanitize_error_message(str(exc), max_length=240)

        shared_headers: dict[str, str] = {
            "Authorization": "Bearer <YOUR_MCP_TOKEN>",
        }

        return {
            "tools": discovered_tools or ["solve", "baseline", "trace"],
            "discovered_tools": discovered_tools,
            "http_endpoint": exposed_url,
            "http_reachable": http_reachable,
            "handshake_error": handshake_error,
            "session_id": session_id or "default",
            "remote_https": exposed_url.startswith("https://"),
            "session_db_path": str(self.db_path),
            "session_budget_db_path": str(self.budget_db_path),
            "shared_http_config": {
                "mcpServers": {
                    "crewforge-shared": {
                        "url": exposed_url,
                        "headers": shared_headers,
                    }
                }
            },
            "stdio_config": {
                "mcpServers": {
                    "crewforge-standalone": {
                        "command": "uv",
                        "args": ["run", "--directory", "/path/to/CrewForge", "crewforge-mcp"],
                        "env": {
                            "CREWFORGE_DB_PATH": str(self.db_path),
                            "CREWFORGE_BUDGET_AUTHORITY_PATH": str(self.budget_db_path),
                        },
                    }
                }
            },
            "http_command": (
                f"CREWFORGE_DB_PATH={self.db_path} "
                f"CREWFORGE_BUDGET_AUTHORITY_PATH={self.budget_db_path} "
                "uv run crewforge-mcp --transport streamable-http --host 127.0.0.1 --port 8000"
            ),
        }


class JudgeSessionRegistry:
    """Isolate judge sessions, credentials, chats, and SQLite trace/budget DBs per session ID."""

    def __init__(
        self,
        *,
        base_db_path: str | Path = ".crewforge/judge_session.db",
        base_budget_db_path: str | Path | None = None,
        provider_factory: Callable[[CrewForgeSettings], ModelProvider] | None = None,
    ) -> None:
        self.base_db_path = Path(base_db_path)
        self.base_budget_db_path = (
            Path(base_budget_db_path)
            if base_budget_db_path is not None
            else self.base_db_path.parent / "judge_budget.db"
        )
        self.auth_store = AuthStore(self.base_db_path.parent / "auth.db")
        self.provider_factory = provider_factory
        self._lock = threading.Lock()
        self._sessions: dict[str, JudgeSessionManager] = {}
        self._mcp_dbs: dict[str, Database] = {}
        self._initialized_mcp_dbs: set[str] = set()

    def issue_judge_token(
        self,
        label: str = "Judge",
        *,
        user_id: str | None = None,
        revoke_previous: bool = False,
    ) -> dict[str, Any]:
        """Issue or rotate a revocable MCP token bound to a permanent user account."""
        target_uid = user_id
        if target_uid is None:
            target_uid = self.auth_store.create_service_user(label=label)["user_id"]
        return self.auth_store.issue_mcp_token(
            user_id=target_uid,
            label=label,
            revoke_previous=revoke_previous,
        )

    def register_judge_token(
        self,
        raw_token: str,
        *,
        label: str = "Judge",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a pre-issued MCP token bound to a permanent user account."""
        target_uid = validate_session_id(session_id) if session_id else None
        if target_uid is None or target_uid == "default":
            if session_id == "default":
                raise ValueError("MCP tokens cannot bind to the default development session.")
            target_uid = self.auth_store.create_service_user(label=label)["user_id"]
        return self.auth_store.issue_mcp_token(
            user_id=target_uid,
            label=label,
            raw_token=raw_token,
        )

    def list_judge_tokens(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """List non-secret metadata for issued MCP tokens."""
        return self.auth_store.list_mcp_tokens(user_id=user_id)

    def revoke_judge_token(self, identifier: str, *, owner_user_id: str | None = None) -> bool:
        """Revoke an active MCP token by token_id (or user_id when admin)."""
        return self.auth_store.revoke_mcp_token(identifier, owner_user_id=owner_user_id)

    def authenticate_judge_token(self, raw_token: str) -> str | None:
        """Validate an MCP Bearer token and derive its permanent user_id."""
        return self.auth_store.authenticate_mcp_token(raw_token)

    def _paths_for(self, session_id: str) -> tuple[Path, Path]:
        if session_id == "default":
            return self.base_db_path, self.base_budget_db_path
        folder = "users" if session_id.startswith("usr_") else "sessions"
        session_dir = self.base_db_path.parent / folder / session_id
        return session_dir / "session.db", session_dir / "budget.db"

    def get_session(self, session_id: str | None = None) -> JudgeSessionManager:
        """Return the isolated JudgeSessionManager for the validated session ID."""
        sid = validate_session_id(session_id)
        with self._lock:
            mgr = self._sessions.get(sid)
            if mgr is None:
                db_path, budget_db_path = self._paths_for(sid)
                mgr = JudgeSessionManager(
                    db_path=db_path,
                    budget_db_path=budget_db_path,
                    provider_factory=self.provider_factory,
                )
                self._sessions[sid] = mgr
            return mgr

    def get_current_session(self) -> JudgeSessionManager:
        """Return the JudgeSessionManager bound to the active request/MCP context."""
        return self.get_session(current_session_id_var.get())

    async def get_mcp_database(self, session_id: str | None = None) -> Database:
        """Return an initialized isolated Database for MCP calls in this session."""
        sid = validate_session_id(session_id or current_session_id_var.get())
        with self._lock:
            db = self._mcp_dbs.get(sid)
            if db is None:
                db_path, _ = self._paths_for(sid)
                db = Database(db_path)
                self._mcp_dbs[sid] = db
            needs_init = sid not in self._initialized_mcp_dbs
        if needs_init:
            await db.initialize()
            with self._lock:
                self._initialized_mcp_dbs.add(sid)
        return db

    async def close_all(self) -> None:
        """Close all cached MCP Database connections cleanly."""
        with self._lock:
            dbs = list(self._mcp_dbs.values())
            self._mcp_dbs.clear()
            self._initialized_mcp_dbs.clear()
        for db in dbs:
            with contextlib.suppress(Exception):
                await db.close()
