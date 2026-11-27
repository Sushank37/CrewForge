"""Server-verified Google, verified-email, cookie/CSRF session, and user MCP token store."""

from __future__ import annotations

import hashlib
import inspect
import re
import secrets
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
_GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_iso(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).isoformat()


def normalize_email(raw: str) -> str:
    """Validate and normalize an email address."""
    cleaned = (raw or "").strip().lower()
    if not cleaned or len(cleaned) > 254 or not _EMAIL_RE.match(cleaned):
        raise ValueError("Invalid email address.")
    return cleaned


def hash_secret(raw: str) -> str:
    """Return SHA-256 hex digest for storing tokens/codes without plaintext persistence."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def verify_google_id_token(
    id_token: str,
    *,
    client_id: str | None = None,
    custom_verifier: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Verify a Google OIDC ID token and return verified identity claims."""
    token = (id_token or "").strip()
    if not token:
        raise ValueError("Google ID token is required.")

    if custom_verifier is not None:
        raw_claims = custom_verifier(token)
        claims = await raw_claims if inspect.isawaitable(raw_claims) else raw_claims
    else:
        if not client_id:
            raise ValueError("Google OAuth client ID is not configured on the server.")
        jwk_client = jwt.PyJWKClient(_GOOGLE_JWKS_URL)
        signing_key = jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=_GOOGLE_ISSUERS,
        )

    if not isinstance(claims, dict):
        raise ValueError("Invalid Google token claims.")
    sub = str(claims.get("sub") or "").strip()
    email_verified = claims.get("email_verified") is True
    raw_email = str(claims.get("email") or "")
    if not sub:
        raise ValueError("Google token missing subject identifier.")
    if not email_verified:
        raise ValueError("Google account email must be verified.")
    email = normalize_email(raw_email)
    display_name = str(claims.get("name") or email.split("@")[0]).strip()[:80]
    return {
        "sub": sub,
        "email": email,
        "email_verified": True,
        "name": display_name,
    }


class AuthStore:
    """SQLite-backed user accounts, email challenges, HttpOnly sessions, and MCP tokens."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE,
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    google_sub TEXT UNIQUE,
                    display_name TEXT,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS email_challenges (
                    challenge_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS browser_sessions (
                    session_token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    csrf_token TEXT NOT NULL,
                    auth_provider TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS mcp_tokens (
                    token_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _user_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "user_id": str(row["user_id"]),
            "email": row["email"],
            "email_verified": bool(row["email_verified"]),
            "google_linked": bool(row["google_sub"]),
            "display_name": row["display_name"],
            "created_at": row["created_at"],
            "last_login_at": row["last_login_at"],
        }

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Fetch public user profile by permanent user_id."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return self._user_dict(row) if row is not None else None
        finally:
            conn.close()

    def create_service_user(self, label: str = "Judge") -> dict[str, Any]:
        """Create a permanent non-default user account for an admin-provisioned judge."""
        now = _utc_iso()
        user_id = f"usr_{secrets.token_hex(8)}"
        display_name = (label or "Judge").strip()[:80] or "Judge"
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO users "
                    "(user_id, email, email_verified, google_sub, display_name, created_at, "
                    "last_login_at) VALUES (?, NULL, 0, NULL, ?, ?, ?)",
                    (user_id, display_name, now, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
                assert row is not None
                return self._user_dict(row)
            finally:
                conn.close()

    def resolve_google_user(
        self,
        *,
        sub: str,
        email: str,
        email_verified: bool,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        """Resolve or create a permanent user account from verified Google OIDC claims."""
        if not sub or not email_verified:
            raise ValueError("Verified Google account subject and email are required.")
        norm_email = normalize_email(email)
        now = _utc_iso()
        clean_name = (display_name or norm_email.split("@")[0]).strip()[:80]
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM users WHERE google_sub = ?", (sub,)
                ).fetchone()
                if row is not None:
                    conn.execute(
                        "UPDATE users SET email = ?, email_verified = 1, last_login_at = ? "
                        "WHERE user_id = ?",
                        (norm_email, now, row["user_id"]),
                    )
                    conn.commit()
                    updated = conn.execute(
                        "SELECT * FROM users WHERE user_id = ?", (row["user_id"],)
                    ).fetchone()
                    assert updated is not None
                    return self._user_dict(updated)

                existing_email = conn.execute(
                    "SELECT * FROM users WHERE email = ? AND email_verified = 1",
                    (norm_email,),
                ).fetchone()
                if existing_email is not None:
                    conn.execute(
                        "UPDATE users SET google_sub = ?, last_login_at = ? WHERE user_id = ?",
                        (sub, now, existing_email["user_id"]),
                    )
                    conn.commit()
                    updated = conn.execute(
                        "SELECT * FROM users WHERE user_id = ?",
                        (existing_email["user_id"],),
                    ).fetchone()
                    assert updated is not None
                    return self._user_dict(updated)

                user_id = f"usr_{secrets.token_hex(8)}"
                conn.execute(
                    "INSERT INTO users "
                    "(user_id, email, email_verified, google_sub, display_name, created_at, "
                    "last_login_at) VALUES (?, ?, 1, ?, ?, ?, ?)",
                    (user_id, norm_email, sub, clean_name, now, now),
                )
                conn.commit()
                created = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
                assert created is not None
                return self._user_dict(created)
            finally:
                conn.close()

    def request_email_verification(
        self,
        email: str,
        *,
        delivery_fn: Callable[[str, str], Any] | None = None,
        ttl_seconds: int = 600,
    ) -> dict[str, Any]:
        """Create a single-use hashed email verification challenge and dispatch via delivery_fn."""
        norm_email = normalize_email(email)
        code = f"{secrets.randbelow(1_000_000):06d}"
        now_dt = _utc_now()
        expires_dt = now_dt + timedelta(seconds=ttl_seconds)
        challenge_id = f"eml_{secrets.token_hex(8)}"
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE email_challenges SET consumed_at = ? "
                    "WHERE email = ? AND consumed_at IS NULL",
                    (_utc_iso(now_dt), norm_email),
                )
                conn.execute(
                    "INSERT INTO email_challenges "
                    "(challenge_id, email, code_hash, expires_at, consumed_at, attempts, "
                    "created_at) VALUES (?, ?, ?, ?, NULL, 0, ?)",
                    (
                        challenge_id,
                        norm_email,
                        hash_secret(code),
                        _utc_iso(expires_dt),
                        _utc_iso(now_dt),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        if delivery_fn is not None:
            delivery_fn(norm_email, code)
        return {
            "challenge_sent": True,
            "email": norm_email,
            "expires_in_seconds": ttl_seconds,
        }

    def verify_email_code(self, email: str, code: str) -> dict[str, Any]:
        """Verify a single-use email code and resolve or create the permanent user account."""
        norm_email = normalize_email(email)
        cleaned_code = (code or "").strip()
        if not cleaned_code:
            raise ValueError("Verification code is required.")
        now_dt = _utc_now()
        now_iso = _utc_iso(now_dt)
        candidate_hash = hash_secret(cleaned_code)

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM email_challenges "
                    "WHERE email = ? AND consumed_at IS NULL "
                    "ORDER BY created_at DESC LIMIT 1",
                    (norm_email,),
                ).fetchone()
                if row is None:
                    raise ValueError("Invalid or expired verification code.")
                if int(row["attempts"]) >= 5:
                    raise ValueError("Too many failed verification attempts.")
                if datetime.fromisoformat(str(row["expires_at"])) <= now_dt:
                    raise ValueError("Verification code has expired.")

                conn.execute(
                    "UPDATE email_challenges SET attempts = attempts + 1 WHERE challenge_id = ?",
                    (row["challenge_id"],),
                )
                if not secrets.compare_digest(candidate_hash, str(row["code_hash"])):
                    conn.commit()
                    raise ValueError("Invalid verification code.")

                conn.execute(
                    "UPDATE email_challenges SET consumed_at = ? WHERE challenge_id = ?",
                    (now_iso, row["challenge_id"]),
                )

                existing = conn.execute(
                    "SELECT * FROM users WHERE email = ?", (norm_email,)
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        "UPDATE users SET email_verified = 1, last_login_at = ? WHERE user_id = ?",
                        (now_iso, existing["user_id"]),
                    )
                    conn.commit()
                    updated = conn.execute(
                        "SELECT * FROM users WHERE user_id = ?", (existing["user_id"],)
                    ).fetchone()
                    assert updated is not None
                    return self._user_dict(updated)

                user_id = f"usr_{secrets.token_hex(8)}"
                display_name = norm_email.split("@")[0][:80]
                conn.execute(
                    "INSERT INTO users "
                    "(user_id, email, email_verified, google_sub, display_name, created_at, "
                    "last_login_at) VALUES (?, ?, 1, NULL, ?, ?, ?)",
                    (user_id, norm_email, display_name, now_iso, now_iso),
                )
                conn.commit()
                created = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
                assert created is not None
                return self._user_dict(created)
            finally:
                conn.close()

    def create_browser_session(
        self,
        *,
        user_id: str,
        auth_provider: str,
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> dict[str, str]:
        """Create an HttpOnly browser session bound to a permanent user_id and CSRF token."""
        if not user_id or user_id == "default":
            raise ValueError("Browser sessions cannot bind to the default development workspace.")
        raw_session = f"cfb_{secrets.token_urlsafe(32)}"
        csrf_token = secrets.token_urlsafe(24)
        now_dt = _utc_now()
        expires_dt = now_dt + timedelta(seconds=ttl_seconds)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO browser_sessions "
                    "(session_token_hash, user_id, csrf_token, auth_provider, created_at, "
                    "expires_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (
                        hash_secret(raw_session),
                        user_id,
                        csrf_token,
                        auth_provider,
                        _utc_iso(now_dt),
                        _utc_iso(expires_dt),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return {
            "session_cookie": raw_session,
            "csrf_token": csrf_token,
            "user_id": user_id,
        }

    def authenticate_browser_session(self, raw_cookie: str) -> dict[str, str] | None:
        """Validate a browser session cookie and return user_id + csrf_token."""
        cleaned = (raw_cookie or "").strip()
        if not cleaned:
            return None
        cookie_hash = hash_secret(cleaned)
        now_dt = _utc_now()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT session_token_hash, user_id, csrf_token, auth_provider, expires_at "
                "FROM browser_sessions WHERE session_token_hash = ? AND revoked_at IS NULL",
                (cookie_hash,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        if datetime.fromisoformat(str(row["expires_at"])) <= now_dt:
            return None
        uid = str(row["user_id"])
        if not uid or uid == "default":
            return None
        return {
            "user_id": uid,
            "csrf_token": str(row["csrf_token"]),
            "auth_provider": str(row["auth_provider"]),
        }

    def revoke_browser_session(self, raw_cookie: str) -> bool:
        """Revoke an active browser session cookie."""
        cleaned = (raw_cookie or "").strip()
        if not cleaned:
            return False
        cookie_hash = hash_secret(cleaned)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE browser_sessions SET revoked_at = ? "
                    "WHERE session_token_hash = ? AND revoked_at IS NULL",
                    (_utc_iso(), cookie_hash),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def issue_mcp_token(
        self,
        *,
        user_id: str,
        label: str = "MCP Client",
        revoke_previous: bool = False,
        raw_token: str | None = None,
    ) -> dict[str, Any]:
        """Issue or rotate an MCP Bearer token bound to an existing permanent user_id."""
        if not user_id or user_id == "default":
            raise ValueError("MCP tokens cannot be bound to the default development workspace.")
        token_val = (raw_token or f"cfj_{secrets.token_urlsafe(24)}").strip()
        if len(token_val) < 16:
            raise ValueError("MCP token must be at least 16 characters.")
        token_id = f"tok_{secrets.token_hex(6)}"
        clean_label = (label or "MCP Client").strip()[:80] or "MCP Client"
        now = _utc_iso()
        with self._lock:
            conn = self._connect()
            try:
                if revoke_previous:
                    conn.execute(
                        "UPDATE mcp_tokens SET revoked_at = ? "
                        "WHERE user_id = ? AND revoked_at IS NULL",
                        (now, user_id),
                    )
                conn.execute(
                    "INSERT INTO mcp_tokens "
                    "(token_id, user_id, label, token_hash, created_at, revoked_at) "
                    "VALUES (?, ?, ?, ?, ?, NULL)",
                    (token_id, user_id, clean_label, hash_secret(token_val), now),
                )
                conn.commit()
            finally:
                conn.close()
        return {
            "token_id": token_id,
            "user_id": user_id,
            "session_id": user_id,
            "label": clean_label,
            "token": token_val,
            "created_at": now,
        }

    def list_mcp_tokens(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """List non-secret metadata for MCP tokens (scoped to user_id when provided)."""
        conn = self._connect()
        try:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT token_id, user_id, label, created_at, revoked_at "
                    "FROM mcp_tokens WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT token_id, user_id, label, created_at, revoked_at "
                    "FROM mcp_tokens ORDER BY created_at DESC"
                ).fetchall()
            return [
                {
                    "token_id": r["token_id"],
                    "user_id": r["user_id"],
                    "session_id": r["user_id"],
                    "label": r["label"],
                    "created_at": r["created_at"],
                    "revoked_at": r["revoked_at"],
                    "active": r["revoked_at"] is None,
                }
                for r in rows
            ]
        finally:
            conn.close()

    def revoke_mcp_token(
        self,
        identifier: str,
        *,
        owner_user_id: str | None = None,
    ) -> bool:
        """Revoke an active MCP token by token_id (or user_id when admin)."""
        ident = (identifier or "").strip()
        if not ident:
            return False
        now = _utc_iso()
        with self._lock:
            conn = self._connect()
            try:
                if owner_user_id is not None:
                    cur = conn.execute(
                        "UPDATE mcp_tokens SET revoked_at = ? "
                        "WHERE token_id = ? AND user_id = ? AND revoked_at IS NULL",
                        (now, ident, owner_user_id),
                    )
                else:
                    cur = conn.execute(
                        "UPDATE mcp_tokens SET revoked_at = ? "
                        "WHERE (token_id = ? OR user_id = ?) AND revoked_at IS NULL",
                        (now, ident, ident),
                    )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def authenticate_mcp_token(self, raw_token: str) -> str | None:
        """Validate an MCP Bearer token and return its permanent user_id."""
        cleaned = (raw_token or "").strip()
        if not cleaned:
            return None
        candidate_hash = hash_secret(cleaned)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT user_id, token_hash FROM mcp_tokens WHERE revoked_at IS NULL"
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            if secrets.compare_digest(candidate_hash, str(row["token_hash"])):
                uid = str(row["user_id"])
                return uid if uid and uid != "default" else None
        return None
