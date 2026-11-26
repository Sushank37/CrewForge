"""Shared quota-aware scheduler for Groq free-tier rate limits across Chat and MCP."""

from __future__ import annotations

import asyncio
import contextvars
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

_DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_RETRY_MSG_RE = re.compile(
    r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|s|m|seconds?|minutes?)",
    re.IGNORECASE,
)
_LIMIT_MSG_RE = re.compile(r"\bLimit\s+(\d+)\b")

# Context vars propagated from run execution to provider calls without altering core signatures.
run_deadline_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "crewforge_run_deadline_monotonic",
    default=None,
)
remaining_calls_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "crewforge_remaining_calls",
    default=None,
)
wait_observer_var: contextvars.ContextVar[
    Callable[[float, str], Awaitable[None] | None] | None
] = contextvars.ContextVar(
    "crewforge_provider_wait_observer",
    default=None,
)


def parse_duration_seconds(raw: str | None) -> float | None:
    """Parse HTTP Retry-After or Groq duration strings like '22.8225s', '1m2s', '500ms'."""
    if not raw:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    try:
        val = float(text)
        return max(0.0, val)
    except ValueError:
        pass
    parts = _DURATION_PART_RE.findall(text)
    if not parts:
        return None
    total = 0.0
    for num_str, unit in parts:
        val = float(num_str)
        if unit == "ms":
            total += val / 1000.0
        elif unit == "s":
            total += val
        elif unit == "m":
            total += val * 60.0
        elif unit == "h":
            total += val * 3600.0
    return max(0.0, total)


def _header_get(headers: Mapping[str, Any] | None, name: str) -> str | None:
    if not headers:
        return None
    target = name.lower()
    for k, v in headers.items():
        if str(k).lower() == target and v is not None:
            return str(v)
    return None


def extract_retry_after_seconds(
    headers: Mapping[str, Any] | None,
    message: str = "",
) -> float | None:
    """Extract wait duration from response headers or Groq 429 error message."""
    candidates: list[float] = []
    for hdr in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        parsed = parse_duration_seconds(_header_get(headers, hdr))
        if parsed is not None and parsed > 0:
            candidates.append(parsed)
    if message:
        m = _RETRY_MSG_RE.search(message)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit == "ms":
                val /= 1000.0
            elif unit.startswith("m"):
                val *= 60.0
            if val > 0:
                candidates.append(val)
    return max(candidates) if candidates else None


def extract_tpm_limit(headers: Mapping[str, Any] | None, message: str = "") -> int | None:
    """Extract tokens-per-minute limit from headers or Groq 429 error message."""
    hdr_val = _header_get(headers, "x-ratelimit-limit-tokens")
    if hdr_val:
        try:
            limit = int(float(hdr_val))
            if limit > 0:
                return limit
        except ValueError:
            pass
    if message:
        m = _LIMIT_MSG_RE.search(message)
        if m:
            limit = int(m.group(1))
            if limit > 0:
                return limit
    return None


class GroqQuotaScheduler:
    """Process-wide quota-aware scheduler shared across Chat and MCP for a Groq model."""

    def __init__(self, *, key: str, default_tpm_limit: int = 8000) -> None:
        self.key = key
        self._lock = asyncio.Lock()
        self._tpm_limit = default_tpm_limit
        self._next_available_at = 0.0
        self._token_window: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        while self._token_window and self._token_window[0][0] < cutoff:
            self._token_window.popleft()

    def current_window_tokens(self, now: float | None = None) -> int:
        ts = time.monotonic() if now is None else now
        self._prune(ts)
        return sum(tok for _, tok in self._token_window)

    def record_rate_limit(
        self,
        *,
        headers: Mapping[str, Any] | None = None,
        message: str = "",
    ) -> float:
        """Record a 429 rate-limit response and compute the required cooldown."""
        limit = extract_tpm_limit(headers, message)
        if limit is not None:
            self._tpm_limit = limit
        retry_after = extract_retry_after_seconds(headers, message)
        wait_s = (retry_after if retry_after is not None else 10.0) + 0.5
        now = time.monotonic()
        self._next_available_at = max(self._next_available_at, now + wait_s)
        return wait_s

    def record_usage(
        self,
        total_tokens: int,
        *,
        headers: Mapping[str, Any] | None = None,
    ) -> None:
        """Record completed request tokens and header-reported remaining quota."""
        now = time.monotonic()
        self._prune(now)
        if total_tokens > 0:
            self._token_window.append((now, int(total_tokens)))
        limit = extract_tpm_limit(headers)
        if limit is not None:
            self._tpm_limit = limit
        rem_str = _header_get(headers, "x-ratelimit-remaining-tokens")
        reset_str = _header_get(headers, "x-ratelimit-reset-tokens")
        if rem_str is not None and reset_str is not None:
            try:
                remaining = int(float(rem_str))
            except ValueError:
                remaining = -1
            reset_s = parse_duration_seconds(reset_str)
            # If remaining tokens are too low for another structured call, pace proactively.
            if 0 <= remaining < 1200 and reset_s is not None and reset_s > 0:
                self._next_available_at = max(self._next_available_at, now + reset_s + 0.25)

    async def wait_for_slot(
        self,
        estimated_tokens: int = 1200,
        *,
        deadline_monotonic: float | None = None,
        on_wait: Callable[[float, str], Awaitable[None] | None] | None = None,
    ) -> float:
        """Wait in a serialized queue until quota is available or raise TimeoutError on deadline."""
        total_waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                self._prune(now)
                wait_reset = max(0.0, self._next_available_at - now)
                used = sum(tok for _, tok in self._token_window)
                wait_window = 0.0
                safe_budget = int(self._tpm_limit * 0.92)
                if self._token_window and (used + estimated_tokens) > safe_budget:
                    oldest_ts = self._token_window[0][0]
                    wait_window = max(0.5, (oldest_ts + 60.5) - now)
                wait_s = max(wait_reset, wait_window)
                if wait_s <= 0.05:
                    return total_waited

                if deadline_monotonic is not None and (now + wait_s) >= (deadline_monotonic - 2.0):
                    raise TimeoutError(
                        f"Groq quota wait ({wait_s:.1f}s) exceeds remaining execution deadline."
                    )

                capped_wait = min(wait_s, 65.0)
                reason = (
                    f"Waiting {capped_wait:.1f}s for Groq quota reset "
                    f"({used}/{self._tpm_limit} TPM used)."
                )
                if on_wait is not None:
                    res = on_wait(capped_wait, reason)
                    if asyncio.iscoroutine(res):
                        await res
                await asyncio.sleep(capped_wait)
                total_waited += capped_wait


_SCHEDULERS: dict[str, GroqQuotaScheduler] = {}


def get_groq_scheduler(*, base_url: str | None, model: str) -> GroqQuotaScheduler:
    """Return the shared process-wide GroqQuotaScheduler for a base_url + model."""
    normalized_url = (base_url or "https://api.groq.com/openai/v1").rstrip("/").lower()
    key = f"{normalized_url}::{model.strip()}"
    sched = _SCHEDULERS.get(key)
    if sched is None:
        sched = GroqQuotaScheduler(key=key)
        _SCHEDULERS[key] = sched
    return sched


def reset_groq_schedulers() -> None:
    """Clear shared schedulers (for isolated unit tests)."""
    _SCHEDULERS.clear()
