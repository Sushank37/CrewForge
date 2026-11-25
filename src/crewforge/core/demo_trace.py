"""Safe, portable trace export and historical replay foundation for CrewForge."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewforge.config import sanitize_error_message
from crewforge.storage.database import Database

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|authorization|auth[_-]?header|bearer|secret|password|credential|headers)",
    re.IGNORECASE,
)


def _sanitize_recursive(
    value: Any,
    *,
    redact_prompt_text: str | None = None,
) -> Any:
    """Recursively redact secrets, auth headers, and optionally the prompt across nested data."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            key_str = str(k)
            if _SENSITIVE_KEY_RE.search(key_str) or redact_prompt_text and key_str == "prompt":
                sanitized[key_str] = "[REDACTED]"
            else:
                sanitized[key_str] = _sanitize_recursive(v, redact_prompt_text=redact_prompt_text)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_recursive(item, redact_prompt_text=redact_prompt_text) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_recursive(item, redact_prompt_text=redact_prompt_text) for item in value]
    if isinstance(value, str):
        cleaned = sanitize_error_message(value, max_length=20000)
        if redact_prompt_text and redact_prompt_text in cleaned:
            cleaned = cleaned.replace(redact_prompt_text, "[REDACTED_PROMPT]")
        return cleaned
    return value


class DemoTraceExport(BaseModel):
    """Complete, self-contained run trace exported for evaluation or dashboard replay."""

    export_version: str = "1.1"
    exported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    is_historical_replay: bool = True

    run_id: str
    mode: str
    status: str
    provider: str
    model: str
    returned_model: str | None = None
    prompt: str = "[REDACTED]"
    error: str | None = None
    created_at: str
    completed_at: str | None = None
    latency_ms: float = 0.0

    usage: dict[str, Any] = Field(default_factory=dict)
    roster: list[dict[str, Any]] = Field(default_factory=list)
    execution_plan: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    task_transitions: list[dict[str, Any]] = Field(default_factory=list)
    execution_events: list[dict[str, Any]] = Field(default_factory=list)
    peer_messages: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    review_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    tool_executions: list[dict[str, Any]] = Field(default_factory=list)
    honesty_gate: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None


async def export_demo_trace(
    database: Database,
    run_id: str,
    output_path: str | Path | None = None,
    *,
    include_prompt: bool = False,
) -> DemoTraceExport:
    """Extract a persisted run from SQLite, recursively sanitize secrets, and export."""
    trace = await database.get_run_trace(run_id)
    if trace is None:
        raise ValueError(f"Run '{run_id}' not found in database '{database.db_path}'.")

    redact_prompt = trace.prompt if (not include_prompt and trace.prompt) else None

    usage_metrics = (
        trace.result.usage
        if trace.result and trace.result.usage
        else await database.aggregate_run_usage(run_id)
    )

    raw_plan = trace.plan.model_dump() if trace.plan else None
    if redact_prompt and isinstance(raw_plan, dict) and raw_plan.get("goal") == redact_prompt:
        raw_plan["goal"] = "[REDACTED_PROMPT]"

    raw_final = trace.result.model_dump() if trace.result else None

    export = DemoTraceExport(
        run_id=trace.run_id,
        mode=trace.mode,
        status=trace.status.value,
        provider=trace.provider,
        model=trace.model,
        returned_model=trace.result.returned_model if trace.result else None,
        prompt=trace.prompt if include_prompt else "[REDACTED]",
        error=_sanitize_recursive(
            trace.result.error if trace.result and trace.result.error else None,
            redact_prompt_text=redact_prompt,
        ),
        created_at=trace.created_at.isoformat(),
        completed_at=trace.completed_at.isoformat() if trace.completed_at else None,
        latency_ms=trace.result.latency_ms if trace.result else 0.0,
        usage=_sanitize_recursive(usage_metrics.model_dump(), redact_prompt_text=redact_prompt),
        roster=_sanitize_recursive(
            [a.model_dump() for a in trace.agents], redact_prompt_text=redact_prompt
        ),
        execution_plan=_sanitize_recursive(raw_plan, redact_prompt_text=redact_prompt),
        tasks=_sanitize_recursive(
            [t.model_dump() for t in trace.tasks], redact_prompt_text=redact_prompt
        ),
        task_transitions=_sanitize_recursive(
            trace.task_transitions, redact_prompt_text=redact_prompt
        ),
        execution_events=_sanitize_recursive(
            [ev.model_dump() for ev in trace.events], redact_prompt_text=redact_prompt
        ),
        peer_messages=_sanitize_recursive(
            [msg.model_dump() for msg in trace.messages], redact_prompt_text=redact_prompt
        ),
        evidence=_sanitize_recursive(
            [e.model_dump() for e in trace.evidence], redact_prompt_text=redact_prompt
        ),
        claims=_sanitize_recursive(
            [c.model_dump() for c in trace.claims], redact_prompt_text=redact_prompt
        ),
        review_verdicts=_sanitize_recursive(
            [v.model_dump() for v in trace.review_verdicts], redact_prompt_text=redact_prompt
        ),
        tool_executions=_sanitize_recursive(
            [t.model_dump() for t in trace.tool_executions], redact_prompt_text=redact_prompt
        ),
        honesty_gate=_sanitize_recursive(
            trace.honesty_gate.model_dump() if trace.honesty_gate else None,
            redact_prompt_text=redact_prompt,
        ),
        final_result=_sanitize_recursive(raw_final, redact_prompt_text=redact_prompt),
    )

    if output_path:
        out_p = Path(output_path)

        def _write_file() -> None:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(export.model_dump_json(indent=2), encoding="utf-8")

        await asyncio.to_thread(_write_file)

    return export
