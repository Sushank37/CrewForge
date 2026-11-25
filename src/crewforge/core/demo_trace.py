"""Safe, portable trace export and historical replay foundation for CrewForge."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewforge.config import sanitize_error_message
from crewforge.storage.database import Database


class DemoTraceExport(BaseModel):
    """Complete, self-contained run trace exported for evaluation or dashboard replay."""

    export_version: str = "1.0"
    exported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    is_historical_replay: bool = True

    run_id: str
    mode: str
    status: str
    provider: str
    model: str
    returned_model: str | None = None
    prompt: str = ""
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
    include_prompt: bool = True,
) -> DemoTraceExport:
    """Extract a genuine persisted run from SQLite, sanitize credentials, and export."""
    trace = await database.get_run_trace(run_id)
    if trace is None:
        raise ValueError(f"Run '{run_id}' not found in database '{database.db_path}'.")

    # Sanitize events and messages to guarantee zero raw credentials exist
    sanitized_events: list[dict[str, Any]] = []
    for ev in trace.events:
        d = ev.model_dump()
        d["message"] = sanitize_error_message(d.get("message", ""))
        sanitized_events.append(d)

    sanitized_messages: list[dict[str, Any]] = []
    for msg in trace.messages:
        d = msg.model_dump()
        d["content"] = sanitize_error_message(d.get("content", ""))
        sanitized_messages.append(d)

    usage_metrics = (
        trace.result.usage
        if trace.result and trace.result.usage
        else await database.aggregate_run_usage(run_id)
    )

    export = DemoTraceExport(
        run_id=trace.run_id,
        mode=trace.mode,
        status=trace.status.value,
        provider=trace.provider,
        model=trace.model,
        returned_model=trace.result.returned_model if trace.result else None,
        prompt=trace.prompt if include_prompt else "[REDACTED]",
        error=sanitize_error_message(trace.result.error)
        if trace.result and trace.result.error
        else None,
        created_at=trace.created_at.isoformat(),
        completed_at=trace.completed_at.isoformat() if trace.completed_at else None,
        latency_ms=trace.result.latency_ms if trace.result else 0.0,
        usage=usage_metrics.model_dump(),
        roster=[a.model_dump() for a in trace.agents],
        execution_plan=trace.plan.model_dump() if trace.plan else None,
        tasks=[t.model_dump() for t in trace.tasks],
        task_transitions=trace.task_transitions,
        execution_events=sanitized_events,
        peer_messages=sanitized_messages,
        claims=[c.model_dump() for c in trace.claims],
        review_verdicts=[v.model_dump() for v in trace.review_verdicts],
        tool_executions=[t.model_dump() for t in trace.tool_executions],
        honesty_gate=trace.honesty_gate.model_dump() if trace.honesty_gate else None,
        final_result=trace.result.model_dump() if trace.result else None,
    )

    if output_path:
        out_p = Path(output_path)

        def _write_file() -> None:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(export.model_dump_json(indent=2), encoding="utf-8")

        await asyncio.to_thread(_write_file)

    return export
