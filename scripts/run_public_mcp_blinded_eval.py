"""Blinded V3 evaluation harness against the public unauthenticated CrewForge MCP endpoint.

Executes held-out manifest pairs (`run_task` vs `run_single_agent`) sequentially in
alternating order with persistent idempotency keys, atomic per-side and per-pair
checkpoints, resumable execution, one-pair limit support, and safe stopping on
budget exhaustion or interrupted execution.

Never accesses or reads any private grading key.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from crewforge.evaluation.anonymize import (
    export_blind_key_file,
    export_evaluator_packet,
    generate_blinding_seed,
    sanitize_confidence_notes,
    strip_orchestrator_metadata,
)
from crewforge.evaluation.contracts import (
    BenchmarkItem,
    BlindAnswer,
    BlindEvaluationItem,
    BlindKey,
)

MCPToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

_TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled", "interrupted"})
_INTERRUPTED_STATUSES = frozenset({"cancelled", "interrupted"})
_BUDGET_STOP_MARKERS = (
    "budget",
    "spending cap",
    "daily cap",
    "quota",
    "insufficient_quota",
    "rate limit",
    "429",
    "exhausted",
)


class EvalHaltedError(RuntimeError):
    """Raised when evaluation halts safely due to budget exhaustion or interruption."""

    def __init__(self, reason: str, *, category: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.category = category  # "budget_exhausted" | "interrupted"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_and_verify_item_docs(
    item: BenchmarkItem,
    manifest_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    """Resolve document paths relative to manifest_dir, verify SHA-256, and format for MCP."""
    mcp_docs: list[dict[str, Any]] = []
    doc_names: list[str] = []
    contents_by_name: dict[str, str] = {}
    for idx, rel_path in enumerate(item.reference_documents, start=1):
        doc_path = (manifest_dir / rel_path).resolve()
        if not doc_path.exists() or not doc_path.is_file():
            raise FileNotFoundError(
                f"Missing reference document '{rel_path}' (resolved: {doc_path}) "
                f"for prompt '{item.prompt_id}'."
            )
        raw_bytes = doc_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        expected_sha = item.document_hashes.get(rel_path)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for '{rel_path}' in '{item.prompt_id}': "
                f"expected {expected_sha}, got {actual_sha}."
            )
        text = raw_bytes.decode("utf-8")
        mcp_docs.append(
            {
                "title": doc_path.name,
                "content": text,
                "source_ref": f"doc_{idx}",
            }
        )
        doc_names.append(doc_path.name)
        contents_by_name[doc_path.name] = text
    return mcp_docs, doc_names, contents_by_name


def _parse_mcp_tool_result(res: Any) -> dict[str, Any]:
    """Extract JSON dict from an MCP CallToolResult."""
    if isinstance(res, dict):
        return res
    structured = getattr(res, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(res, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"error": text.strip()}
    if getattr(res, "isError", False):
        return {"error": "MCP tool returned an unspecified error."}
    return {}


def _is_budget_exhausted_message(msg: str | None) -> bool:
    if not msg:
        return False
    lowered = msg.lower()
    return any(marker in lowered for marker in _BUDGET_STOP_MARKERS)


def _load_or_create_idempotency_keys(
    keys_path: Path,
    *,
    manifest_id: str,
    items: list[BenchmarkItem],
) -> dict[str, str]:
    """Load persistent idempotency keys or create & atomically persist new ones."""
    existing: dict[str, str] = {}
    if keys_path.exists():
        loaded = json.loads(keys_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = {str(k): str(v) for k, v in loaded.items()}

    changed = False
    for item in items:
        for mode in ("network", "baseline"):
            key_name = f"{item.prompt_id}:{mode}"
            if key_name not in existing:
                nonce = secrets.token_hex(6)
                existing[key_name] = f"eval_{manifest_id}_{item.prompt_id}_{mode}_{nonce}"
                changed = True

    if changed or not keys_path.exists():
        _atomic_write_json(keys_path, existing)
    return existing


def _build_blinded_pair_from_results(
    *,
    prompt_id: str,
    domain: str,
    prompt: str,
    network_record: dict[str, Any],
    baseline_record: dict[str, Any],
    doc_names: list[str],
    doc_contents: dict[str, str],
    seed: str,
) -> tuple[BlindEvaluationItem, BlindKey]:
    """Anonymize a completed MCP result pair into Candidate A/B without leaking system identity."""
    h = hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()
    a_is_network = int(h[-1], 16) % 2 == 1

    def _to_blind_answer(rec: dict[str, Any], label: str) -> BlindAnswer:
        res = rec.get("result") or {}
        return BlindAnswer(
            label=label,
            content=strip_orchestrator_metadata(str(res.get("answer") or "")),
            assumptions=[
                s
                for a in (res.get("assumptions") or [])
                if (s := strip_orchestrator_metadata(str(a)))
            ],
            missing_information=[
                s
                for m in (res.get("missing_information") or [])
                if (s := strip_orchestrator_metadata(str(m)))
            ],
            confidence_notes=sanitize_confidence_notes(res.get("confidence_notes")),
        )

    net_ans = _to_blind_answer(network_record, "A" if a_is_network else "B")
    base_ans = _to_blind_answer(baseline_record, "B" if a_is_network else "A")

    item = BlindEvaluationItem(
        prompt_id=prompt_id,
        domain=domain,
        prompt=prompt,
        reference_documents=doc_names,
        reference_document_contents=doc_contents,
        answer_a=net_ans if a_is_network else base_ans,
        answer_b=base_ans if a_is_network else net_ans,
    )
    key = BlindKey(
        prompt_id=prompt_id,
        label_a_system="network" if a_is_network else "baseline",
        label_b_system="baseline" if a_is_network else "network",
        run_id_a=str(
            (network_record if a_is_network else baseline_record).get("task_id") or ""
        ),
        run_id_b=str(
            (baseline_record if a_is_network else network_record).get("task_id") or ""
        ),
        salt=h[:8],
    )
    return item, key


def _extract_side_metrics(record: dict[str, Any]) -> dict[str, Any]:
    res = record.get("result") or {}
    usage = res.get("usage") or {}
    return {
        "task_id": record.get("task_id"),
        "idempotency_key": record.get("idempotency_key"),
        "status": res.get("status") or record.get("status"),
        "model": res.get("model") or record.get("model"),
        "returned_model": res.get("returned_model"),
        "wall_seconds": round(float(record.get("wall_seconds") or 0.0), 3),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "model_calls": int(usage.get("model_calls") or 0),
        "estimated_cost_usd": float(usage.get("estimated_cost_usd") or 0.0),
        "actual_cost_usd": float(usage.get("actual_cost_usd") or 0.0),
        "verification": res.get("verification") or {},
        "unresolved_requirements": res.get("unresolved_requirements") or [],
        "error": res.get("error") or record.get("error"),
    }


async def _execute_mcp_side(
    *,
    caller: MCPToolCaller,
    prompt_id: str,
    mode: str,
    prompt: str,
    mcp_docs: list[dict[str, Any]],
    idempotency_key: str,
    side_ckpt_path: Path,
    poll_interval_s: float,
    max_wait_s: float,
) -> dict[str, Any]:
    """Submit one side (`run_task` or `run_single_agent`), poll to completion, and checkpoint."""
    cached = _read_json_if_exists(side_ckpt_path)
    if isinstance(cached, dict):
        cached_status = str(
            (cached.get("result") or {}).get("status") or cached.get("status") or ""
        )
        if cached_status in {"completed", "partial"}:
            return cached

    tool_name = "run_task" if mode == "network" else "run_single_agent"
    submit_args: dict[str, Any] = {
        "prompt": prompt,
        "idempotency_key": idempotency_key,
    }
    if mcp_docs:
        submit_args["reference_documents"] = mcp_docs

    t0 = time.monotonic()
    submit_res = await caller(tool_name, submit_args)
    submit_err = str(submit_res.get("error") or submit_res.get("message") or "")
    if not submit_res.get("accepted") and not submit_res.get("task_id"):
        if _is_budget_exhausted_message(submit_err):
            raise EvalHaltedError(
                f"Budget exhausted submitting {prompt_id} ({mode}): {submit_err}",
                category="budget_exhausted",
            )
        raise EvalHaltedError(
            f"Submission rejected for {prompt_id} ({mode}): {submit_err or submit_res}",
            category="interrupted",
        )

    task_id = str(submit_res["task_id"])
    last_status: dict[str, Any] = {}
    while True:
        status_res = await caller("get_task_status", {"task_id": task_id})
        last_status = status_res
        cur_status = str(status_res.get("status") or "")
        if cur_status in _TERMINAL_STATUSES:
            break
        if (time.monotonic() - t0) >= max_wait_s:
            raise EvalHaltedError(
                f"Timed out waiting for {prompt_id} ({mode}, task_id={task_id}) "
                f"after {max_wait_s}s (last_status={cur_status}).",
                category="interrupted",
            )
        await asyncio.sleep(poll_interval_s)

    wall_s = time.monotonic() - t0
    result_res = await caller("get_task_result", {"task_id": task_id})
    trace_res = await caller("get_task_trace", {"task_id": task_id})

    final_status = str(result_res.get("status") or last_status.get("status") or "")
    final_error = str(result_res.get("error") or last_status.get("error") or "")

    record = {
        "prompt_id": prompt_id,
        "mode": mode,
        "task_id": task_id,
        "idempotency_key": idempotency_key,
        "status": final_status,
        "model": result_res.get("model") or last_status.get("model"),
        "wall_seconds": round(wall_s, 3),
        "result": result_res,
        "trace": trace_res,
        "error": final_error or None,
    }
    _atomic_write_json(side_ckpt_path, record)

    if final_status in _INTERRUPTED_STATUSES:
        raise EvalHaltedError(
            f"Execution interrupted for {prompt_id} ({mode}, task_id={task_id}): {final_status}.",
            category="interrupted",
        )
    if final_status == "failed":
        if _is_budget_exhausted_message(final_error):
            raise EvalHaltedError(
                f"Budget exhausted for {prompt_id} ({mode}, task_id={task_id}): {final_error}",
                category="budget_exhausted",
            )
        raise EvalHaltedError(
            f"Execution failed for {prompt_id} ({mode}, task_id={task_id}): "
            f"{final_error or 'unknown error'}",
            category="interrupted",
        )
    return record


async def run_public_mcp_blinded_eval(
    *,
    endpoint: str,
    items: list[BenchmarkItem],
    manifest_dir: Path,
    out_dir: Path,
    manifest_id: str = "crewforge_v3",
    max_pairs: int = 0,
    poll_interval_s: float = 3.0,
    max_wait_s: float = 600.0,
    inter_side_delay_s: float = 2.0,
    inter_case_delay_s: float = 3.0,
    caller_override: MCPToolCaller | None = None,
) -> dict[str, Any]:
    """Execute blinded public-MCP evaluation with checkpoints, resume, and safe halts."""
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    verified_docs: dict[str, tuple[list[dict[str, Any]], list[str], dict[str, str]]] = {}
    for item in items:
        verified_docs[item.prompt_id] = load_and_verify_item_docs(item, manifest_dir)

    seed_path = checkpoints_dir / "blinding_seed.txt"
    if seed_path.exists():
        seed = seed_path.read_text(encoding="utf-8").strip()
    else:
        seed = generate_blinding_seed()
        _atomic_write_text(seed_path, seed + "\n")

    idempotency_keys = _load_or_create_idempotency_keys(
        checkpoints_dir / "idempotency_keys.json",
        manifest_id=manifest_id,
        items=items,
    )

    blinded_items: list[BlindEvaluationItem] = []
    key_entries: list[BlindKey] = []
    metrics_rows: list[dict[str, Any]] = []
    traces_rows: list[dict[str, Any]] = []

    halted_reason: str | None = None
    halted_category: str | None = None
    newly_completed_pairs = 0

    async def _run_with_caller(caller: MCPToolCaller) -> None:
        nonlocal halted_reason, halted_category, newly_completed_pairs
        for idx, item in enumerate(items):
            pair_ckpt_path = checkpoints_dir / f"{item.prompt_id}.json"
            if pair_ckpt_path.exists():
                cached = json.loads(pair_ckpt_path.read_text(encoding="utf-8"))
                blinded_items.append(BlindEvaluationItem.model_validate(cached["blind_item"]))
                key_entries.append(BlindKey.model_validate(cached["blind_key"]))
                metrics_rows.append(cached["metrics"])
                traces_rows.append(cached["traces"])
                continue

            if max_pairs > 0 and newly_completed_pairs >= max_pairs:
                break

            mcp_docs, doc_names, doc_contents = verified_docs[item.prompt_id]
            network_first = idx % 2 == 0
            order = ("network", "baseline") if network_first else ("baseline", "network")
            side_records: dict[str, dict[str, Any]] = {}

            try:
                for side_idx, mode in enumerate(order):
                    side_ckpt = checkpoints_dir / f"{item.prompt_id}__{mode}.json"
                    rec = await _execute_mcp_side(
                        caller=caller,
                        prompt_id=item.prompt_id,
                        mode=mode,
                        prompt=item.prompt,
                        mcp_docs=mcp_docs,
                        idempotency_key=idempotency_keys[f"{item.prompt_id}:{mode}"],
                        side_ckpt_path=side_ckpt,
                        poll_interval_s=poll_interval_s,
                        max_wait_s=max_wait_s,
                    )
                    side_records[mode] = rec
                    if side_idx == 0 and inter_side_delay_s > 0:
                        await asyncio.sleep(inter_side_delay_s)
            except EvalHaltedError as exc:
                halted_reason = exc.reason
                halted_category = exc.category
                break

            net_rec = side_records["network"]
            base_rec = side_records["baseline"]
            b_item, b_key = _build_blinded_pair_from_results(
                prompt_id=item.prompt_id,
                domain=item.domain.value,
                prompt=item.prompt,
                network_record=net_rec,
                baseline_record=base_rec,
                doc_names=doc_names,
                doc_contents=doc_contents,
                seed=seed,
            )
            m_row = {
                "prompt_id": item.prompt_id,
                "domain": item.domain.value,
                "execution_order": "network_first" if network_first else "baseline_first",
                "network": _extract_side_metrics(net_rec),
                "baseline": _extract_side_metrics(base_rec),
            }
            t_row = {
                "prompt_id": item.prompt_id,
                "network_task_id": net_rec.get("task_id"),
                "baseline_task_id": base_rec.get("task_id"),
                "network_trace": net_rec.get("trace"),
                "baseline_trace": base_rec.get("trace"),
            }

            _atomic_write_json(
                pair_ckpt_path,
                {
                    "blind_item": b_item.model_dump(mode="json"),
                    "blind_key": b_key.model_dump(mode="json"),
                    "metrics": m_row,
                    "traces": t_row,
                },
            )
            blinded_items.append(b_item)
            key_entries.append(b_key)
            metrics_rows.append(m_row)
            traces_rows.append(t_row)
            newly_completed_pairs += 1

            if (
                idx < len(items) - 1
                and (max_pairs <= 0 or newly_completed_pairs < max_pairs)
                and inter_case_delay_s > 0
            ):
                await asyncio.sleep(inter_case_delay_s)

    if caller_override is not None:
        await _run_with_caller(caller_override)
    else:
        async with (
            streamable_http_client(endpoint) as (read_stream, write_stream, _get_session_id),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            async def _mcp_caller(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
                raw = await session.call_tool(tool_name, args)
                return _parse_mcp_tool_result(raw)

            await _run_with_caller(_mcp_caller)

    packet_path = out_dir / "blinded_evaluator_packet.json"
    key_path = out_dir / "private_blind_key.json"
    metrics_path = out_dir / "private_system_metrics.json"
    traces_path = out_dir / "private_traces.json"

    if blinded_items:
        export_evaluator_packet(blinded_items, packet_path, packet_id=manifest_id)
        export_blind_key_file(
            key_entries,
            blinding_seed=seed,
            output_path=key_path,
            experiment_fingerprint=manifest_id,
        )
    _atomic_write_json(metrics_path, metrics_rows)
    _atomic_write_json(traces_path, traces_rows)

    return {
        "manifest_id": manifest_id,
        "completed_pairs": len(blinded_items),
        "newly_completed_pairs": newly_completed_pairs,
        "halted": halted_reason is not None,
        "halted_category": halted_category,
        "halted_reason": halted_reason,
        "packet_path": str(packet_path) if blinded_items else None,
        "key_path": str(key_path) if blinded_items else None,
        "metrics_path": str(metrics_path),
        "traces_path": str(traces_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run blinded V3 evaluation via the public CrewForge MCP endpoint."
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Public Streamable HTTP MCP URL (e.g. https://<host>/mcp).",
    )
    parser.add_argument(
        "--manifest",
        default=".crewforge/heldout_v3/manifest.json",
        help="Path to held-out manifest JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default=".crewforge/v3_public_mcp_eval",
        help="Output directory for checkpoints, packet, key, metrics, and traces.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of new pairs to execute in this invocation (0 = all remaining).",
    )
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--max-wait", type=float, default=600.0)
    parser.add_argument("--inter-side-delay", type=float, default=2.0)
    parser.add_argument("--inter-case-delay", type=float, default=3.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    manifest_id = (
        raw.get("manifest_id", "crewforge_v3") if isinstance(raw, dict) else "crewforge_v3"
    )
    items = [BenchmarkItem.model_validate(entry) for entry in raw_items]

    summary = asyncio.run(
        run_public_mcp_blinded_eval(
            endpoint=args.endpoint,
            items=items,
            manifest_dir=manifest_path.parent,
            out_dir=Path(args.out_dir).resolve(),
            manifest_id=manifest_id,
            max_pairs=args.max_pairs,
            poll_interval_s=args.poll_interval,
            max_wait_s=args.max_wait,
            inter_side_delay_s=args.inter_side_delay,
            inter_case_delay_s=args.inter_case_delay,
        )
    )
    print(json.dumps(summary, indent=2))
    if summary["halted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
