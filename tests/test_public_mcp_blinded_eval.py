"""Offline tests for scripts/run_public_mcp_blinded_eval.py using mocked MCP calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_public_mcp_blinded_eval import run_public_mcp_blinded_eval

from crewforge.evaluation.contracts import BenchmarkItem


def _make_manifest(tmp_path: Path) -> tuple[Path, list[BenchmarkItem]]:
    manifest_dir = tmp_path / "manifest"
    docs_dir = manifest_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_text = "Policy: Maximum fee is $10,000 USD."
    doc_path = docs_dir / "policy.md"
    doc_path.write_text(doc_text, encoding="utf-8")
    doc_sha = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()

    items = [
        BenchmarkItem(
            prompt_id="case_1",
            title="Case 1",
            domain="financial",
            difficulty="hard",
            prompt="Calculate total cost.",
            reference_documents=["docs/policy.md"],
            document_hashes={"docs/policy.md": doc_sha},
        ),
        BenchmarkItem(
            prompt_id="case_2",
            title="Case 2",
            domain="causation",
            difficulty="hard",
            prompt="Assess causal confounding.",
            reference_documents=[],
            document_hashes={},
        ),
    ]
    return manifest_dir, items


async def test_public_mcp_blinded_eval_one_pair_limit_alternation_and_resume(
    tmp_path: Path,
) -> None:
    manifest_dir, items = _make_manifest(tmp_path)
    out_dir = tmp_path / "eval_out"
    submitted: list[tuple[str, str, str]] = []
    tasks: dict[str, dict[str, Any]] = {}
    counter = 0

    async def _mock_caller(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        if tool_name in {"run_task", "run_single_agent"}:
            counter += 1
            tid = f"tsk_mock_{counter}"
            mode = "network" if tool_name == "run_task" else "baseline"
            submitted.append((tool_name, args["prompt"], args["idempotency_key"]))
            tasks[tid] = {
                "mode": mode,
                "prompt": args["prompt"],
                "docs": args.get("reference_documents") or [],
            }
            return {"accepted": True, "task_id": tid, "status": "queued", "mode": mode}

        tid = args["task_id"]
        meta = tasks[tid]
        if tool_name == "get_task_status":
            return {
                "found": True,
                "task_id": tid,
                "mode": meta["mode"],
                "status": "completed",
                "model": "gpt-6-luna",
                "current_stage": "completed",
                "progress": 1.0,
            }
        if tool_name == "get_task_result":
            return {
                "found": True,
                "ready": True,
                "task_id": tid,
                "mode": meta["mode"],
                "status": "completed",
                "model": "gpt-6-luna",
                "returned_model": "gpt-6-luna",
                "answer": f"Answer from {meta['mode']} [clm_abc123@v1] via CrewForge.",
                "sources": [],
                "evidence": [],
                "verification": {"honesty_gate_passed": meta["mode"] == "network"},
                "unresolved_requirements": [],
                "assumptions": [],
                "missing_information": [],
                "confidence_notes": "Honesty Gate: full (passed=True).",
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 60,
                    "total_tokens": 180,
                    "model_calls": 3 if meta["mode"] == "network" else 1,
                    "estimated_cost_usd": 0.002,
                    "actual_cost_usd": 0.0015,
                },
            }
        if tool_name == "get_task_trace":
            return {
                "found": True,
                "trace_available": True,
                "task_id": tid,
                "mode": meta["mode"],
                "agents": [{"role": "Analyst"}] if meta["mode"] == "network" else [],
            }
        raise AssertionError(f"Unexpected tool {tool_name}")

    # 1. Run with one-pair limit
    s1 = await run_public_mcp_blinded_eval(
        endpoint="https://example.invalid/mcp",
        items=items,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        manifest_id="heldout_v3_test",
        max_pairs=1,
        poll_interval_s=0.0,
        inter_side_delay_s=0.0,
        inter_case_delay_s=0.0,
        caller_override=_mock_caller,
    )
    assert s1["completed_pairs"] == 1
    assert s1["newly_completed_pairs"] == 1
    assert s1["halted"] is False
    # Case 1 (idx=0) runs network_first: run_task then run_single_agent
    assert [s[0] for s in submitted] == ["run_task", "run_single_agent"]

    # 2. Resume remaining pairs
    submitted.clear()
    s2 = await run_public_mcp_blinded_eval(
        endpoint="https://example.invalid/mcp",
        items=items,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        manifest_id="heldout_v3_test",
        max_pairs=0,
        poll_interval_s=0.0,
        inter_side_delay_s=0.0,
        inter_case_delay_s=0.0,
        caller_override=_mock_caller,
    )
    assert s2["completed_pairs"] == 2
    assert s2["newly_completed_pairs"] == 1
    assert s2["halted"] is False
    # Case 2 (idx=1) alternates to baseline_first: run_single_agent then run_task
    assert [s[0] for s in submitted] == ["run_single_agent", "run_task"]

    packet = json.loads((out_dir / "blinded_evaluator_packet.json").read_text(encoding="utf-8"))
    key_file = json.loads((out_dir / "private_blind_key.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "private_system_metrics.json").read_text(encoding="utf-8"))
    traces = json.loads((out_dir / "private_traces.json").read_text(encoding="utf-8"))

    assert len(packet["items"]) == 2
    assert packet["items"][0]["reference_document_contents"]["policy.md"] == (
        "Policy: Maximum fee is $10,000 USD."
    )
    packet_text = json.dumps(packet)
    assert "clm_abc123" not in packet_text
    assert "CrewForge" not in packet_text
    assert "label_a_system" not in packet_text
    assert "actual_cost_usd" not in packet_text

    assert len(key_file["keys"]) == 2
    assert metrics[0]["execution_order"] == "network_first"
    assert metrics[1]["execution_order"] == "baseline_first"
    assert metrics[0]["network"]["total_tokens"] == 180
    assert metrics[0]["network"]["actual_cost_usd"] == 0.0015
    assert len(traces) == 2
    assert traces[0]["network_trace"]["trace_available"] is True


async def test_public_mcp_blinded_eval_resume_reuses_idempotency_key_after_interruption(
    tmp_path: Path,
) -> None:
    manifest_dir, items = _make_manifest(tmp_path)
    out_dir = tmp_path / "eval_interrupt"
    seen_keys: list[tuple[str, str]] = []
    fail_second_case_network = True
    counter = 0
    tasks: dict[str, str] = {}

    async def _mock_caller(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        if tool_name in {"run_task", "run_single_agent"}:
            counter += 1
            tid = f"tsk_int_{counter}"
            mode = "network" if tool_name == "run_task" else "baseline"
            seen_keys.append((f"{args['prompt']}:{mode}", args["idempotency_key"]))
            is_cancelled = (
                fail_second_case_network
                and "confounding" in args["prompt"]
                and mode == "network"
            )
            tasks[tid] = "cancelled" if is_cancelled else "completed"
            return {"accepted": True, "task_id": tid, "status": "queued", "mode": mode}

        tid = args["task_id"]
        st = tasks[tid]
        if tool_name == "get_task_status":
            return {"found": True, "task_id": tid, "status": st, "model": "gpt-6-luna"}
        if tool_name == "get_task_result":
            return {
                "found": True,
                "ready": True,
                "task_id": tid,
                "status": st,
                "model": "gpt-6-luna",
                "answer": "Ok." if st == "completed" else "",
                "usage": {"total_tokens": 10, "actual_cost_usd": 0.0001},
            }
        if tool_name == "get_task_trace":
            return {"found": True, "trace_available": True, "task_id": tid}
        raise AssertionError(f"Unexpected tool {tool_name}")

    s1 = await run_public_mcp_blinded_eval(
        endpoint="https://example.invalid/mcp",
        items=items,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        poll_interval_s=0.0,
        inter_side_delay_s=0.0,
        inter_case_delay_s=0.0,
        caller_override=_mock_caller,
    )
    assert s1["halted"] is True
    assert s1["halted_category"] == "interrupted"
    assert s1["completed_pairs"] == 1
    first_net_key = dict(seen_keys)["Assess causal confounding.:network"]

    # Resume with failure resolved: case_1 and case_2:baseline must NOT rerun,
    # and case_2:network must reuse the exact same idempotency_key.
    seen_keys.clear()
    fail_second_case_network = False
    s2 = await run_public_mcp_blinded_eval(
        endpoint="https://example.invalid/mcp",
        items=items,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        poll_interval_s=0.0,
        inter_side_delay_s=0.0,
        inter_case_delay_s=0.0,
        caller_override=_mock_caller,
    )
    assert s2["halted"] is False
    assert s2["completed_pairs"] == 2
    assert len(seen_keys) == 1
    assert seen_keys[0] == ("Assess causal confounding.:network", first_net_key)


async def test_public_mcp_blinded_eval_stops_safely_on_budget_exhaustion(
    tmp_path: Path,
) -> None:
    manifest_dir, items = _make_manifest(tmp_path)
    out_dir = tmp_path / "eval_budget"
    calls = 0

    async def _mock_caller(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "accepted": False,
            "error": "Daily spending cap exhausted; try again later.",
        }

    summary = await run_public_mcp_blinded_eval(
        endpoint="https://example.invalid/mcp",
        items=items,
        manifest_dir=manifest_dir,
        out_dir=out_dir,
        poll_interval_s=0.0,
        inter_side_delay_s=0.0,
        inter_case_delay_s=0.0,
        caller_override=_mock_caller,
    )
    assert summary["halted"] is True
    assert summary["halted_category"] == "budget_exhausted"
    assert summary["completed_pairs"] == 0
    assert calls == 1
