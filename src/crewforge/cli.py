"""Command-line interface for CrewForge operations and smoke testing."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from crewforge.config import CrewForgeSettings, ProviderType, configure_logging
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import LocalDocumentRetriever, ReferenceDocument
from crewforge.mcp_server import run_mcp_server
from crewforge.schemas import RunResult, RunStatus
from crewforge.storage.database import Database


def _print_run_result(result: RunResult, *, json_output: bool) -> None:
    if json_output:
        print(result.model_dump_json(indent=2))
        return

    print(f"Run ID:   {result.run_id}")
    print(f"Mode:     {result.mode}")
    print(f"Status:   {result.status.value}")
    print(f"Model:    {result.provider}/{result.model}")
    print(
        f"Calls:    dispatched={result.usage.model_calls} "
        f"succeeded={result.usage.successful_operations} retries={result.usage.retries}"
    )
    print(
        f"Tokens:   in={result.usage.input_tokens} "
        f"out={result.usage.output_tokens} total={result.usage.total_tokens}"
    )
    print(f"Latency:  {result.latency_ms:.1f} ms")
    if result.specialist_roles:
        print(f"Team:     {', '.join(result.specialist_roles)}")
    if result.verification is not None:
        v = result.verification
        print(
            f"Verified: {v.supported_claim_count}/{v.material_claim_count} material claims | "
            f"unresolved={v.unresolved_claim_count} rejected={v.rejected_claim_count} "
            f"stale={v.stale_claim_count} revisions={v.revision_count} | "
            f"gate_passed={v.honesty_gate_passed}"
        )
    if result.error:
        print(f"Error:    {result.error}", file=sys.stderr)
    if result.answer:
        print(f"\nAnswer:\n{result.answer}")
    if result.missing_information:
        print("\nMissing Information:")
        for item in result.missing_information:
            print(f"- {item}")
    if result.assumptions:
        print("\nAssumptions:")
        for item in result.assumptions:
            print(f"- {item}")
    if result.confidence_notes:
        print(f"\nConfidence Notes:\n{result.confidence_notes}")


async def _run_mode_cmd(
    prompt: str,
    *,
    mode: str,
    settings: CrewForgeSettings,
    json_output: bool,
    doc_paths: list[str] | None = None,
    compute_matched: bool = False,
) -> int:
    db = Database(settings.effective_db_path)
    await db.initialize()
    provider = build_provider_from_settings(settings)
    ref_docs: list[ReferenceDocument] | None = None
    if doc_paths:
        ref_docs = LocalDocumentRetriever.from_paths(doc_paths).documents
    try:
        budget = BudgetManager(db, settings)
        if mode == "network":
            runner = NetworkRunner(
                provider=provider,
                database=db,
                budget_manager=budget,
                settings=settings,
            )
            result = await runner.run(prompt, reference_documents=ref_docs)
        else:
            b_runner = BaselineRunner(
                provider=provider,
                database=db,
                budget_manager=budget,
                settings=settings,
            )
            result = await b_runner.run(
                prompt,
                reference_documents=ref_docs,
                compute_matched=compute_matched or None,
            )
    finally:
        await provider.aclose()
        await db.close()

    _print_run_result(result, json_output=json_output)
    return 0 if result.status in {RunStatus.COMPLETED, RunStatus.PARTIAL} else 1


async def _run_trace_cmd(run_id: str, *, settings: CrewForgeSettings) -> int:
    db = Database(settings.effective_db_path)
    await db.initialize()
    try:
        trace = await db.get_run_trace(run_id)
    finally:
        await db.close()

    if trace is None:
        print(json.dumps({"found": False, "run_id": run_id, "error": "Run not found."}, indent=2))
        return 1
    print(trace.model_dump_json(indent=2))
    return 0


async def _run_budget_cmd(*, settings: CrewForgeSettings) -> int:
    db = Database(settings.effective_db_path)
    await db.initialize()
    try:
        summary: dict[str, Any] = await db.get_budget_summary()
    finally:
        await db.close()
    summary["db_path"] = str(settings.effective_db_path)
    summary["allow_paid_runs"] = settings.allow_paid_runs
    summary["max_cost_per_run_usd"] = settings.max_cost_per_run_usd
    summary["max_cumulative_cost_usd"] = settings.max_cumulative_cost_usd
    summary["max_model_calls_per_run"] = settings.max_model_calls_per_run
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="crewforge",
        description="CrewForge CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a prompt with CrewForge's verified multi-agent network.",
    )
    solve_parser.add_argument("prompt", help="Prompt to solve.")
    solve_parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="Operator-supplied UTF-8 reference document path (repeatable).",
    )
    solve_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Execute a single-agent baseline run.",
    )
    baseline_parser.add_argument("prompt", help="Prompt to evaluate.")
    baseline_parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="Operator-supplied UTF-8 reference document path (repeatable).",
    )
    baseline_parser.add_argument(
        "--compute-matched",
        action="store_true",
        help="Enable iterative single-agent self-check refinement pass.",
    )
    baseline_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    trace_parser = subparsers.add_parser(
        "trace",
        help="Print persisted execution trace for a run ID.",
    )
    trace_parser.add_argument("run_id", help="Run identifier.")

    subparsers.add_parser("budget", help="Show persistent cumulative budget usage.")

    smoke_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a single bounded real-provider smoke test (incurs API usage if using OpenAI).",
    )
    smoke_parser.add_argument(
        "--provider",
        choices=[p.value for p in ProviderType],
        default=None,
        help="Override configured provider.",
    )
    smoke_parser.add_argument("--model", default=None, help="Override configured model.")
    smoke_parser.add_argument(
        "--mode",
        choices=["baseline", "network"],
        default="baseline",
        help="Execution mode for smoke test (default: baseline).",
    )
    smoke_parser.add_argument(
        "--prompt",
        default=(
            "Summarize three practical risks of deploying an uncalibrated ML classifier, "
            "and list any key context missing from this prompt."
        ),
        help="Smoke-test prompt.",
    )
    smoke_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    mcp_parser = subparsers.add_parser("mcp", help="Launch the CrewForge MCP server.")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    mcp_parser.add_argument("--host", default="127.0.0.1")
    mcp_parser.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the `crewforge` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = CrewForgeSettings()
    configure_logging(settings.log_level)

    if args.command == "solve":
        return asyncio.run(
            _run_mode_cmd(
                args.prompt,
                mode="network",
                settings=settings,
                json_output=args.json,
                doc_paths=args.doc,
            )
        )
    if args.command == "baseline":
        return asyncio.run(
            _run_mode_cmd(
                args.prompt,
                mode="baseline",
                settings=settings,
                json_output=args.json,
                doc_paths=args.doc,
                compute_matched=args.compute_matched,
            )
        )
    if args.command == "trace":
        return asyncio.run(_run_trace_cmd(args.run_id, settings=settings))
    if args.command == "budget":
        return asyncio.run(_run_budget_cmd(settings=settings))
    if args.command == "smoke-test":
        updates: dict[str, Any] = {}
        if args.provider is not None:
            updates["provider"] = ProviderType(args.provider)
        if args.model is not None:
            updates["model"] = args.model
        smoke_settings = settings.model_copy(update=updates) if updates else settings
        return asyncio.run(
            _run_mode_cmd(
                args.prompt,
                mode=args.mode,
                settings=smoke_settings,
                json_output=args.json,
            )
        )
    if args.command == "mcp":
        run_mcp_server(
            transport=args.transport,
            host=args.host,
            port=args.port,
            settings=settings,
        )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
