"""Blinded A/B evaluation harness for Groq free-tier GPT-OSS-120B (Network vs Baseline).

Does not generate questions or grading keys. Accepts a user-supplied manifest JSON file
and runs both CrewForge Network and Single-Agent Baseline on the same Groq free-tier model,
recording answer correctness inputs, tokens, request attempts, latency, quota wait time,
and $0.00 expenditure, then exports anonymized Candidate A/B packets for independent grading.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from crewforge.config import CrewForgeSettings, ModelPricing, ProviderType
from crewforge.core.baseline import BaselineRunner
from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import ReferenceDocument
from crewforge.evaluation.anonymize import (
    anonymize_pair,
    export_blind_key_file,
    export_evaluator_packet,
    generate_blinding_seed,
)
from crewforge.evaluation.contracts import BenchmarkItem
from crewforge.providers.groq_quota import wait_observer_var
from crewforge.providers.openai_provider import OpenAIProvider
from crewforge.storage.database import Database


def build_groq_free_eval_settings(
    *,
    db_path: Path,
    budget_db_path: Path,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
) -> CrewForgeSettings:
    """Build isolated $0.00 Groq free-tier settings for Network vs Baseline comparison."""
    return CrewForgeSettings(
        db_path=db_path,
        budget_authority_path=budget_db_path,
        provider=ProviderType.OPENAI_COMPATIBLE,
        model=model,
        openai_api_key=api_key,
        openai_base_url="https://api.groq.com/openai/v1",
        allow_paid_runs=False,
        max_cost_per_run_usd=0.0,
        max_cumulative_cost_usd=0.0,
        pricing_json={model: ModelPricing(input_per_1m=0.0, output_per_1m=0.0)},
        max_output_tokens=8192,
        planner_max_output_tokens=6144,
        worker_max_output_tokens=8192,
        review_max_output_tokens=6144,
        revision_max_output_tokens=8192,
        synthesis_max_output_tokens=6144,
        max_workers=3,
        max_tasks=3,
        max_concurrency=3,
        max_communication_rounds=1,
        max_revision_rounds=1,
        max_model_calls_per_run=16,
        max_retries=0,
        request_timeout_seconds=180.0,
        run_deadline_seconds=480.0,
        enable_http_retrieval=False,
    )


def _load_item_docs(item: BenchmarkItem) -> list[ReferenceDocument]:
    docs: list[ReferenceDocument] = []
    for idx, p_str in enumerate(item.reference_documents):
        p = Path(p_str)
        if p.exists():
            docs.append(
                ReferenceDocument(
                    doc_id=f"doc_{idx + 1}",
                    source_ref=p.name,
                    content=p.read_text(encoding="utf-8"),
                )
            )
    return docs


async def run_blinded_benchmark(
    *,
    items: list[BenchmarkItem],
    out_dir: Path,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
) -> Path:
    """Execute Network vs Baseline on supplied items and export anonymized Candidate A/B files."""
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
    db_path = out_dir / "eval_runs.db"
    budget_path = out_dir / "eval_budget.db"
    settings = build_groq_free_eval_settings(
        db_path=db_path,
        budget_db_path=budget_path,
        api_key=api_key,
        model=model,
    )
    db = Database(db_path)
    await db.initialize()
    authority = UnifiedBudgetAuthority(settings=settings, migrate_default_dbs=False)
    budget = BudgetManager(database=db, settings=settings, authority=authority)
    provider = OpenAIProvider.from_settings(settings)

    seed = generate_blinding_seed()
    blinded_items = []
    key_entries = []
    metrics_rows: list[dict[str, Any]] = []

    try:
        for item in items:
            ref_docs = await asyncio.to_thread(_load_item_docs, item)
            wait_acc = {"seconds": 0.0}

            def _track_wait(wait_s: float, _reason: str, acc: dict[str, float] = wait_acc) -> None:
                acc["seconds"] += float(wait_s)

            tok = wait_observer_var.set(_track_wait)
            try:
                t0 = time.monotonic()
                net_runner = NetworkRunner(
                    provider=provider, database=db, budget_manager=budget, settings=settings
                )
                net_res = await net_runner.run(
                    item.prompt, reference_documents=ref_docs or None
                )
                net_wall_s = time.monotonic() - t0
                net_wait_s = wait_acc["seconds"]

                wait_acc["seconds"] = 0.0
                t1 = time.monotonic()
                base_runner = BaselineRunner(
                    provider=provider, database=db, budget_manager=budget, settings=settings
                )
                base_res = await base_runner.run(
                    item.prompt, reference_documents=ref_docs or None
                )
                base_wall_s = time.monotonic() - t1
                base_wait_s = wait_acc["seconds"]
            finally:
                wait_observer_var.reset(tok)

            b_item, b_key = anonymize_pair(
                prompt_id=item.prompt_id,
                domain=item.domain.value,
                prompt=item.prompt,
                baseline_result=base_res,
                network_result=net_res,
                seed=seed,
                reference_documents=item.reference_documents,
            )
            blinded_items.append(b_item)
            key_entries.append(b_key)
            metrics_rows.append(
                {
                    "prompt_id": item.prompt_id,
                    "domain": item.domain.value,
                    "network": {
                        "run_id": net_res.run_id,
                        "status": net_res.status.value,
                        "wall_seconds": round(net_wall_s, 2),
                        "quota_wait_seconds": round(net_wait_s, 2),
                        "model_calls": net_res.usage.model_calls,
                        "retries": net_res.usage.retries,
                        "total_tokens": net_res.usage.total_tokens,
                        "cost_usd": net_res.usage.actual_cost_usd,
                    },
                    "baseline": {
                        "run_id": base_res.run_id,
                        "status": base_res.status.value,
                        "wall_seconds": round(base_wall_s, 2),
                        "quota_wait_seconds": round(base_wait_s, 2),
                        "model_calls": base_res.usage.model_calls,
                        "retries": base_res.usage.retries,
                        "total_tokens": base_res.usage.total_tokens,
                        "cost_usd": base_res.usage.actual_cost_usd,
                    },
                }
            )
    finally:
        await provider.aclose()
        await db.close()

    packet_path = out_dir / "blinded_evaluator_packet.json"
    key_path = out_dir / "private_blind_key.json"
    metrics_path = out_dir / "benchmark_metrics.json"
    export_evaluator_packet(blinded_items, packet_path)
    export_blind_key_file(key_entries, key_path)
    metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    return packet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blinded Groq free-tier Network vs Baseline.")
    parser.add_argument("--manifest", required=True, help="Path to JSON list of BenchmarkItem.")
    parser.add_argument("--out-dir", default=".crewforge/groq_eval", help="Output directory.")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq free-tier model.")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GROQ_API_KEY environment variable is required for CLI execution.")

    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    raw_items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    items = [BenchmarkItem.model_validate(x) for x in raw_items]
    packet = asyncio.run(
        run_blinded_benchmark(
            items=items,
            out_dir=Path(args.out_dir),
            api_key=api_key,
            model=args.model,
        )
    )
    print(f"Exported anonymized Candidate A/B packet to {packet}")


if __name__ == "__main__":
    main()
