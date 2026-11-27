"""Blinded V3 evaluation harness for Groq free-tier GPT-OSS-120B (Network vs Baseline).

Does not generate questions or grading keys. Accepts a user-supplied manifest JSON file,
resolves and SHA-256 verifies documents relative to the manifest, runs both Network and
Baseline in alternating order with quota recovery and per-pair checkpointing, and exports
the blinded Candidate A/B evaluator packet separately from the private unblinding key
and system-specific performance metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

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
from crewforge.evaluation.contracts import (
    BenchmarkItem,
    BlindEvaluationItem,
    BlindKey,
)
from crewforge.providers.base import ModelProvider
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
    """Build isolated $0.00 Groq free-tier settings without .env or team budget fallback."""
    return CrewForgeSettings(
        _env_file=None,
        db_path=db_path,
        budget_authority_path=budget_db_path,
        provider=ProviderType.OPENAI_COMPATIBLE,
        model=model,
        openai_api_key=SecretStr(api_key),
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


def load_and_verify_item_docs(
    item: BenchmarkItem,
    manifest_dir: Path,
) -> tuple[list[ReferenceDocument], dict[str, str]]:
    """Resolve document paths relative to manifest_dir, verify existence & SHA-256."""
    docs: list[ReferenceDocument] = []
    contents_by_name: dict[str, str] = {}
    for idx, rel_path in enumerate(item.reference_documents):
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
        docs.append(
            ReferenceDocument(
                doc_id=f"doc_{idx + 1}",
                source_ref=doc_path.name,
                content=text,
            )
        )
        contents_by_name[doc_path.name] = text
    return docs, contents_by_name


async def run_blinded_benchmark(
    *,
    items: list[BenchmarkItem],
    manifest_dir: Path,
    out_dir: Path,
    api_key: str,
    model: str = "openai/gpt-oss-120b",
    manifest_id: str = "crewforge_v3_fresh_groq",
    inter_side_delay_s: float = 10.0,
    inter_case_delay_s: float = 15.0,
    provider_override: ModelProvider | None = None,
) -> tuple[Path, Path, Path]:
    """Execute Network vs Baseline in alternating order with per-pair checkpointing."""
    checkpoints_dir = out_dir / "checkpoints"
    await asyncio.to_thread(checkpoints_dir.mkdir, parents=True, exist_ok=True)

    # Pre-verify all documents upfront before spending any quota
    verified_docs: dict[str, tuple[list[ReferenceDocument], dict[str, str]]] = {}
    for item in items:
        verified_docs[item.prompt_id] = await asyncio.to_thread(
            load_and_verify_item_docs, item, manifest_dir
        )

    seed_path = checkpoints_dir / "blinding_seed.txt"
    if seed_path.exists():
        seed = seed_path.read_text(encoding="utf-8").strip()
    else:
        seed = generate_blinding_seed()
        await asyncio.to_thread(seed_path.write_text, seed, "utf-8")

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
    provider: ModelProvider = provider_override or OpenAIProvider.from_settings(settings)

    blinded_items: list[BlindEvaluationItem] = []
    key_entries: list[BlindKey] = []
    metrics_rows: list[dict[str, Any]] = []

    try:
        for idx, item in enumerate(items):
            ckpt_path = checkpoints_dir / f"{item.prompt_id}.json"
            if ckpt_path.exists():
                cached = json.loads(await asyncio.to_thread(ckpt_path.read_text, "utf-8"))
                blinded_items.append(BlindEvaluationItem.model_validate(cached["blind_item"]))
                key_entries.append(BlindKey.model_validate(cached["blind_key"]))
                metrics_rows.append(cached["metrics"])
                continue

            ref_docs, doc_contents = verified_docs[item.prompt_id]
            network_first = idx % 2 == 0
            wait_acc = {"seconds": 0.0}

            def _track_wait(
                wait_s: float, _reason: str, acc: dict[str, float] = wait_acc
            ) -> None:
                acc["seconds"] += float(wait_s)

            async def _run_network(
                cur_item: BenchmarkItem = item,
                cur_docs: list[ReferenceDocument] = ref_docs,
                acc: dict[str, float] = wait_acc,
            ) -> tuple[Any, float, float]:
                acc["seconds"] = 0.0
                t0 = time.monotonic()
                runner = NetworkRunner(
                    provider=provider, database=db, budget_manager=budget, settings=settings
                )
                res = await runner.run(cur_item.prompt, reference_documents=cur_docs or None)
                return res, time.monotonic() - t0, acc["seconds"]

            async def _run_baseline(
                cur_item: BenchmarkItem = item,
                cur_docs: list[ReferenceDocument] = ref_docs,
                acc: dict[str, float] = wait_acc,
            ) -> tuple[Any, float, float]:
                acc["seconds"] = 0.0
                t0 = time.monotonic()
                runner = BaselineRunner(
                    provider=provider, database=db, budget_manager=budget, settings=settings
                )
                res = await runner.run(cur_item.prompt, reference_documents=cur_docs or None)
                return res, time.monotonic() - t0, acc["seconds"]

            tok = wait_observer_var.set(_track_wait)
            try:
                if network_first:
                    net_res, net_wall_s, net_wait_s = await _run_network()
                    if inter_side_delay_s > 0:
                        await asyncio.sleep(inter_side_delay_s)
                    base_res, base_wall_s, base_wait_s = await _run_baseline()
                else:
                    base_res, base_wall_s, base_wait_s = await _run_baseline()
                    if inter_side_delay_s > 0:
                        await asyncio.sleep(inter_side_delay_s)
                    net_res, net_wall_s, net_wait_s = await _run_network()
            finally:
                wait_observer_var.reset(tok)

            b_item, b_key = anonymize_pair(
                prompt_id=item.prompt_id,
                domain=item.domain.value,
                prompt=item.prompt,
                baseline_result=base_res,
                network_result=net_res,
                reference_documents=item.reference_documents,
                reference_document_contents=doc_contents,
                seed=seed,
            )
            m_row = {
                "prompt_id": item.prompt_id,
                "domain": item.domain.value,
                "execution_order": "network_first" if network_first else "baseline_first",
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

            # Checkpoint immediately so export failures cannot lose completed pairs
            ckpt_payload = json.dumps(
                {
                    "blind_item": b_item.model_dump(mode="json"),
                    "blind_key": b_key.model_dump(mode="json"),
                    "metrics": m_row,
                },
                indent=2,
            )
            await asyncio.to_thread(ckpt_path.write_text, ckpt_payload, "utf-8")

            blinded_items.append(b_item)
            key_entries.append(b_key)
            metrics_rows.append(m_row)

            if idx < len(items) - 1 and inter_case_delay_s > 0:
                await asyncio.sleep(inter_case_delay_s)
    finally:
        if provider_override is None:
            await provider.aclose()
        await db.close()

    packet_path = out_dir / "blinded_evaluator_packet.json"
    key_path = out_dir / "private_blind_key.json"
    metrics_path = out_dir / "private_system_metrics.json"
    export_evaluator_packet(blinded_items, packet_path, packet_id=manifest_id)
    export_blind_key_file(
        key_entries,
        blinding_seed=seed,
        output_path=key_path,
        experiment_fingerprint=manifest_id,
    )
    metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    return packet_path, key_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blinded Groq free-tier Network vs Baseline.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file.")
    parser.add_argument("--out-dir", default=".crewforge/v3_eval", help="Output directory.")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Groq free-tier model.")
    parser.add_argument("--inter-side-delay", type=float, default=10.0)
    parser.add_argument("--inter-case-delay", type=float, default=15.0)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass(
            "Enter Groq free-tier API key (transient, not persisted): "
        ).strip()
    if not api_key:
        raise SystemExit("Transient Groq API key is required.")

    manifest_path = Path(args.manifest).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_items = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    manifest_id = (
        raw.get("manifest_id", "crewforge_v3") if isinstance(raw, dict) else "crewforge_v3"
    )
    items = [BenchmarkItem.model_validate(x) for x in raw_items]
    packet, _key, _metrics = asyncio.run(
        run_blinded_benchmark(
            items=items,
            manifest_dir=manifest_path.parent,
            out_dir=Path(args.out_dir),
            api_key=api_key,
            model=args.model,
            manifest_id=manifest_id,
            inter_side_delay_s=args.inter_side_delay,
            inter_case_delay_s=args.inter_case_delay,
        )
    )
    print(f"Blinded evaluator packet exported to {packet}")


if __name__ == "__main__":
    main()
