"""Reproducible benchmark harness runner with experiment fingerprinting and blind export."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from crewforge.config import (
    DEFAULT_MODEL_PRICING,
    BudgetAllocationCategory,
    CrewForgeSettings,
    ProviderType,
    sanitize_error_message,
)
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import (
    BudgetConstraintError,
    BudgetError,
    BudgetManager,
    UnifiedBudgetAuthority,
)
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import LocalDocumentRetriever, ReferenceDocument
from crewforge.evaluation.ablation import AblationConfig
from crewforge.evaluation.anonymize import (
    anonymize_pair,
    export_blind_key_file,
    export_evaluator_packet,
    generate_blinding_seed,
)
from crewforge.evaluation.contracts import (
    BenchmarkItem,
    BenchmarkManifest,
    BlindEvaluationItem,
    BlindKey,
    CachedRunRecord,
    EvaluationRubric,
    EvaluationScore,
    ExperimentFingerprint,
    ExperimentResult,
    HumanGradeInput,
)
from crewforge.evaluation.reporting import (
    export_experiment_results_csv,
    export_experiment_results_json,
    generate_markdown_report,
)
from crewforge.evaluation.scoring import compute_evaluation_score
from crewforge.providers.base import ModelProvider, ProviderError
from crewforge.schemas import RunResult, RunStatus, UsageMetrics
from crewforge.storage.database import Database

_SYSTEMIC_ERROR_RE = re.compile(
    r"(AttributeError|TypeError|AuthenticationError|invalid_api_key|"
    r"HTTP\s*401|HTTP\s*403|insufficient_quota|BudgetConstraintError|"
    r"CREWFORGE_ALLOW_PAID_RUNS)",
    re.IGNORECASE,
)


class SystemicBenchmarkError(RuntimeError):
    """Raised on systemic integration, auth, config, or budget guard failures."""


def get_current_git_commit() -> str:
    """Safely obtain current repository git commit SHA."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3.0,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown_commit"


def compute_manifest_hash(manifest: BenchmarkManifest) -> str:
    """Compute canonical SHA-256 hash of a manifest's immutable items."""
    payload = [
        {
            "prompt_id": it.prompt_id,
            "domain": it.domain.value,
            "prompt": it.prompt,
            "prompt_hash": it.prompt_hash,
            "reference_documents": it.reference_documents,
            "document_hashes": it.document_hashes,
        }
        for it in manifest.items
    ]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_and_validate_manifest(manifest_path: str | Path) -> tuple[BenchmarkManifest, Path]:
    """Load an external or local manifest JSON and verify all referenced documents exist."""
    m_path = Path(manifest_path).resolve()
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")
    manifest = BenchmarkManifest.model_validate_json(m_path.read_text(encoding="utf-8"))
    manifest_dir = m_path.parent
    for item in manifest.items:
        resolve_item_documents(item, manifest_dir=manifest_dir)
    return manifest, manifest_dir


def resolve_item_documents(
    item: BenchmarkItem,
    *,
    manifest_dir: Path | None = None,
) -> tuple[list[ReferenceDocument], dict[str, str], dict[str, str]]:
    """Resolve relative document paths relative to manifest_dir and verify SHA-256 hashes."""
    if not item.reference_documents:
        return [], {}, {}

    base_dir = manifest_dir.resolve() if manifest_dir is not None else Path.cwd()
    resolved_paths: list[Path] = []
    doc_hashes: dict[str, str] = {}
    doc_contents: dict[str, str] = {}

    for raw_ref in item.reference_documents:
        ref_path = Path(raw_ref)
        candidate = ref_path if ref_path.is_absolute() else (base_dir / ref_path)
        if not candidate.exists():
            raise FileNotFoundError(
                f"Required reference document '{raw_ref}' for prompt '{item.prompt_id}' "
                f"not found at '{candidate}'."
            )
        content = candidate.read_text(encoding="utf-8")
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        expected_hash = item.document_hashes.get(raw_ref) or item.document_hashes.get(
            candidate.name
        )
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"Reference document hash mismatch for '{raw_ref}' in prompt '{item.prompt_id}': "
                f"expected {expected_hash}, got {actual_hash}."
            )

        resolved_paths.append(candidate)
        doc_hashes[raw_ref] = actual_hash
        doc_contents[candidate.name] = content

    retriever = LocalDocumentRetriever.from_paths(resolved_paths)
    return retriever.documents, doc_hashes, doc_contents


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text file via temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def build_david_vs_goliath_settings(
    base_settings: CrewForgeSettings | None = None,
    *,
    goliath_cost_per_run_usd: float = 0.20,
) -> tuple[CrewForgeSettings, CrewForgeSettings]:
    """Construct verified settings for David (ollama/qwen3:4b) vs Goliath (openai/gpt-6.2-astra)."""
    base = base_settings or CrewForgeSettings()
    pricing = dict(DEFAULT_MODEL_PRICING)
    pricing.update(base.pricing_json)

    david_settings = base.model_copy(
        update={
            "provider": ProviderType.OLLAMA,
            "model": "qwen3:4b",
            "small_model_profile": True,
            "max_concurrency": 1,
            "max_workers": 2,
            "max_tasks": 3,
            "max_revision_rounds": 1,
            "ollama_num_ctx": 4096,
            "budget_allocation_category": BudgetAllocationCategory.DAVID_VS_GOLIATH,
        }
    )

    goliath_settings = base.model_copy(
        update={
            "provider": ProviderType.OPENAI,
            "model": "gpt-6.2-astra",
            "openai_reasoning_effort": base.openai_reasoning_effort or "medium",
            "max_cost_per_run_usd": goliath_cost_per_run_usd,
            "david_vs_goliath_budget_cap_usd": 1.10,
            "budget_allocation_category": BudgetAllocationCategory.DAVID_VS_GOLIATH,
            "pricing_json": pricing,
        }
    )
    return goliath_settings, david_settings


def _build_side_fingerprint(
    *,
    manifest_hash: str,
    item: BenchmarkItem,
    doc_hashes: dict[str, str],
    git_sha: str,
    settings: CrewForgeSettings,
    baseline_mode: str,
    ablation_variant: str,
) -> ExperimentFingerprint:
    return ExperimentFingerprint(
        manifest_hash=manifest_hash,
        prompt_hash=item.prompt_hash,
        reference_document_hashes=doc_hashes,
        engine_commit_sha=git_sha,
        provider_identity=settings.provider.value,
        configured_model=settings.model,
        reasoning_effort=settings.openai_reasoning_effort,
        tool_configuration={
            "arithmetic_tool": "DeterministicArithmeticTool",
            "http_retrieval": settings.enable_http_retrieval,
            "deterministic_fast_path": settings.enable_deterministic_synthesis_fast_path,
        },
        baseline_mode=baseline_mode,
        ablation_variant=ablation_variant,
        runtime_limits={
            "max_output_tokens": settings.max_output_tokens,
            "max_model_calls_per_run": settings.max_model_calls_per_run,
            "max_cost_per_run_usd": settings.max_cost_per_run_usd,
            "run_deadline_seconds": settings.run_deadline_seconds,
            "baseline_max_calls": settings.baseline_max_calls,
            "baseline_max_output_tokens": settings.baseline_max_output_tokens,
            "enable_independent_review": settings.enable_independent_review,
            "enable_targeted_repair": settings.enable_targeted_repair,
            "enable_peer_messaging": settings.enable_peer_messaging,
        },
    )


class BenchmarkRunner:
    """Orchestrates reproducible benchmark comparisons and exports blind evaluation packets."""

    def __init__(
        self,
        *,
        settings: CrewForgeSettings,
        baseline_settings: CrewForgeSettings | None = None,
        network_settings: CrewForgeSettings | None = None,
        baseline_provider: ModelProvider | None = None,
        network_provider: ModelProvider | None = None,
        output_dir: str | Path = "benchmarks/results",
        database: Database | None = None,
        authority: UnifiedBudgetAuthority | None = None,
    ) -> None:
        self._settings = settings
        self._baseline_settings = baseline_settings or settings
        self._network_settings = network_settings or settings
        self._baseline_provider = baseline_provider
        self._network_provider = network_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._db = database or Database(settings.effective_db_path)
        self._authority = authority or UnifiedBudgetAuthority.get_instance(settings=settings)

    def _preflight_check(
        self,
        *,
        manifest: BenchmarkManifest,
        manifest_dir: Path | None,
        base_runner: BaselineRunner | None,
        net_runner: NetworkRunner,
        base_provider: ModelProvider | None,
        net_provider: ModelProvider,
        net_settings: CrewForgeSettings,
    ) -> list[tuple[BenchmarkItem, list[ReferenceDocument], dict[str, str], dict[str, str]]]:
        """Validate runner interfaces, providers, documents, and budget before any dispatch."""
        if base_runner is not None:
            if not callable(getattr(base_runner, "run", None)):
                raise SystemicBenchmarkError(
                    "Preflight failed: BaselineRunner.run is not callable."
                )
            if base_provider is None:
                raise SystemicBenchmarkError("Preflight failed: baseline_provider is None.")
            if base_provider.is_paid_provider and not self._baseline_settings.allow_paid_runs:
                raise SystemicBenchmarkError(
                    "Preflight failed: paid baseline execution requires allow_paid_runs=true."
                )

        if not callable(getattr(net_runner, "run", None)):
            raise SystemicBenchmarkError("Preflight failed: NetworkRunner.run is not callable.")
        if net_provider is None:
            raise SystemicBenchmarkError("Preflight failed: network_provider is None.")
        if net_provider.is_paid_provider and not net_settings.allow_paid_runs:
            raise SystemicBenchmarkError(
                "Preflight failed: paid network execution requires allow_paid_runs=true."
            )

        resolved_items: list[
            tuple[BenchmarkItem, list[ReferenceDocument], dict[str, str], dict[str, str]]
        ] = []
        for item in manifest.items:
            ref_docs, doc_hashes, doc_contents = resolve_item_documents(
                item, manifest_dir=manifest_dir
            )
            resolved_items.append((item, ref_docs, doc_hashes, doc_contents))
        return resolved_items

    async def _extract_stages_executed(self, n_res: RunResult) -> list[str]:
        """Derive actually executed orchestration stages from persisted trace events."""
        if n_res.status == RunStatus.FAILED and (
            n_res.usage is None or n_res.usage.model_calls == 0
        ):
            return []
        trace = await self._db.get_run_trace(n_res.run_id)
        if trace is None:
            return []
        for ev in reversed(trace.events):
            if ev.event_type == "stages_executed_summary":
                stages = ev.payload.get("stages_executed")
                if isinstance(stages, list):
                    return [str(s) for s in stages]
        observed: list[str] = []
        event_types = {ev.event_type for ev in trace.events}
        if "plan_created" in event_types:
            observed.append("planner")
        if "task_completed" in event_types:
            observed.append("dag_workers")
        if trace.review_verdicts:
            observed.append("independent_review")
        if "honesty_gate_evaluated" in event_types:
            observed.append("honesty_gate_and_synthesis")
        return observed

    async def evaluate_manifest(
        self,
        manifest: BenchmarkManifest,
        rubrics: dict[str, EvaluationRubric] | None = None,
        *,
        manifest_dir: Path | None = None,
        mode: Literal[
            "baseline_vs_network", "compute_matched_vs_network", "ablation"
        ] = "baseline_vs_network",
        ablation_variant: str = "full",
        resume: bool = True,
        blind_seed: str | None = None,
        human_grades: dict[tuple[str, str], HumanGradeInput] | None = None,
    ) -> tuple[list[ExperimentResult], str]:
        """Execute manifest items with fingerprint isolation; grading is strictly optional."""
        await self._db.initialize()
        base_provider = self._baseline_provider or build_provider_from_settings(
            self._baseline_settings
        )
        net_provider = self._network_provider or build_provider_from_settings(
            self._network_settings
        )

        base_budget = BudgetManager(self._db, self._baseline_settings, authority=self._authority)
        net_settings = self._network_settings
        ab_cfg = AblationConfig.from_variant(ablation_variant if mode == "ablation" else "full")
        if mode == "ablation":
            net_settings = ab_cfg.apply_to_settings(self._network_settings)
        net_budget = BudgetManager(self._db, net_settings, authority=self._authority)

        base_runner = BaselineRunner(
            provider=base_provider,
            database=self._db,
            budget_manager=base_budget,
            settings=self._baseline_settings,
        )
        net_runner = NetworkRunner(
            provider=net_provider,
            database=self._db,
            settings=net_settings,
            budget_manager=net_budget,
        )

        manifest_hash = compute_manifest_hash(manifest)
        git_sha = get_current_git_commit()
        baseline_mode_str = (
            "compute_budget_matched" if mode == "compute_matched_vs_network" else "standard"
        )
        resolved_blind_seed = blind_seed or generate_blinding_seed()

        results: list[ExperimentResult] = []
        blind_items: list[BlindEvaluationItem] = []
        blind_keys: list[BlindKey] = []

        try:
            resolved_items = self._preflight_check(
                manifest=manifest,
                manifest_dir=manifest_dir,
                base_runner=base_runner,
                net_runner=net_runner,
                base_provider=base_provider,
                net_provider=net_provider,
                net_settings=net_settings,
            )

            for item, ref_docs, doc_hashes, doc_contents in resolved_items:
                base_fp = _build_side_fingerprint(
                    manifest_hash=manifest_hash,
                    item=item,
                    doc_hashes=doc_hashes,
                    git_sha=git_sha,
                    settings=self._baseline_settings,
                    baseline_mode=baseline_mode_str,
                    ablation_variant="none",
                )
                net_fp = _build_side_fingerprint(
                    manifest_hash=manifest_hash,
                    item=item,
                    doc_hashes=doc_hashes,
                    git_sha=git_sha,
                    settings=net_settings,
                    baseline_mode="none",
                    ablation_variant=ab_cfg.variant.value,
                )
                pair_fp_id = hashlib.sha256(
                    f"{base_fp.fingerprint_id}:{net_fp.fingerprint_id}".encode()
                ).hexdigest()[:16]

                base_file = (
                    self.output_dir
                    / "runs"
                    / base_fp.fingerprint_id
                    / f"{item.prompt_id}_baseline.json"
                )
                net_file = (
                    self.output_dir
                    / "runs"
                    / net_fp.fingerprint_id
                    / f"{item.prompt_id}_network.json"
                )

                b_res = await self._run_or_load_cached(
                    cache_file=base_file,
                    fingerprint=base_fp,
                    prompt_id=item.prompt_id,
                    system_label="baseline",
                    resume=resume,
                    coro_fn=lambda p=item.prompt, d=ref_docs: base_runner.run(
                        p,
                        reference_documents=d,
                        compute_matched=(mode == "compute_matched_vs_network"),
                    ),
                    provider_name=self._baseline_settings.provider.value,
                    model_name=self._baseline_settings.model,
                    prompt=item.prompt,
                )

                n_res = await self._run_or_load_cached(
                    cache_file=net_file,
                    fingerprint=net_fp,
                    prompt_id=item.prompt_id,
                    system_label="network",
                    resume=resume,
                    coro_fn=lambda p=item.prompt, d=ref_docs: net_runner.run(
                        p,
                        reference_documents=d,
                    ),
                    provider_name=net_settings.provider.value,
                    model_name=net_settings.model,
                    prompt=item.prompt,
                )

                stages_executed = await self._extract_stages_executed(n_res)
                exp_result, b_it, b_key = self._build_pair_outputs(
                    item=item,
                    doc_hashes=doc_hashes,
                    doc_contents=doc_contents,
                    b_res=b_res,
                    n_res=n_res,
                    pair_fp_id=pair_fp_id,
                    base_fp_id=base_fp.fingerprint_id,
                    net_fp_id=net_fp.fingerprint_id,
                    baseline_settings=self._baseline_settings,
                    network_settings=net_settings,
                    baseline_mode_str=baseline_mode_str,
                    ablation_variant=ab_cfg.variant.value,
                    stages_executed=stages_executed,
                    git_sha=git_sha,
                    rubrics=rubrics,
                    human_grades=human_grades,
                    blind_seed=resolved_blind_seed,
                )
                results.append(exp_result)
                blind_items.append(b_it)
                blind_keys.append(b_key)

            self._write_experiment_artifacts(
                results=results,
                blind_items=blind_items,
                blind_keys=blind_keys,
                blind_seed=resolved_blind_seed,
                manifest_hash=manifest_hash,
                title=f"CrewForge Evaluation — {manifest.name}",
            )

            md_report = (self.output_dir / "report.md").read_text(encoding="utf-8")
            return results, md_report
        finally:
            if self._baseline_provider is None:
                await base_provider.aclose()
            if self._network_provider is None:
                await net_provider.aclose()

    async def recover_from_results(
        self,
        *,
        original_results_path: str | Path,
        manifest: BenchmarkManifest,
        manifest_dir: Path | None = None,
        mode: Literal[
            "baseline_vs_network", "compute_matched_vs_network", "ablation"
        ] = "baseline_vs_network",
        ablation_variant: str = "full",
        resume: bool = True,
        blind_seed: str | None = None,
        recovery_reason: str = "network_runner_solve_to_run_hotfix",
    ) -> tuple[list[ExperimentResult], str]:
        """Recover failed zero-call network runs while reusing verified original baselines."""
        await self._db.initialize()
        raw_text = await asyncio.to_thread(Path(original_results_path).read_text, encoding="utf-8")
        raw_orig = json.loads(raw_text)
        orig_by_id = {
            r.prompt_id: r for r in (ExperimentResult.model_validate(x) for x in raw_orig)
        }

        net_provider = self._network_provider or build_provider_from_settings(
            self._network_settings
        )
        net_settings = self._network_settings
        ab_cfg = AblationConfig.from_variant(ablation_variant if mode == "ablation" else "full")
        if mode == "ablation":
            net_settings = ab_cfg.apply_to_settings(self._network_settings)
        net_budget = BudgetManager(self._db, net_settings, authority=self._authority)
        net_runner = NetworkRunner(
            provider=net_provider,
            database=self._db,
            settings=net_settings,
            budget_manager=net_budget,
        )

        manifest_hash = compute_manifest_hash(manifest)
        new_git_sha = get_current_git_commit()
        baseline_mode_str = (
            "compute_budget_matched" if mode == "compute_matched_vs_network" else "standard"
        )
        resolved_blind_seed = blind_seed or generate_blinding_seed()

        try:
            resolved_items = self._preflight_check(
                manifest=manifest,
                manifest_dir=manifest_dir,
                base_runner=None,
                net_runner=net_runner,
                base_provider=None,
                net_provider=net_provider,
                net_settings=net_settings,
            )

            # Strict validation of all original baseline records before any network dispatch
            validated_originals: list[
                tuple[
                    BenchmarkItem,
                    list[ReferenceDocument],
                    dict[str, str],
                    dict[str, str],
                    ExperimentResult,
                    ExperimentFingerprint,
                ]
            ] = []
            for item, ref_docs, doc_hashes, doc_contents in resolved_items:
                orig = orig_by_id.get(item.prompt_id)
                if orig is None:
                    raise ValueError(
                        f"Recovery aborted: prompt '{item.prompt_id}' "
                        "missing from original results."
                    )
                b_res = orig.baseline_result
                if b_res.status != RunStatus.COMPLETED or not b_res.answer.strip():
                    raise ValueError(
                        f"Recovery aborted: baseline for '{item.prompt_id}' is not completed."
                    )
                if b_res.prompt != item.prompt:
                    raise ValueError(
                        f"Recovery aborted: baseline prompt mismatch for '{item.prompt_id}'."
                    )
                if (
                    b_res.provider != self._baseline_settings.provider.value
                    or b_res.model != self._baseline_settings.model
                ):
                    raise ValueError(
                        f"Recovery aborted: baseline provider/model mismatch for "
                        f"'{item.prompt_id}' ({b_res.provider}/{b_res.model})."
                    )
                if b_res.usage is None or b_res.usage.model_calls < 1:
                    raise ValueError(
                        f"Recovery aborted: baseline for '{item.prompt_id}' has 0 model calls."
                    )
                n_orig = orig.network_result
                if n_orig.status != RunStatus.FAILED or (
                    n_orig.usage is not None and n_orig.usage.model_calls != 0
                ):
                    raise ValueError(
                        f"Recovery aborted: original network run for '{item.prompt_id}' "
                        f"is not a zero-call failed attempt."
                    )

                orig_base_fp = _build_side_fingerprint(
                    manifest_hash=manifest_hash,
                    item=item,
                    doc_hashes=doc_hashes,
                    git_sha=orig.git_commit_sha or "unknown_commit",
                    settings=self._baseline_settings,
                    baseline_mode=baseline_mode_str,
                    ablation_variant="none",
                )
                validated_originals.append(
                    (item, ref_docs, doc_hashes, doc_contents, orig, orig_base_fp)
                )

            results: list[ExperimentResult] = []
            blind_items: list[BlindEvaluationItem] = []
            blind_keys: list[BlindKey] = []

            for item, ref_docs, doc_hashes, doc_contents, orig, orig_base_fp in validated_originals:
                b_res = orig.baseline_result
                net_fp = _build_side_fingerprint(
                    manifest_hash=manifest_hash,
                    item=item,
                    doc_hashes=doc_hashes,
                    git_sha=new_git_sha,
                    settings=net_settings,
                    baseline_mode="none",
                    ablation_variant=ab_cfg.variant.value,
                )
                pair_fp_id = hashlib.sha256(
                    f"{orig_base_fp.fingerprint_id}:{net_fp.fingerprint_id}".encode()
                ).hexdigest()[:16]

                net_file = (
                    self.output_dir
                    / "runs"
                    / net_fp.fingerprint_id
                    / f"{item.prompt_id}_network.json"
                )
                n_res = await self._run_or_load_cached(
                    cache_file=net_file,
                    fingerprint=net_fp,
                    prompt_id=item.prompt_id,
                    system_label="network",
                    resume=resume,
                    coro_fn=lambda p=item.prompt, d=ref_docs: net_runner.run(
                        p,
                        reference_documents=d,
                    ),
                    provider_name=net_settings.provider.value,
                    model_name=net_settings.model,
                    prompt=item.prompt,
                )

                stages_executed = await self._extract_stages_executed(n_res)
                exp_result, b_it, b_key = self._build_pair_outputs(
                    item=item,
                    doc_hashes=doc_hashes,
                    doc_contents=doc_contents,
                    b_res=b_res,
                    n_res=n_res,
                    pair_fp_id=pair_fp_id,
                    base_fp_id=orig_base_fp.fingerprint_id,
                    net_fp_id=net_fp.fingerprint_id,
                    baseline_settings=self._baseline_settings,
                    network_settings=net_settings,
                    baseline_mode_str=baseline_mode_str,
                    ablation_variant=ab_cfg.variant.value,
                    stages_executed=stages_executed,
                    git_sha=new_git_sha,
                    rubrics=None,
                    human_grades=None,
                    blind_seed=resolved_blind_seed,
                    original_baseline_engine_commit=orig.git_commit_sha,
                    recovered_network_engine_commit=new_git_sha,
                    original_baseline_run_id=b_res.run_id,
                    original_failed_network_run_id=orig.network_run_id,
                    recovered_network_run_id=n_res.run_id,
                    recovery_reason=recovery_reason,
                )
                results.append(exp_result)
                blind_items.append(b_it)
                blind_keys.append(b_key)

            self._write_experiment_artifacts(
                results=results,
                blind_items=blind_items,
                blind_keys=blind_keys,
                blind_seed=resolved_blind_seed,
                manifest_hash=manifest_hash,
                title=f"CrewForge Recovered Evaluation — {manifest.name}",
            )
            md_report = (self.output_dir / "report.md").read_text(encoding="utf-8")
            return results, md_report
        finally:
            if self._network_provider is None:
                await net_provider.aclose()

    def _build_pair_outputs(
        self,
        *,
        item: BenchmarkItem,
        doc_hashes: dict[str, str],
        doc_contents: dict[str, str],
        b_res: RunResult,
        n_res: RunResult,
        pair_fp_id: str,
        base_fp_id: str,
        net_fp_id: str,
        baseline_settings: CrewForgeSettings,
        network_settings: CrewForgeSettings,
        baseline_mode_str: str,
        ablation_variant: str,
        stages_executed: list[str],
        git_sha: str,
        rubrics: dict[str, EvaluationRubric] | None,
        human_grades: dict[tuple[str, str], HumanGradeInput] | None,
        blind_seed: str,
        original_baseline_engine_commit: str = "",
        recovered_network_engine_commit: str = "",
        original_baseline_run_id: str = "",
        original_failed_network_run_id: str = "",
        recovered_network_run_id: str = "",
        recovery_reason: str = "",
    ) -> tuple[ExperimentResult, BlindEvaluationItem, BlindKey]:
        b_score: EvaluationScore | None = None
        n_score: EvaluationScore | None = None
        rubric = rubrics.get(item.prompt_id) if rubrics else None
        if rubric is not None:
            b_grade = human_grades.get((item.prompt_id, "baseline")) if human_grades else None
            n_grade = human_grades.get((item.prompt_id, "network")) if human_grades else None
            b_score = compute_evaluation_score(
                rubric=rubric,
                system_label="baseline",
                answer_text=b_res.answer,
                assumptions=b_res.assumptions,
                missing_information=b_res.missing_information,
                confidence_notes=b_res.confidence_notes,
                human_grade=b_grade,
            )
            n_score = compute_evaluation_score(
                rubric=rubric,
                system_label="network",
                answer_text=n_res.answer,
                assumptions=n_res.assumptions,
                missing_information=n_res.missing_information,
                confidence_notes=n_res.confidence_notes,
                human_grade=n_grade,
            )

        blind_it, b_key = anonymize_pair(
            prompt_id=item.prompt_id,
            domain=item.domain,
            prompt=item.prompt,
            baseline_result=b_res,
            network_result=n_res,
            reference_documents=item.reference_documents,
            reference_document_contents=doc_contents,
            seed=blind_seed,
        )

        b_cost = b_res.usage.actual_cost_usd if b_res.usage and b_res.usage.actual_cost_usd else 0.0
        n_cost = n_res.usage.actual_cost_usd if n_res.usage and n_res.usage.actual_cost_usd else 0.0
        both_graded = (
            b_score is not None
            and n_score is not None
            and b_score.grading_status == "graded"
            and n_score.grading_status == "graded"
        )

        exp_result = ExperimentResult(
            prompt_id=item.prompt_id,
            domain=item.domain,
            experiment_fingerprint=pair_fp_id,
            baseline_fingerprint=base_fp_id,
            network_fingerprint=net_fp_id,
            baseline_run_id=b_res.run_id,
            network_run_id=n_res.run_id,
            original_baseline_engine_commit=original_baseline_engine_commit or git_sha,
            recovered_network_engine_commit=recovered_network_engine_commit,
            original_baseline_run_id=original_baseline_run_id or b_res.run_id,
            original_failed_network_run_id=original_failed_network_run_id,
            recovered_network_run_id=recovered_network_run_id,
            recovery_reason=recovery_reason,
            prompt_hash=item.prompt_hash,
            reference_document_hashes=doc_hashes,
            baseline_provider=baseline_settings.provider.value,
            baseline_model=baseline_settings.model,
            network_provider=network_settings.provider.value,
            network_model=network_settings.model,
            baseline_mode=baseline_mode_str,
            ablation_variant=ablation_variant,
            stages_executed=stages_executed,
            baseline_result=b_res,
            network_result=n_res,
            baseline_score=b_score,
            network_score=n_score,
            grading_status="graded" if both_graded else "pending_independent_grading",
            latency_difference_ms=n_res.latency_ms - b_res.latency_ms,
            cost_difference_usd=(n_cost or 0.0) - (b_cost or 0.0),
            git_commit_sha=git_sha,
        )
        return exp_result, blind_it, b_key

    def _write_experiment_artifacts(
        self,
        *,
        results: list[ExperimentResult],
        blind_items: list[BlindEvaluationItem],
        blind_keys: list[BlindKey],
        blind_seed: str,
        manifest_hash: str,
        title: str,
    ) -> None:
        export_evaluator_packet(blind_items, self.output_dir / "evaluator_packet.json")
        export_blind_key_file(
            blind_keys,
            blinding_seed=blind_seed,
            output_path=self.output_dir / "blind_key.json",
            experiment_fingerprint=manifest_hash[:16],
        )
        export_experiment_results_json(results, self.output_dir / "experiment_results.json")
        export_experiment_results_csv(results, self.output_dir / "experiment_results.csv")
        md_report = generate_markdown_report(results, title=title)
        _atomic_write_text(self.output_dir / "report.md", md_report)

    async def _run_or_load_cached(
        self,
        *,
        cache_file: Path,
        fingerprint: ExperimentFingerprint,
        prompt_id: str,
        system_label: str,
        resume: bool,
        coro_fn: Any,
        provider_name: str,
        model_name: str,
        prompt: str,
    ) -> RunResult:
        if resume:

            def _load_if_exists() -> CachedRunRecord | None:
                if not cache_file.exists():
                    return None
                return CachedRunRecord.model_validate_json(cache_file.read_text(encoding="utf-8"))

            record = await asyncio.to_thread(_load_if_exists)
            if record is not None:
                if record.experiment_fingerprint != fingerprint.fingerprint_id:
                    raise ValueError(
                        f"Cached record fingerprint mismatch for '{prompt_id}' ({system_label}): "
                        f"expected {fingerprint.fingerprint_id}, "
                        f"got {record.experiment_fingerprint}."
                    )
                return record.result

        systemic_exc: Exception | None = None
        try:
            res = await coro_fn()
            if (
                res.status == RunStatus.FAILED
                and res.error
                and _SYSTEMIC_ERROR_RE.search(res.error)
            ):
                systemic_exc = SystemicBenchmarkError(
                    f"Systemic failure in {system_label} for '{prompt_id}': {res.error}"
                )
        except (
            AttributeError,
            TypeError,
            BudgetError,
            BudgetConstraintError,
            SystemicBenchmarkError,
        ) as exc:
            systemic_exc = exc
            res = RunResult(
                run_id=f"run_{uuid4().hex[:12]}",
                mode="baseline" if system_label == "baseline" else "network",
                status=RunStatus.FAILED,
                prompt=prompt,
                provider=provider_name,
                model=model_name,
                answer="",
                usage=UsageMetrics(),
                error=f"systemic_failure: {type(exc).__name__}: {sanitize_error_message(str(exc))}",
            )
        except ProviderError as exc:
            msg = sanitize_error_message(str(exc))
            if _SYSTEMIC_ERROR_RE.search(msg):
                systemic_exc = exc
            res = RunResult(
                run_id=f"run_{uuid4().hex[:12]}",
                mode="baseline" if system_label == "baseline" else "network",
                status=RunStatus.FAILED,
                prompt=prompt,
                provider=provider_name,
                model=model_name,
                answer="",
                usage=UsageMetrics(),
                error=f"unavailable_or_failed: {msg}",
            )
        except Exception as exc:
            res = RunResult(
                run_id=f"run_{uuid4().hex[:12]}",
                mode="baseline" if system_label == "baseline" else "network",
                status=RunStatus.FAILED,
                prompt=prompt,
                provider=provider_name,
                model=model_name,
                answer="",
                usage=UsageMetrics(),
                error=f"unavailable_or_failed: {sanitize_error_message(str(exc))}",
            )

        cached = CachedRunRecord(
            experiment_fingerprint=fingerprint.fingerprint_id,
            fingerprint_details=fingerprint,
            prompt_id=prompt_id,
            system_label=system_label,
            result=res,
        )
        await asyncio.to_thread(_atomic_write_text, cache_file, cached.model_dump_json(indent=2))
        if systemic_exc is not None:
            raise SystemicBenchmarkError(str(systemic_exc)) from systemic_exc
        return res
