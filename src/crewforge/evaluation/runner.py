"""Benchmark harness runner with resume support, git tracking, and blind packet export."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal

from crewforge.config import CrewForgeSettings
from crewforge.core.baseline import BaselineRunner, build_provider_from_settings
from crewforge.core.budget import BudgetManager, UnifiedBudgetAuthority
from crewforge.core.orchestrator import NetworkRunner
from crewforge.core.retrieval import LocalDocumentRetriever, ReferenceDocument
from crewforge.evaluation.ablation import AblationConfig
from crewforge.evaluation.anonymize import anonymize_pair, export_evaluator_packet
from crewforge.evaluation.contracts import (
    BenchmarkManifest,
    BlindEvaluationItem,
    BlindKey,
    EvaluationRubric,
    ExperimentResult,
)
from crewforge.evaluation.reporting import (
    export_experiment_results_csv,
    export_experiment_results_json,
    generate_markdown_report,
)
from crewforge.evaluation.scoring import compute_evaluation_score
from crewforge.schemas import RunResult
from crewforge.storage.database import Database


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


class BenchmarkRunner:
    """Orchestrates reproducible benchmark comparisons and exports evaluation packets."""

    def __init__(
        self,
        *,
        settings: CrewForgeSettings,
        output_dir: str | Path = "benchmarks/results",
        database: Database | None = None,
        authority: UnifiedBudgetAuthority | None = None,
    ) -> None:
        self._settings = settings
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._db = database or Database(settings.effective_db_path)
        self._authority = authority or UnifiedBudgetAuthority.get_instance(settings=settings)

    async def evaluate_manifest(
        self,
        manifest: BenchmarkManifest,
        rubrics: dict[str, EvaluationRubric],
        *,
        mode: Literal[
            "baseline_vs_network", "compute_matched_vs_network", "ablation"
        ] = "baseline_vs_network",
        ablation_variant: str = "full",
        resume: bool = True,
        blind_seed: str = "crewforge_eval_2026",
    ) -> tuple[list[ExperimentResult], str]:
        """Execute all items in the manifest, score objectively, and export results."""
        await self._db.initialize()
        provider = build_provider_from_settings(self._settings)
        budget = BudgetManager(self._db, self._settings, authority=self._authority)

        base_runner = BaselineRunner(
            provider=provider,
            database=self._db,
            budget_manager=budget,
            settings=self._settings,
        )

        net_settings = self._settings
        if mode == "ablation":
            ab_cfg = AblationConfig.from_variant(ablation_variant)
            net_settings = ab_cfg.apply_to_settings(self._settings)

        net_runner = NetworkRunner(
            database=self._db,
            settings=net_settings,
            budget_manager=budget,
        )

        results: list[ExperimentResult] = []
        blind_items: list[BlindEvaluationItem] = []
        blind_keys: list[BlindKey] = []
        git_sha = get_current_git_commit()

        for item in manifest.items:
            rubric = rubrics.get(item.prompt_id)
            if rubric is None:
                continue

            ref_docs: list[ReferenceDocument] = []
            if item.reference_documents:
                try:
                    retriever = LocalDocumentRetriever.from_paths(item.reference_documents)
                    ref_docs = retriever.documents
                except Exception:
                    pass

            base_file = self.output_dir / f"{item.prompt_id}_baseline.json"
            net_file = self.output_dir / f"{item.prompt_id}_network.json"

            # 1. Baseline Run (with resume check)
            if resume and base_file.exists():
                b_res = RunResult.model_validate_json(base_file.read_text(encoding="utf-8"))
            else:
                is_compute_matched = mode == "compute_matched_vs_network"
                b_res = await base_runner.run(
                    item.prompt,
                    reference_documents=ref_docs,
                    compute_matched=is_compute_matched,
                )
                base_file.write_text(b_res.model_dump_json(indent=2), encoding="utf-8")

            # 2. Network Run (with resume check)
            if resume and net_file.exists():
                n_res = RunResult.model_validate_json(net_file.read_text(encoding="utf-8"))
            else:
                n_res = await net_runner.solve(
                    item.prompt,
                    reference_documents=ref_docs,
                )
                net_file.write_text(n_res.model_dump_json(indent=2), encoding="utf-8")

            # 3. Objective Rubric Scoring
            b_score = compute_evaluation_score(
                rubric=rubric,
                system_label="baseline",
                answer_text=b_res.answer,
                assumptions=b_res.assumptions,
                missing_information=b_res.missing_information,
                confidence_notes=b_res.confidence_notes,
            )
            n_score = compute_evaluation_score(
                rubric=rubric,
                system_label="network",
                answer_text=n_res.answer,
                assumptions=n_res.assumptions,
                missing_information=n_res.missing_information,
                confidence_notes=n_res.confidence_notes,
            )

            # 4. A/B Blind Anonymization
            blind_it, b_key = anonymize_pair(
                prompt_id=item.prompt_id,
                domain=item.domain,
                prompt=item.prompt,
                baseline_result=b_res,
                network_result=n_res,
                reference_documents=item.reference_documents,
                seed=blind_seed,
            )
            blind_items.append(blind_it)
            blind_keys.append(b_key)

            b_cost = (
                b_res.usage.actual_cost_usd if b_res.usage and b_res.usage.actual_cost_usd else 0.0
            )
            n_cost = (
                n_res.usage.actual_cost_usd if n_res.usage and n_res.usage.actual_cost_usd else 0.0
            )

            exp_result = ExperimentResult(
                prompt_id=item.prompt_id,
                domain=item.domain,
                baseline_run_id=b_res.run_id,
                network_run_id=n_res.run_id,
                baseline_result=b_res,
                network_result=n_res,
                baseline_score=b_score,
                network_score=n_score,
                latency_difference_ms=n_res.latency_ms - b_res.latency_ms,
                cost_difference_usd=(n_cost or 0.0) - (b_cost or 0.0),
                git_commit_sha=git_sha,
            )
            results.append(exp_result)

        # Export Evaluator Packet and Blind Key
        export_evaluator_packet(blind_items, self.output_dir / "evaluator_packet.json")
        key_blob = [k.model_dump() for k in blind_keys]
        (self.output_dir / "blind_key.json").write_text(
            json.dumps(key_blob, indent=2), encoding="utf-8"
        )

        # Export Result Formats
        export_experiment_results_json(results, self.output_dir / "experiment_results.json")
        export_experiment_results_csv(results, self.output_dir / "experiment_results.csv")
        md_report = generate_markdown_report(
            results, title=f"CrewForge Evaluation — {manifest.name}"
        )
        (self.output_dir / "report.md").write_text(md_report, encoding="utf-8")

        return results, md_report
