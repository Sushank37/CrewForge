"""CrewForge competitive evaluation, blind benchmarking, and scoring framework."""

from __future__ import annotations

from crewforge.evaluation.ablation import AblationConfig
from crewforge.evaluation.anonymize import (
    anonymize_pair,
    export_evaluator_packet,
    strip_orchestrator_metadata,
)
from crewforge.evaluation.contracts import (
    AblationVariant,
    BenchmarkItem,
    BenchmarkManifest,
    BlindEvaluationItem,
    BlindKey,
    EvaluationRubric,
    EvaluationScore,
    ExpectedLogicalCheck,
    ExpectedMissingInfoCheck,
    ExpectedNumericalCheck,
    ExperimentResult,
    ObjectiveCheckResult,
    PromptDifficulty,
    PromptDomain,
)
from crewforge.evaluation.objective_checks import evaluate_objective_checks
from crewforge.evaluation.reporting import (
    export_experiment_results_csv,
    export_experiment_results_json,
    export_experiment_results_jsonl,
    generate_markdown_report,
)
from crewforge.evaluation.runner import BenchmarkRunner
from crewforge.evaluation.scoring import compute_evaluation_score

__all__ = [
    "AblationConfig",
    "AblationVariant",
    "BenchmarkItem",
    "BenchmarkManifest",
    "BenchmarkRunner",
    "BlindEvaluationItem",
    "BlindKey",
    "EvaluationRubric",
    "EvaluationScore",
    "ExpectedLogicalCheck",
    "ExpectedMissingInfoCheck",
    "ExpectedNumericalCheck",
    "ExperimentResult",
    "ObjectiveCheckResult",
    "PromptDifficulty",
    "PromptDomain",
    "anonymize_pair",
    "compute_evaluation_score",
    "evaluate_objective_checks",
    "export_evaluator_packet",
    "export_experiment_results_csv",
    "export_experiment_results_json",
    "export_experiment_results_jsonl",
    "generate_markdown_report",
    "strip_orchestrator_metadata",
]
