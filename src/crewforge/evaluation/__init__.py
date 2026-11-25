"""CrewForge competitive evaluation, blind benchmarking, and scoring framework."""

from __future__ import annotations

from crewforge.evaluation.ablation import AblationConfig
from crewforge.evaluation.anonymize import (
    anonymize_pair,
    export_blind_key_file,
    export_evaluator_packet,
    generate_blinding_seed,
    sanitize_confidence_notes,
    strip_orchestrator_metadata,
)
from crewforge.evaluation.contracts import (
    AblationVariant,
    BenchmarkItem,
    BenchmarkManifest,
    BlindEvaluationItem,
    BlindKey,
    BlindKeyFile,
    CachedRunRecord,
    EvaluationRubric,
    EvaluationScore,
    ExpectedLogicalCheck,
    ExpectedMissingInfoCheck,
    ExpectedNumericalCheck,
    ExperimentFingerprint,
    ExperimentResult,
    HumanGradeInput,
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
from crewforge.evaluation.runner import (
    BenchmarkRunner,
    build_david_vs_goliath_settings,
    load_and_validate_manifest,
    resolve_item_documents,
)
from crewforge.evaluation.scoring import (
    compute_evaluation_score,
    compute_objective_diagnostics,
    finalize_human_score,
    import_human_grades_from_file,
)

__all__ = [
    "AblationConfig",
    "AblationVariant",
    "BenchmarkItem",
    "BenchmarkManifest",
    "BenchmarkRunner",
    "BlindEvaluationItem",
    "BlindKey",
    "BlindKeyFile",
    "CachedRunRecord",
    "EvaluationRubric",
    "EvaluationScore",
    "ExpectedLogicalCheck",
    "ExpectedMissingInfoCheck",
    "ExpectedNumericalCheck",
    "ExperimentFingerprint",
    "ExperimentResult",
    "HumanGradeInput",
    "ObjectiveCheckResult",
    "PromptDifficulty",
    "PromptDomain",
    "anonymize_pair",
    "build_david_vs_goliath_settings",
    "compute_evaluation_score",
    "compute_objective_diagnostics",
    "evaluate_objective_checks",
    "export_blind_key_file",
    "export_evaluator_packet",
    "export_experiment_results_csv",
    "export_experiment_results_json",
    "export_experiment_results_jsonl",
    "finalize_human_score",
    "generate_blinding_seed",
    "generate_markdown_report",
    "import_human_grades_from_file",
    "load_and_validate_manifest",
    "resolve_item_documents",
    "sanitize_confidence_notes",
    "strip_orchestrator_metadata",
]
