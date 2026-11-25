"""100-point rubric scoring engine separating deterministic diagnostics from semantic grading."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from crewforge.evaluation.contracts import (
    BlindKeyFile,
    EvaluationRubric,
    EvaluationScore,
    HumanGradeInput,
    ObjectiveCheckResult,
)
from crewforge.evaluation.objective_checks import evaluate_objective_checks

REQUIRED_RUBRIC_FIELDS = (
    "factual_correctness",
    "requirement_coverage",
    "evidence_grounding",
    "internal_consistency",
    "usefulness_and_uncertainty",
)


def compute_objective_diagnostics(
    *,
    rubric: EvaluationRubric,
    system_label: str,
    answer_text: str,
    missing_information: list[str] | None = None,
) -> EvaluationScore:
    """Run deterministic diagnostics only; leave 100-point semantic score pending."""
    obj_results = evaluate_objective_checks(
        rubric, answer_text=answer_text, missing_info=missing_information
    )
    total_obj_score = sum(r.score for r in obj_results)
    total_obj_max = sum(r.max_score for r in obj_results)

    return EvaluationScore(
        prompt_id=rubric.prompt_id,
        system_label=system_label,
        grading_status="pending_independent_grading",
        evaluator_id=None,
        grading_method="pending",
        graded_at=None,
        objective_score=round(total_obj_score, 2),
        objective_max_score=round(total_obj_max, 2),
        objective_checks=obj_results,
        evaluator_notes="Awaiting independent semantic rubric grading.",
        is_fully_objective=False,
    )


def finalize_human_score(
    *,
    rubric: EvaluationRubric,
    system_label: str,
    human_grade: HumanGradeInput,
    objective_checks: list[ObjectiveCheckResult] | None = None,
) -> EvaluationScore:
    """Validate and compute the final 100-point grade from complete independent rubric input."""
    if human_grade.prompt_id != rubric.prompt_id:
        raise ValueError(
            f"Human grade prompt_id '{human_grade.prompt_id}' does not match "
            f"rubric '{rubric.prompt_id}'."
        )

    f_score = human_grade.factual_correctness
    cov_score = human_grade.requirement_coverage
    g_score = human_grade.evidence_grounding
    c_score = human_grade.internal_consistency
    u_score = human_grade.usefulness_and_uncertainty

    # Validate against rubric maximums
    if f_score > rubric.factual_correctness_max:
        raise ValueError(
            f"factual_correctness {f_score} exceeds max {rubric.factual_correctness_max}"
        )
    if cov_score > rubric.requirement_coverage_max:
        raise ValueError(
            f"requirement_coverage {cov_score} exceeds max {rubric.requirement_coverage_max}"
        )
    if g_score > rubric.evidence_grounding_max:
        raise ValueError(
            f"evidence_grounding {g_score} exceeds max {rubric.evidence_grounding_max}"
        )
    if c_score > rubric.internal_consistency_max:
        raise ValueError(
            f"internal_consistency {c_score} exceeds max {rubric.internal_consistency_max}"
        )
    if u_score > rubric.usefulness_and_uncertainty_max:
        raise ValueError(
            f"usefulness_and_uncertainty {u_score} exceeds max "
            f"{rubric.usefulness_and_uncertainty_max}"
        )

    # 5 points per distinct unsupported material assertion, capped at 20 points
    raw_penalty = 5.0 * human_grade.unsupported_material_assertion_count
    penalty = min(raw_penalty, rubric.max_unsupported_assertion_penalty, 20.0)

    raw_total = f_score + cov_score + g_score + c_score + u_score - penalty
    final_total = max(0.0, min(100.0, raw_total))

    checks = objective_checks or []
    total_obj_score = sum(r.score for r in checks)
    total_obj_max = sum(r.max_score for r in checks)

    return EvaluationScore(
        prompt_id=rubric.prompt_id,
        system_label=system_label,
        factual_correctness=round(f_score, 2),
        requirement_coverage=round(cov_score, 2),
        evidence_grounding=round(g_score, 2),
        internal_consistency=round(c_score, 2),
        usefulness_and_uncertainty=round(u_score, 2),
        unsupported_material_assertion_count=human_grade.unsupported_material_assertion_count,
        unsupported_assertion_penalty=round(penalty, 2),
        total_score=round(final_total, 2),
        grading_status="graded",
        evaluator_id=human_grade.evaluator_id,
        grading_method=human_grade.grading_method,
        graded_at=datetime.now(UTC).isoformat(),
        objective_score=round(total_obj_score, 2),
        objective_max_score=round(total_obj_max, 2),
        objective_checks=checks,
        evaluator_notes=human_grade.evaluator_notes,
        is_fully_objective=False,
    )


def compute_evaluation_score(
    *,
    rubric: EvaluationRubric,
    system_label: str,
    answer_text: str,
    assumptions: list[str] | None = None,
    missing_information: list[str] | None = None,
    confidence_notes: str | None = None,
    human_grade: HumanGradeInput | None = None,
) -> EvaluationScore:
    """Compute diagnostics and apply independent semantic grading iff complete human_grade given."""
    diag = compute_objective_diagnostics(
        rubric=rubric,
        system_label=system_label,
        answer_text=answer_text,
        missing_information=missing_information,
    )
    if human_grade is None:
        return diag
    return finalize_human_score(
        rubric=rubric,
        system_label=system_label,
        human_grade=human_grade,
        objective_checks=diag.objective_checks,
    )


def import_human_grades_from_file(
    grades_path: str | Path,
    blind_key_path: str | Path,
) -> dict[tuple[str, str], HumanGradeInput]:
    """Load validated human grades and unblind answer labels using BlindKeyFile."""
    grades_raw = json.loads(Path(grades_path).read_text(encoding="utf-8"))
    if isinstance(grades_raw, dict) and "grades" in grades_raw:
        grades_raw = grades_raw["grades"]
    if not isinstance(grades_raw, list):
        raise ValueError("Grades JSON must be a list of grade objects or {'grades': [...]}.")

    key_file = BlindKeyFile.model_validate_json(Path(blind_key_path).read_text(encoding="utf-8"))
    keys_by_prompt = {k.prompt_id: k for k in key_file.keys}

    resolved: dict[tuple[str, str], HumanGradeInput] = {}
    for entry in grades_raw:
        grade = HumanGradeInput.model_validate(entry)
        if grade.system_label is not None:
            sys_lbl = grade.system_label
        elif grade.answer_label is not None:
            b_key = keys_by_prompt.get(grade.prompt_id)
            if b_key is None:
                raise ValueError(f"No blind key found for prompt_id '{grade.prompt_id}'.")
            sys_lbl = b_key.label_a_system if grade.answer_label == "A" else b_key.label_b_system
        else:
            raise ValueError(
                f"Grade for '{grade.prompt_id}' must specify 'answer_label' or 'system_label'."
            )
        resolved[(grade.prompt_id, sys_lbl)] = grade
    return resolved
