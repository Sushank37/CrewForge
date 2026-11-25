"""100-point rubric scoring engine and penalty evaluation for benchmark answers."""

from __future__ import annotations

import re

from crewforge.evaluation.contracts import (
    EvaluationRubric,
    EvaluationScore,
)
from crewforge.evaluation.objective_checks import evaluate_objective_checks


def compute_evaluation_score(
    *,
    rubric: EvaluationRubric,
    system_label: str,
    answer_text: str,
    assumptions: list[str] | None = None,
    missing_information: list[str] | None = None,
    confidence_notes: str | None = None,
    subjective_override: dict[str, float] | None = None,
) -> EvaluationScore:
    """Compute 100-point score using deterministic checks and rubric weights."""
    obj_results = evaluate_objective_checks(
        rubric, answer_text=answer_text, missing_info=missing_information
    )

    total_obj_score = sum(r.score for r in obj_results)
    total_obj_max = sum(r.max_score for r in obj_results) or 1.0
    obj_fraction = total_obj_score / total_obj_max

    # 1. Factual and numerical correctness (30 points)
    num_results = [r for r in obj_results if r.check_name.startswith("num_")]
    if num_results:
        f_score = sum(r.score for r in num_results) / sum(r.max_score for r in num_results) * 30.0
    else:
        f_score = obj_fraction * 30.0

    # 2. Requirement coverage (25 points)
    # Scaled by fraction of checks passed
    cov_score = obj_fraction * 25.0

    # 3. Evidence and grounding (20 points)
    # Check if factual statements cite evidence or identify limits
    has_grounding = bool(
        re.search(
            r"(\[|cited|excerpt|according to|based on|policy|data)", answer_text, re.IGNORECASE
        )
    )
    g_score = 20.0 if has_grounding else 10.0

    # 4. Internal consistency (15 points)
    logic_results = [r for r in obj_results if r.check_name.startswith("logic_")]
    if logic_results:
        c_score = (
            sum(r.score for r in logic_results) / sum(r.max_score for r in logic_results) * 15.0
        )
    else:
        c_score = 15.0

    # 5. Usefulness and uncertainty handling (10 points)
    miss_results = [r for r in obj_results if r.check_name.startswith("missing_")]
    has_missing = bool(
        missing_information
        or miss_results
        or re.search(r"(missing|unknown|unspecified|need|require)", answer_text, re.IGNORECASE)
    )
    u_score = 10.0 if has_missing else 5.0

    # Unsupported assertion penalty (up to 20 points)
    penalty = 0.0
    for r in logic_results:
        if not r.passed and "Contradiction:" in r.details:
            penalty += 10.0
    penalty = min(penalty, rubric.max_unsupported_assertion_penalty)

    # Allow subjective override if human/independent grading is supplied
    if subjective_override:
        f_score = subjective_override.get("factual_correctness", f_score)
        cov_score = subjective_override.get("requirement_coverage", cov_score)
        g_score = subjective_override.get("evidence_grounding", g_score)
        c_score = subjective_override.get("internal_consistency", c_score)
        u_score = subjective_override.get("usefulness_and_uncertainty", u_score)
        penalty = subjective_override.get("unsupported_assertion_penalty", penalty)

    raw_total = f_score + cov_score + g_score + c_score + u_score - penalty
    final_total = max(0.0, min(100.0, raw_total))

    return EvaluationScore(
        prompt_id=rubric.prompt_id,
        system_label=system_label,
        factual_correctness=round(f_score, 2),
        requirement_coverage=round(cov_score, 2),
        evidence_grounding=round(g_score, 2),
        internal_consistency=round(c_score, 2),
        usefulness_and_uncertainty=round(u_score, 2),
        unsupported_assertion_penalty=round(penalty, 2),
        total_score=round(final_total, 2),
        objective_score=round(total_obj_score, 2),
        objective_max_score=round(total_obj_max, 2),
        objective_checks=obj_results,
        evaluator_notes="",
        is_fully_objective=subjective_override is None,
    )
