"""Deterministic objective scoring checks for benchmark answers."""

from __future__ import annotations

import re

from crewforge.evaluation.contracts import (
    EvaluationRubric,
    ExpectedLogicalCheck,
    ExpectedMissingInfoCheck,
    ExpectedNumericalCheck,
    ObjectiveCheckResult,
)
from crewforge.tools.arithmetic import extract_numbers_from_text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def run_numerical_check(check: ExpectedNumericalCheck, answer_text: str) -> ObjectiveCheckResult:
    """Evaluate whether an expected numeric value is present with units/percentage."""
    found_nums = extract_numbers_from_text(answer_text)
    norm = _normalize_text(answer_text)

    target_val = check.expected_value
    tol = check.tolerance
    passed = False
    details = ""

    # Check for direct decimal match
    for num in found_nums:
        f_num = float(num)
        if abs(f_num - target_val) <= tol or (
            target_val != 0 and abs((f_num - target_val) / target_val) <= tol
        ):
            passed = True
            break
        # If check specifies percentage points vs relative percent
        if check.is_percentage and abs(f_num * 100.0 - target_val) <= tol:
            passed = True
            break

    # If unit check is required, verify unit token appears near the matched number
    if passed and check.expected_units:
        unit_token = check.expected_units.lower()
        if unit_token not in norm:
            passed = False
            details = (
                f"Value {target_val} found but missing required units '{check.expected_units}'."
            )

    # Verify percentage-point distinction if check specifies it
    if (
        passed
        and check.is_percentage_points
        and ("percentage point" not in norm and "pp" not in norm and "percentage-point" not in norm)
    ):
        passed = False
        details = f"Value {target_val} found but failed percentage-point distinction."

    if passed:
        details = f"Verified expected value {target_val} within tolerance."
        score = check.points
    else:
        if not details:
            details = f"Expected {target_val} (units='{check.expected_units}') not found in answer."
        score = 0.0

    return ObjectiveCheckResult(
        check_name=f"num_{check.variable_name}",
        passed=passed,
        score=score,
        max_score=check.points,
        details=details,
    )


def run_logical_check(check: ExpectedLogicalCheck, answer_text: str) -> ObjectiveCheckResult:
    """Check required conclusions, forbidden contradictions, and explicit assumptions."""
    norm = _normalize_text(answer_text)
    passed = True
    reasons: list[str] = []

    # 1. Required phrases or concepts
    for phrase in check.required_phrases:
        if phrase.lower() not in norm:
            passed = False
            reasons.append(f"Missing required conclusion concept: '{phrase}'.")

    # 2. Forbidden phrases (e.g., claiming proof when not proven)
    for forbidden in check.forbidden_phrases:
        if forbidden.lower() in norm:
            passed = False
            reasons.append(f"Contradiction: found forbidden phrase '{forbidden}'.")

    # 3. Required assumptions or conditions
    for req_assump in check.required_assumptions_or_conditions:
        if req_assump.lower() not in norm:
            passed = False
            reasons.append(f"Missing required qualification/assumption: '{req_assump}'.")

    score = check.points if passed else 0.0
    details = "Passed logical check." if passed else "; ".join(reasons)

    return ObjectiveCheckResult(
        check_name=f"logic_{check.check_id}",
        passed=passed,
        score=score,
        max_score=check.points,
        details=details,
    )


def run_missing_info_check(
    check: ExpectedMissingInfoCheck, answer_text: str, missing_list: list[str]
) -> ObjectiveCheckResult:
    """Check that important unstated or missing context was explicitly surfaced."""
    combined_missing = " ".join([answer_text, *missing_list]).lower()
    passed = True
    reasons: list[str] = []

    for concept in check.required_missing_concepts:
        if concept.lower() not in combined_missing:
            passed = False
            reasons.append(f"Did not identify missing context concept: '{concept}'.")

    score = check.points if passed else 0.0
    details = "Identified all expected missing context items." if passed else "; ".join(reasons)

    return ObjectiveCheckResult(
        check_name=f"missing_{check.check_id}",
        passed=passed,
        score=score,
        max_score=check.points,
        details=details,
    )


def evaluate_objective_checks(
    rubric: EvaluationRubric,
    answer_text: str,
    missing_info: list[str] | None = None,
) -> list[ObjectiveCheckResult]:
    """Execute all deterministic assertions defined in the rubric."""
    results: list[ObjectiveCheckResult] = []
    missing_list = missing_info or []

    for num_chk in rubric.numerical_checks:
        results.append(run_numerical_check(num_chk, answer_text))

    for log_chk in rubric.logical_checks:
        results.append(run_logical_check(log_chk, answer_text))

    for miss_chk in rubric.missing_info_checks:
        results.append(run_missing_info_check(miss_chk, answer_text, missing_list))

    return results
