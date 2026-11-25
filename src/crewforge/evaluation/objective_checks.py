"""Deterministic objective diagnostic checks for benchmark answers."""

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

_NEGATION_OR_REJECTION_RE = re.compile(
    r"\b(not|never|rejected|incorrect|false|wrong|invalid|contradicted|erroneous|disproved)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _split_clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<=[.!?\n;])\s+|\n+", text) if c.strip()]


def run_numerical_check(check: ExpectedNumericalCheck, answer_text: str) -> ObjectiveCheckResult:
    """Run deterministic numerical diagnostic check with unit & negation validation."""
    clauses = _split_clauses(answer_text)
    target_val = check.expected_value
    tol = check.tolerance

    valid_matches: list[str] = []
    ambiguous_matches: list[str] = []
    rejected_matches: list[str] = []

    for clause in clauses:
        clause_nums = extract_numbers_from_text(clause)
        clause_norm = _normalize_text(clause)
        matched_in_clause = False
        for num in clause_nums:
            f_num = float(num)
            if abs(f_num - target_val) <= tol or (
                target_val != 0 and abs((f_num - target_val) / target_val) <= tol
            ):
                matched_in_clause = True
                break
            if check.is_percentage and abs(f_num * 100.0 - target_val) <= tol:
                matched_in_clause = True
                break

        if not matched_in_clause:
            continue

        # Check if clause explicitly rejects or negates the number
        if _NEGATION_OR_REJECTION_RE.search(clause_norm):
            rejected_matches.append(clause[:120])
            continue

        # Validate unit association in the SAME clause
        if check.expected_units:
            unit_token = check.expected_units.lower()
            if unit_token not in clause_norm:
                ambiguous_matches.append(
                    f"Matched {target_val} without unit '{check.expected_units}' in clause: "
                    f"{clause[:100]}"
                )
                continue

        # Validate percentage-point distinction in the SAME clause
        if check.is_percentage_points and (
            "percentage point" not in clause_norm
            and "pp" not in clause_norm
            and "percentage-point" not in clause_norm
        ):
            ambiguous_matches.append(
                f"Matched {target_val} without percentage-point distinction in clause: "
                f"{clause[:100]}"
            )
            continue

        valid_matches.append(clause[:120])

    if valid_matches:
        return ObjectiveCheckResult(
            check_name=f"num_{check.variable_name}",
            passed=True,
            score=check.points,
            max_score=check.points,
            status="passed",
            diagnostic_only=False,
            needs_manual_review=False,
            candidate_matches=valid_matches,
            details=f"Diagnostic verified expected value {target_val} with unit/context match.",
        )

    if ambiguous_matches:
        return ObjectiveCheckResult(
            check_name=f"num_{check.variable_name}",
            passed=False,
            score=0.0,
            max_score=check.points,
            status="ambiguous_manual_review",
            diagnostic_only=True,
            needs_manual_review=True,
            candidate_matches=ambiguous_matches,
            details=(
                f"Ambiguous candidate match for {target_val} (units='{check.expected_units}'); "
                "marked for manual evaluation."
            ),
        )

    if rejected_matches:
        return ObjectiveCheckResult(
            check_name=f"num_{check.variable_name}",
            passed=False,
            score=0.0,
            max_score=check.points,
            status="failed",
            diagnostic_only=False,
            needs_manual_review=False,
            candidate_matches=rejected_matches,
            details=(
                f"Expected {target_val} appeared only in negated/rejected context; "
                "cannot establish numerical correctness."
            ),
        )

    return ObjectiveCheckResult(
        check_name=f"num_{check.variable_name}",
        passed=False,
        score=0.0,
        max_score=check.points,
        status="failed",
        diagnostic_only=False,
        needs_manual_review=False,
        details=f"Expected {target_val} (units='{check.expected_units}') not found in answer.",
    )


def run_logical_check(check: ExpectedLogicalCheck, answer_text: str) -> ObjectiveCheckResult:
    """Diagnostic phrase heuristic only — never treated as verified semantic correctness."""
    norm = _normalize_text(answer_text)
    passed = True
    reasons: list[str] = []

    for phrase in check.required_phrases:
        if phrase.lower() not in norm:
            passed = False
            reasons.append(f"Missing diagnostic phrase: '{phrase}'.")

    for forbidden in check.forbidden_phrases:
        if forbidden.lower() in norm:
            passed = False
            reasons.append(f"Contradiction candidate: found forbidden phrase '{forbidden}'.")

    for req_assump in check.required_assumptions_or_conditions:
        if req_assump.lower() not in norm:
            passed = False
            reasons.append(f"Missing diagnostic qualification/assumption: '{req_assump}'.")

    score = check.points if passed else 0.0
    prefix = "Diagnostic phrase check only (requires independent semantic grading): "
    details = prefix + "all heuristic phrases matched." if passed else prefix + "; ".join(reasons)

    return ObjectiveCheckResult(
        check_name=f"logic_{check.check_id}",
        passed=passed,
        score=score,
        max_score=check.points,
        status="diagnostic_only",
        diagnostic_only=True,
        needs_manual_review=True,
        details=details,
    )


def run_missing_info_check(
    check: ExpectedMissingInfoCheck, answer_text: str, missing_list: list[str]
) -> ObjectiveCheckResult:
    """Diagnostic check that missing-context keywords appeared (requires semantic review)."""
    combined_missing = " ".join([answer_text, *missing_list]).lower()
    passed = True
    reasons: list[str] = []

    for concept in check.required_missing_concepts:
        if concept.lower() not in combined_missing:
            passed = False
            reasons.append(f"Did not surface missing-context term: '{concept}'.")

    score = check.points if passed else 0.0
    prefix = "Diagnostic missing-info check only (requires independent semantic grading): "
    details = (
        prefix + "candidate missing-context terms present."
        if passed
        else prefix + "; ".join(reasons)
    )

    return ObjectiveCheckResult(
        check_name=f"missing_{check.check_id}",
        passed=passed,
        score=score,
        max_score=check.points,
        status="diagnostic_only",
        diagnostic_only=True,
        needs_manual_review=True,
        details=details,
    )


def evaluate_objective_checks(
    rubric: EvaluationRubric,
    answer_text: str,
    missing_info: list[str] | None = None,
) -> list[ObjectiveCheckResult]:
    """Execute all deterministic diagnostic assertions defined in the rubric."""
    results: list[ObjectiveCheckResult] = []
    missing_list = missing_info or []

    for num_chk in rubric.numerical_checks:
        results.append(run_numerical_check(num_chk, answer_text))

    for log_chk in rubric.logical_checks:
        results.append(run_logical_check(log_chk, answer_text))

    for miss_chk in rubric.missing_info_checks:
        results.append(run_missing_info_check(miss_chk, answer_text, missing_list))

    return results
