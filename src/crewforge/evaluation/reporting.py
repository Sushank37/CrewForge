"""Result export and comparison reporting across JSON, JSONL, CSV, and Markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from crewforge.evaluation.contracts import ExperimentResult


def export_experiment_results_json(results: list[ExperimentResult], path: str | Path) -> None:
    """Export complete experiment results with raw RunResult outputs to JSON."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    blob = [r.model_dump(mode="json") for r in results]
    out_p.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def export_experiment_results_jsonl(results: list[ExperimentResult], path: str | Path) -> None:
    """Export results as JSONL for streaming ingestion."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")


def export_experiment_results_csv(results: list[ExperimentResult], path: str | Path) -> None:
    """Export benchmark comparison summary to CSV."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "domain",
        "experiment_fingerprint",
        "grading_status",
        "baseline_run_id",
        "baseline_model",
        "baseline_status",
        "baseline_score",
        "baseline_cost_usd",
        "baseline_tokens",
        "baseline_latency_ms",
        "network_run_id",
        "network_model",
        "network_status",
        "network_score",
        "network_cost_usd",
        "network_tokens",
        "network_latency_ms",
        "score_delta",
        "cost_delta_usd",
    ]
    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            b_graded = (
                r.baseline_score is not None
                and r.baseline_score.grading_status == "graded"
                and r.baseline_score.total_score is not None
            )
            n_graded = (
                r.network_score is not None
                and r.network_score.grading_status == "graded"
                and r.network_score.total_score is not None
            )
            b_score_val = (
                r.baseline_score.total_score if b_graded and r.baseline_score is not None else None
            )
            n_score_val = (
                r.network_score.total_score if n_graded and r.network_score is not None else None
            )
            b_cost = r.baseline_result.usage.actual_cost_usd if r.baseline_result.usage else 0.0
            n_cost = r.network_result.usage.actual_cost_usd if r.network_result.usage else 0.0
            b_tok = r.baseline_result.usage.total_tokens if r.baseline_result.usage else 0
            n_tok = r.network_result.usage.total_tokens if r.network_result.usage else 0

            delta_val = (
                round(n_score_val - b_score_val, 2)
                if b_score_val is not None and n_score_val is not None
                else "pending_independent_grading"
            )

            writer.writerow(
                {
                    "prompt_id": r.prompt_id,
                    "domain": r.domain.value,
                    "experiment_fingerprint": r.experiment_fingerprint,
                    "grading_status": r.grading_status,
                    "baseline_run_id": r.baseline_run_id,
                    "baseline_model": f"{r.baseline_provider}/{r.baseline_model}",
                    "baseline_status": r.baseline_result.status.value,
                    "baseline_score": (
                        b_score_val if b_score_val is not None else "pending_independent_grading"
                    ),
                    "baseline_cost_usd": b_cost or 0.0,
                    "baseline_tokens": b_tok,
                    "baseline_latency_ms": round(r.baseline_result.latency_ms, 2),
                    "network_run_id": r.network_run_id,
                    "network_model": f"{r.network_provider}/{r.network_model}",
                    "network_status": r.network_result.status.value,
                    "network_score": (
                        n_score_val if n_score_val is not None else "pending_independent_grading"
                    ),
                    "network_cost_usd": n_cost or 0.0,
                    "network_tokens": n_tok,
                    "network_latency_ms": round(r.network_result.latency_ms, 2),
                    "score_delta": delta_val,
                    "cost_delta_usd": round((n_cost or 0.0) - (b_cost or 0.0), 6),
                }
            )


def generate_markdown_report(
    results: list[ExperimentResult], *, title: str = "Benchmark Evaluation Report"
) -> str:
    """Generate Markdown summary table without fabricating ungraded wins/losses."""
    lines: list[str] = [
        f"# {title}",
        "",
        (
            "| Prompt ID | Domain | Baseline Status | Base Score | "
            "Network Status | Net Score | Delta | Base Cost | Net Cost | Latency Diff |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    graded_b_scores: list[float] = []
    graded_n_scores: list[float] = []
    total_b_cost = 0.0
    total_n_cost = 0.0

    for r in results:
        b_score_val = (
            r.baseline_score.total_score
            if r.baseline_score is not None and r.baseline_score.grading_status == "graded"
            else None
        )
        n_score_val = (
            r.network_score.total_score
            if r.network_score is not None and r.network_score.grading_status == "graded"
            else None
        )
        b_cost = (
            r.baseline_result.usage.actual_cost_usd
            if r.baseline_result.usage and r.baseline_result.usage.actual_cost_usd
            else 0.0
        )
        n_cost = (
            r.network_result.usage.actual_cost_usd
            if r.network_result.usage and r.network_result.usage.actual_cost_usd
            else 0.0
        )

        total_b_cost += b_cost
        total_n_cost += n_cost

        b_str = f"{b_score_val:.1f}" if b_score_val is not None else "pending independent grading"
        n_str = f"{n_score_val:.1f}" if n_score_val is not None else "pending independent grading"

        if b_score_val is not None and n_score_val is not None:
            graded_b_scores.append(b_score_val)
            graded_n_scores.append(n_score_val)
            delta = n_score_val - b_score_val
            delta_str = f"**{delta:+.1f}**"
        else:
            delta_str = "pending"

        lines.append(
            f"| `{r.prompt_id}` | {r.domain.value} | {r.baseline_result.status.value} | "
            f"{b_str} | {r.network_result.status.value} | {n_str} | "
            f"{delta_str} | ${b_cost:.4f} | ${n_cost:.4f} | "
            f"{round(r.latency_difference_ms / 1000.0, 1)}s |"
        )

    lines.append("")
    lines.append(f"**Graded Cases:** {len(graded_b_scores)} / {len(results)}")
    if graded_b_scores:
        avg_b = sum(graded_b_scores) / len(graded_b_scores)
        avg_n = sum(graded_n_scores) / len(graded_n_scores)
        lines.append(f"**Mean Baseline Score (Graded):** {avg_b:.2f} / 100")
        lines.append(
            f"**Mean Network Score (Graded):** {avg_n:.2f} / 100 (Delta: {avg_n - avg_b:+.2f})"
        )
    else:
        lines.append("**Grading Status:** Pending independent semantic grading.")
    lines.append(f"**Total Baseline Spend:** ${total_b_cost:.6f}")
    lines.append(f"**Total Network Spend:** ${total_n_cost:.6f}")
    lines.append("")

    return "\n".join(lines)
