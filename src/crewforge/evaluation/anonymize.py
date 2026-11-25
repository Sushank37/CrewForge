"""A/B anonymization and evaluator packet generation for reproducible blind judging."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from crewforge.evaluation.contracts import (
    BlindAnswer,
    BlindEvaluationItem,
    BlindKey,
    EvaluatorPacket,
)
from crewforge.schemas import RunResult


def strip_orchestrator_metadata(text: str) -> str:
    """Remove internal claim tags, agent names, and internal gate markers."""
    # Strip claim tags: [clm_4d9332138e68@v1], [advisory; clm_...]
    cleaned = re.sub(r"\[(?:advisory;\s*)?clm_[a-f0-9]+(?:@v\d+)?(?:\s*\([^\)]*\))?\]", "", text)
    # Strip requirement reference markers: [req_1], [req_2]
    cleaned = re.sub(r"\[req_\d+\]", "", cleaned)
    # Strip trailing whitespace on lines
    lines = [re.sub(r"\s+$", "", line) for line in cleaned.splitlines()]
    # Normalize multiple blank lines
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def anonymize_pair(
    *,
    prompt_id: str,
    domain: Any,
    prompt: str,
    baseline_result: RunResult,
    network_result: RunResult,
    reference_documents: list[str] | None = None,
    seed: str = "crewforge_eval_2026",
) -> tuple[BlindEvaluationItem, BlindKey]:
    """Create reproducible blind A/B pair using a deterministic hash seed."""
    # Deterministically decide whether A is baseline or network
    h = hashlib.sha256(f"{seed}:{prompt_id}".encode()).hexdigest()
    # If lowest bit is 0: A is baseline, B is network; if 1: A is network, B is baseline
    a_is_network = int(h[-1], 16) % 2 == 1

    clean_base_content = strip_orchestrator_metadata(baseline_result.answer)
    clean_net_content = strip_orchestrator_metadata(network_result.answer)

    base_ans = BlindAnswer(
        label="A" if not a_is_network else "B",
        content=clean_base_content,
        assumptions=baseline_result.assumptions,
        missing_information=baseline_result.missing_information,
        confidence_notes=baseline_result.confidence_notes,
    )
    net_ans = BlindAnswer(
        label="B" if not a_is_network else "A",
        content=clean_net_content,
        assumptions=network_result.assumptions,
        missing_information=network_result.missing_information,
        confidence_notes=network_result.confidence_notes,
    )

    answer_a = net_ans if a_is_network else base_ans
    answer_b = base_ans if a_is_network else net_ans

    item = BlindEvaluationItem(
        prompt_id=prompt_id,
        domain=domain,
        prompt=prompt,
        reference_documents=reference_documents or [],
        answer_a=answer_a,
        answer_b=answer_b,
    )

    key = BlindKey(
        prompt_id=prompt_id,
        label_a_system="network" if a_is_network else "baseline",
        label_b_system="baseline" if a_is_network else "network",
        run_id_a=network_result.run_id if a_is_network else baseline_result.run_id,
        run_id_b=baseline_result.run_id if a_is_network else network_result.run_id,
        salt=h[:8],
    )

    return item, key


def export_evaluator_packet(
    items: list[BlindEvaluationItem],
    output_path: str | Path,
    *,
    packet_id: str = "eval_packet_blind",
) -> EvaluatorPacket:
    """Write blind evaluation packet JSON and clean Markdown presentation for judges."""
    packet = EvaluatorPacket(packet_id=packet_id, items=items)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(packet.model_dump_json(indent=2), encoding="utf-8")

    # Also render clean markdown for human evaluators
    md_lines: list[str] = [
        f"# CrewForge Blind Evaluation Packet — {packet_id}",
        "",
        "Instructions: Evaluate Answer A and Answer B independently using the 100-point rubric.",
        "System identities, run IDs, and model metadata have been stripped.",
        "",
        "---",
        "",
    ]
    for idx, it in enumerate(items, start=1):
        md_lines.append(f"## Case {idx}: [{it.domain.value}] (ID: {it.prompt_id})")
        md_lines.append("")
        md_lines.append(f"**Prompt:**\n> {it.prompt}")
        md_lines.append("")
        if it.reference_documents:
            md_lines.append(
                f"**Attached Reference Documents:** {', '.join(it.reference_documents)}"
            )
            md_lines.append("")
        md_lines.append("### Answer A")
        md_lines.append(it.answer_a.content)
        if it.answer_a.missing_information:
            md_lines.append("\n*Missing Information Identified (A):*")
            for m in it.answer_a.missing_information:
                md_lines.append(f"- {m}")
        md_lines.append("")
        md_lines.append("### Answer B")
        md_lines.append(it.answer_b.content)
        if it.answer_b.missing_information:
            md_lines.append("\n*Missing Information Identified (B):*")
            for m in it.answer_b.missing_information:
                md_lines.append(f"- {m}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_path = out_p.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return packet
