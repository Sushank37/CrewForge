"""A/B anonymization and evaluator packet generation for reproducible blind judging."""

from __future__ import annotations

import hashlib
import re
import secrets
from pathlib import Path
from typing import Any

from crewforge.evaluation.contracts import (
    BlindAnswer,
    BlindEvaluationItem,
    BlindKey,
    BlindKeyFile,
    EvaluatorPacket,
)
from crewforge.schemas import RunResult

# Matches bracketed internal IDs including multi-ref lists like:
# [clm_123@v1, clm_456@v2], [advisory; clm_123@v1; clm_456@v2], [req_1, req_2], [evd_123], [tlx_123]
_BRACKETED_INTERNAL_RE = re.compile(
    r"\[(?:[^\[\]]*\b(?:clm_|evd_|tlx_|req_\d+|up_\d+|doc_\d+|advisory;)[^\[\]]*)\]",
    re.IGNORECASE,
)

# Standalone internal IDs
_STANDALONE_ID_RE = re.compile(
    r"\b(?:run_[a-z0-9_]{6,}|clm_[a-z0-9_]{4,}(?:@v\d+)?|evd_[a-z0-9_]{4,}|"
    r"tlx_[a-z0-9_]{4,}|agent_[a-z0-9_]{4,})\b",
    re.IGNORECASE,
)

# CrewForge component / model / provider orchestration metadata
_ORCHESTRATION_TOKEN_RE = re.compile(
    r"\b(?:Honesty\s*Gate|ReviewerPool|LeadSynthesizer|SpecialistWorker|NetworkRunner|"
    r"BaselineRunner|ModelExecutor|CrewForge|honesty_gate_passed\s*=\s*\w+|"
    r"gate_passed\s*=\s*\w+|supported_claim_count\s*=\s*\d+|"
    r"material_claim_count\s*=\s*\d+|verification_status\s*=\s*\w+|"
    r"qwen3:4b|gpt-6-luna|gpt-6\.2-astra|fake-model-v1)\b",
    re.IGNORECASE,
)


def generate_blinding_seed() -> str:
    """Generate a cryptographically random private blinding seed once per experiment."""
    return secrets.token_hex(32)


def strip_orchestrator_metadata(text: str) -> str:
    """Remove internal claim tags, run IDs, model names, and CrewForge verification markers."""
    if not text:
        return ""
    cleaned = _BRACKETED_INTERNAL_RE.sub("", text)
    cleaned = _STANDALONE_ID_RE.sub("", cleaned)
    cleaned = _ORCHESTRATION_TOKEN_RE.sub("", cleaned)
    # Clean up leftover empty parentheses or repeated spaces created by removals
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip(" \t") for line in cleaned.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def sanitize_confidence_notes(notes: str | None) -> str | None:
    """Strip CrewForge-specific verification metadata from confidence notes."""
    if not notes:
        return None
    # Split on sentences/semicolons and drop clauses that are pure orchestration metadata
    parts = re.split(r"(?<=[.;])\s+|\n+", notes)
    kept: list[str] = []
    for part in parts:
        if _ORCHESTRATION_TOKEN_RE.search(part) or _STANDALONE_ID_RE.search(part):
            stripped = strip_orchestrator_metadata(part)
            # Only keep if substantive words remain (not just punctuation/fragments)
            if len(re.findall(r"[A-Za-z]{3,}", stripped)) >= 4:
                kept.append(stripped)
        else:
            cleaned = strip_orchestrator_metadata(part)
            if cleaned:
                kept.append(cleaned)
    joined = " ".join(kept).strip()
    return joined or None


def anonymize_pair(
    *,
    prompt_id: str,
    domain: Any,
    prompt: str,
    baseline_result: RunResult,
    network_result: RunResult,
    reference_documents: list[str] | None = None,
    reference_document_contents: dict[str, str] | None = None,
    seed: str | None = None,
) -> tuple[BlindEvaluationItem, BlindKey]:
    """Create reproducible blind A/B pair using a private random seed."""
    resolved_seed = seed or generate_blinding_seed()
    h = hashlib.sha256(f"{resolved_seed}:{prompt_id}".encode()).hexdigest()
    # If lowest bit is 1: A is network, B is baseline; if 0: A is baseline, B is network
    a_is_network = int(h[-1], 16) % 2 == 1

    clean_base_content = strip_orchestrator_metadata(baseline_result.answer)
    clean_net_content = strip_orchestrator_metadata(network_result.answer)

    base_assumptions = [
        s for a in baseline_result.assumptions if (s := strip_orchestrator_metadata(a))
    ]
    net_assumptions = [
        s for a in network_result.assumptions if (s := strip_orchestrator_metadata(a))
    ]
    base_missing = [
        s for m in baseline_result.missing_information if (s := strip_orchestrator_metadata(m))
    ]
    net_missing = [
        s for m in network_result.missing_information if (s := strip_orchestrator_metadata(m))
    ]

    base_ans = BlindAnswer(
        label="A" if not a_is_network else "B",
        content=clean_base_content,
        assumptions=base_assumptions,
        missing_information=base_missing,
        confidence_notes=sanitize_confidence_notes(baseline_result.confidence_notes),
    )
    net_ans = BlindAnswer(
        label="B" if not a_is_network else "A",
        content=clean_net_content,
        assumptions=net_assumptions,
        missing_information=net_missing,
        confidence_notes=sanitize_confidence_notes(network_result.confidence_notes),
    )

    answer_a = net_ans if a_is_network else base_ans
    answer_b = base_ans if a_is_network else net_ans

    item = BlindEvaluationItem(
        prompt_id=prompt_id,
        domain=domain,
        prompt=prompt,
        reference_documents=reference_documents or [],
        reference_document_contents=reference_document_contents or {},
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


def export_blind_key_file(
    keys: list[BlindKey],
    *,
    blinding_seed: str,
    output_path: str | Path,
    experiment_fingerprint: str = "",
) -> BlindKeyFile:
    """Persist the private unblinding key file separately from evaluator packets."""
    key_file = BlindKeyFile(
        experiment_fingerprint=experiment_fingerprint,
        blinding_seed=blinding_seed,
        keys=keys,
    )
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(key_file.model_dump_json(indent=2), encoding="utf-8")
    return key_file


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

    md_lines: list[str] = [
        f"# Blind Evaluation Packet — {packet_id}",
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
        for doc_name, doc_content in it.reference_document_contents.items():
            md_lines.append(f"#### Reference Document: `{doc_name}`")
            md_lines.append("```markdown")
            md_lines.append(doc_content.strip())
            md_lines.append("```")
            md_lines.append("")

        for lbl, ans in (("A", it.answer_a), ("B", it.answer_b)):
            md_lines.append(f"### Answer {lbl}")
            md_lines.append(ans.content)
            if ans.assumptions:
                md_lines.append(f"\n*Assumptions ({lbl}):*")
                for a in ans.assumptions:
                    md_lines.append(f"- {a}")
            if ans.missing_information:
                md_lines.append(f"\n*Missing Information Identified ({lbl}):*")
                for m in ans.missing_information:
                    md_lines.append(f"- {m}")
            if ans.confidence_notes:
                md_lines.append(f"\n*Confidence Notes ({lbl}):* {ans.confidence_notes}")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    md_path = out_p.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return packet
