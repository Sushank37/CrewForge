"""Safe, topic-agnostic deterministic arithmetic tool with strict provenance lineage."""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from crewforge.schemas import (
    ArithmeticRequest,
    Claim,
    ClaimCategory,
    ClaimStatus,
    Evidence,
    ExecutionEvent,
    ToolExecutionRecord,
)
from crewforge.storage.database import Database

_MAX_EXPRESSION_CHARS = 256
_MAX_AST_DEPTH = 12
_MAX_AST_NODES = 64
_MAX_MAGNITUDE = Decimal("1e18")
_MAX_POWER_ABS = Decimal("100")

_CURRENCY_CHARS = "$₹€£¥"
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![A-Za-z0-9_])"
)
_WORD_RE = re.compile(r"[a-z]+")

_SOURCE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[kKmMbB](?![A-Za-z])|\s*%)?(?![A-Za-z0-9_])"
)
_CURRENCY_UNITS = {"$": "usd", "₹": "inr", "€": "eur", "£": "gbp", "¥": "jpy"}
_FIXED_CONSTANTS = {
    "zero": Decimal(0),
    "one": Decimal(1),
    "percent_scale": Decimal(100),
    "percentage_scale": Decimal(100),
    "percent_factor": Decimal(100),
    "percent_conversion": Decimal(100),
    "months_per_year": Decimal(12),
    "hours_per_day": Decimal(24),
    "minutes_per_hour": Decimal(60),
    "seconds_per_minute": Decimal(60),
    "seconds_per_hour": Decimal(3600),
    "milliseconds_per_second": Decimal(1000),
}
_FAIR_COIN = re.compile(
    r"(?<!not )\bfair (?:(?:and )?independent )?coins?\b|\bcoins? (?:is|are) fair\b", re.I,
)
_ISO_DATE = re.compile(r"(?<![\w-])([0-9]{4})-([0-9]{2})-([0-9]{2})(?![\w-])")
_COIN_INDEPENDENCE = re.compile(
    r"(?<!not )\bindependent (?:fair )?coins?\b|"
    r"\bcoin (?:tosses|flips) (?:are )?(?:fair and )?independent\b",
    re.I,
)
_BINOMIAL_FUNCTIONS = {"binomial_pmf", "binomial_cdf", "binomial_sf"}

_TIME_CONVERSIONS = {
    "months_per_year": ("years", "months"),
    "hours_per_day": ("days", "hours"),
    "minutes_per_hour": ("hours", "minutes"),
    "seconds_per_minute": ("minutes", "seconds"),
    "seconds_per_hour": ("hours", "seconds"),
    "milliseconds_per_second": ("seconds", "ms"),
}
_EMPIRICAL_ROLES = frozenset({
    "budget", "charge", "cost", "count", "fee", "price", "rate", "revenue", "volume",
})
_ROLE_WORDS = _EMPIRICAL_ROLES | {
    "amount", "case", "forecast", "limit", "scenario", "value",
}


_EXACT_UNIT_NORMALIZATIONS = {
    "date": "date",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "sec": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "min": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "hr": "hours",
    "hour": "hours",
    "hours": "hours",
    "day": "days",
    "days": "days",
    "month": "months",
    "months": "months",
    "yr": "years",
    "year": "years",
    "years": "years",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "gbp": "gbp",
    "jpy": "jpy",
    "inr": "inr",
    "rupee": "inr",
    "rupees": "inr",
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "point": "points",
    "points": "points",
    "gr": "g",
    "g": "g",
    "kg": "kg",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "ratio": "ratio",
    "probability": "ratio",
}


class UnsafeExpressionError(ValueError):
    """Raised when an arithmetic expression uses disallowed syntax or exceeds bounds."""


class MalformedExpressionError(ValueError):
    """Raised when an arithmetic expression or numeric literal is malformed."""


@dataclass
class ProvenanceTarget:
    """Trusted source target resolved by application provenance catalog."""

    alias: str
    kind: str  # prompt_span | verified_evidence | supported_claim | tool_result | assumption
    target_id: str
    text: str
    units: str = ""
    is_trusted: bool = True
    rejection_reason: str | None = None
    numeric_value: Decimal | None = None
    tool_execution_id: str | None = None
    counted_noun: str = ""
    description: str = ""
    source_type: str = ""


@dataclass
class ProvenanceCatalog:
    """Run-scoped mapping from compact model-facing aliases to verified provenance targets."""

    original_prompt: str = ""
    targets_by_key: dict[str, ProvenanceTarget] = field(default_factory=dict)
    prompt_clauses: list[ProvenanceTarget] = field(default_factory=list)
    assumptions: set[str] = field(default_factory=set)


    def prompt_summary_lines(self) -> list[str]:
        lines: list[str] = []
        for target in self.prompt_clauses:
            lines.append(f"- [{target.alias}] {target.text}")
        for key, target in self.targets_by_key.items():
            if key != target.alias or not target.is_trusted:
                continue
            if target.kind == "verified_evidence":
                if target.source_type == "user_prompt" and target.text in self.original_prompt:
                    continue
                lines.append(f"- [{key}] {target.text}")
            elif target.kind == "supported_claim":
                lines.append(f"- [{key}] {target.description}")
        return lines


def split_prompt_clauses(prompt: str) -> list[str]:
    """Split input prompt deterministically into compact numbered clauses."""
    raw_parts = re.split(r"(?<=[.!?;])\s+|\n+", prompt.strip())
    clauses = [part.strip() for part in raw_parts if part.strip()]
    if not clauses and prompt.strip():
        clauses = [prompt.strip()]
    return clauses


def build_provenance_catalog(
    *,
    original_prompt: str,
    verified_evidence: list[Evidence] | None = None,
    all_claims: list[Claim] | None = None,
    tool_records: list[ToolExecutionRecord] | None = None,
    assumptions: list[str] | None = None,
    run_id: str | None = None,
) -> ProvenanceCatalog:
    """Construct strict provenance catalog for exact variable-to-source resolution."""
    catalog = ProvenanceCatalog(
        original_prompt=original_prompt,
        assumptions={a.strip().lower() for a in (assumptions or []) if a.strip()},
    )

    for idx, clause in enumerate(split_prompt_clauses(original_prompt), start=1):
        alias = f"up_{idx}"
        target = ProvenanceTarget(
            alias=alias,
            kind="prompt_span",
            target_id=alias,
            text=clause,
            is_trusted=True,
        )
        if catalog.prompt_clauses and re.match(r"Of (these|those)\b", clause, re.I):
            previous = catalog.prompt_clauses[-1].text
            preceding = list(_SOURCE_NUMBER_RE.finditer(previous))
            if len(preceding) == 1:
                noun = re.match(r"\s+([a-z]+s)\b", previous[preceding[0].end():], re.I)
                if noun:
                    target.counted_noun = noun[1].lower()
        catalog.prompt_clauses.append(target)
        catalog.targets_by_key[alias] = target

    doc_source_texts: dict[str, list[str]] = {}
    doc_source_trusted: dict[str, bool] = {}
    doc_alias_by_source: dict[str, str] = {}
    explicit_doc_refs = {
        e.source_ref for e in verified_evidence or []
        if (run_id is None or e.run_id == run_id) and re.fullmatch(r"doc_[1-9]\d*", e.source_ref)
        and not str(e.created_by).startswith("agt_")
        and e.source_type == "external_retrieved"
        and not (e.source_location or "").startswith("excerpt:")
    }
    used_doc_aliases: set[str] = set()
    next_doc_idx = 1

    def _alloc_doc_alias(source_ref: str) -> str:
        nonlocal next_doc_idx
        if source_ref in doc_alias_by_source:
            return doc_alias_by_source[source_ref]
        if (
            re.fullmatch(r"doc_[1-9]\d*", source_ref)
            and source_ref not in used_doc_aliases
        ):
            doc_alias_by_source[source_ref] = source_ref
            used_doc_aliases.add(source_ref)
            return source_ref
        while f"doc_{next_doc_idx}" in (used_doc_aliases | explicit_doc_refs):
            next_doc_idx += 1
        candidate = f"doc_{next_doc_idx}"
        next_doc_idx += 1
        doc_alias_by_source[source_ref] = candidate
        used_doc_aliases.add(candidate)
        return candidate

    for evd in verified_evidence or []:
        if run_id is not None and evd.run_id != run_id:
            continue
        is_ok = evd.verified_in_input and evd.source_type in {"user_prompt", "external_retrieved"}
        reason = (
            None if is_ok else
            f"Evidence '{evd.evidence_id}' is not a verified primary source. "
            "Cite the original source, a supported claim, or its verified calculation."
        )
        # Preserve stable doc_1..doc_N aliases for primary ingested/retrieved documents
        # without letting specialist sub-excerpts overwrite them.
        is_specialist = str(evd.created_by or "").startswith("agt_")
        is_primary_doc = (
            evd.source_type == "external_retrieved" and not is_specialist
        ) or (
            evd.created_by in {"lead", "baseline", "operator", "retriever"}
            and evd.source_type != "user_prompt"
        )
        if (evd.source_location or "").startswith("excerpt:"):
            is_primary_doc = False
        if is_primary_doc:
            source_key = evd.source_ref or evd.evidence_id
            alias = _alloc_doc_alias(source_key)
        else:
            source_key = ""
            alias = evd.evidence_id

        target = ProvenanceTarget(
            alias=alias,
            kind="verified_evidence",
            target_id=evd.evidence_id,
            source_type=evd.source_type,
            text=evd.excerpt,
            is_trusted=is_ok,
            rejection_reason=reason,
        )
        catalog.targets_by_key[evd.evidence_id] = target
        if is_primary_doc:
            if alias not in catalog.targets_by_key:
                catalog.targets_by_key[alias] = target
            if source_key and source_key not in catalog.targets_by_key:
                catalog.targets_by_key[source_key] = target
            if source_key:
                doc_source_texts.setdefault(source_key, []).append(evd.excerpt)
                doc_source_trusted[source_key] = (
                    doc_source_trusted.get(source_key, True) and is_ok
                )

    for source_ref, excerpts in doc_source_texts.items():
        if len(excerpts) > 1:
            full_text = "\n\n".join(excerpts)
            is_agg_ok = doc_source_trusted.get(source_ref, False)
            agg_target = ProvenanceTarget(
                alias=source_ref,
                kind="verified_evidence",
                target_id=source_ref,
                source_type="external_retrieved",
                text=full_text,
                is_trusted=is_agg_ok,
                rejection_reason=None
                if is_agg_ok
                else f"Evidence '{source_ref}' contains unverified chunks.",
            )
            current = catalog.targets_by_key.get(source_ref)
            if current is None or current.kind == "verified_evidence":
                catalog.targets_by_key[source_ref] = agg_target
            doc_alias = doc_alias_by_source.get(source_ref)
            if doc_alias:
                catalog.targets_by_key[doc_alias] = agg_target

    scoped_claims = [c for c in all_claims or [] if run_id is None or c.run_id == run_id]
    latest_by_id: dict[str, Claim] = {}
    for claim in scoped_claims:
        previous = latest_by_id.get(claim.claim_id)
        if previous is None or claim.version > previous.version:
            latest_by_id[claim.claim_id] = claim

    scoped_tools = {
        r.tool_execution_id: r for r in tool_records or [] if run_id is None or r.run_id == run_id
    }
    owners = {tid: c for c in scoped_claims for tid in c.tool_execution_ids}
    for tid, record in scoped_tools.items():
        owner = owners.get(tid)
        latest = latest_by_id.get(owner.claim_id) if owner is not None else None
        if latest is None:
            issue = f"Calculation '{tid}' is not attached to a persisted claim."
        elif (
            not latest.is_material or tid not in latest.tool_execution_ids or latest.status in {
                ClaimStatus.REJECTED, ClaimStatus.STALE, ClaimStatus.UNRESOLVED,
            }
        ):
            issue = f"Calculation '{tid}' belongs to a superseded or unverified claim."
        else:
            issue = arithmetic_lineage_issue(record, scoped_tools, latest_by_id, owners)
        catalog.targets_by_key[tid] = make_tool_target(record, rejection_reason=issue)

    for claim in scoped_claims:
        is_latest = claim.version == latest_by_id[claim.claim_id].version
        is_supported = (
            claim.status == ClaimStatus.SUPPORTED
            and claim.verification_status.value in {
                "provenance_validated", "computationally_verified",
                "independently_reviewed_inference",
            }
        )
        valid_results = [
            catalog.targets_by_key[tid] for tid in claim.tool_execution_ids
            if tid in catalog.targets_by_key and catalog.targets_by_key[tid].is_trusted
        ]
        result = valid_results[0] if len(valid_results) == 1 else None
        ver_key = f"{claim.claim_id}@v{claim.version}"
        issue = None
        if not is_latest:
            issue = f"Stale claim reference '{ver_key}'. Cite the current version."
        elif not result and (
            claim.category == ClaimCategory.COMPUTATION or claim.tool_execution_ids
        ):
            issue = f"Claim '{ver_key}' lacks a unique current, grounded calculation."
        elif not claim.is_material or claim.category == ClaimCategory.ASSUMPTION:
            issue = f"Claim '{ver_key}' is not a verified factual source."
        elif not is_supported and not (
            claim.status == ClaimStatus.ASSERTED
            and claim.category == ClaimCategory.COMPUTATION and result
        ):
            issue = f"Premise claim '{ver_key}' has status '{claim.status.value}'."
        if not issue and is_supported:
            pending = list(claim.premise_refs)
            seen: set[str] = set()
            while pending:
                ref = pending.pop()
                parent = latest_by_id.get(ref.claim_id)
                if (
                    parent is None or parent.version != ref.claim_version
                    or parent.status != ClaimStatus.SUPPORTED or not parent.is_material
                    or (parent.category == ClaimCategory.COMPUTATION
                        and not parent.tool_execution_ids)
                    or any(tid not in scoped_tools or not catalog.targets_by_key[tid].is_trusted
                           for tid in parent.tool_execution_ids)
                ):
                    issue = f"Claim '{ver_key}' relies on an unverified or superseded premise."
                    break
                if parent.claim_id not in seen:
                    seen.add(parent.claim_id)
                    pending.extend(parent.premise_refs)
        target = ProvenanceTarget(
            alias=ver_key,
            kind="supported_claim",
            target_id=ver_key,
            text=result.text if result else claim.statement,
            units=result.units if result else "",
            is_trusted=issue is None,
            rejection_reason=issue,
            numeric_value=result.numeric_value if result else None,
            tool_execution_id=result.tool_execution_id if result else None,
        )
        if is_latest and target.is_trusted:
            status = "reviewed" if is_supported else "pending semantic review"
            calculation = (
                f" | result {result.text} [{result.tool_execution_id}]" if result else ""
            )
            target.description = (
                f"({status}, {', '.join(claim.requirement_refs) or 'premise'}) "
                f"{claim.statement}{calculation}"
            )
        catalog.targets_by_key[ver_key] = target
        if is_latest:
            catalog.targets_by_key[claim.claim_id] = target

    return catalog


def make_tool_target(
    record: ToolExecutionRecord, *, rejection_reason: str | None = None,
) -> ProvenanceTarget:
    value = record.computed_result
    valid = (
        record.success and record.inputs_grounded and not record.has_ungrounded_literals
        and record.matches_proposed is not False and record.formula_appropriate is not False
        and value is not None
    )
    return ProvenanceTarget(
        alias=record.tool_execution_id,
        kind="tool_result",
        target_id=record.tool_execution_id,
        text=f"{value} {record.units}".strip() if valid else "",
        units=record.units,
        is_trusted=valid and rejection_reason is None,
        rejection_reason=rejection_reason or (
            None if valid else f"Calculation '{record.tool_execution_id}' is not verified."
        ),
        numeric_value=Decimal(value) if valid and value is not None else None,
        tool_execution_id=record.tool_execution_id,
    )


def arithmetic_lineage_issue(
    record: ToolExecutionRecord,
    tools: dict[str, ToolExecutionRecord],
    latest_claims: dict[str, Claim],
    owners: dict[str, Claim],
) -> str | None:
    """Check the recorded operand sources against current claim versions and tool results."""
    pending = [record]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        tid = current.tool_execution_id
        if tid in seen:
            continue
        seen.add(tid)
        owner = owners.get(tid)
        if owner is not None and current is not record:
            latest = latest_claims[owner.claim_id]
            if not latest.is_material or tid not in latest.tool_execution_ids or latest.status in {
                ClaimStatus.REJECTED, ClaimStatus.STALE, ClaimStatus.UNRESOLVED,
            }:
                return f"Calculation '{tid}' belongs to a superseded or unverified claim."
        if owner is not None:
            for premise_ref in latest_claims[owner.claim_id].premise_refs:
                premise = latest_claims.get(premise_ref.claim_id)
                if (
                    premise is None or premise.version != premise_ref.claim_version
                    or premise.category == ClaimCategory.ASSUMPTION
                    or premise.status in {
                        ClaimStatus.REJECTED, ClaimStatus.STALE, ClaimStatus.UNRESOLVED,
                    }
                ):
                    return (
                        f"Calculation '{tid}' depends on a stale or unverified premise "
                        f"'{premise_ref.claim_id}'."
                    )
        for name, ref in current.input_provenance_refs.items():
            ref = ref.strip()
            detail = current.provenance_details.get(name, {})
            if ref.startswith("assumption:") or (
                ref.startswith("const:") and detail.get("usage_validated") is not True
            ):
                return f"Calculation '{tid}' lacks verified usage for operand '{name}'."
        for detail in current.provenance_details.values():
            if detail.get("kind") == "supported_claim":
                ref = str(detail.get("resolved_id", ""))
                cid, _, version = ref.partition("@v")
                premise = latest_claims.get(cid)
                if (
                    premise is None or str(premise.version) != version or not premise.is_material
                    or premise.category == ClaimCategory.ASSUMPTION
                    or (premise.status != ClaimStatus.SUPPORTED and not (
                        premise.status == ClaimStatus.ASSERTED
                        and detail.get("source_tool_execution_id") in premise.tool_execution_ids
                    ))
                ):
                    return f"Operand source '{ref}' is stale or unsupported. Recompute the claim."
            source_id = detail.get("source_tool_execution_id")
            if source_id is None and detail.get("kind") == "tool_result":
                source_id = detail.get("resolved_id")
            if source_id:
                source = tools.get(source_id)
                if (
                    source_id not in owners or source is None
                    or not make_tool_target(source).is_trusted
                ):
                    return f"Operand calculation '{source_id}' is unavailable or invalid."
                pending.append(source)
    return None


def provenance_failures(record: ToolExecutionRecord) -> list[str]:
    """Render the persisted per-operand failures for review and targeted repair."""
    return [
        f"{name}={detail.get('value', ', '.join(detail.get('values', [])))} "
        f"[{detail.get('ref') or 'no reference'}]: {detail.get('note', 'Invalid provenance.')}"
        + (f" Candidates: {', '.join(detail['candidates'])}." if detail.get("candidates") else "")
        for name, detail in record.provenance_details.items()
        if detail.get("grounded") is False
    ]


def _parse_decimal(raw: str | int | float, *, field_name: str = "value") -> Decimal:
    if isinstance(raw, bool):
        raise MalformedExpressionError(f"Boolean is not valid numeric {field_name}.")
    if isinstance(raw, (int, float)):
        text = str(raw)
    elif isinstance(raw, str):
        text = raw.strip().replace(",", "").strip(_CURRENCY_CHARS).strip()
        if not text:
            raise MalformedExpressionError(f"Empty numeric {field_name}.")
        if text.endswith("%"):
            inner = text[:-1].strip()
            try:
                val = Decimal(inner) / Decimal("100")
            except InvalidOperation as exc:
                raise MalformedExpressionError(
                    f"Malformed percentage {field_name}: '{raw}'."
                ) from exc
            if not val.is_finite():
                raise MalformedExpressionError(f"Non-finite {field_name}: '{raw}'.")
            return val
    else:
        raise MalformedExpressionError(f"Unsupported type for {field_name}: {type(raw).__name__}.")

    try:
        val = Decimal(text)
    except InvalidOperation as exc:
        raise MalformedExpressionError(f"Malformed numeric {field_name}: '{raw}'.") from exc
    if not val.is_finite():
        raise MalformedExpressionError(f"Non-finite numeric {field_name}: '{raw}'.")
    return val


_WORD_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_WORD_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
_WORD_FRACTIONS: dict[str, Decimal] = {
    "half": Decimal("0.5"),
    "quarter": Decimal("0.25"),
}
_WORD_NUM_TOKEN_RE = re.compile(r"[A-Za-z]+|[-]")


def _extract_word_number_spans(text: str) -> list[tuple[Decimal, int, int, str]]:
    """Extract numbers expressed in English words with their character spans."""
    if not text:
        return []
    tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_NUM_TOKEN_RE.finditer(text)]
    results: list[tuple[Decimal, int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok, start, end = tokens[i]
        if tok in _WORD_FRACTIONS:
            # Handle standalone 'half' / 'quarter' (e.g. 'half an hour', 'a quarter')
            results.append((_WORD_FRACTIONS[tok], start, end, text[start:end]))
            i += 1
            continue
        if tok not in _WORD_UNITS and tok not in _WORD_TENS and tok not in _WORD_SCALES:
            i += 1
            continue

        # Accumulate multi-word cardinal: e.g. 'two thousand five hundred', 'forty-five'
        total = 0
        current = 0
        span_end = end
        j = i
        consumed_any = False
        while j < n:
            w, w_start, w_end = tokens[j]
            if j > i and text[tokens[j - 1][2]:w_start].strip():
                break
            if w == "-":
                j += 1
                continue
            if w in _WORD_UNITS:
                current += _WORD_UNITS[w]
                span_end = w_end
                consumed_any = True
                j += 1
            elif w in _WORD_TENS:
                current += _WORD_TENS[w]
                span_end = w_end
                consumed_any = True
                j += 1
            elif w == "hundred":
                current = (current or 1) * 100
                span_end = w_end
                consumed_any = True
                j += 1
            elif w in _WORD_SCALES:
                scale = _WORD_SCALES[w]
                total += (current or 1) * scale
                current = 0
                span_end = w_end
                consumed_any = True
                j += 1
            elif w == "and" and consumed_any and j + 1 < n:
                nxt = tokens[j + 1][0]
                if nxt in _WORD_UNITS or nxt in _WORD_TENS:
                    j += 1
                    continue
                if nxt in _WORD_FRACTIONS:
                    val = Decimal(total + current) + _WORD_FRACTIONS[nxt]
                    results.append((val, start, tokens[j + 1][2], text[start : tokens[j + 1][2]]))
                    consumed_any = False
                    j += 2
                    break
                if nxt in ("a", "an") and j + 2 < n and tokens[j + 2][0] in _WORD_FRACTIONS:
                    frac = _WORD_FRACTIONS[tokens[j + 2][0]]
                    val = Decimal(total + current) + frac
                    results.append((val, start, tokens[j + 2][2], text[start : tokens[j + 2][2]]))
                    consumed_any = False
                    j += 3
                    break
                break
            else:
                break
        if consumed_any:
            results.append((Decimal(total + current), start, span_end, text[start:span_end]))
        i = max(i + 1, j)
    return results


def extract_numbers_from_text(text: str) -> set[Decimal]:
    """Extract normalized Decimal values (including percentages and number words)."""
    found: set[Decimal] = set()
    if not text:
        return found
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        try:
            cleaned = token.replace(",", "")
            if cleaned.endswith("%"):
                pct_val = Decimal(cleaned[:-1])
                if pct_val.is_finite():
                    found.add(pct_val)
                    found.add(pct_val / Decimal("100"))
            else:
                val = Decimal(cleaned)
                if val.is_finite():
                    found.add(val)
        except InvalidOperation:
            continue
    for val, _s, _e, _raw in _extract_word_number_spans(text):
        if val.is_finite():
            found.add(val)
    return found


_SENTENCE_OR_PARA_BREAK_RE = re.compile(r"(?:(?<!\d)\.(?!\d)|;|\n\s*\n)")
_COMMA_SEP_RE = re.compile(r"(?<!\d),(?!\d)")
_COPULA_RE = re.compile(r"(?:\b(?:is|was|are|were|equals)\b|=|:)", re.IGNORECASE)


@dataclass(frozen=True)
class _NumericOccurrence:
    value: Decimal
    ratio: Decimal | None
    token: str
    context: str
    start: int
    end: int
    units: str

    def matches(self, value: Decimal, units: str) -> bool:
        if self.ratio is not None:
            return (self.value == value and units != "ratio") or (
                self.ratio == value and units != "%"
            )
        if self.value == value:
            return True
        scale = self.units.split(" ", 1)[0]
        return scale in _WORD_SCALES and self.value * _WORD_SCALES[scale] == value


def _normalize_operand_units(units: str) -> str:
    text = re.sub(r"\s+", " ", units.strip().lower())
    text = {"percentage point": "points", "percentage points": "points",
            "percent points": "points", "usd thousands": "thousand usd"}.get(text, text)
    return _EXACT_UNIT_NORMALIZATIONS.get(text, text)


def _units_conflict(declared: str, found: str, *, scaled: bool = False) -> bool:
    if not declared or not found or declared == found or (declared == "ratio" and found == "%"):
        return False
    if scaled and found.partition(" ")[2] == declared:
        return False
    # The source parser only emits known explicit units. Compare simple units directly,
    # leaving compound dimensions (e.g. USD per shipment) to formula review.
    known = set(_EXACT_UNIT_NORMALIZATIONS.values()) | set(_CURRENCY_UNITS.values())
    numerator = declared.split(" per ", 1)[0].split("/", 1)[0]
    return numerator != found and (
        numerator in known or found.split(" ", 1)[0] in _WORD_SCALES
    )


def _extract_numeric_occurrences(text: str) -> list[_NumericOccurrence]:
    """Extract source quantities with exact offsets and local operand metadata."""
    raw_spans: list[tuple[Decimal, Decimal | None, int, int, str, str]] = []
    for match in _ISO_DATE.finditer(text):
        year, month, day = map(int, match.groups())
        try:
            date(year, month, day)
        except ValueError:
            continue
        raw_spans.append((
            Decimal(year * 10000 + month * 100 + day), None, match.start(), match.end(),
            match[0], "date",
        ))
    date_spans = [(start, end) for _, _, start, end, _, _ in raw_spans]
    for match in _SOURCE_NUMBER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        token = match.group(0)
        cleaned = token.replace(",", "").replace("−", "-").strip()
        suffix = cleaned[-1].lower()
        before = text[:match.start()]
        currency = _CURRENCY_UNITS.get(before[-1:], "")
        prefix = int(bool(currency))
        if currency and before[-2:-1] in {"-", "−", "+"}:
            cleaned = ("-" if before[-2:-1] in {"-", "−"} else "+") + cleaned
            prefix += 1
        scale_word = re.match(r"\s+(thousand|million|billion)\b", text[match.end():], re.I)
        scale = {"k": "thousand", "m": "million", "b": "billion"}.get(suffix)
        # Single-letter scales are recognized for currency, avoiding units such as metres.
        if scale and not currency:
            continue
        if scale:
            cleaned = cleaned[:-1]
        end = match.end()
        if scale_word and not scale:
            scale = scale_word[1].lower()
            end += scale_word.end()
        percent_word = re.match(r"\s+percent\b", text[end:], re.I)
        is_percent = suffix == "%" or bool(percent_word)
        if percent_word:
            end += percent_word.end()
        raw_val = Decimal(cleaned.rstrip("%").strip())
        ratio_val = raw_val / Decimal(100) if is_percent else None
        units = "%" if is_percent else currency
        if scale:
            units = f"{scale} {units}".strip()
        raw_spans.append((raw_val, ratio_val, match.start() - prefix, end,
                          text[match.start() - prefix:end], units))
    for raw_val, start, end, _token in _extract_word_number_spans(text):
        if any(start < span[3] and end > span[2] for span in raw_spans):
            continue
        percent = re.match(r"\s+percent\b", text[end:], re.I)
        if percent:
            end += percent.end()
        raw_spans.append((raw_val, raw_val / Decimal(100) if percent else None,
                          start, end, text[start:end], "%" if percent else ""))
    raw_spans.sort(key=lambda item: item[2])

    results: list[_NumericOccurrence] = []
    for idx, (raw_val, ratio_val, start, end, token, units) in enumerate(raw_spans):
        prev_end = raw_spans[idx - 1][3] if idx > 0 else 0
        next_start = raw_spans[idx + 1][2] if idx + 1 < len(raw_spans) else len(text)

        left_bound = max(0, start - 65, prev_end)
        right_bound = min(len(text), end + 65, next_start)
        left_slice = text[left_bound:start]
        right_slice = text[end:right_bound]

        # Break only on sentence/paragraph boundaries (not decimals, thousands commas, or \n).
        left_breaks = list(_SENTENCE_OR_PARA_BREAK_RE.finditer(left_slice))
        if left_breaks:
            left_slice = left_slice[left_breaks[-1].end() :]
        elif prev_end > 0 and left_bound == prev_end:
            comma_match = _COMMA_SEP_RE.search(left_slice)
            if comma_match is not None:
                left_slice = left_slice[comma_match.end() :]
            elif _COPULA_RE.search(left_slice):
                # If the prior number was a qualifier inside the subject phrase before `is`/`:`
                # (e.g. "cost at 100,000 transactions is $50,000"), include the subject prefix.
                prev_start = raw_spans[idx - 1][2]
                subj_slice = text[max(0, prev_start - 65) : prev_start]
                subj_breaks = list(_SENTENCE_OR_PARA_BREAK_RE.finditer(subj_slice))
                if subj_breaks:
                    subj_slice = subj_slice[subj_breaks[-1].end() :]
                if not _COPULA_RE.search(subj_slice):
                    left_slice = f"{subj_slice} {left_slice}"

        right_break = _SENTENCE_OR_PARA_BREAK_RE.search(right_slice)
        if right_break is not None:
            right_slice = right_slice[: right_break.start()]
        elif next_start < len(text) and right_bound == next_start:
            comma_match = _COMMA_SEP_RE.search(right_slice)
            if comma_match is not None:
                right_slice = right_slice[: comma_match.start()]

        norm_ctx = re.sub(r"\s+", " ", f"{left_slice} {token} {right_slice}").strip().lower()
        suffix_words = re.match(r"\s*(?:an?\s+)?([A-Za-z]+)", right_slice)
        if not units and suffix_words:
            units = _EXACT_UNIT_NORMALIZATIONS.get(suffix_words[1].lower(), "")
        results.append(_NumericOccurrence(raw_val, ratio_val, token, norm_ctx, start, end, units))
    return results


def _var_stem_matches_context(var_name: str, context: str) -> bool:
    parts = [p for p in re.split(r"[_\d]+", var_name.lower()) if len(p) >= 3]
    if not parts:
        return False
    ctx_words = _WORD_RE.findall(context)
    for part in parts:
        for w in ctx_words:
            if (
                w == part
                or (len(part) >= 3 and w.startswith(part))
                or (len(w) >= 3 and part.startswith(w))
            ):
                return True
    return False


def _check_ast_limits(node: ast.AST, depth: int = 1, counter: list[int] | None = None) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > _MAX_AST_NODES:
        raise UnsafeExpressionError(f"Expression exceeds max AST node count ({_MAX_AST_NODES}).")
    if depth > _MAX_AST_DEPTH:
        raise UnsafeExpressionError(f"Expression exceeds max AST depth ({_MAX_AST_DEPTH}).")
    for child in ast.iter_child_nodes(node):
        _check_ast_limits(child, depth + 1, counter)


def _operand_role_words(name: str) -> set[str]:
    split_name = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return {word for word in _WORD_RE.findall(split_name.lower()) if len(word) >= 3}


def _node_has_input(node: ast.AST, details: dict[str, dict[str, Any]]) -> bool:
    return any(
        detail.get("grounded") and detail["kind"] != "constant"
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and (detail := details.get(child.id))
    )


def _identity_use(
    value: int, node: ast.AST, parent: ast.AST | None, details: dict[str, dict[str, Any]],
) -> bool:
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        if value == 0 and parent.func.id == "range" and len(parent.args) == 2:
            return node is parent.args[0] and _node_has_input(parent.args[1], details)
        if value == 0 and parent.func.id in {"min", "max"} and node in parent.args:
            return any(arg is not node and _node_has_input(arg, details) for arg in parent.args)
        if parent.func.id == "round" and len(parent.args) == 2 and node is parent.args[1]:
            return _node_has_input(parent.args[0], details)
    if isinstance(parent, ast.BinOp):
        other = parent.right if node is parent.left else parent.left
        if (
            value == 1 and node is parent.left and isinstance(parent.op, ast.Sub)
            and isinstance(other, ast.Name)
            and details.get(other.id, {}).get("resolved_id") == "fair_coin_probability"
        ):
            return True
        if _node_has_input(other, details):
            if value == 0:
                return isinstance(parent.op, (ast.Add, ast.Sub))
            return isinstance(parent.op, (ast.Mult, ast.Div)) or (
                node is parent.left and isinstance(parent.op, ast.Sub)
            )
    return False


def _valid_constant_usage(
    name: str,
    label: str,
    tree: ast.AST | None,
    result_units: str,
    details: dict[str, dict[str, Any]],
    catalog: ProvenanceCatalog,
) -> bool:
    words = _operand_role_words(name)
    if tree is None or words & (_EMPIRICAL_ROLES - {"rate"}):
        return False
    if "rate" in words and not words & {"conversion", "scale"}:
        return False
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name]
    result = _normalize_operand_units(result_units)
    operand_units = details[name]["units"]

    def valid(node: ast.Name) -> bool:
        parent = parents.get(node)
        if label == "fair_coin_probability":
            fair_in_prompt = bool(_FAIR_COIN.search(catalog.original_prompt))
            fair_in_evidence = any(
                target.is_trusted and target.kind == "verified_evidence"
                and any(
                    entry.get("grounded") and entry.get("resolved_id") == target.target_id
                    for entry in details.values()
                )
                and _FAIR_COIN.search(target.text)
                for target in catalog.targets_by_key.values()
            )
            if operand_units not in {"", "ratio"} or not (fair_in_prompt or fair_in_evidence):
                return False
            if fair_in_prompt and result in {"", "ratio", "%"}:
                if isinstance(parent, ast.Expression):
                    return result != "%"
                if isinstance(parent, ast.BinOp) and (
                    (isinstance(parent.op, ast.Pow) and node is parent.left
                     and _node_has_input(parent.right, details))
                    or (isinstance(parent.op, ast.Sub) and node is parent.right
                        and isinstance(parent.left, ast.Constant) and parent.left.value == 1)
                ):
                    return True
            return (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id in _BINOMIAL_FUNCTIONS
                and len(parent.args) == 3 and node is parent.args[2]
                and all(_node_has_input(arg, details) for arg in parent.args[:2])
            )
        if label in {"zero", "one"}:
            return not operand_units and _identity_use(int(_FIXED_CONSTANTS[label]), node,
                                                       parent, details)
        if not isinstance(parent, ast.BinOp):
            return False
        other = parent.right if parent.left is node else parent.left
        if not _node_has_input(other, details):
            return False

        other_units = ""
        if isinstance(other, ast.Name) and other.id in details:
            source = details[other.id]
            other_units = source["units"] or source.get("source_units", "")
            if (
                other_units == "%" and source.get("source_value") is not None
                and Decimal(source["value"]) != Decimal(source["source_value"])
            ):
                other_units = "ratio"

        multiply = isinstance(parent.op, ast.Mult)
        divide = isinstance(parent.op, ast.Div) and node is parent.right
        if label in _TIME_CONVERSIONS:
            large, small = _TIME_CONVERSIONS[label]
            conversion_units = tuple(
                _normalize_operand_units(part) for part in re.split(r"/| per ", operand_units)
            )
            return (not operand_units or conversion_units == (small, large)) and (
                (multiply and result == small and other_units in {"", large})
                or (divide and result == large and other_units in {"", small})
            )

        if operand_units not in {"", "ratio"}:
            return False
        return (
            multiply and result in {"%", "points"}
            and (other_units == "ratio" or (
                not other_units and any(isinstance(n, ast.Div) for n in ast.walk(other))
            ))
        ) or (divide and result in {"", "ratio"} and other_units == "%")

    return bool(uses) and all(valid(node) for node in uses)


def _formula_source_issue(
    tree: ast.AST, details: dict[str, dict[str, Any]], catalog: ProvenanceCatalog,
) -> str | None:
    day_operands = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "days_between":
            if len(node.args) != 2 or any(
                not isinstance(arg, ast.Name)
                or details.get(arg.id, {}).get("source_units") != "date"
                or details[arg.id].get("units") != "date"
                for arg in node.args
            ):
                return "days_between requires two named operands grounded in cited ISO dates."
            day_operands.update(node.args)
        if node.func.id in _BINOMIAL_FUNCTIONS:
            coin_contexts = [catalog.original_prompt] if _FAIR_COIN.search(
                catalog.original_prompt
            ) else []
            coin_contexts.extend(
                target.text for entry in details.values()
                if (target := catalog.targets_by_key.get(str(entry.get("ref"))))
                and target.kind == "verified_evidence" and _FAIR_COIN.search(target.text)
            )
            if coin_contexts and not any(_COIN_INDEPENDENCE.search(text) for text in coin_contexts):
                return (
                    "A binomial coin model needs independence as well as fairness. "
                    "Supply an independent-toss premise or qualify the unknown result."
                )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name) and details.get(node.id, {}).get("source_units") == "date"
            and node not in day_operands
        ):
            return "Use ISO date operands inside days_between rather than as ordinary numbers."
    return None


def _ungrounded_literals(
    tree: ast.AST, result_units: str, details: dict[str, dict[str, Any]]
) -> list[str]:
    allowed_nodes: set[ast.AST] = set()
    named_uses = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    legacy_values = Counter(
        Decimal(detail["value"]) for name, detail in details.items()
        if name not in named_uses and detail["grounded"]
    )
    percent_output = bool(re.search(r"%|\bpercent(?:age)?\b", result_units, re.I))
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        if (
            isinstance(node.op, ast.Sub) and isinstance(node.left, ast.Constant)
            and node.left.value == 1 and isinstance(node.right, ast.Constant)
            and type(node.right.value) in {int, float}
        ):
            ratio = Decimal(str(node.right.value))
            if legacy_values[ratio] == 1 and any(
                d.get("grounded") and d.get("source_units") == "%"
                and Decimal(d["source_value"]) / 100 == ratio for d in details.values()
            ):
                allowed_nodes.add(node.left)
        if percent_output and isinstance(node.op, ast.Mult):
            for literal, other in ((node.left, node.right), (node.right, node.left)):
                if (
                    isinstance(literal, ast.Constant) and literal.value == 100
                    and any(isinstance(n, ast.Div) for n in ast.walk(other))
                ):
                    allowed_nodes.add(literal)
        # Legacy rate formulas can divide by 1000 only when the multiplier is bound
        # to a source rate whose own context explicitly specifies the denominator.
        if not isinstance(node.op, ast.Mult):
            continue
        for quotient, rate in ((node.left, node.right), (node.right, node.left)):
            if (
                isinstance(quotient, ast.BinOp) and isinstance(quotient.op, ast.Div)
                and isinstance(quotient.right, ast.Constant) and quotient.right.value == 1000
                and isinstance(rate, ast.Name)
            ):
                source = details.get(rate.id, {})
                if source.get("grounded") and source.get("per_denominator") == "1000":
                    allowed_nodes.add(quotient.right)

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    range_offsets = {
        child for call in ast.walk(tree)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        for arg in call.args
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Add, ast.Sub))
        for child, other in ((arg.left, arg.right), (arg.right, arg.left))
        if isinstance(child, ast.Constant) and child.value == 1
        and _node_has_input(other, details)
    }
    signed_literals = {
        node.operand: -1 if isinstance(node.op, ast.USub) else 1
        for node in ast.walk(tree)
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)
    }
    failures = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant) and type(node.value) in {int, float}
            and node not in allowed_nodes and node not in range_offsets
        ):
            if node.value in {0, 1} and _identity_use(node.value, node, parents.get(node), details):
                continue
            value = Decimal(str(node.value)) * signed_literals.get(node, 1)
            if legacy_values[value] != 1:
                failures.append(str(value))
    return list(dict.fromkeys(failures))


def _to_bounded_int(val: Decimal, *, name: str, min_val: int = 0, max_val: int = 1000) -> int:
    if val != val.to_integral_value():
        raise MalformedExpressionError(f"{name} must be an integer, got {val}.")
    iv = int(val)
    if iv < min_val or iv > max_val:
        raise UnsafeExpressionError(f"{name} ({iv}) outside safe bounds [{min_val}, {max_val}].")
    return iv


_MAX_CUMULATIVE_OPS = 2500


@dataclass
class _EvalBudget:
    """Tracks cumulative AST evaluations and loop steps across an expression."""

    remaining: int = _MAX_CUMULATIVE_OPS

    def consume(self, cost: int = 1) -> None:
        self.remaining -= cost
        if self.remaining < 0:
            raise UnsafeExpressionError(
                f"Expression exceeds cumulative computation limit ({_MAX_CUMULATIVE_OPS} ops)."
            )


def _eval_ast_node(
    node: ast.AST, env: dict[str, Decimal], budget: _EvalBudget | None = None
) -> Decimal:
    if budget is None:
        budget = _EvalBudget()
    budget.consume(1)
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, env, budget)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpressionError(
                f"Disallowed literal type in expression: {type(node.value).__name__}."
            )
        val = _parse_decimal(node.value, field_name="literal")
        if abs(val) > _MAX_MAGNITUDE:
            raise UnsafeExpressionError("Literal magnitude exceeds safe limit.")
        return val

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise MalformedExpressionError(f"Undefined variable '{node.id}' in expression.")
        return env[node.id]

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, env, budget)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise UnsafeExpressionError(f"Disallowed unary operator: {type(node.op).__name__}.")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, env, budget)
        right = _eval_ast_node(node.right, env, budget)
        with localcontext() as ctx:
            ctx.prec = 28
            if isinstance(node.op, ast.Add):
                res = left + right
            elif isinstance(node.op, ast.Sub):
                res = left - right
            elif isinstance(node.op, ast.Mult):
                res = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise MalformedExpressionError("Division by zero.")
                res = left / right
            elif isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise MalformedExpressionError("Division by zero.")
                res = left // right
            elif isinstance(node.op, ast.Mod):
                if right == 0:
                    raise MalformedExpressionError("Modulo by zero.")
                res = left % right
            elif isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_POWER_ABS:
                    raise UnsafeExpressionError(
                        f"Exponent magnitude exceeds safe limit ({_MAX_POWER_ABS})."
                    )
                if right != right.to_integral_value() and left < 0:
                    raise MalformedExpressionError("Negative base with fractional exponent.")
                res = left**right
            else:
                raise UnsafeExpressionError(
                    f"Disallowed binary operator: {type(node.op).__name__}."
                )
        if not res.is_finite() or abs(res) > _MAX_MAGNITUDE:
            raise UnsafeExpressionError("Computed value exceeds safe numerical bounds.")
        return res

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsafeExpressionError("Chained comparisons are not supported.")
        left = _eval_ast_node(node.left, env, budget)
        right = _eval_ast_node(node.comparators[0], env, budget)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            ok = left < right
        elif isinstance(op, ast.LtE):
            ok = left <= right
        elif isinstance(op, ast.Gt):
            ok = left > right
        elif isinstance(op, ast.GtE):
            ok = left >= right
        elif isinstance(op, ast.Eq):
            ok = left == right
        elif isinstance(op, ast.NotEq):
            ok = left != right
        else:
            raise UnsafeExpressionError(f"Disallowed comparison operator: {type(op).__name__}.")
        return Decimal(1) if ok else Decimal(0)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise UnsafeExpressionError("Only simple named allowlisted function calls are allowed.")
        fn_name = node.func.id
        allowed_funcs = {
            "sum",
            "min",
            "max",
            "avg",
            "abs",
            "round",
            "comb",
            "C",
            "perm",
            "P",
            "factorial",
            "days_between",
            "binomial_pmf",
            "binomial_cdf",
            "binomial_sf",
        }
        if fn_name not in allowed_funcs:
            raise UnsafeExpressionError(f"Disallowed function '{fn_name}'.")

        # Support bounded `sum(expr for var in range(start, stop))`
        if (
            fn_name == "sum"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            gen = node.args[0]
            if len(gen.generators) != 1:
                raise UnsafeExpressionError("Only single-loop generator sum() is allowed.")
            comp = gen.generators[0]
            if comp.ifs or not isinstance(comp.target, ast.Name):
                raise UnsafeExpressionError(
                    "Generator sum() requires simple `for var in range(...)`."
                )
            if (
                not isinstance(comp.iter, ast.Call)
                or not isinstance(comp.iter.func, ast.Name)
                or comp.iter.func.id != "range"
                or comp.iter.keywords
                or len(comp.iter.args) not in {1, 2}
            ):
                raise UnsafeExpressionError(
                    "Generator sum() only supports range(stop) or range(start, stop)."
                )
            r_args = [
                _to_bounded_int(
                    _eval_ast_node(a, env, budget), name="range bound", min_val=-1000, max_val=1000
                )
                for a in comp.iter.args
            ]
            start_i, stop_i = (0, r_args[0]) if len(r_args) == 1 else (r_args[0], r_args[1])
            if stop_i - start_i > 200:
                raise UnsafeExpressionError("Generator range exceeds safe limit (200 iterations).")
            loop_var = comp.target.id
            total = Decimal(0)
            with localcontext() as ctx:
                ctx.prec = 28
                for idx in range(start_i, stop_i):
                    budget.consume(1)
                    child_env = {**env, loop_var: Decimal(idx)}
                    total += _eval_ast_node(gen.elt, child_env, budget)
            if not total.is_finite() or abs(total) > _MAX_MAGNITUDE:
                raise UnsafeExpressionError("Computed value exceeds safe numerical bounds.")
            return total

        args: list[Decimal] = []
        for arg in node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                args.extend(_eval_ast_node(elem, env, budget) for elem in arg.elts)
            else:
                args.append(_eval_ast_node(arg, env, budget))

        if not args:
            raise MalformedExpressionError(f"Function '{fn_name}' requires at least one argument.")

        with localcontext() as ctx:
            ctx.prec = 28
            if fn_name == "sum":
                return sum(args, Decimal(0))
            if fn_name == "min":
                return min(args)
            if fn_name == "max":
                return max(args)
            if fn_name == "avg":
                return sum(args, Decimal(0)) / Decimal(len(args))
            if fn_name == "abs":
                if len(args) != 1:
                    raise MalformedExpressionError("abs() takes exactly 1 argument.")
                return abs(args[0])
            if fn_name == "days_between":
                if len(args) != 2:
                    raise MalformedExpressionError("days_between(start, end) takes two ISO dates.")
                dates = []
                for arg in args:
                    packed = _to_bounded_int(arg, name="ISO date", min_val=10101, max_val=99991231)
                    year, month_day = divmod(packed, 10000)
                    month, day = divmod(month_day, 100)
                    dates.append(date(year, month, day))
                return Decimal((dates[1] - dates[0]).days)
            if fn_name == "round":
                if len(args) not in {1, 2}:
                    raise MalformedExpressionError("round() takes 1 or 2 arguments.")
                digits = int(args[1]) if len(args) == 2 else 0
                if abs(digits) > 12:
                    raise UnsafeExpressionError("round() precision exceeds safe bounds.")
                quant = Decimal("1").scaleb(-digits)
                return args[0].quantize(quant)
            if fn_name in {"comb", "C"}:
                if len(args) != 2:
                    raise MalformedExpressionError(f"{fn_name}() takes exactly 2 arguments (n, k).")
                n_i = _to_bounded_int(args[0], name="n", min_val=0, max_val=1000)
                k_i = _to_bounded_int(args[1], name="k", min_val=0, max_val=1000)
                res = Decimal(math.comb(n_i, k_i))
                if abs(res) > _MAX_MAGNITUDE:
                    raise UnsafeExpressionError("Computed value exceeds safe numerical bounds.")
                return res
            if fn_name in {"perm", "P"}:
                if len(args) != 2:
                    raise MalformedExpressionError(f"{fn_name}() takes exactly 2 arguments (n, k).")
                n_i = _to_bounded_int(args[0], name="n", min_val=0, max_val=1000)
                k_i = _to_bounded_int(args[1], name="k", min_val=0, max_val=1000)
                res = Decimal(math.perm(n_i, k_i))
                if abs(res) > _MAX_MAGNITUDE:
                    raise UnsafeExpressionError("Computed value exceeds safe numerical bounds.")
                return res
            if fn_name == "factorial":
                if len(args) != 1:
                    raise MalformedExpressionError("factorial() takes exactly 1 argument.")
                n_i = _to_bounded_int(args[0], name="n", min_val=0, max_val=170)
                res = Decimal(math.factorial(n_i))
                if abs(res) > _MAX_MAGNITUDE:
                    raise UnsafeExpressionError("Computed value exceeds safe numerical bounds.")
                return res
            if fn_name in _BINOMIAL_FUNCTIONS:
                if len(args) != 3:
                    raise MalformedExpressionError(
                        f"{fn_name}() takes exactly 3 arguments (n, k, p)."
                    )
                n_i = _to_bounded_int(args[0], name="n", min_val=0, max_val=500)
                k_i = _to_bounded_int(args[1], name="k", min_val=0, max_val=n_i)
                p_val = args[2]
                if p_val < 0 or p_val > 1:
                    raise MalformedExpressionError("Probability p must be in [0, 1].")
                q_val = Decimal(1) - p_val
                # binomial_cdf(n, k, p) = P(X <= k)
                # binomial_sf(n, k, p) = P(X >= k) (inclusive of threshold k)
                if fn_name == "binomial_pmf":
                    budget.consume(1)
                    return Decimal(math.comb(n_i, k_i)) * (p_val**k_i) * (q_val ** (n_i - k_i))
                if fn_name == "binomial_cdf":
                    budget.consume(k_i + 1)
                    return sum(
                        (
                            Decimal(math.comb(n_i, i)) * (p_val**i) * (q_val ** (n_i - i))
                            for i in range(0, k_i + 1)
                        ),
                        Decimal(0),
                    )
                budget.consume((n_i - k_i) + 1)
                return sum(
                    (
                        Decimal(math.comb(n_i, i)) * (p_val**i) * (q_val ** (n_i - i))
                        for i in range(k_i, n_i + 1)
                    ),
                    Decimal(0),
                )

    raise UnsafeExpressionError(f"Disallowed AST construct: {type(node).__name__}.")


def evaluate_decimal_expression(
    expression: str,
    variables: dict[str, str | int | float],
) -> Decimal:
    """Safely parse and evaluate a bounded Decimal arithmetic expression.

    Combinatorial helpers:
    - `binomial_pmf(n, k, p)`: exact P(X = k)
    - `binomial_cdf(n, k, p)`: cumulative P(X <= k)
    - `binomial_sf(n, k, p)`: inclusive upper tail P(X >= k)
    """
    cleaned = expression.strip()
    if not cleaned:
        raise MalformedExpressionError("Expression cannot be empty.")
    if len(cleaned) > _MAX_EXPRESSION_CHARS:
        raise UnsafeExpressionError(
            f"Expression exceeds max length ({_MAX_EXPRESSION_CHARS} chars)."
        )

    env: dict[str, Decimal] = {}
    for key, raw_val in variables.items():
        if not key.isidentifier():
            raise MalformedExpressionError(f"Invalid variable name '{key}'.")
        val = _parse_decimal(raw_val, field_name=f"variable '{key}'")
        if abs(val) > _MAX_MAGNITUDE:
            raise UnsafeExpressionError(f"Variable '{key}' exceeds safe magnitude limit.")
        env[key] = val

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise MalformedExpressionError(f"Malformed expression syntax: {exc.msg}") from exc

    _check_ast_limits(tree)
    return _eval_ast_node(tree, env)


def format_decimal(value: Decimal) -> str:
    """Format a Decimal cleanly without trailing zeros or scientific notation."""
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _decimals_match(computed: Decimal, proposed: Decimal) -> bool:
    diff = abs(computed - proposed)
    tolerance = Decimal("1e-6") * max(Decimal(1), abs(computed))
    return diff <= tolerance


class DeterministicArithmeticTool:
    """Evaluates, validates resolvable provenance lineage, and persists immutable tool records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def _match_operand(
        target: ProvenanceTarget,
        name: str,
        value: Decimal,
        units: str,
        quote: str | None,
        catalog: ProvenanceCatalog,
    ) -> tuple[str, str, dict[str, Any]]:
        text = target.text
        if target.numeric_value is not None:
            source_units = _normalize_operand_units(target.units)
            ratio = target.numeric_value / Decimal(100) if source_units == "%" else None
            occurrences = [
                _NumericOccurrence(target.numeric_value, ratio, text, text, 0, len(text),
                                   source_units)
            ]
        else:
            occurrences = _extract_numeric_occurrences(text)

        # An explicit standalone exclusion can ground its own zero-valued term.
        # Inspect the whole primary clause so a truncated quotation cannot remove a condition.
        exclusion = re.fullmatch(
            r"No (discount|rebate|surcharge|fee|credit|commission)\s*[.!]?", text.strip(), re.I,
        ) if target.kind == "prompt_span" else None
        role = _operand_role_words(name)
        if (
            exclusion and value == 0 and quote
            and quote.strip().casefold() == text.strip().casefold()
            and exclusion[1].lower() in role
            and role <= {exclusion[1].lower(), "rate", "amount"}
            and ("rate" not in role or units in {"", "%", "ratio"})
            and units in {"", "%", "ratio", *_CURRENCY_UNITS.values()}
        ):
            occurrences = [_NumericOccurrence(
                Decimal(0), Decimal(0) if units in {"%", "ratio"} else None,
                text, text.lower(), 0, len(text), units,
            )]

        candidates = occurrences
        if quote is not None:
            quote = quote.strip()
            starts = [m.start() for m in re.finditer(re.escape(quote), text, re.I)] if quote else []
            if not starts:
                return "quote_not_found", "Use a verbatim passage from the cited source.", {}
            candidates = [
                occ for occ in occurrences
                if any(start <= occ.start and occ.end <= start + len(quote) for start in starts)
            ]

        matching = [occ for occ in candidates if occ.matches(value, units)]
        details = {"candidates": [occ.token for occ in candidates[:12]]}
        if not matching:
            return (
                "value_not_found",
                f"{value} is absent from this source selection. Cite the correct passage or "
                "derive the operand in the expression or a verified prior calculation.",
                details,
            )

        with_units = [
            occ for occ in matching
            if not _units_conflict(units, occ.units, scaled=occ.value != value)
            and (
                units != "%" or occ.ratio is not None or occ.units == "%"
            )
            and (
                units != "ratio" or occ.ratio is not None or (
                    occ.units in {"", "ratio"} and 0 <= occ.value <= 1
                )
            )
        ]
        if not with_units:
            found_units = ", ".join(dict.fromkeys(occ.units or "unspecified" for occ in matching))
            return "unit_conflict", f"Operand units '{units}' conflict with {found_units}.", details

        role_parts = role
        if target.kind in {"prompt_span", "verified_evidence"}:
            qualifiers = role_parts - _ROLE_WORDS
            if units == target.counted_noun:
                qualifiers.discard(target.counted_noun)
            for qualifier in sorted(qualifiers - set(_WORD_RE.findall(text.lower()))):
                # Require positive evidence that the label belongs elsewhere. Ordinary
                # descriptive variable names and unmatched synonyms are left to formula review.
                peer_has_label = any(
                    peer.is_trusted
                    and peer.kind in {"prompt_span", "verified_evidence"}
                    and peer.target_id != target.target_id
                    and qualifier in _WORD_RE.findall(peer.text.lower())
                    and any(
                        qualifier in _WORD_RE.findall(occ.context)
                        and not (
                            target.counted_noun and units == target.counted_noun
                            and occ.units in {"%", *_CURRENCY_UNITS.values()}
                        )
                        and all(_var_stem_matches_context(part, occ.context) for part in role_parts)
                        for occ in _extract_numeric_occurrences(peer.text)
                    )
                    for peer in catalog.targets_by_key.values()
                )
                if peer_has_label:
                    return (
                        "role_conflict",
                        f"Source does not identify '{qualifier}' for operand '{name}'. "
                        "Use the alias for that entity or scenario.",
                        details,
                    )
        if target.numeric_value is None and role_parts:
            exact_role = [
                occ for occ in occurrences
                if all(_var_stem_matches_context(part, occ.context) for part in role_parts)
            ]
            if exact_role:
                role_matches = [occ for occ in with_units if occ in exact_role]
                if not role_matches:
                    return (
                        "role_conflict", f"The selected occurrence conflicts with '{name}'.",
                        details,
                    )
                with_units = role_matches
        if quote is None and target.kind != "tool_result":
            role_matches = [
                occ for occ in with_units if _var_stem_matches_context(name, occ.context)
            ]
            if role_matches:
                with_units = role_matches
            elif any(_var_stem_matches_context(name, occ.context) for occ in occurrences):
                return (
                    "role_conflict",
                    f"The inferred role of '{name}' points to a different number in this source. "
                    "Provide a verbatim source_quote identifying the intended operand.",
                    details,
                )
            elif target.kind == "prompt_span" and text != catalog.original_prompt and any(
                occ.matches(value, units) and _var_stem_matches_context(name, occ.context)
                for occ in _extract_numeric_occurrences(catalog.original_prompt)
            ):
                return (
                    "role_conflict",
                    f"'{name}' matches a different prompt passage. Cite its correct alias or "
                    "provide a verbatim source_quote.",
                    details,
                )
        if quote is not None and target.numeric_value is None and len(with_units) != 1:
            return (
                "ambiguous_quote", "Quote a passage identifying one numeric occurrence.",
                details,
            )

        matched = with_units[0]
        rate_denominator = re.match(r"\s+per\s+(1,?000)\b", text[matched.end:], re.I)
        return "matched", "Matched referenced quantity.", {
            "per_denominator": rate_denominator[1].replace(",", "") if rate_denominator else None,
            "source_token": matched.token,
            "source_value": str(matched.value),
            "source_units": matched.units,
            "source_context": matched.context,
            "source_start": matched.start,
            "source_end": matched.end,
        }

    @classmethod
    def check_inputs_grounded(
        cls,
        *,
        variables: dict[str, Decimal],
        input_provenance_refs: dict[str, str],
        grounded_texts: list[str] | None = None,
        catalog: ProvenanceCatalog | None = None,
        declared_assumptions: list[str] | None = None,
        declared_units: str = "",
        operand_units: dict[str, str] | None = None,
        source_quotes: dict[str, str] | None = None,
        expression_tree: ast.AST | None = None,
        result_units: str = "",
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Bind each value exactly to a trusted source or a validated constant."""
        active_catalog = catalog if catalog is not None else build_provenance_catalog(
            original_prompt="\n".join(grounded_texts or []),
            assumptions=declared_assumptions,
        )
        details: dict[str, dict[str, Any]] = {}
        for name, value in variables.items():
            ref = input_provenance_refs.get(name, "").strip()
            units = _normalize_operand_units((operand_units or {}).get(name, declared_units))
            quote = (source_quotes or {}).get(name)
            kind = "unknown"
            resolved_id = ref
            extra: dict[str, Any] = {}
            if not ref:
                kind, code, note = "missing", "missing_reference", "Supply a provenance reference."
            elif ref.startswith("const:"):
                kind = "constant"
                resolved_id = ref.partition(":")[2].strip().lower()
                expected = _FIXED_CONSTANTS.get(resolved_id)
                if resolved_id == "fair_coin_probability":
                    expected = Decimal("0.5")
                if expected is None:
                    code = "unknown_constant"
                    note = "Use a supported mathematical constant or cite a trusted input."
                elif value != expected:
                    code, note = "constant_mismatch", f"{ref} requires the exact value {expected}."
                else:
                    code, note = "matched", f"Validated {ref} = {expected}."
            else:
                target = active_catalog.targets_by_key.get(ref)
                if ref == "user_prompt" or ref.startswith(("user_prompt:", "span:")):
                    if ref != "user_prompt":
                        quote = ref.partition(":")[2].strip()
                    target = ProvenanceTarget(
                        "user_prompt", "prompt_span", "original_prompt",
                        active_catalog.original_prompt,
                    )
                elif ref.startswith("assumption:"):
                    label = ref.partition(":")[2].strip().lower()
                    target = ProvenanceTarget(
                        ref, "assumption", label, label,
                        is_trusted=False,
                        rejection_reason=(
                            "This plan assumption is conditional. "
                            if label in active_catalog.assumptions else ""
                        ) + "Cite a supplied or verified source for the empirical operand.",
                    )
                if target is None:
                    code, note = "unknown_reference", f"'{ref}' is absent from the source catalog."
                    prefix, _, suffix = ref.partition("_")
                    if prefix in {"tlx", "clm"} and suffix:
                        alternate = ("clm_" if prefix == "tlx" else "tlx_") + suffix
                        candidate = active_catalog.targets_by_key.get(alternate)
                        if candidate and candidate.is_trusted:
                            note += f" Check the catalog entry '{candidate.alias}'."
                else:
                    kind, resolved_id = target.kind, target.target_id
                    if target.tool_execution_id:
                        extra["source_tool_execution_id"] = target.tool_execution_id
                    if target.is_trusted:
                        code, note, match = cls._match_operand(
                            target, name, value, units, quote, active_catalog,
                        )
                        extra.update(match)
                    else:
                        code = "untrusted_reference"
                        note = target.rejection_reason or "Use a trusted, current source."
            details[name] = {
                "ref": ref,
                "kind": kind,
                "resolved_id": resolved_id,
                "value": str(value),
                "units": units,
                "source_quote": quote,
                "grounded": code == "matched",
                "code": code,
                "note": note,
                **extra,
            }

        for name, detail in details.items():
            if detail["kind"] != "constant" or not detail["grounded"]:
                continue
            if _valid_constant_usage(
                name, detail["resolved_id"], expression_tree, result_units, details, active_catalog,
            ):
                detail["usage_validated"] = True
            else:
                detail.update(
                    grounded=False, code="invalid_constant_usage",
                    note="Use this constant in its mathematical role, or cite an empirical source.",
                )

        return all(detail["grounded"] for detail in details.values()), details

    async def execute(
        self,
        *,
        run_id: str,
        actor: str,
        request: ArithmeticRequest,
        grounded_texts: list[str] | None = None,
        catalog: ProvenanceCatalog | None = None,
        task_id: str | None = None,
        invalid_premise_refs: list[str] | None = None,
    ) -> ToolExecutionRecord:
        """Evaluate an ArithmeticRequest deterministically and persist an immutable record."""
        str_vars = {k: str(v) for k, v in request.variables.items()}
        proposed_str = None if request.proposed_result is None else str(request.proposed_result)

        try:
            parsed_vars: dict[str, Decimal] = {}
            for key, raw_val in request.variables.items():
                if not key.isidentifier():
                    raise MalformedExpressionError(f"Invalid variable name '{key}'.")
                parsed_vars[key] = _parse_decimal(raw_val, field_name=f"variable '{key}'")

            tree = ast.parse(request.expression.strip(), mode="eval")
            _check_ast_limits(tree)
            if len(str_vars) != len(request.variable_bindings):
                raise MalformedExpressionError("Duplicate variable binding names.")
            runtime_catalog = catalog if catalog is not None else build_provenance_catalog(
                original_prompt="\n".join(grounded_texts or []),
            )
            computed = evaluate_decimal_expression(request.expression, request.variables)
            computed_str = format_decimal(computed)

            matches_proposed: bool | None = None
            if request.proposed_result is not None:
                proposed_dec = _parse_decimal(request.proposed_result, field_name="proposed_result")
                if (
                    isinstance(request.proposed_result, str)
                    and request.proposed_result.strip().endswith("%")
                    and _normalize_operand_units(request.units) == "%"
                ):
                    proposed_dec *= 100
                matches_proposed = _decimals_match(computed, proposed_dec)

            inputs_grounded, prov_details = self.check_inputs_grounded(
                variables=parsed_vars,
                input_provenance_refs=request.input_provenance_refs,
                catalog=runtime_catalog,
                expression_tree=tree,
                result_units=request.units,
                operand_units={b.name: b.units for b in request.variable_bindings},
                source_quotes={
                    b.name: b.source_quote for b in request.variable_bindings
                    if b.source_quote is not None
                },
            )
            formula_issue = _formula_source_issue(tree, prov_details, runtime_catalog)
            if formula_issue:
                inputs_grounded = False
                prov_details["__formula__"] = {
                    "kind": "formula", "grounded": False,
                    "code": "formula_source_mismatch", "note": formula_issue,
                }
            ungrounded_literals = _ungrounded_literals(tree, request.units, prov_details)
            has_ungrounded_literals = bool(ungrounded_literals)
            if has_ungrounded_literals:
                inputs_grounded = False
                prov_details["__expression_literals__"] = {
                    "kind": "inline_literal",
                    "grounded": False,
                    "code": "unbound_literal",
                    "values": ungrounded_literals,
                    "note": "Replace each listed literal with a named, source-bound operand.",
                }

            if invalid_premise_refs:
                inputs_grounded = False
                prov_details["__claim_premises__"] = {
                    "kind": "claim_dependency",
                    "grounded": False,
                    "code": "invalid_premise_reference",
                    "values": invalid_premise_refs,
                    "note": "Use existing, current material premises for this calculation.",
                }

            record = ToolExecutionRecord(
                run_id=run_id,
                tool_name="arithmetic",
                actor=actor,
                task_id=task_id,
                expression=request.expression.strip(),
                variables=str_vars,
                input_provenance_refs=dict(request.input_provenance_refs),
                provenance_details=prov_details,
                proposed_result=proposed_str,
                computed_result=computed_str,
                units=request.units.strip(),
                assumptions=list(request.assumptions),
                formula_appropriate=False if formula_issue else None,
                matches_proposed=matches_proposed,
                inputs_grounded=inputs_grounded,
                has_ungrounded_literals=has_ungrounded_literals,
                success=True,
                error=None,
            )
        except (UnsafeExpressionError, MalformedExpressionError, SyntaxError, ValueError) as exc:
            err_msg = exc.msg if isinstance(exc, SyntaxError) else str(exc)
            record = ToolExecutionRecord(
                run_id=run_id,
                tool_name="arithmetic",
                actor=actor,
                task_id=task_id,
                expression=request.expression.strip(),
                variables=str_vars,
                input_provenance_refs=dict(request.input_provenance_refs),
                provenance_details={},
                proposed_result=proposed_str,
                computed_result=None,
                units=request.units.strip(),
                assumptions=list(request.assumptions),
                matches_proposed=False if proposed_str is not None else None,
                inputs_grounded=False,
                has_ungrounded_literals=False,
                success=False,
                error=err_msg,
            )

        await self._db.insert_tool_execution(record)
        await self._db.append_event(
            ExecutionEvent(
                run_id=run_id,
                event_type="tool_executed",
                actor=actor,
                message=(
                    f"Arithmetic tool {'succeeded' if record.success else 'failed'}: "
                    f"{record.expression} -> {record.computed_result or record.error}"
                ),
                payload={
                    "tool_execution_id": record.tool_execution_id,
                    "success": record.success,
                    "matches_proposed": record.matches_proposed,
                    "inputs_grounded": record.inputs_grounded,
                    "has_ungrounded_literals": record.has_ungrounded_literals,
                    "computed_result": record.computed_result,
                },
            )
        )
        return record


def tool_record_to_dict(record: ToolExecutionRecord) -> dict[str, Any]:
    """Compact summary dictionary for prompts and traces."""
    return {
        "tool_execution_id": record.tool_execution_id,
        "expression": record.expression,
        "computed_result": record.computed_result,
        "proposed_result": record.proposed_result,
        "units": record.units,
        "matches_proposed": record.matches_proposed,
        "inputs_grounded": record.inputs_grounded,
        "has_ungrounded_literals": record.has_ungrounded_literals,
        "success": record.success,
        "error": record.error,
    }
