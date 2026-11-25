"""Safe, topic-agnostic deterministic arithmetic tool with strict provenance lineage."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from crewforge.schemas import (
    ArithmeticRequest,
    Claim,
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

_UNIT_CONFLICT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "ms",
            "millisecond",
            "milliseconds",
            "sec",
            "second",
            "seconds",
            "min",
            "minute",
            "minutes",
            "hr",
            "hour",
            "hours",
            "day",
            "days",
            "month",
            "months",
            "yr",
            "year",
            "years",
        }
    ),
    frozenset(
        {"usd", "dollar", "dollars", "inr", "rupee", "rupees", "eur", "euro", "euros", "gbp"}
    ),
    frozenset({"%", "percent", "percentage", "point", "points"}),
)
_EXACT_UNIT_NORMALIZATIONS = {
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
    "inr": "inr",
    "rupee": "inr",
    "rupees": "inr",
    "%": "%",
    "percent": "%",
    "percentage": "%",
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
            if key == target.alias and target.kind != "prompt_span" and target.is_trusted:
                lines.append(f"- [{target.alias}] ({target.kind}): {target.text}")
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
        catalog.prompt_clauses.append(target)
        catalog.targets_by_key[alias] = target

    doc_source_texts: dict[str, list[str]] = {}
    for idx, evd in enumerate(verified_evidence or [], start=1):
        doc_alias = f"doc_{idx}"
        is_ok = bool(evd.verified_in_input)
        reason = None if is_ok else f"Evidence '{evd.evidence_id}' is unverified."
        target = ProvenanceTarget(
            alias=doc_alias,
            kind="verified_evidence",
            target_id=evd.evidence_id,
            text=evd.excerpt,
            is_trusted=is_ok,
            rejection_reason=reason,
        )
        catalog.targets_by_key[evd.evidence_id] = target
        catalog.targets_by_key[doc_alias] = target
        if evd.source_ref:
            if evd.source_ref not in catalog.targets_by_key:
                catalog.targets_by_key[evd.source_ref] = target
            doc_source_texts.setdefault(evd.source_ref, []).append(evd.excerpt)

    # If any document has multiple chunks, build an aggregate target under source_ref and doc_1
    for source_ref, excerpts in doc_source_texts.items():
        if len(excerpts) > 1:
            full_text = "\n\n".join(excerpts)
            agg_target = ProvenanceTarget(
                alias=source_ref,
                kind="verified_evidence",
                target_id=source_ref,
                text=full_text,
                is_trusted=True,
            )
            catalog.targets_by_key[source_ref] = agg_target
            if "doc_1" in catalog.targets_by_key and len(doc_source_texts) == 1:
                catalog.targets_by_key["doc_1"] = agg_target

    # Compute latest version per claim_id to detect stale claim versions immediately.
    latest_ver_by_id: dict[str, int] = {}
    for claim in all_claims or []:
        prev_ver = latest_ver_by_id.get(claim.claim_id, 0)
        if claim.version > prev_ver:
            latest_ver_by_id[claim.claim_id] = claim.version

    for claim in all_claims or []:
        is_latest = claim.version == latest_ver_by_id.get(claim.claim_id, -1)
        is_ok = (
            is_latest
            and claim.status == ClaimStatus.SUPPORTED
            and claim.verification_status.value
            in {
                "provenance_validated",
                "computationally_verified",
                "independently_reviewed_inference",
            }
        )
        if not is_latest:
            reason = f"Stale claim reference '{claim.claim_id}' v{claim.version}."
        elif claim.status != ClaimStatus.SUPPORTED:
            reason = (
                f"Premise claim '{claim.claim_id}' has status "
                f"'{claim.status.value}', not supported."
            )
        else:
            reason = None
        ver_key = f"{claim.claim_id}@v{claim.version}"
        t_ver = ProvenanceTarget(
            alias=ver_key,
            kind="supported_claim",
            target_id=ver_key,
            text=claim.statement,
            is_trusted=is_ok,
            rejection_reason=reason,
        )
        catalog.targets_by_key[ver_key] = t_ver
        if is_latest:
            catalog.targets_by_key[claim.claim_id] = t_ver

    for rec in tool_records or []:
        if rec.success and rec.inputs_grounded and rec.computed_result is not None:
            t_tool = ProvenanceTarget(
                alias=rec.tool_execution_id,
                kind="tool_result",
                target_id=rec.tool_execution_id,
                text=f"{rec.computed_result} {rec.units}".strip(),
                units=rec.units,
                is_trusted=True,
            )
            catalog.targets_by_key[rec.tool_execution_id] = t_tool
        else:
            t_bad = ProvenanceTarget(
                alias=rec.tool_execution_id,
                kind="tool_result",
                target_id=rec.tool_execution_id,
                text=rec.error or "",
                units=rec.units,
                is_trusted=False,
                rejection_reason=f"Tool execution '{rec.tool_execution_id}' is not grounded/valid.",
            )
            catalog.targets_by_key[rec.tool_execution_id] = t_bad

    return catalog


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


def extract_numbers_from_text(text: str) -> set[Decimal]:
    """Extract normalized Decimal values (including percentages) from grounded text."""
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
    return found


def _extract_numeric_occurrences(text: str) -> list[tuple[Decimal, Decimal | None, str, str]]:
    """Return list of (raw_value, pct_ratio_value, raw_token, local_context_window)."""
    results: list[tuple[Decimal, Decimal | None, str, str]] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        cleaned = token.replace(",", "")
        try:
            if cleaned.endswith("%"):
                raw_val = Decimal(cleaned[:-1])
                ratio_val = raw_val / Decimal("100")
            else:
                raw_val = Decimal(cleaned)
                ratio_val = None
        except InvalidOperation:
            continue
        if not raw_val.is_finite():
            continue
        start = match.start()
        end = match.end()
        # Bound local context to current clause/comma separator for accurate variable binding.
        left_slice = text[max(0, start - 45) : start]
        for sep in (".", ";", "\n", ","):
            pos = left_slice.rfind(sep)
            if pos != -1:
                left_slice = left_slice[pos + 1 :]
        right_slice = text[end : min(len(text), end + 20)]
        for sep in (".", ";", "\n", ","):
            pos = right_slice.find(sep)
            if pos != -1:
                right_slice = right_slice[:pos]
        results.append((raw_val, ratio_val, token, f"{left_slice} {token} {right_slice}".lower()))
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


def _detect_unit_conflict(declared_units: str, context: str) -> bool:
    if not declared_units.strip():
        return False
    decl_tokens = _WORD_RE.findall(declared_units.lower())
    if "%" in declared_units:
        decl_tokens.append("%")
    ctx_tokens = _WORD_RE.findall(context.lower())
    if "%" in context:
        ctx_tokens.append("%")
    decl_norm = {_EXACT_UNIT_NORMALIZATIONS.get(t, t) for t in decl_tokens}
    ctx_norm = {_EXACT_UNIT_NORMALIZATIONS.get(t, t) for t in ctx_tokens}
    for group in _UNIT_CONFLICT_GROUPS:
        d_in = decl_norm & group
        c_in = ctx_norm & group
        if d_in and c_in and not (d_in & c_in):
            # Only reject incompatible physical dimensions when the context explicitly binds
            # a contradicting dimension unit on the input/quantity itself (not "monthly" rate).
            if d_in & {"ms", "seconds", "minutes", "hours", "days"} and c_in & {
                "months",
                "years",
                "usd",
                "inr",
            }:
                return True
            if c_in & {"ms", "seconds"} and d_in & {"months", "years", "usd", "inr"}:
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


def _extract_ast_numeric_literals(node: ast.AST) -> list[Decimal]:
    literals: list[Decimal] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Constant)
            and isinstance(child.value, (int, float))
            and not isinstance(child.value, bool)
        ):
            literals.append(_parse_decimal(child.value, field_name="literal"))
    return literals


def _eval_ast_node(node: ast.AST, env: dict[str, Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, env)

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
        operand = _eval_ast_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise UnsafeExpressionError(f"Disallowed unary operator: {type(node.op).__name__}.")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, env)
        right = _eval_ast_node(node.right, env)
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
        left = _eval_ast_node(node.left, env)
        right = _eval_ast_node(node.comparators[0], env)
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
        if fn_name not in {"sum", "min", "max", "avg", "abs", "round"}:
            raise UnsafeExpressionError(f"Disallowed function '{fn_name}'.")

        args: list[Decimal] = []
        for arg in node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                args.extend(_eval_ast_node(elem, env) for elem in arg.elts)
            else:
                args.append(_eval_ast_node(arg, env))

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
            if fn_name == "round":
                if len(args) not in {1, 2}:
                    raise MalformedExpressionError("round() takes 1 or 2 arguments.")
                digits = int(args[1]) if len(args) == 2 else 0
                if abs(digits) > 12:
                    raise UnsafeExpressionError("round() precision exceeds safe bounds.")
                quant = Decimal("1").scaleb(-digits)
                return args[0].quantize(quant)

    raise UnsafeExpressionError(f"Disallowed AST construct: {type(node).__name__}.")


def evaluate_decimal_expression(
    expression: str,
    variables: dict[str, str | int | float],
) -> Decimal:
    """Safely parse and evaluate a bounded Decimal arithmetic expression."""
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
    def _validate_target_text_match(
        *,
        var_name: str,
        var_val: Decimal,
        target_text: str,
        declared_units: str = "",
    ) -> tuple[bool, str]:
        occurrences = _extract_numeric_occurrences(target_text)
        matching_occ: list[tuple[Decimal, Decimal | None, str, str]] = []
        for occ in occurrences:
            raw_v, ratio_v, _tok, _ctx = occ
            if _decimals_match(var_val, raw_v) or (
                ratio_v is not None and _decimals_match(var_val, ratio_v)
            ):
                matching_occ.append(occ)

        if not matching_occ:
            return (
                False,
                f"Value '{var_val}' for variable '{var_name}' not found in referenced target.",
            )

        # Check sign preservation explicitly.
        if all(
            (occ[0] < 0) != (var_val < 0) for occ in matching_occ if var_val != 0 and occ[0] != 0
        ):
            return False, f"Sign mismatch for variable '{var_name}' against referenced target."

        # Detect unit conflict on the matched occurrence.
        valid_unit_occ = [
            occ for occ in matching_occ if not _detect_unit_conflict(declared_units, occ[3])
        ]
        if not valid_unit_occ:
            return (
                False,
                f"Unit conflict for variable '{var_name}' ({declared_units}) against target.",
            )

        # Variable-to-source binding check when multiple distinct numbers sit in target text.
        if len(var_name) >= 3:
            matched_with_stem = any(
                _var_stem_matches_context(var_name, occ[3]) for occ in valid_unit_occ
            )
            if not matched_with_stem:
                other_claims_stem = any(
                    _var_stem_matches_context(var_name, occ[3])
                    and not (
                        _decimals_match(var_val, occ[0])
                        or (occ[1] is not None and _decimals_match(var_val, occ[1]))
                    )
                    for occ in occurrences
                )
                if other_claims_stem:
                    return (
                        False,
                        f"Variable '{var_name}' appears bound to wrong number in target.",
                    )

        return True, "Matched referenced target."

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
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Validate every variable against a specific resolvable provenance target."""
        details: dict[str, dict[str, Any]] = {}
        if not variables:
            return True, details

        fallback_prompt = "\n".join(grounded_texts or [])
        active_catalog = catalog or build_provenance_catalog(
            original_prompt=fallback_prompt,
            assumptions=declared_assumptions,
        )
        all_grounded = True

        for var_name, var_val in variables.items():
            ref = input_provenance_refs.get(var_name, "").strip()
            if not ref:
                details[var_name] = {
                    "ref": ref,
                    "kind": "missing",
                    "grounded": False,
                    "note": "Missing provenance reference.",
                }
                all_grounded = False
                continue

            # 1. Explicit mathematical constant (`const:...`)
            if ref.startswith("const:"):
                label = ref[len("const:") :].strip()
                if not label:
                    details[var_name] = {
                        "ref": ref,
                        "kind": "constant",
                        "grounded": False,
                        "note": "Mathematical constant ref must include explicit label.",
                    }
                    all_grounded = False
                else:
                    details[var_name] = {
                        "ref": ref,
                        "kind": "constant",
                        "resolved_id": label,
                        "grounded": True,
                        "note": f"Explicit mathematical constant: {label}",
                    }
                continue

            # 2. Explicit assumption (`assumption:...`)
            if ref.startswith("assumption:"):
                label = ref[len("assumption:") :].strip().lower()
                known_assumptions = active_catalog.assumptions | {
                    a.strip().lower() for a in (declared_assumptions or []) if a.strip()
                }
                matched_assump = bool(label) and (
                    label in known_assumptions
                    or any(label in a or a in label for a in known_assumptions)
                )
                details[var_name] = {
                    "ref": ref,
                    "kind": "assumption",
                    "resolved_id": label,
                    "grounded": matched_assump,
                    "note": (
                        "Resolved to declared assumption."
                        if matched_assump
                        else "Assumption reference is not declared in request/plan assumptions."
                    ),
                }
                if not matched_assump:
                    all_grounded = False
                continue

            # 3. Direct quoted user-prompt span (`user_prompt:<exact_quote>` or `span:<quote>`)
            if ref.startswith("user_prompt:") or ref.startswith("span:"):
                quote = ref.split(":", 1)[1].strip()
                if not quote or quote.lower() not in active_catalog.original_prompt.lower():
                    details[var_name] = {
                        "ref": ref,
                        "kind": "prompt_span",
                        "grounded": False,
                        "note": "Quoted prompt span not found in user prompt.",
                    }
                    all_grounded = False
                    continue
                ok, note = cls._validate_target_text_match(
                    var_name=var_name,
                    var_val=var_val,
                    target_text=quote,
                    declared_units=declared_units,
                )
                details[var_name] = {
                    "ref": ref,
                    "kind": "prompt_span",
                    "resolved_id": "quoted_span",
                    "grounded": ok,
                    "note": note,
                }
                if not ok:
                    all_grounded = False
                continue

            # 4. General `user_prompt`: only matches prompt text with strict variable/unit checks
            if ref == "user_prompt":
                ok, note = cls._validate_target_text_match(
                    var_name=var_name,
                    var_val=var_val,
                    target_text=active_catalog.original_prompt,
                    declared_units=declared_units,
                )
                details[var_name] = {
                    "ref": ref,
                    "kind": "prompt_span",
                    "resolved_id": "original_prompt",
                    "grounded": ok,
                    "note": note,
                }
                if not ok:
                    all_grounded = False
                continue

            # 5. Direct catalog alias / verified ID (`up_k`, `doc_k`, `evd_...`, `clm_...`, `tlx_`)
            target = active_catalog.targets_by_key.get(ref)
            if target is None:
                details[var_name] = {
                    "ref": ref,
                    "kind": "unknown",
                    "grounded": False,
                    "note": f"Fabricated or unknown provenance target '{ref}'.",
                }
                all_grounded = False
                continue

            if not target.is_trusted:
                details[var_name] = {
                    "ref": ref,
                    "kind": target.kind,
                    "resolved_id": target.target_id,
                    "grounded": False,
                    "note": target.rejection_reason or f"Target '{ref}' is not trusted/supported.",
                }
                all_grounded = False
                continue

            ok, note = cls._validate_target_text_match(
                var_name=var_name,
                var_val=var_val,
                target_text=target.text,
                declared_units=declared_units,
            )
            details[var_name] = {
                "ref": ref,
                "kind": target.kind,
                "resolved_id": target.target_id,
                "grounded": ok,
                "note": note,
            }
            if not ok:
                all_grounded = False

        return all_grounded, details

    async def execute(
        self,
        *,
        run_id: str,
        actor: str,
        request: ArithmeticRequest,
        grounded_texts: list[str] | None = None,
        catalog: ProvenanceCatalog | None = None,
        task_id: str | None = None,
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
            inline_literals = _extract_ast_numeric_literals(tree)
            is_percent_expr = "%" in request.units or "percent" in request.units.lower()
            ungrounded_literals = [
                lit
                for lit in inline_literals
                if lit not in {Decimal(0), Decimal(1)}
                and not (lit == Decimal(100) and is_percent_expr)
                and format_decimal(lit) not in request.input_provenance_refs
                and not any(_decimals_match(lit, v) for v in parsed_vars.values())
            ]
            has_ungrounded_literals = bool(ungrounded_literals)

            computed = evaluate_decimal_expression(request.expression, request.variables)
            computed_str = format_decimal(computed)

            matches_proposed: bool | None = None
            if request.proposed_result is not None:
                proposed_dec = _parse_decimal(request.proposed_result, field_name="proposed_result")
                matches_proposed = _decimals_match(computed, proposed_dec)

            inputs_grounded, prov_details = self.check_inputs_grounded(
                variables=parsed_vars,
                input_provenance_refs=request.input_provenance_refs,
                grounded_texts=grounded_texts,
                catalog=catalog,
                declared_assumptions=request.assumptions,
                declared_units=request.units,
            )
            if has_ungrounded_literals:
                inputs_grounded = False
                prov_details["__expression_literals__"] = {
                    "kind": "inline_literal",
                    "grounded": False,
                    "values": [format_decimal(x) for x in ungrounded_literals],
                    "note": (
                        "Empirical numeric constants inside expression must be grounded variables."
                    ),
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
