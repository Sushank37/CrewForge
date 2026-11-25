"""Safe, topic-agnostic deterministic arithmetic tool using Decimal and allowlisted AST."""

from __future__ import annotations

import ast
import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from crewforge.schemas import ArithmeticRequest, ExecutionEvent, ToolExecutionRecord
from crewforge.storage.database import Database

_MAX_EXPRESSION_CHARS = 256
_MAX_AST_DEPTH = 12
_MAX_AST_NODES = 64
_MAX_MAGNITUDE = Decimal("1e18")
_MAX_POWER_ABS = Decimal("100")

_CURRENCY_CHARS = "$₹€£¥"
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


class UnsafeExpressionError(ValueError):
    """Raised when an arithmetic expression uses disallowed syntax or exceeds bounds."""


class MalformedExpressionError(ValueError):
    """Raised when an arithmetic expression or numeric literal is malformed."""


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
    """Evaluates, checks input grounding, and persists immutable arithmetic tool records."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def check_inputs_grounded(
        *,
        variables: dict[str, Decimal],
        input_provenance_refs: dict[str, str],
        grounded_texts: list[str],
    ) -> bool:
        """Verify every variable has a declared provenance ref and numeric grounding."""
        if not variables:
            return True
        grounded_numbers: set[Decimal] = set()
        for text in grounded_texts:
            grounded_numbers.update(extract_numbers_from_text(text))

        for var_name, var_val in variables.items():
            ref = input_provenance_refs.get(var_name, "").strip()
            if not ref:
                return False
            if not any(_decimals_match(var_val, cand) for cand in grounded_numbers):
                return False
        return True

    async def execute(
        self,
        *,
        run_id: str,
        actor: str,
        request: ArithmeticRequest,
        grounded_texts: list[str],
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

            computed = evaluate_decimal_expression(request.expression, request.variables)
            computed_str = format_decimal(computed)

            matches_proposed: bool | None = None
            if request.proposed_result is not None:
                proposed_dec = _parse_decimal(request.proposed_result, field_name="proposed_result")
                matches_proposed = _decimals_match(computed, proposed_dec)

            inputs_grounded = self.check_inputs_grounded(
                variables=parsed_vars,
                input_provenance_refs=request.input_provenance_refs,
                grounded_texts=grounded_texts,
            )

            record = ToolExecutionRecord(
                run_id=run_id,
                tool_name="arithmetic",
                actor=actor,
                task_id=task_id,
                expression=request.expression.strip(),
                variables=str_vars,
                input_provenance_refs=dict(request.input_provenance_refs),
                proposed_result=proposed_str,
                computed_result=computed_str,
                units=request.units.strip(),
                assumptions=list(request.assumptions),
                matches_proposed=matches_proposed,
                inputs_grounded=inputs_grounded,
                success=True,
                error=None,
            )
        except (UnsafeExpressionError, MalformedExpressionError, ValueError) as exc:
            record = ToolExecutionRecord(
                run_id=run_id,
                tool_name="arithmetic",
                actor=actor,
                task_id=task_id,
                expression=request.expression.strip(),
                variables=str_vars,
                input_provenance_refs=dict(request.input_provenance_refs),
                proposed_result=proposed_str,
                computed_result=None,
                units=request.units.strip(),
                assumptions=list(request.assumptions),
                matches_proposed=False if proposed_str is not None else None,
                inputs_grounded=False,
                success=False,
                error=str(exc),
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
        "success": record.success,
        "error": record.error,
    }
