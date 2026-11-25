"""LangChain tools for arithmetic evaluation, text metrics, and Asia/Kolkata time."""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPONENT = 1000
_MAX_EXPRESSION_LENGTH = 500
_IST_FALLBACK = timezone(timedelta(hours=5, minutes=30), name="IST")


def _eval_ast_node(node: ast.AST) -> int | float:
    """Recursively evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported literal value: {node.value!r}")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_ast_node(node.operand))

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(
                f"Unsupported operator: {type(node.op).__name__}. "
                "Supported operators are +, -, *, /, //, %, and **."
            )
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent magnitude exceeds limit of {_MAX_EXPONENT}.")
        return op_fn(left, right)

    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression string (e.g., '(12 + 8) * 3 / 4')."""
    cleaned = expression.strip()
    if not cleaned:
        return "Error: Expression must not be empty."
    if len(cleaned) > _MAX_EXPRESSION_LENGTH:
        return f"Error: Expression exceeds maximum length of {_MAX_EXPRESSION_LENGTH} characters."

    try:
        parsed = ast.parse(cleaned, mode="eval")
        result = _eval_ast_node(parsed)
    except SyntaxError as exc:
        return f"Error: Invalid arithmetic syntax ({exc.msg})."
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except (ValueError, ArithmeticError, OverflowError) as exc:
        return f"Error: {exc}"

    return str(result)


@tool
def word_count(text: str) -> str:
    """Count the number of characters, words, and lines in the provided text."""
    char_count = len(text)
    words = len(text.split())
    line_count = len(text.splitlines()) if text else 0
    return f"characters: {char_count}, words: {words}, lines: {line_count}"


@tool
def current_time() -> str:
    """Return the current date and time in the Asia/Kolkata timezone (IST)."""
    try:
        tz = ZoneInfo("Asia/Kolkata")
    except ZoneInfoNotFoundError:
        tz = _IST_FALLBACK

    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


for _tool in (calculator, word_count, current_time):
    _tool.handle_tool_error = True
    _tool.handle_validation_error = True

TOOLS = [calculator, word_count, current_time]
