"""Safe, LLM-free expression evaluation for judge:expr rules and computed views.

Grammar: python expressions restricted to literals, names, attribute/subscript
access on the state namespaces, comparisons, boolean ops, unary not, calls to a
small helper whitelist. No comprehensions, lambdas, dunders, or imports.
"""

from __future__ import annotations

import ast
from typing import Any

from .state import ConversationState


class ExprError(ValueError):
    pass


_ALLOWED_CALLS = {"len", "first", "last", "pluck", "unique", "min", "max", "any", "all"}


def _first(xs: Any) -> Any:
    return _DotDict.wrap(xs[0]) if xs else None


def _last(xs: Any) -> Any:
    return _DotDict.wrap(xs[-1]) if xs else None


def _pluck(xs: Any, key: str) -> list[Any]:
    return [
        x.get(key) if isinstance(x, dict) else getattr(x, key, None) for x in (xs or [])
    ]


def _unique(xs: Any) -> list[Any]:
    seen: list[Any] = []
    for x in xs or []:
        if x not in seen:
            seen.append(x)
    return seen


_HELPERS = {
    "len": lambda xs: len(xs) if xs is not None else 0,
    "first": _first,
    "last": _last,
    "pluck": _pluck,
    "unique": _unique,
    "min": min,
    "max": max,
    "any": any,
    "all": all,
}


class _DotDict:
    """Attribute access over dicts so exprs read results.x.data.slots."""

    def __init__(self, data: Any) -> None:
        self._data = data

    @classmethod
    def wrap(cls, value: Any) -> Any:
        return cls(value) if isinstance(value, dict) else value

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise ExprError(f"forbidden attribute {item!r}")
        if isinstance(self._data, dict):
            return _DotDict.wrap(self._data.get(item))
        return None

    def __getitem__(self, item: Any) -> Any:
        return _DotDict.wrap(self._data[item])

    def __eq__(self, other: object) -> bool:
        return self._data == other

    __hash__ = None  # type: ignore[assignment]  # intentionally unhashable

    def __bool__(self) -> bool:
        return bool(self._data)


class _Slots:
    def __init__(self, state: ConversationState) -> None:
        self._state = state

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise ExprError(f"forbidden attribute {key!r}")
        return _DotDict.wrap(self._state.slot_value(key))


class _Results:
    def __init__(self, state: ConversationState) -> None:
        self._state = state

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise ExprError(f"forbidden attribute {key!r}")
        tr = self._state.tool_results.get(key)
        if tr is None:
            return _DotDict.wrap(None)
        return _DotDict.wrap(
            {"ok": tr.ok, "status": tr.status, "data": tr.data, "error": tr.error}
        )


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Attribute,
    ast.Subscript,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.USub,
)


def _check(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise ExprError(f"forbidden syntax: {type(child).__name__}")
        if isinstance(child, ast.Attribute) and child.attr.startswith("_"):
            raise ExprError(f"forbidden attribute {child.attr!r}")
        if isinstance(child, ast.Name) and child.id.startswith("_"):
            raise ExprError(f"forbidden name {child.id!r}")
        if isinstance(child, ast.Call):
            if (
                not isinstance(child.func, ast.Name)
                or child.func.id not in _ALLOWED_CALLS
            ):
                raise ExprError("only whitelisted helper calls allowed")


def evaluate(
    expr: str, state: ConversationState, extra: dict[str, Any] | None = None
) -> Any:
    """Evaluate a restricted expression against state. Missing values -> None.

    Runtime errors (e.g. a KeyError from subscripting absent data) also yield
    None: missing data is falsy, not fatal. ``extra`` injects additional
    namespace entries (e.g. ``pipeline``) supplied by the Director.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(str(exc)) from exc
    _check(tree)
    namespace: dict[str, Any] = {
        **_HELPERS,
        "slots": _Slots(state),
        "results": _Results(state),
        "env": _DotDict(dict(state.env)),
        **(extra or {}),
    }
    try:
        # Safety: the tree was validated by _check (strict node whitelist, no
        # dunders/lambdas/comprehensions/imports), builtins are stripped, and
        # the namespace only exposes guarded wrappers + pure helpers.
        value = eval(  # noqa: S307 - AST-whitelisted above
            compile(tree, "<expr>", "eval"), {"__builtins__": {}}, namespace
        )
    except ExprError:
        raise
    except Exception:
        return None
    return value._data if isinstance(value, _DotDict) else value
