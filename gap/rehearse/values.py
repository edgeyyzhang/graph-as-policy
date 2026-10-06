"""Reduce node values to small JSON-safe values and to one-line text.

Node inputs, node outputs and subgraph outputs can hold anything: poses,
boxes, images, point clouds, candidate lists. Feedback files need them small
and readable. The rules here depend only on a value's structure (its type,
size and keys), never on what the value means for a task.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_MAX_TEXT = 200
_SMALL_ARRAY = 16


def _number(x: float, digits: int = 4) -> Any:
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    return round(x, digits)


def compact(value: Any, *, max_items: int = 8, depth: int = 6) -> Any:
    """JSON-safe reduction: large arrays become their shape, long lists their
    length and first entries, numbers are rounded."""
    if depth < 0:
        return "<...>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + "..."
    if isinstance(value, np.ndarray):
        if value.size <= _SMALL_ARRAY and value.dtype.kind in "biuf":
            return compact(value.tolist(), max_items=_SMALL_ARRAY, depth=depth)
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    if isinstance(value, dict):
        return {str(k): compact(v, max_items=max_items, depth=depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        if len(items) > max_items:
            return {
                "count": len(items),
                "first": [compact(v, max_items=max_items, depth=depth - 1) for v in items[:3]],
            }
        return [compact(v, max_items=max_items, depth=depth - 1) for v in items]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return compact(to_dict(), max_items=max_items, depth=depth - 1)
        except Exception:
            pass
    return compact(str(value))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _num(x: Any, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def text(value: Any) -> str:
    """One-line text for a value already reduced by :func:`compact`."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_number(value):
        return _num(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and len(value) <= 4 and all(_is_number(v) for v in value):
            return "(" + ", ".join(_num(v) for v in value) + ")"
        return "[" + ", ".join(text(v) for v in value) + "]"
    if isinstance(value, dict):
        keys = list(value)
        if keys and set(keys) <= set("wxyz") and all(_is_number(v) for v in value.values()):
            ordered = [k for k in "wxyz" if k in value] if "w" in value else [k for k in "xyz" if k in value]
            return "(" + ", ".join(f"{k} {_num(value[k])}" for k in ordered) + ")"
        if set(keys) == {"count", "first"}:
            return f"list of {value['count']}, first: " + "; ".join(text(v) for v in value["first"])
        parts = []
        for k, v in value.items():
            rendered = text(v)
            if isinstance(v, dict) and not (set(v) <= set("wxyz")):
                rendered = "{" + rendered + "}"
            parts.append(f"{k} {rendered}")
        return ", ".join(parts)
    return str(value)


def mapping_text(values: dict[str, Any] | None) -> str:
    """Text for a name -> value mapping, one ``name: value`` per entry."""
    if not values:
        return "none"
    return "; ".join(f"{k}: {text(v)}" for k, v in values.items())


__all__ = ["compact", "text", "mapping_text"]
