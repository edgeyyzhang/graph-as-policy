"""Predicate registry for the rehearsal goal-AST evaluator.

The LLM-authored goal is a JSON-AST tree (``{"and": [...], "on": [...], ...}``)
walked by :mod:`gap.runtime.predicates.goal_eval` against a :class:`SimTrace` produced by
the rollout. Each *leaf* AST node names a predicate registered here; each
*composition* node uses one of the built-in operators ``and``/``or``/``not``/
``eventually``/``always``/``at_end``.

This module exposes:

- :class:`PredicateSpec` — descriptor for one registered predicate.
- :data:`REGISTRY` — name → spec map.
- :func:`register_predicate` — decorator used by
  :mod:`gap.runtime.predicates.predicate_evaluators` to publish built-ins.
- :func:`validate_predicate_call` — schema check used by goal-AST validation.
- :data:`COMPOSITION_OPS` — reserved keywords routed to the composition walker.

The actual evaluator bodies live in :mod:`gap.runtime.predicates.predicate_evaluators`,
which is imported on first registry use to populate the built-ins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gap.runtime.predicates.goal_eval import SimState

# ---------------------------------------------------------------------------
# Reserved composition operators (handled by goal_eval, not the registry)
# ---------------------------------------------------------------------------


COMPOSITION_OPS: frozenset[str] = frozenset({
    "and", "or", "not",
    "eventually", "always", "at_end",
})


# ---------------------------------------------------------------------------
# Predicate spec + registry
# ---------------------------------------------------------------------------


# A predicate evaluator takes a single-step ``SimState`` plus the resolved
# kwargs (body names + scalar args) and returns a (success, diagnostics) tuple.
PredicateEvaluator = Callable[["SimState", Mapping[str, Any]], tuple[bool, dict[str, Any]]]


@dataclass(frozen=True)
class PredicateSpec:
    """Descriptor for one registered predicate."""

    name: str
    """The AST keyword (e.g. ``"on"``)."""

    body_args: tuple[str, ...]
    """Required positional argument names that must resolve to body names in
    the simulated scene. The AST may pass these positionally as a list
    (``{"on": ["X", "Y"]}``) or by name as a mapping
    (``{"on": {"target": "X", "container": "Y"}}``)."""

    optional_args: dict[str, Any] = field(default_factory=dict)
    """Optional kwargs with defaults (e.g. ``{"tol_m": 0.05}``)."""

    evaluator: PredicateEvaluator = field(default=None)  # populated by decorator
    """Callable ``(SimState, kwargs) -> (bool, diagnostics)``."""

    description: str = ""


REGISTRY: dict[str, PredicateSpec] = {}


def register_predicate(
    name: str,
    *,
    body_args: Sequence[str],
    optional_args: Mapping[str, Any] | None = None,
    description: str = "",
) -> Callable[[PredicateEvaluator], PredicateEvaluator]:
    """Decorator: register a predicate evaluator under ``name``.

    Usage::

        @register_predicate("on", body_args=("target", "container"),
                            optional_args={"tol_m": 0.05})
        def on(state, args):
            ...
            return success, diagnostics
    """

    if name in COMPOSITION_OPS:
        raise ValueError(
            f"predicate name {name!r} collides with reserved composition op; "
            f"choose a different name (composition ops: {sorted(COMPOSITION_OPS)})"
        )

    def deco(fn: PredicateEvaluator) -> PredicateEvaluator:
        if name in REGISTRY:
            raise ValueError(f"predicate {name!r} already registered")
        REGISTRY[name] = PredicateSpec(
            name=name,
            body_args=tuple(body_args),
            optional_args=dict(optional_args or {}),
            evaluator=fn,
            description=description,
        )
        return fn

    return deco


# ---------------------------------------------------------------------------
# AST → kwargs resolution + validation
# ---------------------------------------------------------------------------


def resolve_predicate_args(spec: PredicateSpec, payload: Any) -> dict[str, Any]:
    """Turn an AST payload (list or dict) into kwargs for a predicate.

    - List form (``["X", "Y"]``): values are matched against ``spec.body_args``
      by position. Optional args take their defaults.
    - Dict form (``{"target": "X", "container": "Y", "tol_m": 0.05}``): values
      are matched by name; missing optional args take their defaults.
    """

    if isinstance(payload, list):
        if len(payload) != len(spec.body_args):
            raise ValueError(
                f"predicate {spec.name!r} expects {len(spec.body_args)} body "
                f"args ({list(spec.body_args)}); got {len(payload)} positional"
            )
        kwargs: dict[str, Any] = dict(zip(spec.body_args, payload, strict=True))
        for k, default in spec.optional_args.items():
            kwargs.setdefault(k, default)
        return kwargs

    if isinstance(payload, dict):
        kwargs = dict(payload)
        missing = [k for k in spec.body_args if k not in kwargs]
        if missing:
            raise ValueError(
                f"predicate {spec.name!r} missing required body args {missing}"
            )
        for k, default in spec.optional_args.items():
            kwargs.setdefault(k, default)
        # Body args must be strings (body names).
        for k in spec.body_args:
            if not isinstance(kwargs[k], str):
                raise ValueError(
                    f"predicate {spec.name!r} arg {k!r} must be a body-name "
                    f"string; got {type(kwargs[k]).__name__}"
                )
        return kwargs

    raise ValueError(
        f"predicate {spec.name!r} payload must be a list or dict; "
        f"got {type(payload).__name__}"
    )


def validate_predicate_call(name: str, payload: Any) -> None:
    """Raise ``ValueError`` if the predicate is unknown or args are malformed.

    Used by goal-AST validation; does NOT execute the evaluator.
    """

    _ensure_builtins_loaded()
    if name in COMPOSITION_OPS:
        raise ValueError(
            f"{name!r} is a composition operator, not a predicate; "
            f"validate via goal_eval.validate_goal_ast"
        )
    spec = REGISTRY.get(name)
    if spec is None:
        raise ValueError(
            f"unknown predicate {name!r}; "
            f"registered: {sorted(REGISTRY.keys())}"
        )
    # Resolve to surface arg-shape errors; result is discarded.
    resolve_predicate_args(spec, payload)


# ---------------------------------------------------------------------------
# Built-ins loading
# ---------------------------------------------------------------------------


_BUILTINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    """Lazy-import the evaluator module so the registry self-populates."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Importing populates REGISTRY via @register_predicate.
    from gap.runtime.predicates import predicate_evaluators  # noqa: F401

    _BUILTINS_LOADED = True


def registered_predicates() -> dict[str, PredicateSpec]:
    """Return a snapshot of the registry (loads built-ins if needed)."""
    _ensure_builtins_loaded()
    return dict(REGISTRY)


__all__ = [
    "COMPOSITION_OPS",
    "PredicateEvaluator",
    "PredicateSpec",
    "REGISTRY",
    "register_predicate",
    "registered_predicates",
    "resolve_predicate_args",
    "validate_predicate_call",
]
