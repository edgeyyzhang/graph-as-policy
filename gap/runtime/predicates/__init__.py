"""Goal-AST predicate system (separate from checkpoint predicates).

The success goal is a JSON-AST tree of composition operators over leaf
predicates registered in :mod:`gap.runtime.predicates.predicates`;
:mod:`gap.runtime.predicates.goal_eval` walks the AST against a
:class:`SimTrace` produced by a rollout.
"""

from __future__ import annotations

from .goal_eval import (
    BodyView,
    EvalResult,
    SimState,
    SimTrace,
    evaluate_goal_ast,
    validate_goal_ast,
)
from .predicates import (
    COMPOSITION_OPS,
    REGISTRY,
    PredicateSpec,
    register_predicate,
    registered_predicates,
)

__all__ = [
    "BodyView",
    "COMPOSITION_OPS",
    "EvalResult",
    "PredicateSpec",
    "REGISTRY",
    "SimState",
    "SimTrace",
    "evaluate_goal_ast",
    "register_predicate",
    "registered_predicates",
    "validate_goal_ast",
]
