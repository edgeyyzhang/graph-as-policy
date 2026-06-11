"""Tests for the goal-AST predicate system (gap.runtime.predicates).

Ported from the source repo's sandbox tests: the tolerance-aware ``in``
predicate plus registry / goal-AST validation coverage.

The legacy ``in`` predicate used a strict AABB-inside-cavity check with
``tol_m=0`` -- which made tasks fail asymmetrically when small
terminal-pose drift moved the target a few centimetres outside the
cavity. The reworked predicate adds anisotropic ``tol_xy_m`` /
``tol_z_m`` slack (default 0.02 m each, matching robosuite) and surfaces
a soft ``contain_score`` so callers can see "barely missed" vs "wildly
off".
"""

from __future__ import annotations

import numpy as np
import pytest

from gap.runtime.predicates import (
    COMPOSITION_OPS,
    SimState,
    SimTrace,
    evaluate_goal_ast,
    registered_predicates,
    validate_goal_ast,
)
from gap.runtime.predicates.goal_eval import BodyView


def _body(name, pos, half, contacts=()):
    pos = np.asarray(pos, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    return BodyView(
        name=name, position=pos,
        quaternion_wxyz=np.array([1., 0., 0., 0.]),
        aabb_lower=pos - half, aabb_upper=pos + half,
        joint_qpos=np.empty(0, dtype=np.float64),
        contacts=frozenset(contacts),
        cavity_lower=None, cavity_upper=None,
    )


def _basket_with_cavity(name, pos, half, contacts=()):
    """Basket with an explicit cavity AABB."""
    pos = np.asarray(pos, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    cavity_h = np.maximum(half - 0.005, 1e-4)
    cavity_centre = pos + np.array([0.0, 0.0, 0.005])
    return BodyView(
        name=name, position=pos,
        quaternion_wxyz=np.array([1., 0., 0., 0.]),
        aabb_lower=pos - half, aabb_upper=pos + half,
        joint_qpos=np.empty(0, dtype=np.float64),
        contacts=frozenset(contacts),
        cavity_lower=cavity_centre - cavity_h,
        cavity_upper=cavity_centre + cavity_h,
    )


def _eval(target_body, container_body, **predicate_args):
    args_payload = {"target": "target", "container": "container", **predicate_args}
    state = SimState(bodies={"target": target_body, "container": container_body})
    res = evaluate_goal_ast(
        {"in": args_payload},
        SimTrace(states=[state]),
    )
    return res.success, res.per_predicate[0]["diagnostics"]


# ---------------------------------------------------------------------------
# ``in`` predicate tolerance behaviour
# ---------------------------------------------------------------------------


def test_target_just_outside_cavity_passes_with_tolerance() -> None:
    """1 cm outside cavity edge in xy -> passes with default tol_xy_m=0.02."""
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts={"target"},
    )
    # Cavity xy ranges roughly [0.42, 0.58] x [-0.08, 0.08]. Centre +1 cm
    # past the upper-y edge: y = 0.085 -> 0.5 cm outside, well within
    # tol_xy_m=0.02 default.
    target = _body("target", [0.5, 0.085, 0.06], [0.02, 0.02, 0.02], contacts={"container"})
    success, diag = _eval(target, container)
    assert success is True
    assert diag["contain"] is True
    assert 0.0 < diag["contain_score"] < 1.0  # near edge


def test_target_well_outside_cavity_fails() -> None:
    """5 cm outside cavity edge -> fails even with default tolerance."""
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts={"target"},
    )
    target = _body("target", [0.5, 0.13, 0.06], [0.02, 0.02, 0.02], contacts={"container"})
    success, diag = _eval(target, container)
    assert success is False
    assert diag["contain"] is False
    assert diag["contain_score"] < 0.0  # negative when outside


def test_score_is_one_at_centre() -> None:
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts={"target"},
    )
    # Right at the cavity centre.
    target = _body("target", [0.5, 0.0, 0.065], [0.02, 0.02, 0.02], contacts={"container"})
    success, diag = _eval(target, container)
    assert success is True
    assert diag["contain_score"] > 0.9


def test_explicit_tol_xy_overrides_default() -> None:
    """A caller can demand stricter containment by passing ``tol_xy_m=0.0``."""
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts={"target"},
    )
    # 0.5 cm outside upper-y -> passes with default 2 cm tol, but should
    # fail when caller asks for tol_xy_m=0.0.
    target = _body("target", [0.5, 0.081, 0.06], [0.02, 0.02, 0.02], contacts={"container"})
    success_default, _ = _eval(target, container)
    success_strict, _ = _eval(target, container, tol_xy_m=0.0, tol_m=0.0)
    assert success_default is True
    assert success_strict is False


def test_no_contact_fails_regardless_of_tolerance() -> None:
    """Tolerance only relaxes containment; contact is still required."""
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts=set(),
    )
    target = _body("target", [0.5, 0.0, 0.065], [0.02, 0.02, 0.02], contacts=set())
    success, diag = _eval(target, container)
    assert success is False
    assert diag["contain"] is True
    assert diag["contact"] is False


def test_diagnostics_expose_tolerances() -> None:
    container = _basket_with_cavity(
        "container", [0.5, 0.0, 0.06], [0.08, 0.08, 0.06], contacts={"target"},
    )
    target = _body("target", [0.5, 0.0, 0.065], [0.02, 0.02, 0.02], contacts={"container"})
    _, diag = _eval(target, container)
    for k in ("tol_m", "tol_xy_m", "tol_z_m", "contain_score", "contain_via"):
        assert k in diag, f"diagnostic missing: {k}"


# ---------------------------------------------------------------------------
# Predicate registry / goal-AST validation
# ---------------------------------------------------------------------------


def test_registry_contains_builtins() -> None:
    registry = registered_predicates()
    for name in ("on", "in", "near", "above", "stack", "grasped", "released",
                 "axis_aligned", "joint_threshold", "at_pose"):
        assert name in registry, f"builtin predicate missing: {name}"
    # Composition ops are reserved, never registered as predicates.
    assert not COMPOSITION_OPS & set(registry)


def test_validate_goal_ast_accepts_composed_goal() -> None:
    validate_goal_ast({
        "and": [
            {"eventually": {"grasped": ["target", "robot"]}},
            {"at_end": {"in": ["target", "container"]}},
        ],
    })


def test_validate_goal_ast_rejects_unknown_predicate() -> None:
    with pytest.raises(ValueError, match="unknown predicate"):
        validate_goal_ast({"levitating": ["target"]})


def test_validate_goal_ast_rejects_bad_arity() -> None:
    with pytest.raises(ValueError, match="expects 2 body args"):
        validate_goal_ast({"on": ["only_one"]})


def test_eventually_over_trace() -> None:
    """``eventually`` walks every snapshot; ``always`` requires all."""
    table = _body("table", [0.0, 0.0, 0.0], [0.5, 0.5, 0.005])
    off = _body("block", [0.0, 0.0, 0.5], [0.03, 0.03, 0.03])
    on_table = _body("block", [0.0, 0.0, 0.035], [0.03, 0.03, 0.03])
    trace = SimTrace(states=[
        SimState(bodies={"block": off, "table": table}, time_s=0.0),
        SimState(bodies={"block": on_table, "table": table}, time_s=1.0),
    ])
    ast_on = {"on": ["block", "table"]}
    assert evaluate_goal_ast({"eventually": ast_on}, trace).success is True
    assert evaluate_goal_ast({"always": ast_on}, trace).success is False
    # Implicit at_end: the final state has the block on the table.
    assert evaluate_goal_ast(ast_on, trace).success is True


def test_missing_body_is_soft_failure() -> None:
    table = _body("table", [0.0, 0.0, 0.0], [0.5, 0.5, 0.005])
    trace = SimTrace(states=[SimState(bodies={"table": table})])
    res = evaluate_goal_ast({"on": ["ghost", "table"]}, trace)
    assert res.success is False
    assert res.per_predicate[0]["diagnostics"]["error"] == "missing_body"
