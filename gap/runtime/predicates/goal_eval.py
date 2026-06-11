"""Goal-AST evaluator for the rehearsal sandbox.

The LLM-authored success goal is a JSON tree (``{"and": [...], "on": [...], …}``)
of composition ops over leaf predicates. This module:

- Provides :class:`SimState` / :class:`BodyView` — a façade over a MuJoCo
  ``(MjModel, MjData)`` snapshot exposing body positions, AABBs, and contacts
  by name. Predicate evaluators registered in
  :mod:`gap.runtime.predicates.predicate_evaluators` only see this façade and never touch
  raw MuJoCo internals.

- Provides :class:`SimTrace` — a sequence of ``SimState`` snapshots over a
  rollout. Composition operators ``eventually`` / ``always`` evaluate over the
  whole sequence; ``at_end`` (the implicit default) evaluates only the last.

- Exposes :func:`validate_goal_ast` (schema-level) and
  :func:`evaluate_goal_ast` (runtime).

The AST grammar:

```
goal       := composition | predicate
composition := { "and"|"or": [goal, ...] } | { "not"|"eventually"|"always"|"at_end": goal }
predicate  := { <name>: list_or_dict_payload }
```

Top-level: a goal may be either a single predicate or a composition op. If the
top-level is a leaf predicate, it is evaluated as ``at_end``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gap.runtime.predicates.predicates import (
    COMPOSITION_OPS,
    REGISTRY,
    _ensure_builtins_loaded,
    resolve_predicate_args,
    validate_predicate_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimState / BodyView — snapshot façade over MjModel + MjData
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyView:
    """One body's pose + AABB + joint qpos at a single timestep."""

    name: str
    position: np.ndarray              # (3,) world-frame
    quaternion_wxyz: np.ndarray       # (4,) world-frame
    aabb_lower: np.ndarray            # (3,) world-frame min
    aabb_upper: np.ndarray            # (3,) world-frame max
    joint_qpos: np.ndarray            # (k,) joint state, empty if no joints
    contacts: frozenset[str]          # body-names in contact with this body
    cavity_lower: np.ndarray | None = None
    """World-frame min of the body's interior cavity AABB, if a sibling
    ``<site name="<body>__cavity">`` is registered. ``None`` for solid
    bodies (no cavity)."""
    cavity_upper: np.ndarray | None = None
    """World-frame max of the cavity AABB; ``None`` if no cavity site."""

    @property
    def x(self) -> float: return float(self.position[0])
    @property
    def y(self) -> float: return float(self.position[1])
    @property
    def z(self) -> float: return float(self.position[2])

    @property
    def xy(self) -> np.ndarray: return self.position[:2]

    @property
    def top_z(self) -> float: return float(self.aabb_upper[2])
    @property
    def bottom_z(self) -> float: return float(self.aabb_lower[2])
    @property
    def left_x(self) -> float: return float(self.aabb_lower[0])
    @property
    def right_x(self) -> float: return float(self.aabb_upper[0])
    @property
    def near_y(self) -> float: return float(self.aabb_lower[1])
    @property
    def far_y(self) -> float: return float(self.aabb_upper[1])

    def half_extents_world(self) -> np.ndarray:
        return 0.5 * (self.aabb_upper - self.aabb_lower)


@dataclass
class SimState:
    """Indexed snapshot of one MuJoCo timestep, addressable by body name."""

    bodies: dict[str, BodyView]
    time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def body(self, name: str) -> BodyView:
        if name not in self.bodies:
            raise KeyError(
                f"body {name!r} not found in scene; available: "
                f"{sorted(self.bodies.keys())}"
            )
        return self.bodies[name]

    def has_body(self, name: str) -> bool:
        return name in self.bodies

    @classmethod
    def from_mujoco(cls, model: Any, data: Any) -> SimState:
        """Snapshot a single timestep from MjModel + MjData.

        Body AABBs are computed by transforming each geom's local box-extent
        to world frame and taking the union. Contacts are collapsed to body
        pairs. Both raw MjData fields are deep-copied so the SimState does not
        alias mutable simulator state.
        """
        return _snapshot_mujoco(model, data)


def _snapshot_mujoco(model: Any, data: Any) -> SimState:
    """Internal: build a SimState from one MjModel/MjData."""
    import mujoco

    bodies: dict[str, BodyView] = {}
    nbody = int(model.nbody)

    # Pre-compute body→geom-id list for AABB.
    body_geoms: dict[int, list[int]] = {}
    for gid in range(int(model.ngeom)):
        bid = int(model.geom_bodyid[gid])
        body_geoms.setdefault(bid, []).append(gid)

    # Pre-compute body→joint-qpos slice.
    body_joints: dict[int, list[int]] = {}
    for jid in range(int(model.njnt)):
        bid = int(model.jnt_bodyid[jid])
        qpos_addr = int(model.jnt_qposadr[jid])
        # joint_type → qpos width: free=7, ball=4, slide=1, hinge=1.
        jtype = int(model.jnt_type[jid])
        width = {
            mujoco.mjtJoint.mjJNT_FREE: 7,
            mujoco.mjtJoint.mjJNT_BALL: 4,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
        }.get(jtype, 1)
        body_joints.setdefault(bid, []).extend(
            range(qpos_addr, qpos_addr + width)
        )

    # Build per-body contact set.
    contacts_by_body: dict[int, set[int]] = {}
    ncon = int(data.ncon)
    for i in range(ncon):
        c = data.contact[i]
        # Skip sentinel contacts (zero penetration distance + zero force).
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        contacts_by_body.setdefault(b1, set()).add(b2)
        contacts_by_body.setdefault(b2, set()).add(b1)

    body_id_to_name = {}
    for bid in range(nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        body_id_to_name[bid] = nm if nm else f"body_{bid}"

    # Pre-compute body→cavity-site AABB if a sibling site named
    # "<body>__cavity" is registered. The site encodes the interior cavity
    # of a hollow container (built by ``make_hollow_container``); the
    # ``in`` predicate uses this to mirror LIBERO's BDDL ``check_contain``
    # against the basket interior rather than the union AABB of the walls.
    body_cavity: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid in range(int(model.nsite)):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if not site_name.endswith("__cavity"):
            continue
        cbid = int(model.site_bodyid[sid])
        spos = np.asarray(data.site_xpos[sid], dtype=np.float64)
        smat = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        ssize = np.asarray(model.site_size[sid], dtype=np.float64)
        world_half = np.abs(smat) @ ssize
        body_cavity[cbid] = (spos - world_half, spos + world_half)

    qpos = np.asarray(data.qpos, dtype=np.float64).copy()

    for bid in range(nbody):
        nm = body_id_to_name[bid]
        # Body 0 is the world; skip.
        if bid == 0:
            continue
        position = np.asarray(data.xpos[bid], dtype=np.float64).copy()
        quat = np.asarray(data.xquat[bid], dtype=np.float64).copy()  # wxyz
        aabb_lower, aabb_upper = _body_aabb(model, data, bid, body_geoms.get(bid, []))
        joint_addrs = body_joints.get(bid, [])
        jqpos = qpos[joint_addrs] if joint_addrs else np.empty(0, dtype=np.float64)
        contact_names = frozenset(
            body_id_to_name[other_bid] for other_bid in contacts_by_body.get(bid, set())
        )
        cavity_lo, cavity_hi = body_cavity.get(bid, (None, None))
        bodies[nm] = BodyView(
            name=nm,
            position=position,
            quaternion_wxyz=quat,
            aabb_lower=aabb_lower,
            aabb_upper=aabb_upper,
            joint_qpos=jqpos,
            contacts=contact_names,
            cavity_lower=cavity_lo,
            cavity_upper=cavity_hi,
        )

    return SimState(bodies=bodies, time_s=float(data.time))


def _body_aabb(
    model: Any, data: Any, bid: int, geom_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a body's AABB in world frame from its geoms.

    Each geom's bounding box is approximated as ``geom_xpos ± |R| @ size``,
    which is exact for boxes and conservative for spheres/capsules.
    """
    if not geom_ids:
        # Body has no geoms — fall back to body origin point.
        p = np.asarray(data.xpos[bid], dtype=np.float64)
        return p.copy(), p.copy()

    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for gid in geom_ids:
        gpos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
        gmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        gsize = np.asarray(model.geom_size[gid], dtype=np.float64)
        # |R| @ size gives the world half-extents of the local box.
        world_half = np.abs(gmat) @ gsize
        lo = np.minimum(lo, gpos - world_half)
        hi = np.maximum(hi, gpos + world_half)
    return lo, hi


# ---------------------------------------------------------------------------
# SimTrace — sequence of SimStates over a rollout
# ---------------------------------------------------------------------------


@dataclass
class SimTrace:
    """Ordered list of single-step ``SimState`` snapshots.

    Composition ops:
      ``at_end``   → child(states[-1])
      ``eventually`` → any(child(s) for s in states)
      ``always``   → all(child(s) for s in states)
    """

    states: list[SimState]

    @property
    def last(self) -> SimState:
        if not self.states:
            raise ValueError("SimTrace is empty; cannot evaluate at_end")
        return self.states[-1]

    @classmethod
    def from_final(cls, model: Any, data: Any) -> SimTrace:
        """Single-snapshot trace from a final MuJoCo state."""
        return cls(states=[SimState.from_mujoco(model, data)])

    @classmethod
    def from_states(cls, states: Iterable[SimState]) -> SimTrace:
        return cls(states=list(states))


# ---------------------------------------------------------------------------
# Validation — schema-level (no runtime state needed)
# ---------------------------------------------------------------------------


def validate_goal_ast(ast: Any) -> None:
    """Recursively check that ``ast`` is a well-formed predicate tree.

    Raises ``ValueError`` on:
      - non-mapping nodes,
      - mappings with != 1 key,
      - composition ops with the wrong child shape,
      - leaf predicates not registered in ``REGISTRY``,
      - leaf predicates whose payload doesn't satisfy
        :func:`gap.runtime.predicates.predicates.resolve_predicate_args`.
    """
    _ensure_builtins_loaded()
    _validate_node(ast)


def _validate_node(node: Any) -> None:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(
            f"goal AST node must be a mapping with exactly one key; "
            f"got {type(node).__name__} = {node!r}"
        )
    [(op, payload)] = node.items()
    if op in COMPOSITION_OPS:
        _validate_composition(op, payload)
    else:
        validate_predicate_call(op, payload)


def _validate_composition(op: str, payload: Any) -> None:
    if op in {"and", "or"}:
        if not isinstance(payload, list) or not payload:
            raise ValueError(
                f"composition op {op!r} requires a non-empty list of children; "
                f"got {payload!r}"
            )
        for child in payload:
            _validate_node(child)
        return
    # not / eventually / always / at_end → single child node
    _validate_node(payload)


# ---------------------------------------------------------------------------
# Evaluation — runtime walk over a SimTrace
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    success: bool
    """Top-level AST verdict."""

    per_predicate: list[dict[str, Any]] = field(default_factory=list)
    """One entry per leaf evaluation: {ast_path, predicate, args, success, diagnostics}."""

    rationale: str = ""


def evaluate_goal_ast(ast: Any, trace: SimTrace) -> EvalResult:
    """Walk the AST against the rollout trace and return a verdict.

    The top-level node is treated as ``at_end`` if it is a leaf predicate and
    not already wrapped in a composition op.
    """
    _ensure_builtins_loaded()
    if not trace.states:
        return EvalResult(success=False, rationale="empty trace")

    per_predicate: list[dict[str, Any]] = []
    success = _eval_node(ast, trace, path="$", per_predicate=per_predicate)
    return EvalResult(success=success, per_predicate=per_predicate)


def _eval_node(
    node: Any,
    trace: SimTrace,
    *,
    path: str,
    per_predicate: list[dict[str, Any]],
) -> bool:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(
            f"AST node at {path}: expected single-key mapping, got {node!r}"
        )
    [(op, payload)] = node.items()

    if op == "and":
        return all(
            _eval_node(child, trace, path=f"{path}.and[{i}]",
                      per_predicate=per_predicate)
            for i, child in enumerate(payload)
        )
    if op == "or":
        return any(
            _eval_node(child, trace, path=f"{path}.or[{i}]",
                      per_predicate=per_predicate)
            for i, child in enumerate(payload)
        )
    if op == "not":
        return not _eval_node(payload, trace, path=f"{path}.not",
                              per_predicate=per_predicate)
    if op == "eventually":
        for i, state in enumerate(trace.states):
            if _eval_node_against_state(payload, state,
                                         path=f"{path}.eventually[t={i}]",
                                         per_predicate=per_predicate):
                return True
        return False
    if op == "always":
        for i, state in enumerate(trace.states):
            if not _eval_node_against_state(payload, state,
                                             path=f"{path}.always[t={i}]",
                                             per_predicate=per_predicate):
                return False
        return True
    if op == "at_end":
        return _eval_node_against_state(payload, trace.last,
                                         path=f"{path}.at_end",
                                         per_predicate=per_predicate)

    # Leaf predicate at the top level → default to at_end semantics.
    return _eval_leaf_against_state(op, payload, trace.last,
                                     path=f"{path}.<{op}>@end",
                                     per_predicate=per_predicate)


def _eval_node_against_state(
    node: Any,
    state: SimState,
    *,
    path: str,
    per_predicate: list[dict[str, Any]],
) -> bool:
    """Like _eval_node but with the active timestep already chosen.

    Composition ops still recurse, but ``eventually`` / ``always`` / ``at_end``
    nested inside a temporal scope simply re-use the same active state.
    """
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(
            f"AST node at {path}: expected single-key mapping, got {node!r}"
        )
    [(op, payload)] = node.items()

    if op == "and":
        return all(
            _eval_node_against_state(child, state, path=f"{path}.and[{i}]",
                                      per_predicate=per_predicate)
            for i, child in enumerate(payload)
        )
    if op == "or":
        return any(
            _eval_node_against_state(child, state, path=f"{path}.or[{i}]",
                                      per_predicate=per_predicate)
            for i, child in enumerate(payload)
        )
    if op == "not":
        return not _eval_node_against_state(
            payload, state, path=f"{path}.not", per_predicate=per_predicate,
        )
    if op in {"eventually", "always", "at_end"}:
        # Re-entrant temporal op: collapse to the active state.
        return _eval_node_against_state(
            payload, state, path=f"{path}.{op}", per_predicate=per_predicate,
        )

    return _eval_leaf_against_state(
        op, payload, state, path=path, per_predicate=per_predicate,
    )


def _eval_leaf_against_state(
    name: str,
    payload: Any,
    state: SimState,
    *,
    path: str,
    per_predicate: list[dict[str, Any]],
) -> bool:
    spec = REGISTRY.get(name)
    if spec is None:
        raise ValueError(
            f"unknown predicate {name!r} at {path}; registered: {sorted(REGISTRY.keys())}"
        )
    args = resolve_predicate_args(spec, payload)
    try:
        success, diagnostics = spec.evaluator(state, args)
    except KeyError as exc:
        # Body name in args doesn't exist in the rolled-out scene.
        # Treat as a soft failure with a diagnostic; the AST itself is well-
        # formed (validate_goal_ast already passed), the runtime scene just
        # diverged from what the LLM author named.
        success = False
        diagnostics = {
            "error": "missing_body",
            "message": str(exc).strip("'"),
            "available_bodies": sorted(state.bodies.keys()),
        }
    per_predicate.append({
        "ast_path": path,
        "predicate": name,
        "args": args,
        "success": bool(success),
        "diagnostics": dict(diagnostics or {}),
    })
    return bool(success)


__all__ = [
    "BodyView",
    "EvalResult",
    "SimState",
    "SimTrace",
    "evaluate_goal_ast",
    "validate_goal_ast",
]
