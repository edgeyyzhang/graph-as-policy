"""Built-in predicate evaluators for :mod:`gap.runtime.predicates.goal_eval`.

Each function takes a :class:`SimState` plus a kwargs mapping (already resolved
from the AST payload by :func:`gap.runtime.predicates.predicates.resolve_predicate_args`)
and returns a ``(bool, diagnostics_dict)`` tuple. Registration via
``@register_predicate`` populates the global registry on first import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from gap.runtime.predicates.predicates import register_predicate

# ---------------------------------------------------------------------------
# Spatial relations: on / in / near / above / stack
# ---------------------------------------------------------------------------


@register_predicate(
    "on",
    body_args=("target", "container"),
    optional_args={"tol_m": 0.05},
    description="Target's bottom face within tol_m of container's top face, "
                "with xy overlap.",
)
def on(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    c = state.body(args["container"])
    tol = float(args["tol_m"])
    dz = t.bottom_z - c.top_z
    xy_overlap = (
        t.right_x  >= c.left_x and
        t.left_x   <= c.right_x and
        t.far_y    >= c.near_y and
        t.near_y   <= c.far_y
    )
    success = abs(dz) <= tol and xy_overlap
    return success, {
        "dz": dz,
        "abs_dz": abs(dz),
        "xy_overlap": bool(xy_overlap),
        "tol_m": tol,
    }


@register_predicate(
    "in",
    body_args=("target", "container"),
    optional_args={
        "tol_m": 0.0,
        "tol_xy_m": 0.02,
        "tol_z_m": 0.02,
    },
    description="LIBERO-BDDL-aligned: target is in contact with container AND "
                "its centre is inside the container's interior cavity (when a "
                "``<container>__cavity`` site is registered) or the union AABB "
                "(otherwise). Mirrors ``check_contact AND check_contain``. "
                "``tol_m`` is the legacy isotropic slack; ``tol_xy_m`` and "
                "``tol_z_m`` (both default 0.02 m, matching robosuite's "
                "contact-success threshold) prevent small terminal-pose drift "
                "from flipping the binary verdict asymmetrically across tasks. "
                "Both ``contain`` and ``contain_score`` (soft 0..1 overlap) "
                "are surfaced in diagnostics.",
)
def in_(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    c = state.body(args["container"])
    tol = float(args["tol_m"])
    tol_xy = float(args["tol_xy_m"])
    tol_z = float(args["tol_z_m"])
    p = t.position

    # Containment: cavity AABB if a sibling "<container>__cavity" site is
    # registered; otherwise fall back to the union AABB (legacy behaviour
    # for non-hollow containers).
    if c.cavity_lower is not None and c.cavity_upper is not None:
        lo, hi = c.cavity_lower, c.cavity_upper
        contain_via = "cavity_site"
    else:
        lo, hi = c.aabb_lower, c.aabb_upper
        contain_via = "union_aabb"

    # Anisotropic slack: xy gets ``tol_xy_m``; z gets ``tol_z_m``. The
    # legacy isotropic ``tol_m`` is added on top so existing callers
    # that bumped only ``tol_m`` keep their old semantics.
    sx = tol + tol_xy
    sy = tol + tol_xy
    sz = tol + tol_z
    contain = (
        lo[0] - sx <= p[0] <= hi[0] + sx
        and lo[1] - sy <= p[1] <= hi[1] + sy
        and lo[2] - sz <= p[2] <= hi[2] + sz
    )

    # Soft score: how far inside the (slack-expanded) cavity the target's
    # centre is, normalized by half-extents. 1.0 at centre, 0.0 at edge,
    # negative outside. Surfaces in diagnostics as ``contain_score`` so
    # callers can see "barely missed" vs "wildly off".
    half = np.maximum((hi - lo) / 2.0 + np.array([sx, sy, sz]), 1e-6)
    centre = (lo + hi) / 2.0
    score_xyz = 1.0 - np.abs(p - centre) / half
    contain_score = float(np.min(score_xyz))

    # Contact: target body in contact with container body (any geom pair).
    # Mirrors LIBERO's ``check_contact``. ``BodyView.contacts`` collapses
    # geom-pair contacts to body-name pairs in goal_eval._snapshot_mujoco.
    contact = c.name in t.contacts

    success = bool(contain and contact)
    return success, {
        "target_pos": p.tolist(),
        "container_lower": lo.tolist(),
        "container_upper": hi.tolist(),
        "contain": bool(contain),
        "contact": bool(contact),
        "contain_via": contain_via,
        "contain_score": contain_score,
        "tol_m": tol,
        "tol_xy_m": tol_xy,
        "tol_z_m": tol_z,
    }


@register_predicate(
    "near",
    body_args=("target", "anchor"),
    optional_args={"tol_m": 0.10},
    description="3D Euclidean distance between target and anchor centers ≤ tol_m.",
)
def near(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    a = state.body(args["anchor"])
    tol = float(args["tol_m"])
    dist = float(np.linalg.norm(t.position - a.position))
    return dist <= tol, {"dist_m": dist, "tol_m": tol}


@register_predicate(
    "above",
    body_args=("target", "anchor"),
    optional_args={"min_clearance_m": 0.0, "require_xy_overlap": True},
    description="Target's bottom face is above anchor's top face by at least "
                "min_clearance_m (default 0). With require_xy_overlap=True, the "
                "two AABBs must overlap in x and y.",
)
def above(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    a = state.body(args["anchor"])
    clearance = float(args["min_clearance_m"])
    z_clear = t.bottom_z - a.top_z >= clearance
    if args["require_xy_overlap"]:
        xy_overlap = (
            t.right_x  >= a.left_x and t.left_x  <= a.right_x and
            t.far_y    >= a.near_y and t.near_y  <= a.far_y
        )
    else:
        xy_overlap = True
    success = bool(z_clear and xy_overlap)
    return success, {
        "z_clearance": t.bottom_z - a.top_z,
        "z_clear": bool(z_clear),
        "xy_overlap": bool(xy_overlap),
    }


@register_predicate(
    "stack",
    body_args=("top", "bottom"),
    optional_args={"tol_m": 0.02},
    description="Top body's bottom face within tol_m of bottom body's top face, "
                "with xy overlap. Tighter than ``on``.",
)
def stack(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    top = state.body(args["top"])
    bot = state.body(args["bottom"])
    tol = float(args["tol_m"])
    dz = top.bottom_z - bot.top_z
    xy_overlap = (
        top.right_x  >= bot.left_x and top.left_x  <= bot.right_x and
        top.far_y    >= bot.near_y and top.near_y  <= bot.far_y
    )
    success = abs(dz) <= tol and xy_overlap
    return success, {
        "dz": dz, "abs_dz": abs(dz),
        "xy_overlap": bool(xy_overlap), "tol_m": tol,
    }


# ---------------------------------------------------------------------------
# Containment: inside_obb / xy_within
# ---------------------------------------------------------------------------


@register_predicate(
    "inside_obb",
    body_args=("target", "reference"),
    optional_args={"tol_m": 0.0},
    description="Target's center inside reference body's world-AABB (with tol).",
)
def inside_obb(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    r = state.body(args["reference"])
    tol = float(args["tol_m"])
    p = t.position
    inside = (
        r.aabb_lower[0] - tol <= p[0] <= r.aabb_upper[0] + tol and
        r.aabb_lower[1] - tol <= p[1] <= r.aabb_upper[1] + tol and
        r.aabb_lower[2] - tol <= p[2] <= r.aabb_upper[2] + tol
    )
    return inside, {
        "target_pos": p.tolist(),
        "ref_aabb_lower": r.aabb_lower.tolist(),
        "ref_aabb_upper": r.aabb_upper.tolist(),
    }


@register_predicate(
    "xy_within",
    body_args=("target", "reference"),
    optional_args={"radius_m": 0.05},
    description="2D Euclidean distance between target and reference centers "
                "(in xy plane) ≤ radius_m.",
)
def xy_within(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    r = state.body(args["reference"])
    radius = float(args["radius_m"])
    dist = float(np.linalg.norm(t.xy - r.xy))
    return dist <= radius, {"xy_dist_m": dist, "radius_m": radius}


# ---------------------------------------------------------------------------
# Manipulation: grasped / released
# ---------------------------------------------------------------------------


def _bodies_belonging_to(state, robot_name: str) -> set[str]:
    """Return all body-names that match ``robot_name`` or are prefixed by it.

    Heuristic: a body belongs to the robot if its name equals ``robot_name``
    or starts with ``robot_name`` (so `panda_` matches `panda_link0`,
    `panda_hand`, etc., and a wrapper body like `sim_robot_base`
    matches all its descendants if the LLM names them with that prefix).
    Also matches a hard-coded set of well-known Panda link prefixes when
    the robot name is the wrapper.
    """
    matches: set[str] = set()
    well_known = ("panda_", "franka_", "left_finger", "right_finger", "gripper")
    for nm in state.bodies.keys():
        if nm == robot_name or nm.startswith(robot_name):
            matches.add(nm)
            continue
        if any(nm.startswith(p) or p in nm for p in well_known):
            matches.add(nm)
    return matches


@register_predicate(
    "grasped",
    body_args=("target", "robot"),
    optional_args={},
    description="Target body is in contact with the robot (any body whose name "
                "matches the robot or starts with its prefix; falls back to "
                "well-known Panda link patterns).",
)
def grasped(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    robot_bodies = _bodies_belonging_to(state, args["robot"])
    overlap = t.contacts & robot_bodies
    return bool(overlap), {
        "robot_contacts": sorted(overlap),
        "robot_bodies_count": len(robot_bodies),
    }


@register_predicate(
    "released",
    body_args=("target", "robot"),
    optional_args={},
    description="Target body is NOT in contact with any robot body. Inverse of grasped.",
)
def released(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    robot_bodies = _bodies_belonging_to(state, args["robot"])
    overlap = t.contacts & robot_bodies
    return not overlap, {"robot_contacts": sorted(overlap)}


# ---------------------------------------------------------------------------
# Orientation: axis_aligned
# ---------------------------------------------------------------------------


@register_predicate(
    "axis_aligned",
    body_args=("body",),
    optional_args={"local_axis": "z", "world_axis": "z", "tol_rad": 0.20},
    description="The named local axis of body (x|y|z), expressed in world "
                "frame, is aligned with the named world axis (x|y|z) within "
                "tol_rad.",
)
def axis_aligned(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    b = state.body(args["body"])
    local = {"x": np.array([1.0, 0.0, 0.0]),
             "y": np.array([0.0, 1.0, 0.0]),
             "z": np.array([0.0, 0.0, 1.0])}[args["local_axis"]]
    world = {"x": np.array([1.0, 0.0, 0.0]),
             "y": np.array([0.0, 1.0, 0.0]),
             "z": np.array([0.0, 0.0, 1.0])}[args["world_axis"]]
    R = _quat_wxyz_to_rotmat(b.quaternion_wxyz)
    local_in_world = R @ local
    cos_angle = float(np.clip(np.dot(local_in_world, world), -1.0, 1.0))
    angle = math.acos(cos_angle)
    tol = float(args["tol_rad"])
    return angle <= tol, {"angle_rad": angle, "tol_rad": tol}


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Articulated joints: joint_threshold (rolls up open/close/turnon/turnoff)
# ---------------------------------------------------------------------------


@register_predicate(
    "joint_threshold",
    body_args=("body",),
    optional_args={"threshold": 0.5, "mode": "gt", "joint_index": 0},
    description="The joint_index-th joint qpos of body satisfies the threshold "
                "comparison: gt (q > threshold), lt (q < threshold), "
                "abs_gt (|q| > threshold), abs_lt (|q| < threshold).",
)
def joint_threshold(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    b = state.body(args["body"])
    if b.joint_qpos.size == 0:
        return False, {"reason": "body has no joints", "joint_qpos_size": 0}
    idx = int(args["joint_index"])
    if idx >= b.joint_qpos.size:
        return False, {
            "reason": "joint_index out of range",
            "joint_index": idx,
            "joint_qpos_size": int(b.joint_qpos.size),
        }
    q = float(b.joint_qpos[idx])
    th = float(args["threshold"])
    mode = args["mode"]
    if mode == "gt":
        success = q > th
    elif mode == "lt":
        success = q < th
    elif mode == "abs_gt":
        success = abs(q) > th
    elif mode == "abs_lt":
        success = abs(q) < th
    else:
        raise ValueError(
            f"joint_threshold mode must be one of gt|lt|abs_gt|abs_lt; got {mode!r}"
        )
    return success, {"q": q, "threshold": th, "mode": mode}


# ---------------------------------------------------------------------------
# Pose: at_pose (kept from old vocabulary, generalized)
# ---------------------------------------------------------------------------


@register_predicate(
    "at_pose",
    body_args=("target",),
    optional_args={
        "position": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "tol_m": 0.05,
        "tol_rad": 0.20,
    },
    description="Target body within tol_m / tol_rad of an explicit world-frame "
                "pose (position + quaternion_wxyz).",
)
def at_pose(state, args: Mapping[str, Any]) -> tuple[bool, dict]:
    t = state.body(args["target"])
    target_pos = np.asarray(args["position"], dtype=np.float64)
    target_quat = np.asarray(args["quaternion_wxyz"], dtype=np.float64)
    pos_err = float(np.linalg.norm(t.position - target_pos))
    # Angular error: dot(q1, q2) (after normalization) → cos(angle/2).
    q1 = t.quaternion_wxyz / max(np.linalg.norm(t.quaternion_wxyz), 1e-12)
    q2 = target_quat / max(np.linalg.norm(target_quat), 1e-12)
    cos_half = abs(float(np.dot(q1, q2)))
    cos_half = min(1.0, max(-1.0, cos_half))
    ang_err = 2.0 * math.acos(cos_half)
    tol_m = float(args["tol_m"])
    tol_rad = float(args["tol_rad"])
    return pos_err <= tol_m and ang_err <= tol_rad, {
        "pos_err_m": pos_err, "ang_err_rad": ang_err,
        "tol_m": tol_m, "tol_rad": tol_rad,
    }


__all__ = [
    "on", "in_", "near", "above", "stack",
    "inside_obb", "xy_within",
    "grasped", "released",
    "axis_aligned",
    "joint_threshold",
    "at_pose",
]
