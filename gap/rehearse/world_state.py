"""Compact, JSON-safe summaries of a privileged :class:`World` snapshot.

Rehearsal records the world at every node boundary. A full ``World`` is far
too large to keep per visit, so this module reduces it to what an author
needs to see: where the end-effector is, how open the gripper is, which body
it holds, and where each task object is and what it touches. Differences
between two summaries (``diff``) give per-object displacement and the
held/gripper transitions that make a node's physical effect legible.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _vec(v: Any, n: int = 3) -> list[float] | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size < n or not np.all(np.isfinite(arr[:n])):
        return None
    return [round(float(x), 4) for x in arr[:n]]


def _is_robot_body(world: Any, name: str) -> bool:
    prefixes = tuple(getattr(world, "robot_link_prefixes", ()) or ())
    robot_name = getattr(world, "robot_body_name", None)
    return name == robot_name or any(name.startswith(p) for p in prefixes)


def object_names(world: Any, objects: list[str] | None = None) -> list[str]:
    """Task-object bodies: everything that is not robot, table, or a region."""
    bodies = getattr(world, "bodies", {}) or {}
    if objects:
        return [n for n in objects if n in bodies]
    table = getattr(world, "tabletop_body_name", None)
    out = []
    for name, body in bodies.items():
        if name == table or _is_robot_body(world, name):
            continue
        if getattr(body, "is_region", False):
            continue
        out.append(name)
    return sorted(out)


def summarize(world: Any, objects: list[str] | None = None) -> dict[str, Any]:
    """Reduce a World snapshot to a small JSON-safe dictionary."""
    summary: dict[str, Any] = {
        "time_s": None,
        "ee": None,
        "joints": None,
        "gripper_open_fraction": None,
        "held": None,
        "objects": {},
        "robot_contacts": [],
    }
    if world is None:
        return summary
    time_s = getattr(world, "time_s", None)
    summary["time_s"] = None if time_s is None else round(float(time_s), 4)
    robot = getattr(world, "robot_view", None)
    if robot is not None:
        summary["ee"] = {
            "position": _vec(getattr(robot, "ee_position", None)),
            "quaternion_wxyz": _vec(getattr(robot, "ee_quaternion_wxyz", None), 4),
        }
        joints = getattr(robot, "joint_pos", None)
        if joints is not None:
            summary["joints"] = [round(float(x), 4) for x in np.asarray(joints, dtype=float).ravel()]
        frac = getattr(robot, "gripper_open_fraction", None)
        summary["gripper_open_fraction"] = None if frac is None else round(float(frac), 4)
    try:
        held = world.held_body()
        summary["held"] = getattr(held, "name", None) if held is not None else None
    except Exception:
        summary["held"] = None
    robot_contacts: set[str] = set()
    for name in object_names(world, objects):
        body = world.bodies[name]
        contacts = sorted(getattr(body, "contacts", ()) or ())
        if any(_is_robot_body(world, c) for c in contacts):
            robot_contacts.add(name)
        entry: dict[str, Any] = {
            "position": _vec(getattr(body, "position", None)),
            "quaternion_wxyz": _vec(getattr(body, "quaternion_wxyz", None), 4),
            "contacts": contacts,
        }
        try:
            entry["settled"] = bool(body.is_settled())
        except Exception:
            pass
        joints = getattr(body, "joints", None)
        if joints:
            entry["joints"] = {str(k): round(float(v), 4) for k, v in dict(joints).items()}
        summary["objects"][name] = entry
    summary["robot_contacts"] = sorted(robot_contacts)
    return summary


def diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Physical effect of a node: displacement per object and held/gripper change."""
    before = before or {}
    after = after or {}
    out: dict[str, Any] = {
        "held_before": before.get("held"),
        "held_after": after.get("held"),
        "gripper_before": before.get("gripper_open_fraction"),
        "gripper_after": after.get("gripper_open_fraction"),
        "displacement_m": {},
    }
    objs_b = before.get("objects", {}) or {}
    objs_a = after.get("objects", {}) or {}
    for name, entry in objs_a.items():
        pa = entry.get("position")
        pb = (objs_b.get(name) or {}).get("position")
        if pa is None or pb is None:
            continue
        out["displacement_m"][name] = round(float(np.linalg.norm(np.subtract(pa, pb))), 4)
    return out


#: A pose component counts as changed beyond these tolerances. They are
#: numerical-noise floors, the same for every body and every task.
POSITION_TOL_M = 1e-3
ANGLE_TOL_DEG = 1.0
#: An interior joint (drawer, door, knob) counts as moved beyond this: 1 mm
#: for a slide, 0.06 degree for a hinge.
JOINT_TOL = 1e-3


def joints_changed(before: dict[str, Any] | None, after: dict[str, Any] | None, *, tol: float = JOINT_TOL) -> bool:
    """True when any interior joint of a body differs beyond the tolerance."""
    jb, ja = (before or {}).get("joints") or {}, (after or {}).get("joints") or {}
    if set(jb) != set(ja):
        return True
    return any(abs(float(ja[k]) - float(jb[k])) > tol for k in jb)


def _angle_deg(qa: list[float] | None, qb: list[float] | None) -> float | None:
    """Rotation angle between two unit quaternions, in degrees."""
    if qa is None or qb is None:
        return None
    a, b = np.asarray(qa, dtype=float), np.asarray(qb, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    dot = min(1.0, abs(float(np.dot(a / na, b / nb))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def pose_changed(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    position_tol: float = POSITION_TOL_M,
    angle_tol: float = ANGLE_TOL_DEG,
) -> bool:
    """True when position or orientation differs beyond the tolerances."""
    before, after = before or {}, after or {}
    pb, pa = before.get("position"), after.get("position")
    if (pb is None) != (pa is None):
        return True
    if pb is not None and float(np.linalg.norm(np.subtract(pa, pb))) > position_tol:
        return True
    angle = _angle_deg(before.get("quaternion_wxyz"), after.get("quaternion_wxyz"))
    return angle is not None and angle > angle_tol


def changed_bodies(states: list[dict[str, Any] | None], **tol: float) -> list[str]:
    """Bodies whose pose, joints or contacts differ from the first state at any later one."""
    states = [s for s in states if s]
    if len(states) < 2:
        return []
    first = states[0].get("objects") or {}
    out = []
    for name in sorted({n for s in states for n in (s.get("objects") or {})}):
        ref = first.get(name)
        for s in states[1:]:
            cur = (s.get("objects") or {}).get(name)
            if (ref is None) != (cur is None):
                out.append(name)
                break
            if cur is None:
                continue
            if (pose_changed(ref, cur, **tol) or joints_changed(ref, cur)
                    or (ref.get("contacts") or []) != (cur.get("contacts") or [])):
                out.append(name)
                break
    return out


def robot_changed(states: list[dict[str, Any] | None], **tol: float) -> bool:
    """True when the hand pose or the gripper opening changes over the states."""
    states = [s for s in states if s]
    if len(states) < 2:
        return False
    ref = states[0]
    for s in states[1:]:
        if pose_changed(ref.get("ee"), s.get("ee"), **tol):
            return True
        ga, gb = ref.get("gripper_open_fraction"), s.get("gripper_open_fraction")
        if (ga is None) != (gb is None) or (ga is not None and abs(ga - gb) > 1e-3):
            return True
    return False


__all__ = [
    "summarize", "diff", "object_names", "pose_changed", "joints_changed", "changed_bodies", "robot_changed",
    "POSITION_TOL_M", "ANGLE_TOL_DEG", "JOINT_TOL",
]
