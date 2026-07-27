"""Compute a reachable place pose for the held object at the destination.

Determines WHERE the held object's centre should rest (dest top face + tape
half-thickness + optional PLACE_OFFSET nudge) and WHICH yaw about the vertical
is reachable for the hover approach — a plain reachability probe
(``plan_tool_move(execute=False)``), no attached-collision-body planning and
no execution: this node never moves the arm. The round tape lays flat
identically at any yaw, so the sweep only trades reach; the first reachable
yaw wins.

Feeds ``tsh-transport-held`` (the collision-aware hover approach) and
``tsh-place`` (the descend/release/retract legs) — factored out so both
consume the SAME probed, reachable pose instead of each re-deriving it.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import Quaternion, Vec3

from ._motion import plan_tool_move
from .constants import DOWN_QUAT, PLACE_OFFSET, PLACE_YAW_SWEEP_DEG, PLACE_Z_APPROACH


def _as_vec3(v: Vec3) -> np.ndarray:
    return np.asarray([v["x"], v["y"], v["z"]], dtype=float)


def _vec3(v) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(q) -> Quaternion:
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


def _q_to_R(quat_wxyz) -> Rotation:
    qw, qx, qy, qz = quat_wxyz
    return Rotation.from_quat([qx, qy, qz, qw])


class Output(TypedDict):
    place_xyz: Vec3       # world OBJECT-centre rest pose (tape flush on the destination)
    hover_xyz: Vec3       # place_xyz raised by PLACE_Z_APPROACH — the approach target
    place_quat: Quaternion  # reachable presentation orientation (wxyz), first yaw that plans
    target_xyz: Vec3      # alias of hover_xyz — matches tsh-transport-held's generic
                       # target_xyz input so cross-subgraph auto-wire can bind it
    target_quat: Quaternion  # alias of place_quat — matches tsh-transport-held's target_quat


def run(ctx: NodeContext, *, held_offset: Vec3, container_xyz: Vec3, tape_half_z: float,
        arm_id: int = 1) -> Output:
    """Probe the first reachable yaw for laying the held tape flat on container_xyz.

    held_offset: tape centre in the holder's TCP frame (measured, no GT) — the
                 tool offset the hover reachability probe must account for.
    container_xyz:    destination top-face centre (the perceive-object subgraph's
                 container_top_xyz output).
    tape_half_z: perceived tape half-thickness; the tape centre rests this far
                 above the destination top face so the tape sits flush.
                 REQUIRED — a missing value raises (no tuned fallback).
    """
    offset_local = _as_vec3(held_offset)
    seq = _as_vec3(container_xyz)
    # Rest the tape flush: centre = dest top face + tape half-thickness (perceived),
    # plus the optional PLACE_OFFSET nudge (default zero).
    desired_tape_centre = (seq + np.asarray(PLACE_OFFSET)
                           + np.array([0.0, 0.0, float(tape_half_z)]))
    hover_dz = np.array([0.0, 0.0, PLACE_Z_APPROACH])

    # Yaw sweep: rotate the gripper about the tape's vertical axis until the
    # hover (the most reach-constrained waypoint) is plannable. Seed at the
    # "offset points toward the arm base" yaw — usually the most reachable.
    base_xy = np.asarray(
        ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"])[:2]
    h = _q_to_R(DOWN_QUAT).apply(offset_local)[:2]
    from_base = desired_tape_centre[:2] - base_xy
    yaw0 = np.arctan2(from_base[1], from_base[0]) - np.arctan2(h[1], h[0])

    for dyaw in PLACE_YAW_SWEEP_DEG:
        R_place = Rotation.from_rotvec([0.0, 0.0, yaw0 + np.radians(dyaw)]) * _q_to_R(DOWN_QUAT)
        qx, qy, qz, qw = R_place.as_quat()
        pq = (qw, qx, qy, qz)
        if plan_tool_move(ctx, arm_id, desired_tape_centre + hover_dz, pq,
                          tool_offset=list(offset_local), execute=False) is not None:
            place_xyz = _vec3(desired_tape_centre)
            hover_xyz = _vec3(desired_tape_centre + hover_dz)
            place_quat = _quat(pq)
            return {"place_xyz": place_xyz, "hover_xyz": hover_xyz, "place_quat": place_quat,
                    "target_xyz": hover_xyz, "target_quat": place_quat}
    raise RuntimeError("place-pose: no reachable yaw found sweeping the tape rotation")
