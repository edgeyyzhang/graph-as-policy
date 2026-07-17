"""Place the held tape on the perceived destination top face, then retract.

The holder does NOT hold the tape centred in its gripper: after a ring grasp
or a rim thread the tape centre sits several cm off the TCP. The tape is
therefore treated as the holder's end effector: the upstream step MEASURED
the held-tape offset (``held_offset`` — the pickup's ``tape_in_giver`` on the
direct route, or the exchange's ``receiver_offset`` after a handover; both
via rigid-grip FK, no ground truth), and every place target here is a
TAPE-centre pose — the offset is composed into the plan (the shared
``_held`` core) or subtracted once for the linear legs.

WHERE to place comes from the destination perceive-object subgraph (RGB-D
top-face centre, read up front while the view was clean — at place time the
held tape + gripper occlude it). The tape is round, so it lays flat
identically at any yaw about the vertical: the yaw sweep pivots the TCP
around the tape centre until the planner finds a reachable pose — the ring's
own symmetry is the reach margin.

Motion: yaw sweep (plan probe) -> approach the hover with the tape attached
as a collision body (``_held.approach_with_attached``, so the face-on ->
gripper-down reorient can't clip the arm) -> straight-down set-down and
straight-up retract (``curobo_linear_move``, orientation locked).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import PointCloud

from ._held import approach_with_attached, as_vec3, held_center_world, q_to_R
from ._motion import curobo_linear_move, plan_tool_move
from .constants import (
    DOWN_QUAT,
    PLACE_DROP_CLEARANCE,
    PLACE_OFFSET,
    PLACE_YAW_SWEEP_DEG,
    PLACE_Z_APPROACH,
    PLAN_POSITION_THRESHOLD,
    PLAN_ROTATION_THRESHOLD,
)


class Output(TypedDict):
    placed: bool
    place_tcp: list  # world TCP pose whose grip matches the RESTING tape,
                     # [x,y,z,qw,qx,qy,qz] (the release pose minus the drop
                     # clearance). A return leg re-closes exactly here to
                     # re-acquire the tape with the same measured grip.
    place_arm: int   # echo of the placing arm (checkpoint anchor)


def run(ctx: NodeContext, *, held_offset: list, dest_xyz: list, arm_id: int = 1,
        tape_half_z: float | None = None, tape_cloud: PointCloud | None = None) -> Output:
    """Lay the held tape flat on the destination top face, release, retract up.

    held_offset: tape centre in the holder's TCP frame — the pickup's
                 ``tape_in_giver`` (direct route) or the exchange's measured
                 ``receiver_offset`` (handover route); the holder's tool
                 offset here.
    dest_xyz:    destination top-face centre (the perceive-object subgraph's
                 ``dest_top_xyz`` output).
    arm_id:      the holding arm (route-decided).
    tape_half_z: perceived tape half-thickness; the tape centre rests this far
                 above the destination top face so the tape sits flush.
                 REQUIRED — a missing value raises (no tuned fallback).
    tape_cloud:  perceived tape point cloud; attached as the collision body
                 for the approach swing. REQUIRED — a missing/degenerate
                 cloud raises (no tuned bounding-sphere fallback).
    """
    if tape_half_z is None:
        raise RuntimeError(
            "tsh-place requires perceived tape_half_z from the perceive "
            "subgraph — none supplied (no tuned fallback)")
    offset_local = as_vec3(held_offset)
    seq = as_vec3(dest_xyz)
    # Rest the tape flush: centre = dest top face + tape half-thickness (perceived),
    # plus the optional PLACE_OFFSET nudge (default zero).
    desired_tape_centre = (seq + np.asarray(PLACE_OFFSET)
                           + np.array([0.0, 0.0, float(tape_half_z)]))
    hover_dz = np.array([0.0, 0.0, PLACE_Z_APPROACH])

    # Where the attached-tape collision body sits right now (holder FK + offset).
    tape_center_world = held_center_world(ctx, arm_id, offset_local)

    # Yaw sweep: rotate the gripper about the tape's vertical axis until the
    # hover (the most reach-constrained waypoint) is plannable. Seed at the
    # "offset points toward the arm base" yaw — usually the most reachable.
    base_xy = np.asarray(
        ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"])[:2]
    h = q_to_R(DOWN_QUAT).apply(offset_local)[:2]
    from_base = desired_tape_centre[:2] - base_xy
    yaw0 = np.arctan2(from_base[1], from_base[0]) - np.arctan2(h[1], h[0])

    chosen = None
    for dyaw in PLACE_YAW_SWEEP_DEG:
        R_place = Rotation.from_rotvec([0.0, 0.0, yaw0 + np.radians(dyaw)]) * q_to_R(DOWN_QUAT)
        qx, qy, qz, qw = R_place.as_quat()
        pq = (qw, qx, qy, qz)
        hover_traj = plan_tool_move(
            ctx, arm_id, desired_tape_centre + hover_dz, pq,
            tool_offset=list(offset_local), execute=False)
        if hover_traj is not None:
            chosen = (pq, R_place, hover_traj)
            break
    if chosen is None:
        raise RuntimeError("place: no reachable yaw found sweeping the tape rotation")
    place_quat, R_place, hover_traj = chosen
    tcp_place = desired_tape_centre - R_place.apply(offset_local)

    # Approach the hover with the tape attached (collision-aware reorient); fall
    # back to the plain probed hover plan so this can't break a reachable place.
    if not approach_with_attached(ctx, arm_id, desired_tape_centre + hover_dz,
                                  place_quat, offset_local,
                                  tape_center_world, tape_cloud,
                                  object_name="held_tape",
                                  position_threshold=PLAN_POSITION_THRESHOLD,
                                  rotation_threshold=PLAN_ROTATION_THRESHOLD):
        ctx.tool("libero-yam.execute_trajectory", trajectory=hover_traj, arm_id=arm_id)

    # Descend most of the way, stop PLACE_DROP_CLEARANCE above the surface so
    # the fingers can open without jamming against the dest, then retract.
    drop_descent = PLACE_Z_APPROACH - PLACE_DROP_CLEARANCE
    tcp_release = tcp_place + np.array([0.0, 0.0, PLACE_DROP_CLEARANCE])
    curobo_linear_move(ctx, arm_id, list(tcp_release), place_quat, allowed_axes=["Z"],
                       distance=drop_descent, direction=(0.0, 0.0, -1.0))
    ctx.tool("robot.open_gripper", arm_id=arm_id)
    curobo_linear_move(ctx, arm_id,
                       list(tcp_place + np.array([0.0, 0.0, PLACE_Z_APPROACH])),
                       place_quat, allowed_axes=["Z"],
                       distance=PLACE_Z_APPROACH - PLACE_DROP_CLEARANCE,
                       direction=(0.0, 0.0, 1.0))

    return {"placed": True,
            "place_tcp": [float(v) for v in (*tcp_place, *place_quat)],
            "place_arm": int(arm_id)}
