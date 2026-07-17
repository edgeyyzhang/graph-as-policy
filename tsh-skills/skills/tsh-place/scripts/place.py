"""Place the held tape on the perceived duct top face, then retract.

The receiver does NOT hold the tape centred in its gripper: it threaded the rim
at a revolved clock position (see bimanual_exchange), so the tape centre sits
several cm off the TCP. The tape is therefore treated as the receiver's end
effector: the exchange MEASURED the held-tape offset (``receiver_offset``, via
the giver's rigid grip + FK, no ground truth), and every place target here is a
TAPE-centre pose — the offset is composed into the plan (``plan_tool_move``) or
subtracted once for the linear legs.

WHERE to place comes from ``perceive_duct`` (RGB-D top-face centre, read up
front while the view was clean — at place time the held tape + gripper occlude
the duct). The tape is round, so it lays flat identically at any yaw about the
vertical: the yaw sweep pivots the TCP around the tape centre until the planner
finds a reachable pose — the ring's own symmetry is the reach margin.

Motion: yaw sweep (plan probe) -> approach the hover with the tape attached as a
collision body (``curobo.plan_with_grasped_object``, so the face-on ->
gripper-down reorient can't clip the arm) -> straight-down set-down and
straight-up retract (``curobo_linear_move``, orientation locked).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import PointCloud

from ._motion import curobo_linear_move, plan_tool_move
from .constants import (
    DOWN_QUAT,
    PLACE_DROP_CLEARANCE,
    PLACE_OFFSET,
    PLACE_RETRACT_OFFSET,
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


def _q_to_R(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return Rotation.from_quat([qx, qy, qz, qw])


def _approach_with_attached_tape(ctx: NodeContext, arm_id: int, tape_target_xyz, quat_wxyz,
                                 tool_offset, tape_center_world, tape_cloud):
    """Plan + execute the hover approach with the held tape attached as a cuRobo
    collision body, so the face-on -> gripper-down reorient can't clip the ring
    into the arm.

    The pickup's perceived cloud is RECENTRED at the tape's CURRENT centre
    (``tape_center_world`` = receiver TCP + measured offset) before attaching —
    the receiver holds the tape far from where the cloud was captured. REQUIRES
    a usable cloud (no tuned bounding-sphere fallback). Returns False when the
    plan fails so the caller can fall back to a plain (unattached) approach.
    """
    import trimesh

    frame = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)
    base = np.asarray(frame["position"], dtype=np.float32)
    center = np.asarray(tape_center_world, dtype=np.float32)

    raw = tape_cloud.get("points") if isinstance(tape_cloud, dict) else tape_cloud
    pts = np.asarray(raw, dtype=np.float32) if raw is not None and len(raw) >= 4 else None
    if pts is None:
        raise RuntimeError(
            "place: tape_cloud missing or degenerate (<4 points) — no tuned "
            "bounding-sphere fallback")
    pts_base = (pts - pts.mean(0) + center) - base
    hull = trimesh.convex.convex_hull(trimesh.PointCloud(pts_base))
    verts, faces = hull.vertices.tolist(), hull.faces.tolist()

    # plan_with_grasped_object plans the EE link, so compose the TCP + held-tape
    # offsets out of the tape-centre target once, here.
    R = _q_to_R(quat_wxyz)
    off = np.asarray(frame["tcp_offset"], float) + np.asarray(tool_offset, float)
    ee = np.asarray(tape_target_xyz, float) - R.apply(off) - base
    qw, qx, qy, qz = quat_wxyz
    try:
        res = ctx.tool(
            "curobo.plan_with_grasped_object",
            world_config={"meshes": [{"name": "held_tape", "vertices": verts, "faces": faces}]},
            start_joint_position={"positions": frame["joints"]},
            target_pose={"position": {"x": float(ee[0]), "y": float(ee[1]), "z": float(ee[2])},
                         "rotation": {"w": qw, "x": qx, "y": qy, "z": qz}},
            object_name="held_tape",
            robot_file="yam.yml",
            position_threshold=PLAN_POSITION_THRESHOLD,
            rotation_threshold=PLAN_ROTATION_THRESHOLD,
        )
    except Exception as exc:  # degrade, don't abort — the plain hover still lands
        print(f"[place] attached-tape plan errored ({exc}); plain hover fallback",
              flush=True)
        return False
    if not res.get("success") or not res.get("trajectory"):
        return False
    ctx.tool("libero-yam.execute_trajectory", trajectory=res["trajectory"], arm_id=arm_id)
    return True


def run(ctx: NodeContext, *, receiver_offset: list, duct_xyz: list, arm_id: int = 1,
        tape_half_z: float | None = None, tape_cloud: PointCloud | None = None) -> Output:
    """Lay the held tape flat on the duct top face, release, retract straight up.

    receiver_offset: tape centre in the receiver TCP frame, measured by the
                     exchange (no GT) — the receiver's tool offset here.
    duct_xyz:        duct top-face centre from the perceive_duct step.
    tape_half_z:     perceived tape half-thickness (perceive_tape); the tape centre
                     rests this far above the duct top face so the tape sits flush.
                     REQUIRED — a missing value raises (no tuned fallback).
    tape_cloud:      perceived tape point cloud from perceive_tape; attached as
                     the collision body for the approach swing. REQUIRED — a
                     missing/degenerate cloud raises (no tuned bounding-sphere
                     fallback).
    """
    if tape_half_z is None:
        raise RuntimeError(
            "tsh-place requires perceived tape_half_z from tsh-perceive "
            "— none supplied (no tuned fallback)")
    offset_local = np.asarray(
        [receiver_offset["x"], receiver_offset["y"], receiver_offset["z"]]
        if isinstance(receiver_offset, dict) else receiver_offset, dtype=float)
    seq = ([duct_xyz["x"], duct_xyz["y"], duct_xyz["z"]]
           if isinstance(duct_xyz, dict) else list(duct_xyz))
    # Rest the tape flush: centre = duct top face + tape half-thickness (perceived),
    # plus the optional PLACE_OFFSET nudge (default zero).
    desired_tape_centre = (np.asarray(seq, dtype=float) + np.asarray(PLACE_OFFSET)
                           + np.array([0.0, 0.0, float(tape_half_z)]))
    hover_dz = np.array([0.0, 0.0, PLACE_Z_APPROACH])

    # Where the attached-tape collision body sits right now (receiver FK + offset).
    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    p_tcp = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
    r = ee["rotation"]
    tape_center_world = p_tcp + Rotation.from_quat(
        [r["x"], r["y"], r["z"], r["w"]]).apply(offset_local)

    # Yaw sweep: rotate the gripper about the tape's vertical axis until the
    # hover (the most reach-constrained waypoint) is plannable. Seed at the
    # "offset points toward the arm base" yaw — usually the most reachable.
    base_xy = np.asarray(
        ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"])[:2]
    h = _q_to_R(DOWN_QUAT).apply(offset_local)[:2]
    from_base = desired_tape_centre[:2] - base_xy
    yaw0 = np.arctan2(from_base[1], from_base[0]) - np.arctan2(h[1], h[0])

    chosen = None
    for dyaw in PLACE_YAW_SWEEP_DEG:
        R_place = Rotation.from_rotvec([0.0, 0.0, yaw0 + np.radians(dyaw)]) * _q_to_R(DOWN_QUAT)
        qx, qy, qz, qw = R_place.as_quat()
        pq = (qw, qx, qy, qz)
        hover_traj = plan_tool_move(
            ctx, arm_id, desired_tape_centre + hover_dz, pq,
            tool_offset=offset_local, execute=False)
        if hover_traj is not None:
            chosen = (pq, R_place, hover_traj)
            break
    if chosen is None:
        raise RuntimeError("place: no reachable yaw found sweeping the tape rotation")
    place_quat, R_place, hover_traj = chosen
    tcp_place = desired_tape_centre - R_place.apply(offset_local)

    # Approach the hover with the tape attached (collision-aware reorient); fall
    # back to the plain probed hover plan so this can't break a reachable place.
    if not _approach_with_attached_tape(ctx, arm_id, desired_tape_centre + hover_dz,
                                        place_quat, offset_local,
                                        tape_center_world, tape_cloud):
        ctx.tool("libero-yam.execute_trajectory", trajectory=hover_traj, arm_id=arm_id)

    # Descend most of the way, stop PLACE_DROP_CLEARANCE above the surface so
    # the fingers can open without jamming against the duct, then retract.
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
            "place_tcp": [float(v) for v in (*tcp_place, *place_quat)]}

