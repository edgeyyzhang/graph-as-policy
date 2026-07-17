"""Shared held-object ("tool-as-EE") motion core for the tape handover.

A held object is planned AS the end effector: the measured rigid offset of
the object centre in the holder's TCP frame (``held_offset`` — from the
pickup's pre-close FK anchor or the exchange's grab-instant measurement) is
composed into the planner's ``tcp_offset``, so "put the tape centre at X" is
one plan — no manual back-solve of the gripper position.

Two consumers share this core so they cannot drift apart:

  * the handover's giver PRESENT leg (tape centre to the meet point), and
  * the place approach (tape centre to the hover above the destination),

plus the standalone ``tsh-transport-held`` skill node for any other
held-object move ("hold the tape above stack N").

``approach_with_attached`` additionally attaches the object's perceived
cloud (recentred at its live FK-tracked centre) as a cuRobo collision body,
so a large reorient swing cannot clip the held object into the arm.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from ._motion import plan_tool_move


def q_to_R(quat_wxyz) -> Rotation:
    qw, qx, qy, qz = quat_wxyz
    return Rotation.from_quat([qx, qy, qz, qw])


def as_vec3(v) -> np.ndarray:
    """Accept a Vec3 dict {x,y,z} (subgraph type coercion) or a sequence."""
    if isinstance(v, dict):
        return np.asarray([v["x"], v["y"], v["z"]], dtype=float)
    return np.asarray(list(v), dtype=float)


def as_wxyz(q) -> tuple:
    """Accept a Quaternion dict {w,x,y,z} or a wxyz sequence; return wxyz tuple."""
    return (q["w"], q["x"], q["y"], q["z"]) if isinstance(q, dict) else tuple(q)


def held_center_world(ctx: NodeContext, arm_id: int, held_offset) -> np.ndarray:
    """Live held-object centre from the holder's FK + the rigid offset — no GT."""
    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    p = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
    r = ee["rotation"]
    R = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])
    return p + R.apply(as_vec3(held_offset))


def plan_held_move(ctx: NodeContext, arm_id: int, held_offset, target_xyz,
                   target_quat_wxyz, *, execute: bool = True, **plan_kwargs):
    """Plan the HELD-object centre to a world pose (offset composed into
    tcp_offset). Same contract as ``_motion.plan_tool_move``."""
    return plan_tool_move(ctx, arm_id, target_xyz, target_quat_wxyz,
                          tool_offset=list(as_vec3(held_offset)),
                          execute=execute, **plan_kwargs)


def approach_with_attached(ctx: NodeContext, arm_id: int, target_xyz, quat_wxyz,
                           held_offset, center_world, held_cloud,
                           *, object_name: str = "held_object",
                           position_threshold: float | None = None,
                           rotation_threshold: float | None = None) -> bool:
    """Plan + execute a held-object move with the object attached as a cuRobo
    collision body, so a large reorient can't clip it into the arm.

    The perceived cloud is RECENTRED at the object's CURRENT centre
    (``center_world`` = holder FK + measured offset) before attaching — the
    holder may hold the object far from where the cloud was captured.
    REQUIRES a usable cloud (no tuned bounding-sphere fallback). Returns
    False when the plan fails so the caller can fall back to a plain
    (unattached) ``plan_held_move``.
    """
    import trimesh

    frame = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)
    base = np.asarray(frame["position"], dtype=np.float32)
    center = np.asarray(center_world, dtype=np.float32)

    raw = held_cloud.get("points") if isinstance(held_cloud, dict) else held_cloud
    pts = np.asarray(raw, dtype=np.float32) if raw is not None and len(raw) >= 4 else None
    if pts is None:
        raise RuntimeError(
            f"{object_name}: held cloud missing or degenerate (<4 points) — "
            "no tuned bounding-sphere fallback")
    pts_base = (pts - pts.mean(0) + center) - base
    hull = trimesh.convex.convex_hull(trimesh.PointCloud(pts_base))
    verts, faces = hull.vertices.tolist(), hull.faces.tolist()

    # plan_with_grasped_object plans the EE link, so compose the TCP + held
    # offsets out of the object-centre target once, here.
    R = q_to_R(quat_wxyz)
    off = np.asarray(frame["tcp_offset"], float) + as_vec3(held_offset)
    ee = np.asarray(target_xyz, float) - R.apply(off) - base
    qw, qx, qy, qz = quat_wxyz
    thresholds = {}
    if position_threshold is not None:
        thresholds["position_threshold"] = position_threshold
    if rotation_threshold is not None:
        thresholds["rotation_threshold"] = rotation_threshold
    try:
        res = ctx.tool(
            "curobo.plan_with_grasped_object",
            world_config={"meshes": [{"name": object_name, "vertices": verts, "faces": faces}]},
            start_joint_position={"positions": frame["joints"]},
            target_pose={"position": {"x": float(ee[0]), "y": float(ee[1]), "z": float(ee[2])},
                         "rotation": {"w": qw, "x": qx, "y": qy, "z": qz}},
            object_name=object_name,
            robot_file="yam.yml",
            **thresholds,
        )
    except Exception as exc:  # degrade, don't abort — a plain move may still land
        print(f"[transport-held] attached plan errored ({exc}); plain fallback",
              flush=True)
        return False
    if not res.get("success") or not res.get("trajectory"):
        return False
    ctx.tool("libero-yam.execute_trajectory", trajectory=res["trajectory"], arm_id=arm_id)
    return True
