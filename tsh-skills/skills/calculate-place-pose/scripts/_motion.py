"""Shared cuRobo motion helper for the tape-handover scripts.

``plan_tool_move`` — joint plan to a world-frame TOOL pose
(``curobo.plan_to_pose``). The tool frame defaults to the gripper TCP; pass
``tool_offset`` (TCP-frame metres) to plan a point of the HELD object as if
it were the end effector: the offset is composed into the planner's
``tcp_offset``, so "put the tape centre at X" is one plan — no manual
back-solve of the gripper position.

All poses are world-frame; the arm-base conversion happens here (YAM bases are
upright, so world->base is a pure translation).
"""

from __future__ import annotations

import numpy as np

from gap import NodeContext
from gap_core.errors import PlanningFailed


def plan_tool_move(ctx: NodeContext, arm_id: int, target_xyz, target_quat_wxyz, *,
                   tool_offset=None, execute: bool = True, **plan_kwargs):
    """Plan the tool frame to a world pose; execute unless ``execute=False``.

    tool_offset: extra TCP-frame offset composed into the planner's tcp_offset —
                 the held-object point (e.g. the measured tape centre) is then
                 planned as the end effector.
    execute:     ``False`` probes reachability: returns the trajectory or None
                 instead of raising, without touching the sim.
    Returns the trajectory dict; raises :class:`PlanningFailed` when ``execute``
    and no plan is found.
    """
    frame = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)
    off = np.asarray(frame["tcp_offset"], dtype=float)
    if tool_offset is not None:
        off = off + np.asarray(tool_offset, dtype=float)
    p = np.asarray(target_xyz, dtype=float) - np.asarray(frame["position"], dtype=float)
    qw, qx, qy, qz = (float(v) for v in target_quat_wxyz)
    res = ctx.tool(
        "curobo.plan_to_pose",
        target_pose={"position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                     "rotation": {"w": qw, "x": qx, "y": qy, "z": qz}},
        start_joint_position={"positions": frame["joints"]},
        robot_file="yam.yml",
        tcp_offset={"x": float(off[0]), "y": float(off[1]), "z": float(off[2])},
        **plan_kwargs,
    )
    traj = res.get("trajectory") if res.get("success") else None
    if traj is None:
        if not execute:
            return None
        raise PlanningFailed(
            f"plan_tool_move: no plan for arm {arm_id} to "
            f"{[round(float(v), 3) for v in target_xyz]}")
    if execute:
        ctx.tool("libero-yam.execute_trajectory", trajectory=traj, arm_id=arm_id)
    return traj
