### HELPER for common YAM CuRoBo motion primitives + skills
"""Shared cuRobo motion helpers for the tape-handover scripts.

Two primitives over the canonical open-robot-skills curobo bundle, executed on
the sim via the connector's ``libero-yam.execute_trajectory``:

  * ``plan_tool_move`` — joint plan to a world-frame TOOL pose
    (``curobo.plan_to_pose``). The tool frame defaults to the gripper TCP; pass
    ``tool_offset`` (TCP-frame metres) to plan a point of the HELD object as if
    it were the end effector: the offset is composed into the planner's
    ``tcp_offset``, so "put the tape centre at X" is one plan — no manual
    back-solve of the gripper position.
  * ``curobo_linear_move`` — straight Cartesian leg (``curobo.plan_directed_linear``,
    orientation LOCK) with the two grocery-packing-benchmark best practices:
    DISTANCE endpoint mode for pure directional legs (descend/retract/insert)
    and a ``plan_to_pose`` fallback so one hard leg can't abort the handover.

All poses are world-frame; the arm-base conversion happens here (YAM bases are
upright, so world->base is a pure translation). ``direction`` is a world-frame
unit vector.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.errors import PlanningFailed

## Used for tape spool handover, plan to desired TAPE pose
## Simple wrapper around curobo.plan_to_pose
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
    frame = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id) ### fetch current physical state of arm
    off = np.asarray(frame["tcp_offset"], dtype=float) ### offset from the tcp (for example, pass the tape offset)
    if tool_offset is not None: 
        off = off + np.asarray(tool_offset, dtype=float) ## add offset
    p = np.asarray(target_xyz, dtype=float) - np.asarray(frame["position"], dtype=float) ### computes in arm base frame
    qw, qx, qy, qz = (float(v) for v in target_quat_wxyz)
    ### Call curobo skill
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


def _tcp_world_to_ee_base(target_world_xyz, target_quat_wxyz, base, tcp_off):
    """World TCP target -> arm-base link_6 pose (plan_directed_linear takes no
    tcp_offset): ee = tcp - R_target @ tcp_offset_local, then minus the arm base."""
    qw, qx, qy, qz = target_quat_wxyz
    ee = np.asarray(target_world_xyz, float) - Rotation.from_quat([qx, qy, qz, qw]).apply(tcp_off)
    ee_base = ee - base
    return {"position": {"x": float(ee_base[0]), "y": float(ee_base[1]), "z": float(ee_base[2])},
            "rotation": {"w": qw, "x": qx, "y": qy, "z": qz}}

## Straight linear Cartesian motion
def curobo_linear_move(ctx: NodeContext, arm_id: int, target_world_xyz, target_quat_wxyz, allowed_axes,
                       *, distance=None, direction=None, fallback: bool = True):
    """Straight Cartesian move via curobo.plan_directed_linear (orientation LOCK).

    ``target_world_xyz`` is the intended world-frame TCP endpoint (always required —
    it is the target for the fallback even in DISTANCE mode).

    Endpoint mode:
      * DISTANCE — pass ``distance`` (m) + ``direction`` (world unit vec, e.g.
        (0,0,-1) to descend). Moves exactly that far along the axis from FK(start);
        preferred for pure single-axis legs. ``target_world_xyz`` should equal
        start + direction*distance so the fallback lands in the same spot.
      * PROJECT_TO_TARGET (default, no distance) — projects onto the target along
        the free ``allowed_axes``; use for multi-axis re-aligns.

    On plan failure, if ``fallback`` (default), retries with an unconstrained
    ``plan_tool_move``. Raises only if BOTH fail.
    """
    frame = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)
    base = np.asarray(frame["position"], float)
    tcp_off = np.asarray(frame["tcp_offset"], float)

    kwargs = dict(
        start_joint_position={"positions": frame["joints"]},
        allowed_axes=list(allowed_axes),
        orientation_mode="LOCK",
        robot_file="yam.yml",
    )
    ## DISTANCE mode moves the arm along a specific unit vector 
    if distance is not None and direction is not None:
        d = np.asarray(direction, float)
        kwargs["endpoint_mode"] = "DISTANCE"
        kwargs["explicit_direction"] = {"x": float(d[0]), "y": float(d[1]), "z": float(d[2])}
        kwargs["distance"] = float(distance)
    ### Target mode, moves to target, allows flexibility
    else:
        kwargs["endpoint_mode"] = "PROJECT_TO_TARGET"
        kwargs["target_pose"] = _tcp_world_to_ee_base(
            target_world_xyz, target_quat_wxyz, base, tcp_off)

    res = ctx.tool("curobo.plan_directed_linear", **kwargs)
    if res.get("success") and res.get("trajectory"):
        ctx.tool("libero-yam.execute_trajectory", trajectory=res["trajectory"], arm_id=arm_id)
        return


    ### Allows a slight curve with plan_tool_move if straight line planning fails
    reason = res.get("failure_reason", "")
    if fallback:
        traj = plan_tool_move(ctx, arm_id, target_world_xyz, target_quat_wxyz,
                              execute=False)
        if traj is not None:
            ctx.tool("libero-yam.execute_trajectory", trajectory=traj, arm_id=arm_id)
            print(f"[linear-move] arm {arm_id} directed_linear failed ({reason}); "
                  f"plan_to_pose cartesian fallback OK", flush=True)
            return
    raise PlanningFailed(
        f"linear-move: arm {arm_id} axes={list(allowed_axes)} to "
        f"{[round(v, 3) for v in target_world_xyz]} failed; fallback "
        f"{'also failed' if fallback else 'disabled'} ({reason})")

