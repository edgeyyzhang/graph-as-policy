"""Ring-grasp the tape from above: one finger in the hole, one outside the rim.

Pickup motion step: the grasp poses come from ``calculate-grasp-ring`` (the
single producer of the ring-grasp geometry, which ``bimanual-route-arms`` also probes) —
this step drives the picking arm through the three legs (hover, seat, lift),
each planned by the canonical curobo bundle and streamed onto the sim
(``libero-yam.execute_trajectory``), so the motion is smooth and jerk-limited.

Before closing — while the fingers are seated on the perceived grasp point —
the tape centre is expressed in the picking arm's TCP frame and emitted as
``giver_held_offset`` (a.k.a. ``tape_in_giver``). The grip is rigid, so that
offset lets every downstream held-tape move (present, place) track and plan
the tape by forward kinematics with no ground truth. Deliberately NOT named
``held_offset`` — that name is reserved for whichever holder is CURRENT at a
given point in the graph (this pickup's grip, or the receiver's after a
handover); exposing it under the giver-specific name here, instead, means a
consumer positioned too early (before the route decision is dispatched) has
no producer to wire at all, rather than silently getting the wrong holder's
offset.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import Se3Pose, Vec3, make_pose

from ._motion import plan_tool_move
from .constants import (
    GRASP_CLOSE_RAMP_STEPS,
    GRASP_CLOSE_SETTLE_STEPS,
)


## save for the return leg
class Output(TypedDict):
    grasped: bool
    giver_held_offset: Vec3  # tape centre in the picking arm's TCP frame (m);
                       # rigid under the grip, so held-tape moves track it by
                       # FK. Anchored to the pickup perception at the seated
                       # pre-close pose. dispatch-route relays this
                       # forward as the canonical held_offset on the direct
                       # exit; the handover rebinds held_offset to the
                       # receiver's measured offset on the other route.
    tape_in_giver: Vec3  # alias of giver_held_offset (bimanual-handover's input name)
    grasp_tcp: Se3Pose  # world TCP pose at close. A return leg replays this
                        # pose to put the tape back where it was picked.
    pick_arm: int      # echo of the arm that holds the tape (checkpoint anchor)


def _as3(v) -> list:
    return [float(v["x"]), float(v["y"]), float(v["z"])]


def _as_wxyz(q) -> list:
    return [float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])]


def _vec3(v: list) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def run(ctx: NodeContext, *, tape_xyz: list, arm_id: int = 0,
        hover_xyz: list, seat_xyz: list, lift_xyz: list, grasp_quat: list) -> Output:
    """Ring-grasp the tape: near finger in the hole, far finger on the outer
    wall, then lift. The three grasp legs come from ``calculate-grasp-ring``.

    tape_xyz:   world grasp point (body-centroid height) from perception — used
                to measure the held offset at the seat, pre-close.
    arm_id:     the picking arm (route-decided; either side works).
    hover_xyz / seat_xyz / lift_xyz: the three grasp-leg positions, and
    grasp_quat: their shared TCP orientation (wxyz) — all from
                ``calculate-grasp-ring``.
    """
    x, y, z = _as3(tape_xyz)
    hover_xyz, seat_xyz, lift_xyz = _as3(hover_xyz), _as3(seat_xyz), _as3(lift_xyz)
    grasp_quat = _as_wxyz(grasp_quat)

    ctx.tool("robot.open_gripper", arm_id=arm_id)
    plan_tool_move(ctx, arm_id, hover_xyz, grasp_quat)
    # Descend: near finger enters the hole, hooks the inner rim.
    plan_tool_move(ctx, arm_id, seat_xyz, grasp_quat)

    ### use end effector pose for tape pose
    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    p_ee = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
    r = ee["rotation"]
    R_ee = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])
    held_offset = R_ee.inv().apply(np.array([x, y, z]) - p_ee).tolist()

    # ramp_steps: close the jaws gently over ~60 sim steps so the finger seats in
    # the hole without kicking the light ring; the rest settles fully closed.
    ctx.tool("robot.close_gripper", arm_id=arm_id,
             settle_steps=GRASP_CLOSE_SETTLE_STEPS, ramp_steps=GRASP_CLOSE_RAMP_STEPS)
    plan_tool_move(ctx, arm_id, lift_xyz, grasp_quat)

    held_offset_vec3 = _vec3(held_offset)
    return {"grasped": True, "giver_held_offset": held_offset_vec3,
            "tape_in_giver": held_offset_vec3,
            "grasp_tcp": make_pose(seat_xyz, grasp_quat),
            "pick_arm": int(arm_id)}
