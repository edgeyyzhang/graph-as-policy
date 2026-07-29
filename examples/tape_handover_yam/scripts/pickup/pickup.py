"""Ring-grasp the tape from above: one finger in the hole, one outside the rim.

Pickup motion step: the grasp point comes from the generic perceive-object
subgraph and the ring radii from tsh-ring-geometry; this step drives the
picking arm to hover, descend, close and lift. Every leg is planned by the
canonical curobo bundle and streamed onto the sim
(``libero-yam.execute_trajectory``), so the motion is smooth, jerk-limited
and generalises across tape positions.

The grasp GEOMETRY (wall-midpoint offsets, revolve angle, side mirror) lives
in the shared ``_ring.ring_grasp_poses`` — the same function tsh-route probes
with ``execute=False`` — so the route's feasibility answer and this grasp can
never disagree.

Before closing — while the fingers are seated on the perceived grasp point —
the tape centre is expressed in the picking arm's TCP frame and emitted as
``held_offset`` (a.k.a. ``tape_in_giver``). The grip is rigid, so that offset
lets every downstream held-tape move (present, place) track and plan the tape
by forward kinematics with no ground truth.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from ._motion import plan_tool_move
from ._ring import ring_grasp_poses
from .constants import (
    GRASP_CLOSE_RAMP_STEPS,
    GRASP_CLOSE_SETTLE_STEPS,
)

## save for the return leg
class Output(TypedDict):
    grasped: bool
    held_offset: list  # tape centre in the picking arm's TCP frame (m); rigid
                       # under the grip, so held-tape moves track it by FK.
                       # Anchored to the pickup perception at the seated
                       # pre-close pose. (The direct-place route consumes this
                       # name; the handover rebinds it to the receiver's
                       # measured offset.)
    tape_in_giver: list  # alias of held_offset (legacy graphs; exchange input)
    grasp_tcp: list  # world TCP pose at close, [x,y,z,qw,qx,qy,qz]. A return leg
                     # replays this pose to put the tape back where it was picked.
    rim_radius: float  # passthrough of the perceived rim radius used for this
                       # grasp — relayed to bimanual-handover so the receiver's thread
                       # offset can derive from the same validated ring geometry.
    pick_arm: int      # echo of the arm that holds the tape (checkpoint anchor)


def run(ctx: NodeContext, *, tape_xyz: list, arm_id: int = 0,
        hole_radius: float | None = None,
        rim_radius: float | None = None,
        fingertip_axial: float | None = None,
        finger_half_gap: float | None = None) -> Output:
    """Ring-grasp the tape at ``tape_xyz``: near finger in the hole, far finger
    on the outer wall, then lift.

    tape_xyz:    world-frame grasp point (body-centroid height) from the
                 perceive-object subgraph (its ``<name>_xyz`` output).
    arm_id:      the picking arm (route-decided; either side works — the grasp
                 geometry mirrors from the arm base, see ``_ring``).
    hole_radius / rim_radius: perceived ring geometry (from tsh-ring-geometry);
                 centres the finger pair on the wall midpoint so both fingers
                 contact simultaneously on close. REQUIRED — a missing value
                 raises (no tuned fallback).
    fingertip_axial / finger_half_gap: the FK-derived gripper offsets from the
                 tsh-gripper-geometry skill; self-configuring. REQUIRED — a
                 missing value raises (no tuned fallback).
    """
    # Gripper offsets come from the tsh-gripper-geometry skill (model FK) and the
    # ring radii from perception — both REQUIRED; fail loud if either is missing.
    if fingertip_axial is None or finger_half_gap is None:
        raise RuntimeError(
            "pickup requires derived gripper offsets (fingertip_axial, "
            "finger_half_gap) from tsh-gripper-geometry — none supplied")
    if hole_radius is None or rim_radius is None:
        raise RuntimeError(
            "pickup requires perceived ring geometry (hole_radius, "
            "rim_radius) from tsh-ring-geometry — none supplied")

    # tape_xyz may arrive as a Vec3 dict {x,y,z} (subgraph type coercion) or a seq.
    if isinstance(tape_xyz, dict):
        tape_xyz = [tape_xyz["x"], tape_xyz["y"], tape_xyz["z"]]
    x, y, z = (float(v) for v in tape_xyz)
    g = ring_grasp_poses(ctx, arm_id, [x, y, z],
                         hole_radius=hole_radius, rim_radius=rim_radius,
                         fingertip_axial=fingertip_axial,
                         finger_half_gap=finger_half_gap)
    gx, gy = g["grasp_xy"]
    grasp_quat = g["grasp_quat"]

    ctx.tool("robot.open_gripper", arm_id=arm_id)
    plan_tool_move(ctx, arm_id, [gx, gy, g["hover_z"]], grasp_quat)
    # Descend: near finger enters the hole, hooks the inner rim.
    plan_tool_move(ctx, arm_id, [gx, gy, g["seat_z"]], grasp_quat)

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
    # Lift relative to the perceived grasp height so this can't break if the table
    # moves — an absolute world-Z target would.
    plan_tool_move(ctx, arm_id, [gx, gy, g["lift_z"]], grasp_quat)

    return {"grasped": True, "held_offset": held_offset,
            "tape_in_giver": held_offset,
            "grasp_tcp": [float(v) for v in (gx, gy, g["seat_z"], *grasp_quat)],
            "rim_radius": rim_radius, "pick_arm": int(arm_id)}
