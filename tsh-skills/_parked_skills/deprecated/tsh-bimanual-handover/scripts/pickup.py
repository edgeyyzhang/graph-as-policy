"""Ring-grasp the tape from above: one finger in the hole, one outside the rim.

Pickup step 2 (motion): the grasp point and ring radii are perceived upstream by
``perceive_tape``; this step drives the giver arm to hover, descend, close and
lift. Every leg is planned by the canonical curobo bundle and streamed onto the
sim (``libero-yam.execute_trajectory``), so the motion is smooth, jerk-limited
and generalises across tape positions.

Before closing — while the fingers are seated on the perceived grasp point — the
tape centre is expressed in the giver TCP frame and emitted as ``tape_in_giver``.
The grip is rigid, so that offset lets the exchange track (and plan) the held
tape by forward kinematics with no ground truth.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from ._gripper_geometry import grip_geometry
from ._motion import plan_tool_move
from .constants import (
    DOWN_QUAT,
    GRASP_CLOSE_RAMP_STEPS,
    GRASP_CLOSE_SETTLE_STEPS,
    GRASP_HOOK_DZ,
    GRASP_LIFT_CLEARANCE,
    GRASP_PRE_DZ,
    GRASP_RING_ANGLE_DEG,
    GRASP_RING_DX,
    GRASP_RING_DY,
    GRIPPER_FINGERTIP_BEHIND,
)

## save for the return leg
class Output(TypedDict):
    grasped: bool
    tape_in_giver: list  # tape centre in the giver TCP frame (m); rigid under the
                         # grip, so the exchange tracks the tape by FK. Anchored to
                         # the pickup perception at the seated pre-close pose.
    grasp_tcp: list  # world TCP pose at close, [x,y,z,qw,qx,qy,qz]. A return leg
                     # replays this pose to put the tape back where it was picked.


def run(ctx: NodeContext, *, tape_xyz: list, arm_id: int = 0,
        hole_radius: float | None = None,
        rim_radius: float | None = None) -> Output:
    """Ring-grasp the tape at ``tape_xyz``: near finger in the hole, far finger
    on the outer wall, then lift.

    tape_xyz:    world-frame grasp point (body-centroid height) from ``perceive_tape``.
    hole_radius / rim_radius: perceived ring geometry; centres the finger pair on
                 the wall midpoint so both fingers contact simultaneously on
                 close. Falls back to the tuned constants when absent.
    """
    # Centre the TCP on the wall midpoint so both fingers meet their wall with
    # the same travel; the fingertip trails the TCP along the approach axis.
    # Gripper offsets DERIVED from the robot model via FK (self-configuring; see
    # _gripper_geometry.py). Fall back to the tuned constants if the model can't
    # be located, so this never introduces a new failure mode.
    try:
        _g = grip_geometry()
        fingertip_behind, ring_dy_default = _g["fingertip_axial"], _g["finger_half_gap"]
    except Exception as exc:  # model unavailable -> tuned constants
        print(f"[pickup] grip_geometry unavailable ({exc}); using tuned constants",
              flush=True)
        fingertip_behind, ring_dy_default = GRIPPER_FINGERTIP_BEHIND, GRASP_RING_DY
    ring_dy = (hole_radius + rim_radius) / 2 \
        if hole_radius is not None and rim_radius is not None else ring_dy_default
    ring_dx = fingertip_behind if hole_radius is not None else GRASP_RING_DX

    # Revolve the grasp around the ring (about the vertical hole axis). CW from
    # above = negative rotation about +Z. Offset and gripper yaw rotate together
    # so the finger pair stays radial at the new clock angle.
    #
    # The clock angle sets where the giver's fingers land on the PRESENTED ring,
    # and the receiver threads the open side (coupled pair — see constants). For a
    # giver on the -Y side, rotate the grasp an extra 90 deg CW about the tape's
    # (vertical) axis so its fingers sit on the mirrored arc, leaving the side that
    # faces the +Y receiver open. A rotation about the tape axis keeps the top-down
    # grasp reachable. Side is read from the arm base, not tuned.
    x, y, z = (float(v) for v in tape_xyz)
    Rz = Rotation.from_rotvec([0.0, 0.0, -np.radians(GRASP_RING_ANGLE_DEG)])
    giver_base_y = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"][1]
    if float(giver_base_y) < 0.0:
        Rz = Rotation.from_rotvec([0.0, 0.0, -np.radians(90.0)]) * Rz
    off = Rz.apply([ring_dx, ring_dy, 0.0])
    gx, gy = x + off[0], y + off[1]
    qw0, qx0, qy0, qz0 = DOWN_QUAT
    rx, ry, rz, rw = (Rz * Rotation.from_quat([qx0, qy0, qz0, qw0])).as_quat()
    grasp_quat = (rw, rx, ry, rz)

    ctx.tool("robot.open_gripper", arm_id=arm_id)
    plan_tool_move(ctx, arm_id, [gx, gy, z + GRASP_PRE_DZ], grasp_quat)
    # Descend: near finger enters the hole, hooks the inner rim.
    plan_tool_move(ctx, arm_id, [gx, gy, z + GRASP_HOOK_DZ], grasp_quat)

    ### use end effector posoe for tape pose
    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    p_ee = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
    r = ee["rotation"]
    R_ee = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])
    tape_in_giver = R_ee.inv().apply(np.array([x, y, z]) - p_ee).tolist()

    # ramp_steps: close the jaws gently over ~60 sim steps so the finger seats in
    # the hole without kicking the light ring; the rest settles fully closed.
    ctx.tool("robot.close_gripper", arm_id=arm_id,
             settle_steps=GRASP_CLOSE_SETTLE_STEPS, ramp_steps=GRASP_CLOSE_RAMP_STEPS)
    # Lift relative to the perceived grasp height so this can't break if the table
    # moves — an absolute world-Z target would.
    plan_tool_move(ctx, arm_id, [gx, gy, z + GRASP_LIFT_CLEARANCE], grasp_quat)

    return {"grasped": True, "tape_in_giver": tape_in_giver,
            "grasp_tcp": [float(v) for v in
                          (gx, gy, z + GRASP_HOOK_DZ, *grasp_quat)]}

