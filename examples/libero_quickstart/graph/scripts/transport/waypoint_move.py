"""Two-segment transport using ``robot.go_to_pose_cartesian``.

Lift the end-effector straight up to a safe height at the current XY,
then translate laterally to above the drop XY. Each segment is a
TCP-frame target driven through the connector's TCP-aware cartesian
motion (cuRobo linear plan with plan_to_pose fallback). Earlier this
called ``curobo.plan_linear`` directly, which is *not* TCP-aware.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose

_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
# TCP-frame fingertip altitude during transit (was 0.45 when this script
# called the bundle's curobo.plan_linear in link frame; matched the original
# panda_hand z=0.45 lift). go_to_pose_cartesian applies the configured TCP
# offset, so a TCP target of 0.353 puts panda_hand at 0.45, preserving the
# original physical altitude that fits inside the Franka workspace.
_LIFT_Z_M = 0.353


class Output(TypedDict):
    done: bool


def run(ctx: NodeContext, drop_x: float, drop_y: float) -> Output:
    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    lift_pose: Se3Pose = {
        "position": {
            "x": ee["pose"]["position"]["x"],
            "y": ee["pose"]["position"]["y"],
            "z": _LIFT_Z_M,
        },
        "rotation": _DOWN,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=lift_pose)

    over_drop: Se3Pose = {
        "position": {"x": float(drop_x), "y": float(drop_y), "z": _LIFT_Z_M},
        "rotation": _DOWN,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=over_drop)

    return {"done": True}
