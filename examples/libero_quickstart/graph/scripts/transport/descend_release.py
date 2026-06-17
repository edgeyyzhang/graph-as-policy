"""Descend to TCP-frame drop position, release, retract.

Routes through ``robot.go_to_pose_cartesian`` (TCP-aware connector path,
cuRobo linear plan with plan_to_pose fallback). Earlier this called
``curobo.plan_linear`` directly, which interprets poses as the IK-link
frame.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose, Vec3

_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    drop_position: Vec3


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    rotation = drop_rotation if drop_rotation is not None else _DOWN
    end_pose: Se3Pose = {"position": drop_position, "rotation": rotation}

    ctx.tool("robot.go_to_pose_cartesian", pose=end_pose)

    ctx.tool("robot.open_gripper", settle_steps=60)
    ctx.tool("robot.go_home")

    return {"drop_position": drop_position}
