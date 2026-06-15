"""Compute the align-pose: the grasp pose lifted to the approach height.

The gripper rotates into the grasp orientation at this pose first
(``rotate_align``), then descends straight down onto the actual grasp
pose — rotation never happens during the descent.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose


class Output(TypedDict):
    align_pose: Se3Pose


def run(ctx: NodeContext, grasp_pose: Se3Pose, approach_height: float) -> Output:
    align_pose: Se3Pose = {
        "position": {
            "x": grasp_pose["position"]["x"],
            "y": grasp_pose["position"]["y"],
            "z": approach_height,
        },
        "rotation": grasp_pose["rotation"],
    }
    return {"align_pose": align_pose}
