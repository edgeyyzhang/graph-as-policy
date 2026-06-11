"""Transport via a 2-waypoint go_to_pose chain (no trajectory planner).

Lifts the end-effector to a safe height at the current XY, then moves
laterally to above the drop position. Uses simple go_to_pose motions that
skip collision avoidance — suitable for uncluttered tabletop scenes (and
as a fallback when a planner cannot find a collision-free path).
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import Quaternion, Se3Pose

# Canonical top-down gripper orientation (z-axis pointing down in world).
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

# Lift height at which to clear the workspace before lateral translation.
_LIFT_Z_M = 0.45


class Output(TypedDict):
    done: bool


def run(ctx: NodeContext, drop_x: float, drop_y: float) -> Output:
    # Lift to safe height at current XY.
    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    pose: Se3Pose = {
        "position": {
            "x": ee["pose"]["position"]["x"],
            "y": ee["pose"]["position"]["y"],
            "z": _LIFT_Z_M,
        },
        "rotation": _DOWN,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    # Move laterally to drop XY at safe height.
    pose = {
        "position": {"x": float(drop_x), "y": float(drop_y), "z": _LIFT_Z_M},
        "rotation": _DOWN,
    }
    ctx.tool("robot.go_to_pose", pose=pose)

    return {"done": True}
