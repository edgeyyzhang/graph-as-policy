"""Move end-effector to a safe height directly above a target XY position.

Three-step motion so rotation happens *before* any descent onto the object:
1. Lift vertically at current XY with the current gripper rotation.
2. Translate laterally to above the target XY, still with current rotation.
3. Rotate in place to the target grasp rotation at safe height.

If no ``rotation`` is provided, the default downward-facing quaternion is used
and the final rotate step is a no-op. When ``target_obb`` is supplied, the
approach height is derived from the OBB top so no magic number needs to appear
in the workflow JSON.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Se3Pose, Vec3

logger = logging.getLogger(__name__)

# Clearance above the OBB top when deriving approach height from a target OBB.
_OBB_APPROACH_CLEARANCE = 0.15


def _try_go(ctx: NodeContext, pose: Se3Pose, label: str) -> bool:
    """Best-effort pre-positioning leg. ``approach`` only *helps* the planner:
    if a leg's single-shot IK can't reach (e.g. a high pose over a far target,
    or a contorted seed left by the previous place), skip it and let the robust
    collision-aware ``plan_grasp`` (128 IK seeds) reach the actual grasp from
    wherever the arm ends up. Aborting the whole grasp on a pre-positioning
    miss — when the lower grasp pose is itself reachable — is the wrong call."""
    try:
        ctx.tool("robot.go_to_pose", pose=pose)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[approach_above] '%s' leg unreachable (%s); skipping — the "
            "collision-aware grasp planner will reach the grasp from the "
            "current pose.", label, exc,
        )
        return False


class Output(TypedDict):
    done: bool


def run(
    ctx: NodeContext,
    target_position: Vec3,
    approach_height: float = 0.35,
    rotation: Quaternion | None = None,
    target_obb: OrientedBoundingBox | None = None,
) -> Output:
    down_rot: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

    if target_obb is not None:
        approach_z = max(
            target_obb["center"]["z"] + target_obb["extent"]["z"]
            + _OBB_APPROACH_CLEARANCE,
            approach_height,
        )
    else:
        approach_z = approach_height

    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    current_pos = ee["pose"]["position"]
    current_rot = ee["pose"]["rotation"]

    # 1. Lift to safe height at current XY, preserving current rotation.
    pose: Se3Pose = {
        "position": {"x": current_pos["x"], "y": current_pos["y"], "z": approach_z},
        "rotation": current_rot,
    }
    _try_go(ctx, pose, "lift")

    # 2. Move above target XY, still with current rotation.
    pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": current_rot,
    }
    _try_go(ctx, pose, "lateral")

    # 3. Rotate in place at safe height to the requested grasp rotation.
    target_rot = rotation if rotation is not None else down_rot
    pose = {
        "position": {
            "x": target_position["x"], "y": target_position["y"], "z": approach_z,
        },
        "rotation": target_rot,
    }
    _try_go(ctx, pose, "rotate")

    return {"done": True}
