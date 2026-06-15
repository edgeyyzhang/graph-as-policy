"""Lift + lateral move above the drop XY via a single cuRobo plan.

Replaces the legacy 2-step ``go_to_pose`` chain (lift in place → translate
laterally) with one ``curobo.plan_to_pose`` call followed by one
``robot.execute_trajectory``. The two go_to_pose calls were each ~1.5–2 min
of PD execution waiting for convergence at every microstep; the
cuRobo path runs TrajOpt once (~1–3 s) and the sim plays the resulting
trajectory back at full speed, cutting the node from ~4 min to ~10–20 s.

The planner is asked to reach ``(drop_x, drop_y, lift_z)`` with the
canonical top-down ``_DOWN`` rotation, starting from the current arm
state. TrajOpt naturally produces a smooth lift-and-translate motion
(the planner sees the start state hovering low over the grasped object
and the goal high above the drop XY).

No collision world is passed — this preserves the legacy script's
"free-space transport" semantics. The robot is still holding the grasp,
so a collision world would have to encode the held object as an
attachment; passing nothing is simpler and matches prior behavior. If a
collision-aware variant is needed, it is the SEPARATE node
``waypoint_move_carve``, wired against a rebuilt world and
``curobo.plan_with_grasped_object``.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed
from gap_core.types import Quaternion, Se3Pose


class Output(TypedDict):
    done: bool


# Canonical top-down orientation — gripper +Z points -Z_world.
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
# Lift height at which to clear the workspace before lateral translation.
# Matches the value the legacy 2-step script used so any downstream node
# that assumed "ee is at z=0.45 after waypoint_move" still holds.
_LIFT_Z_M = 0.45


def run(
    ctx: NodeContext,
    drop_x: float,
    drop_y: float,
    drop_rotation: Quaternion | None = None,
) -> Output:
    obs = ctx.tool("robot.get_observation")
    start_joints = obs["arms"][0]["joint_state"]

    # Use the upstream-supplied drop rotation when available so the
    # planner doesn't unspool grasp-time yaw during the lift+lateral
    # phase — that unspool manifests as a redundant-joint
    # reconfiguration ("circular elbow motion") between pick and place.
    rotation = drop_rotation if drop_rotation is not None else _DOWN
    target_pose: Se3Pose = {
        "position": {"x": float(drop_x), "y": float(drop_y), "z": _LIFT_Z_M},
        "rotation": rotation,
    }

    plan = ctx.tool(
        "curobo.plan_to_pose",
        target_pose=target_pose,
        start_joint_position=start_joints,
    )
    if not plan["success"]:
        raise PlanningFailed(
            f"waypoint_move: CuRobo plan_to_pose failed for drop XY "
            f"=({drop_x:.3f}, {drop_y:.3f}) at lift_z={_LIFT_Z_M:.3f}"
        )

    ctx.tool(
        "robot.execute_trajectory",
        trajectory=plan["trajectory"],
        subsample=4,
        max_steps_per_waypoint=8,
        tolerance=0.02,
    )
    return {"done": True}
