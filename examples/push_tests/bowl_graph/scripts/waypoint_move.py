"""Desend + lateral move above the target XY — the 2-step waypoint chain.

Three straight-line Cartesian legs: descend vertically at the current XY to
the target's height, translate laterally to the target XY at constant height,
then lift vertically at the target XY to retract.

A speed-motivated variant replaced the 2-step chain with a single
free-space ``curobo.plan_to_pose``. Without a collision world (transport
carries the grasped object, which would need an attachment model),
TrajOpt's shortest smooth path routinely dips LOW while translating —
recorded trial videos show the held object dragged through scene clutter,
shoving the container ~12 cm off its perceived pose, with deterministic
per-layout failures. The convergence cost that motivated the single-plan
variant does not apply here: ``robot.go_to_pose_cartesian`` executes a
pre-planned linear trajectory via fast waypoint playback, not per-step PD
convergence. A collision-aware planned variant remains available as the
SEPARATE node ``waypoint_move_carve`` (rebuilt world +
``curobo.plan_with_grasped_object``).
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.errors import PlanningFailed
from gap_core.types import Quaternion, Se3Pose, Vec3

class Output(TypedDict):
    done: bool


# Canonical top-down orientation — gripper +Z points -Z_world.
_DOWN: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
# TCP-frame fingertip lift altitude. ``robot.go_to_pose_cartesian`` applies
# the configured TCP offset (≈0.097 m on Franka), so 0.353 puts the
# panda_hand link at z≈0.45 — the legacy "ee at z=0.45 after waypoint_move"
# invariant downstream nodes assumed, now expressed in fingertip semantics
# instead of link semantics. Going higher (0.45 in TCP frame → link at 0.547)
# pushes Franka near the workspace boundary and the linear plan refuses.
_LIFT_Z_M = 0.353


def run(
    ctx: NodeContext,
    final_pos: Vec3,
    drop_rotation: Quaternion | None = None,
    safe_height: float = _LIFT_Z_M,
) -> Output:
    obs = ctx.tool("robot.get_observation")
    ee = obs["arms"][0]["ee_pose"]["position"]

    # Begin by closing the gripper
    ctx.tool("robot.close_gripper", settle_steps=20)
    initial_height = ee["z"]

    # Use the upstream-supplied drop rotation when available so the
    # lift+lateral phase doesn't unspool grasp-time yaw — that unspool
    # manifests as a redundant-joint reconfiguration ("circular elbow
    # motion") that can fling the held object.
    rotation = drop_rotation if drop_rotation is not None else _DOWN

    # Leg 1: straight vertical descent at the current XY.
    descent_pose: Se3Pose = {
        "position": {"x": float(ee["x"]), "y": float(ee["y"]), "z": float(final_pos["z"])},
        "rotation": rotation,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=descent_pose)

    # Leg 2: lateral translate to the drop XY at constant height.
    lateral_pose: Se3Pose = {
        "position": {"x": float(final_pos["x"]), "y": float(final_pos["y"]), "z": float(final_pos["z"])},
        "rotation": rotation,
    }
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=lateral_pose)
    except Exception as exc:  # linear plan can fail near joint limits
        raise PlanningFailed(
            f"waypoint_move: lateral cartesian leg to "
            f"({final_pos['x']:.3f}, {final_pos['y']:.3f}) at z={safe_height:.3f} failed: {exc}"
        ) from exc

    # Leg 3: straight vertical lift at the current XY.
    lift_pose: Se3Pose = {
        "position": {"x": float(final_pos["x"]), "y": float(final_pos["y"]), "z": float(initial_height)},
        "rotation": rotation,
    }
    ctx.tool("robot.go_to_pose_cartesian", pose=lift_pose)

    return {"done": True}