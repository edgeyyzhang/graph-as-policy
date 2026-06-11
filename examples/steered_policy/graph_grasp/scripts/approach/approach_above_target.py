"""Cartesian pre-positioning above a target OBB before VLA hand-off.

Designed for the OBB+policy hybrid workflow. The OBB-based perception
gives us a robust object pose even on perturbed (OOD) layouts; this
script uses it to translate the end-effector to safely above the
(perturbed) object before invoking the LIBERO pi05 policy. The policy
then handles the dexterous grasp + place from a starting pose that is
much closer to its training distribution than the natural reset pose
on a perturbed layout.

Compared to the canonical ``approach_above.py`` used by graph_cartesian_obb:
- preserves the current end-effector rotation (no forced top-down quat),
  because the LIBERO pi05 checkpoint was trained with whatever rotation
  the env's natural reset gives, and re-orienting to a top-down quat
  pushes the proprio-state OOD.
- uses the OBB center XY directly (no TopDownGraspCandidates step).
- target_z defaults to a libero-friendly height (well within the
  reachable XY-translation envelope at the natural reset).
- passes a tight ``move_tolerance`` / generous ``move_max_steps`` to
  GoToPose so the IK move fully settles — GoToPose's loose default
  convergence stops early and leaves the EE rotation a few degrees off
  the commanded (preserved) rotation, drifting the hand-off proprio.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose


class Output(TypedDict):
    done: bool


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    target_z: float = 0.30,
    obb_clearance: float = 0.12,
    move_tolerance: float = 0.003,
    move_max_steps: int = 400,
) -> Output:
    # ``move_tolerance`` / ``move_max_steps`` are passed through to GoToPose
    # so the closed-loop IK move fully settles before the VLA hand-off.
    # GoToPose's default convergence (0.01 rad, 120 steps) breaks as soon
    # as the joint error dips under 0.01 — it stops early at ~0.009 rad,
    # which leaves the end-effector rotation a few degrees off the
    # commanded (preserved) rotation. That drift shifts the proprio-state
    # the policy receives at hand-off away from the episode-start pose.
    # A tighter tolerance + a generous step budget lets the joints settle
    # so the hand-off rotation stays close to the episode-start rotation.
    approach_z = max(
        target_obb["center"]["z"] + target_obb["extent"]["z"] + obb_clearance,
        target_z,
    )

    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    current_rot = ee["pose"]["rotation"]

    if ee["pose"]["position"]["z"] < approach_z - 1e-3:
        ctx.tool("robot.go_to_pose", pose={
            "position": {"x": ee["pose"]["position"]["x"],
                         "y": ee["pose"]["position"]["y"], "z": approach_z},
            "rotation": current_rot,
        }, tolerance=move_tolerance, max_steps=move_max_steps)

    ctx.tool("robot.go_to_pose", pose={
        "position": {"x": target_obb["center"]["x"],
                     "y": target_obb["center"]["y"], "z": approach_z},
        "rotation": current_rot,
    }, tolerance=move_tolerance, max_steps=move_max_steps)

    return {"done": True}
