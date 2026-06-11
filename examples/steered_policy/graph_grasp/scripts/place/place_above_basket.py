"""Geometric place: drop a VLA-grasped item into a perceived basket.

Used by the VLA-grasp + skill-place hybrid loop (graph_obb_policy_grasp).
After ``run_policy`` has grasped an item (gripper closed, item held), this
moves the end-effector over the basket's OBB centre and opens the gripper
so the item drops in. The VLA does the dexterous grasp it generalizes
well; this geometric step does the precise placement it does not.

The current end-effector rotation is preserved (no forced re-orient,
consistent with approach_above_target.py), and the Cartesian moves use a
tight tolerance / generous step budget so the EE actually settles over
the basket before releasing.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox


class Output(TypedDict):
    done: bool


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    lift_z: float = 0.40,
    hover_clearance: float = 0.25,
    drop_clearance: float = 0.10,
    move_tolerance: float = 0.003,
    move_max_steps: int = 500,
) -> Output:
    ee = ctx.tool("robot.get_ee_pose", arm_id=0)
    rot = ee["pose"]["rotation"]
    cx = float(ee["pose"]["position"]["x"])
    cy = float(ee["pose"]["position"]["y"])

    bx = float(container_obb["center"]["x"])
    by = float(container_obb["center"]["y"])
    b_top = float(container_obb["center"]["z"]) + float(container_obb["extent"]["z"])
    transport_z = max(lift_z, b_top + hover_clearance)

    # 1. Lift the grasped item STRAIGHT UP from where it was grasped --
    #    keep the current XY, just raise Z. A combined lift+translate
    #    keeps the item dragging low across the floor / through the other
    #    items; lifting clear first is what makes transport safe.
    ctx.tool("robot.go_to_pose", pose={
        "position": {"x": cx, "y": cy, "z": transport_z},
        "rotation": rot,
    }, tolerance=move_tolerance, max_steps=move_max_steps)

    # 2. Transport across to above the basket centre, staying high.
    ctx.tool("robot.go_to_pose", pose={
        "position": {"x": bx, "y": by, "z": transport_z},
        "rotation": rot,
    }, tolerance=move_tolerance, max_steps=move_max_steps)

    # 3. Descend to just above the basket rim.
    ctx.tool("robot.go_to_pose", pose={
        "position": {"x": bx, "y": by, "z": b_top + drop_clearance},
        "rotation": rot,
    }, tolerance=move_tolerance, max_steps=move_max_steps)

    # 4. Release -- the held item drops into the basket.
    ctx.tool("robot.open_gripper", settle_steps=40)

    return {"done": True}
