"""Descend onto the wrist-refined grasp pose: XY-correct, then LINEAR descend.

``grasp_pose`` comes from the wrist (top-down) OBB, so its XY is the object's
true footprint centre. We first nudge the XY at hover height (a small cartesian
correction off the biased front-view XY we approached on), then descend straight
down with cuRobo's axis-constrained linear plan (``allowed_axes=["Z"]``) so the
grip lands on the centre, vertically — matching the VAB reference.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    done: bool


def _cartesian(ctx: NodeContext, x: float, y: float, z: float) -> None:
    ctx.tool("robot.go_to_pose_cartesian",
             pose={"position": {"x": float(x), "y": float(y), "z": float(z)},
                   "rotation": _DOWN})


def _descend_linear(ctx: NodeContext, target_z: float) -> None:
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
    ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    dist = float(ee["z"]) - float(target_z)
    if dist <= 0.002:
        return
    res = ctx.tool(
        "curobo.plan_directed_linear",
        start_joint_position=js,
        endpoint_mode="DISTANCE",
        explicit_direction={"x": 0.0, "y": 0.0, "z": -1.0},
        distance=dist,
        allowed_axes=["Z"],
        orientation_mode="LOCK",
    )
    if res.get("success") and res.get("trajectory"):
        ctx.tool("robot.execute_trajectory", trajectory=res["trajectory"])
    else:
        _cartesian(ctx, ee["x"], ee["y"], target_z)


def run(ctx: NodeContext, grasp_pose: Se3Pose, hover_z: float = 0.353) -> Output:
    g = grasp_pose["position"]
    _cartesian(ctx, g["x"], g["y"], hover_z)   # XY-correct to the true centre (at hover)
    _descend_linear(ctx, g["z"])               # constrained vertical descend onto it
    return {"done": True}
