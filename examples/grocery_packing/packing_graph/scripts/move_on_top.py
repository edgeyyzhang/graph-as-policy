"""Move the gripper (and its wrist camera) directly above the object.

Uses the FRONT-view OBB only for its XY (which object / roughly where) — the
front-view depth is biased, so we don't trust its centre for the grip. Rise to
hover (Z), then translate over the object (XY), both straight-line cartesian.
The eye-in-hand wrist camera now looks straight DOWN on the object's top face,
ready for an unbiased top-down re-perception (see ``perceive_wrist.py``).
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    done: bool


def _cartesian(ctx: NodeContext, x: float, y: float, z: float) -> None:
    ctx.tool("robot.go_to_pose_cartesian",
             pose={"position": {"x": float(x), "y": float(y), "z": float(z)},
                   "rotation": _DOWN})


def run(ctx: NodeContext, target_obb: OrientedBoundingBox, hover_z: float = 0.353) -> Output:
    c = target_obb["center"]
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, cur["x"], cur["y"], hover_z)   # rise to hover            (Z)
    _cartesian(ctx, c["x"], c["y"], hover_z)        # translate over the object (XY)
    return {"done": True}
