"""VAB-style transport: lift (Z) → XY over basket → LINEAR descend INTO basket.

Lift/XY are straight-line cartesian (``robot.go_to_pose_cartesian``); the descend
into the basket uses cuRobo's AXIS-CONSTRAINED linear move
(``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``, ``orientation_mode=
"LOCK"``) — a guaranteed straight vertical drop, matching the VAB reference
(segment 5). The descend lowers the held object to ``basket_top − place_offset``
(inside the walls) so it is placed, not dropped from above the rim.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Vec3

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


class Output(TypedDict):
    place_position: Vec3


def _cartesian(ctx: NodeContext, x: float, y: float, z: float) -> None:
    ctx.tool("robot.go_to_pose_cartesian",
             pose={"position": {"x": float(x), "y": float(y), "z": float(z)},
                   "rotation": _DOWN})


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    # FINGERTIP-frame heights. Distance is from_z - target_z, NOT
    # get_ee_pose().z - target_z: get_ee_pose returns the panda_hand link
    # (~0.10 m above the fingertip), which overshoots the descent by the TCP
    # offset and rams the object into the basket floor.
    dist = float(from_z) - float(target_z)
    if dist <= 0.002:
        return
    js = ctx.tool("robot.get_observation")["arms"][0]["joint_state"]
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
        ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]  # XY only for the fallback
        _cartesian(ctx, ee["x"], ee["y"], target_z)


def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    transport_z: float = 0.353,
    place_offset: float = 0.06,
) -> Output:
    c = container_obb["center"]
    e = container_obb["extent"]
    bx, by = float(c["x"]), float(c["y"])
    place_z = float(c["z"]) + float(e["z"]) - float(place_offset)
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, cur["x"], cur["y"], transport_z)   # Seg 3: lift              (Z)
    _cartesian(ctx, bx, by, transport_z)               # Seg 4: XY over basket    (XY)
    _descend_linear(ctx, place_z, transport_z)         # Seg 5: descend INTO basket (constrained Z)
    return {"place_position": {"x": bx, "y": by, "z": place_z}}
