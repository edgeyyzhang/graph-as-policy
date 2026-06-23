"""VAB-style grasp approach: rise (Z) → XY over object → LINEAR descend onto it.

Rise/XY are straight-line cartesian (``robot.go_to_pose_cartesian``). The final
descend uses cuRobo's AXIS-CONSTRAINED linear move
(``curobo.plan_directed_linear``, ``allowed_axes=["Z"]``, ``orientation_mode=
"LOCK"``) so it is a *guaranteed* straight vertical line — matching the VAB
reference (libero/libero/vab/planning.py, segment 2). The connector's plain
``go_to_pose_cartesian`` frees all three axes and falls back to a curved
``plan_to_pose`` when the straight-line solve fails; the constrained Z-only plan
is the robust, truly-vertical descend. ``DISTANCE`` mode (move ``dist`` along
−Z from the current FK) keeps the call frame-agnostic — no world→base math.
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


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float) -> None:
    """Z-only straight-down descend from ``from_z`` to ``target_z`` via cuRobo's
    constrained linear planner; cartesian fallback if the plan can't be found.

    ``from_z``/``target_z`` are FINGERTIP-frame world heights (same frame the
    grasp pose and ``go_to_pose_cartesian`` targets use). We do NOT read
    ``robot.get_ee_pose`` for the distance: that returns the *panda_hand* link,
    ~0.10 m above the fingertip, so ``ee_z − grasp_z`` overshoots by the TCP
    offset and drives the gripper into the table. The fingertip is already at
    ``from_z`` (the hover the rise/XY legs converged to), so the descent is just
    ``from_z − target_z`` and cuRobo moves the rigidly-attached hand down by it."""
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
        ee = ctx.tool("robot.get_ee_pose")["pose"]["position"]  # XY only (top-down: hand XY == fingertip XY)
        _cartesian(ctx, ee["x"], ee["y"], target_z)


def run(ctx: NodeContext, grasp_pose: Se3Pose, hover_z: float = 0.353) -> Output:
    g = grasp_pose["position"]
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, cur["x"], cur["y"], hover_z)   # Seg 0: rise to hover            (Z)
    _cartesian(ctx, g["x"], g["y"], hover_z)       # Seg 1: XY over the object       (XY)
    _descend_linear(ctx, g["z"], hover_z)          # Seg 2: descend onto it (constrained Z)
    return {"done": True}
