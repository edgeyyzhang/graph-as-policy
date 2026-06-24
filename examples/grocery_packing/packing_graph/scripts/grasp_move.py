"""Grasp with a fast path and a robust fallback.

PRIMARY — VAB-style cartesian descend on the chosen (long-axis) pose:
rise (Z) -> XY over the object -> LINEAR descend onto it. Fast, leaves a clean
down-facing config, and (unlike the planner's goalset) happily grasps a flat
object by simply lowering onto it.

FALLBACK — if the cartesian path can't be found (a far-edge item where a
single fixed top-down wrist has no IK, or a straight-line solve is infeasible),
hand off to the collision-aware cuRobo planner, which searches the whole
candidate fan for a reachable, collision-free wrist configuration. This is the
only way to grasp items at the reach limit, where one fixed orientation fails
but another in the fan succeeds.
"""

import logging
from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose

logger = logging.getLogger(__name__)

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

# Grasp depth tuning. Drop this far below the top-down candidate Z for a firmer
# hold, but keep the fingertips at least _BASE_CLEARANCE above the object base
# (~the table) so the jaws never ram into the surface.
_GRASP_DEEPEN = 0.02
_BASE_CLEARANCE = 0.012


class Output(TypedDict):
    done: bool


def _cartesian(ctx: NodeContext, x: float, y: float, z: float,
               rotation: dict | None = None) -> None:
    """Straight-line cartesian move. Raises on an infeasible solve — the
    caller's planner fallback handles that, so we do NOT silently retry here."""
    ctx.tool("robot.go_to_pose_cartesian",
             pose={"position": {"x": float(x), "y": float(y), "z": float(z)},
                   "rotation": rotation or _DOWN})


def _descend_linear(ctx: NodeContext, target_z: float, from_z: float,
                    rotation: dict | None = None) -> None:
    """Z-only straight-down descend via cuRobo's constrained linear planner;
    cartesian fallback if the constrained plan can't be found. FINGERTIP-frame
    heights: distance is ``from_z - target_z`` (NOT ``get_ee_pose().z - target_z``,
    which returns panda_hand ~0.10 m above the fingertip and overshoots)."""
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
        _cartesian(ctx, ee["x"], ee["y"], target_z, rotation)


def _cartesian_grasp(ctx: NodeContext, grasp_pose: Se3Pose,
                     target_obb: OrientedBoundingBox, hover_z: float) -> None:
    """Fast path: rise -> XY over object (rotating to the grasp yaw) -> descend.

    Descend DEEPER than the candidate for a firmer hold: the default -0.03
    z-offset sits just under the OBB top, a shallow grip that lets taller items
    pivot out of the jaws, so drop a further _GRASP_DEEPEN. The floor is
    _BASE_CLEARANCE above the object base so the fingers never ram the table;
    for a very flat box (e.g. the 18 mm cream cheese) that floor is ~mid-height,
    preserving the old "grip the sides, not empty space" behaviour."""
    g = grasp_pose["position"]
    rot = grasp_pose.get("rotation") or _DOWN      # grasp orientation (short-axis yaw)
    base_z = float(target_obb["center"]["z"]) - float(target_obb["extent"]["z"])
    grasp_z = max(float(g["z"]) - _GRASP_DEEPEN, base_z + _BASE_CLEARANCE)  # deeper, still clear of table
    cur = ctx.tool("robot.get_ee_pose")["pose"]["position"]
    _cartesian(ctx, cur["x"], cur["y"], hover_z, _DOWN)   # Seg 0: rise to hover (keep down)
    _cartesian(ctx, g["x"], g["y"], hover_z, rot)         # Seg 1: XY over object + rotate to grasp yaw
    _descend_linear(ctx, grasp_z, hover_z, rot)           # Seg 2: descend INTO it (clamped; LOCK keeps yaw)


def _planner_grasp(
    ctx: NodeContext,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    topk: int = 8,
    num_ik_seeds: int = 128,
) -> None:
    """Collision-aware fallback: build a per-observation collision world (target
    carved out by its OBB volume) and plan to the first reachable, collision-free
    candidate in the fan. Plans from the CURRENT config — no pre-positioning
    needed (the cuRobo solve searches IK seeds + the whole goalset)."""
    obs = ctx.tool("robot.get_observation")
    js = obs["arms"][0]["joint_state"]
    world = ctx.tool(
        "geometry.build_world_config",
        cameras=obs["cameras"],
        robot_joint_state=js,
        target_obb=target_obb,
        target_obb_name="target",
    )["config"]
    poses = list(candidate_poses)[: int(topk)]
    for i, pose in enumerate(poses):
        try:
            plan = ctx.tool(
                "curobo.plan_to_grasp_poses",
                world_config=world,
                start_joint_position=js,
                grasp_poses=[pose],
                grasp_pose_is_fingertip=True,
                use_world_collision=True,
                use_cuda_graph=False,
                robot_collision_sphere_buffer=-0.01,
                collision_activation_distance=0.005,
                ignore_obstacle_names=["target"],
                use_grasp_approach=False,
                num_ik_seeds=int(num_ik_seeds),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[grasp] planner candidate %d/%d errored (%s); next",
                           i, len(poses), exc)
            continue
        if plan.get("success"):
            ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"])
            logger.info("[grasp] planner fallback grasped via candidate %d/%d", i, len(poses))
            return
    raise RuntimeError(
        f"grasp: cartesian path infeasible AND collision-aware planner found "
        f"0/{len(poses)} reachable candidates"
    )


def run(
    ctx: NodeContext,
    grasp_pose: Se3Pose,
    candidate_poses: list[Se3Pose],
    target_obb: OrientedBoundingBox,
    hover_z: float = 0.2,
) -> Output:
    try:
        _cartesian_grasp(ctx, grasp_pose, target_obb, hover_z)
        logger.info("[grasp] cartesian fast-path grasp")
        return {"done": True}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[grasp] cartesian path could not be found (%s); falling back to "
            "the collision-aware planner over the candidate fan.", exc,
        )
        _planner_grasp(ctx, candidate_poses, target_obb)
        return {"done": True}
