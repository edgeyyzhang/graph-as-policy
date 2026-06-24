"""Decide whether the target object is too close to the basket and, if so,
push the basket away from the object to open up grasp clearance.

Router node between ``perceive_next`` and ``grasp``:
  - ``"clear"``  -> enough room, grasp directly.
  - ``"pushed"`` -> nudged the basket; the graph loops back to ``container`` so
    the basket (now moved) and the object are re-perceived before the grasp
    retries.

The push is a closed-gripper poke: hover over the basket centre, lower to just
under the basket RIM (high, not deep inside -- the old mid-height descend
plunged the gripper too far), slide outward along the object->basket direction
past the far wall, lift, and go home so the
agentview camera sees the table cleanly for re-perception. Re-perception is the
existing ``container`` + ``perceive_next`` subgraphs (reached via the back-edge),
so this script stays pure motion + geometry.
"""

from __future__ import annotations

import logging

import numpy as np
from gap import NodeContext
from gap_core.types import OrientedBoundingBox

logger = logging.getLogger(__name__)

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def _xy(c) -> np.ndarray:
    return np.array([float(c["x"]), float(c["y"])], dtype=float)


def _radius_xy(obb: OrientedBoundingBox) -> float:
    # Crude circumscribed XY radius -- simple and conservative for a gate.
    return float(max(obb["extent"]["x"], obb["extent"]["y"]))


def _move(ctx: NodeContext, x: float, y: float, z: float) -> None:
    pose = {"position": {"x": float(x), "y": float(y), "z": float(z)},
            "rotation": _DOWN}
    try:
        ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[push_basket] cartesian leg failed (%s); planned move", exc)
        ctx.tool("robot.go_to_pose", pose=pose)


def _poke(ctx: NodeContext, b, end, hover_z: float, contact_z: float) -> bool:
    """One closed-gripper poke at ``contact_z``: hover -> descend -> slide
    outward -> lift. Returns False (without raising) if any leg is IK-infeasible,
    so the caller can retry lower or skip -- a push must never abort the run."""
    try:
        _move(ctx, b[0], b[1], hover_z)        # hover over basket centre
        _move(ctx, b[0], b[1], contact_z)      # lower to contact height
        _move(ctx, end[0], end[1], contact_z)  # slide outward past the far wall
        _move(ctx, end[0], end[1], hover_z)    # lift clear
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[push_basket] poke at contact_z=%.3f infeasible (%s)", contact_z, exc)
        return False


def run(
    ctx: NodeContext,
    target_obb: OrientedBoundingBox,
    container_obb: OrientedBoundingBox,
    min_clearance: float = 0.04,
    push_distance: float = 0.08,
    hover_z: float = 0.30,
    contact_below_rim: float = 0.02,
) -> dict:
    o = _xy(target_obb["center"])
    b = _xy(container_obb["center"])
    d = b - o
    dist = float(np.linalg.norm(d))
    gap = dist - _radius_xy(target_obb) - _radius_xy(container_obb)
    logger.info("[push_basket] object-basket gap=%.3f m (min=%.3f)", gap, min_clearance)
    if dist < 1e-6 or gap >= min_clearance:
        return {"route": "clear"}

    d = d / dist                                     # unit object -> basket
    # Contact the basket near its RIM, not deep inside (the old mid-height target
    # plunged the gripper far down). But a high+far+down wrist can be IK-
    # infeasible for an extended basket, so poke high BY PREFERENCE and fall back
    # to the mid-height contact that always reached. ``end`` is just past the far
    # wall along object->basket.
    center_z = float(container_obb["center"]["z"])
    top_z = center_z + float(container_obb["extent"]["z"])
    hi_z = top_z - float(contact_below_rim)          # near the rim (preferred, high)
    end = b + d * (_radius_xy(container_obb) + push_distance)

    ctx.tool("robot.close_gripper", settle_steps=20)  # rigid poker
    # High poke first; if unreachable, retry at mid-height; if BOTH fail, skip
    # the push (route 'clear') rather than abort the whole episode.
    pushed = _poke(ctx, b, end, hover_z, hi_z) or _poke(ctx, b, end, hover_z, center_z)
    try:
        ctx.tool("robot.go_home")                     # clear camera view to re-perceive
    except Exception as exc:  # noqa: BLE001
        logger.warning("[push_basket] go_home after poke failed (%s)", exc)
    if pushed:
        logger.info("[push_basket] pushed basket ~%.3f m away from object", push_distance)
        return {"route": "pushed"}
    logger.warning("[push_basket] basket unreachable at high+mid contact; skipping push")
    return {"route": "clear"}
