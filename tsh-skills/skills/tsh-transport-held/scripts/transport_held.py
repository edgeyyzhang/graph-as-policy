"""Move a HELD object's centre to a world target pose (tape-as-EE transport).

The generic held-object transport node: the measured rigid ``held_offset``
(object centre in the holder's TCP frame, from pickup or the exchange) is
composed into the planner's ``tcp_offset``, so the target here is an
OBJECT-centre pose, never a bare TCP pose. When the perceived object cloud is
supplied, the approach is planned with the object attached as a cuRobo
collision body (recentred at its live FK-tracked centre), with a plain
tool-offset plan as the fallback — same policy as the place approach.

Used by the handover subgraph as its giver PRESENT leg, and available
standalone for any "hold the object at X" step (e.g. above a sorting stack).
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose, Vec3, Quaternion, make_pose

from ._held import (
    approach_with_attached,
    as_vec3,
    as_wxyz,
    held_center_world,
    plan_held_move,
)
from .constants import PLAN_POSITION_THRESHOLD, PLAN_ROTATION_THRESHOLD


class Output(TypedDict):
    transported: bool
    held_tcp: Se3Pose  # world TCP pose after the move


def run(ctx: NodeContext, *, arm_id: int, held_offset: Vec3,
        target_xyz: Vec3, target_quat: Quaternion,
        held_cloud=None,
        position_threshold: float = PLAN_POSITION_THRESHOLD,
        rotation_threshold: float = PLAN_ROTATION_THRESHOLD) -> Output:
    """Plan + execute the held-object centre to ``target_xyz`` at ``target_quat``.

    arm_id:       the holding arm.
    held_offset:  object centre in the holder's TCP frame (measured, no GT).
    target_xyz:   world-frame OBJECT-centre target.
    target_quat:  target orientation (wxyz) of the holder's TCP.
    held_cloud:   optional perceived object cloud; when given, the move is
                  planned with the object attached as a collision body first,
                  falling back to the plain tool-offset plan.
    """
    target = list(as_vec3(target_xyz))
    quat = as_wxyz(target_quat)
    thresholds = {"position_threshold": position_threshold,
                  "rotation_threshold": rotation_threshold}

    attached_ok = False
    if held_cloud is not None:
        center = held_center_world(ctx, arm_id, held_offset)
        attached_ok = approach_with_attached(
            ctx, arm_id, target, quat, held_offset, center, held_cloud,
            **thresholds)
    if not attached_ok:
        plan_held_move(ctx, arm_id, held_offset, target, quat, **thresholds)

    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    return {"transported": True, "held_tcp": ee}
