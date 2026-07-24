"""Reach a canonical, in-distribution pre-pose for the tape-handover VLA.

Given the perceived tape OBB and a fixed ``canonical_offset``, compute a target
end-effector pose and move there via the connector's cuRobo-backed IK. This is
what keeps the frozen TSH VLA in-distribution regardless of where the tape sits.
"""

from __future__ import annotations

from typing import Any, TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Se3Pose, make_pose

# Top-down placeholder orientation (gripper Z down). TODO(data): replace with the
# canonical grasp/handover orientation the VLA was trained from.
_TOP_DOWN_WXYZ = (0.0, 1.0, 0.0, 0.0)


class Output(TypedDict):
    reached: bool
    target_pose: Se3Pose


def _xyz(v: Any) -> list[float]:
    """Coerce a Vec3 (dict or sequence) to [x, y, z]."""
    if isinstance(v, dict):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return [float(v[0]), float(v[1]), float(v[2])]


def run(
    ctx: NodeContext,
    *,
    target_obb: OrientedBoundingBox,
    canonical_offset: list[float] | None = None,
    arm_id: int = 0,
) -> Output:
    """Move ``arm_id`` to ``tape_center + canonical_offset`` with a canonical orientation.

    Args:
        target_obb: perceived tape OBB (from an upstream perceive subgraph).
        canonical_offset: ``[dx, dy, dz]`` world offset from the tape center to
            the canonical EE pose. Required — see TODO(data) below.
        arm_id: which arm reaches (0/1 for the bimanual rig).

    Returns:
        ``{"reached": bool, "target_pose": Se3Pose}``.
    """
    if canonical_offset is None:
        raise NotImplementedError(
            "TODO(data): canonical_offset must be fit from in-distribution "
            "analysis (which relative pickup/handover/placement poses the VLA "
            "saw in training). Pass it as a subgraph input for now."
        )

    cx, cy, cz = _xyz(target_obb["center"])
    dx, dy, dz = (float(v) for v in canonical_offset)
    target_pose = make_pose([cx + dx, cy + dy, cz + dz], _TOP_DOWN_WXYZ)

    ctx.tool("robot.go_to_pose", pose=target_pose, arm_id=arm_id)
    return {"reached": True, "target_pose": target_pose}
