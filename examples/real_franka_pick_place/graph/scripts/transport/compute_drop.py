"""Compute the above/drop poses over the container from its OBB."""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import OrientedBoundingBox, Quaternion, Se3Pose


class Output(TypedDict):
    above_pose: Se3Pose
    drop_pose: Se3Pose

def run(
    ctx: NodeContext,
    container_obb: OrientedBoundingBox,
    approach_height: float,
    drop_clearance: float,
) -> Output:
    center = container_obb["center"]
    # OBB.extent stores half-extents (gap.types.OrientedBoundingBox.extent),
    # so the top of the container is at center.z + extent.z.
    top_z = center["z"] + container_obb["extent"]["z"]

    quat: Quaternion = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}

    above_pose: Se3Pose = {
        "position": {"x": center["x"], "y": center["y"], "z": top_z + approach_height},
        "rotation": quat,
    }

    drop_pose: Se3Pose = {
        "position": {"x": center["x"], "y": center["y"], "z": top_z + drop_clearance},
        "rotation": quat,
    }

    return {"above_pose": above_pose, "drop_pose": drop_pose}
