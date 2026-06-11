"""Verify a placement: target OBB centre must land near the container centre.

Not wired into the default workflow (the real-robot graph has no
automatic success check) — kept as an optional verification node: bind
``target_cloud``/``container_cloud`` to fresh perception outputs and a
raise here routes the owning subgraph's ``on_error`` exit.
"""

from typing import TypedDict

from gap import NodeContext
from gap.types import PointCloud


class Output(TypedDict):
    success: bool
    distance: float


def run(
    ctx: NodeContext,
    target_cloud: PointCloud,
    container_cloud: PointCloud,
    max_distance: float = 0.06,
) -> Output:
    import math

    t_obb = ctx.tool(
        "geometry.filter_and_compute_obb",
        points=target_cloud, eps=0.005, min_samples=5,
    )["obb"]
    c_obb = ctx.tool(
        "geometry.filter_and_compute_obb",
        points=container_cloud, eps=0.005, min_samples=5,
    )["obb"]

    dx = t_obb["center"]["x"] - c_obb["center"]["x"]
    dy = t_obb["center"]["y"] - c_obb["center"]["y"]
    distance = math.sqrt(dx * dx + dy * dy)

    if distance > max_distance:
        raise RuntimeError(
            f"placement check failed: distance {distance:.3f}m exceeds "
            f"threshold {max_distance:.3f}m"
        )
    return {"success": True, "distance": distance}
