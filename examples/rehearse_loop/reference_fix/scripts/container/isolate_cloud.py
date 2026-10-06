"""Keep the one object-sized cluster of a perceived cloud that rests on the work surface.

The segmentation mask of a container standing close to the robot can bleed onto the arm and
hand. The fused cloud then holds far more robot points than container points, and the fitted
box lands on the robot instead of the container. Two steps remove them: points higher than
``max_height`` above the work surface are dropped, then only the largest connected cluster of
what remains is kept.
"""

from typing import TypedDict

import numpy as np
from gap import NodeContext
from gap_core.types import PointCloud
from scipy import ndimage


class Output(TypedDict):
    cloud: PointCloud
    kept: int
    dropped: int


def run(
    ctx: NodeContext,
    cloud: PointCloud,
    max_height: float = 0.25,
    voxel: float = 0.02,
    min_points: int = 50,
) -> Output:
    points = np.asarray(cloud["points"], dtype=np.float32).reshape(-1, 3)
    low = points[points[:, 2] <= max_height]
    if len(low) < min_points:
        # Nothing plausible near the surface: hand the cloud on unchanged.
        return {"cloud": {"points": points}, "kept": len(points), "dropped": 0}

    cells = np.floor((low - low.min(axis=0)) / voxel).astype(np.int64)
    grid = np.zeros(cells.max(axis=0) + 1, dtype=bool)
    grid[tuple(cells.T)] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    point_labels = labels[tuple(cells.T)]
    sizes = np.bincount(point_labels, minlength=count + 1)
    largest = low[point_labels == int(np.argmax(sizes))]
    if len(largest) < min_points:
        largest = low
    return {
        "cloud": {"points": largest},
        "kept": len(largest),
        "dropped": len(points) - len(largest),
    }
