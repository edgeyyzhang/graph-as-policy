"""Generic RGB-D object localization — classical-CV variant (no DINO/SAM).

Same contract as ``perceive_object.py`` (the appendix perception pattern:
literal ``object_query``, unprefixed geometry outputs renamed by the
subgraph's ``set_outputs``), but segmentation is colour + height-above-table
in-process (``_perceive_cv``) instead of the learned DINO+SAM tool bundles.
Use it when the target has a nameable colour ("yellow tape", "gray tape",
"red spool") — it is faster (no model servers), has no detector score noise,
and fails loudly instead of mis-detecting a lookalike.

``raise_if_missing=False`` turns an empty segmentation into a clean
``found=False`` return — the clean-all-items loop exit, same as the learned
variant.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap_core.types import PointCloud

from ._perceive import estimate_half_thickness
from ._perceive_cv import perceive_top_face_cv


class Output(TypedDict):
    found: bool
    cloud: PointCloud | None  # world-frame cloud (table/floor removed)
    top_xyz: list | None      # [x, y, top_z] — top-face centre (place targets rest here)
    center_xyz: list | None   # [x, y, top_z - half_z] — body-centroid height (grasp point)
    half_z: float | None      # derived half-thickness (m), (top - bottom) / 2


def run(ctx: NodeContext, *, object_query: str, cameras: list,
        raise_if_missing: bool = True) -> Output:
    """Localize ``object_query`` by colour+height; emit cloud + derived geometry.

    object_query: literal noun phrase; its COLOUR WORD selects the segmenter
                  band ("yellow ..." -> hue band, "gray ..." -> achromatic).
    cameras:      the shared observation's camera list. REQUIRED — must come
                  from an explicit ``observe`` (``robot.get_observation``) node
                  inside THIS subgraph (``Ref("observe.cameras")``).
    raise_if_missing: when False, an empty segmentation returns ``found=False``
                  (loop exit) instead of raising to ``on_error``.
    """
    got = perceive_top_face_cv(ctx, cameras, object_query,
                               raise_if_missing=raise_if_missing)
    if got is None:
        print(f"[perceive_object_cv] no '{object_query}' found (clean exit)",
              flush=True)
        return {"found": False, "cloud": None, "top_xyz": None,
                "center_xyz": None, "half_z": None}
    x, y, top_z, pts = got
    half_z = estimate_half_thickness(pts, top_z)
    print(f"[perceive_object_cv] '{object_query}': top=({x:.4f},{y:.4f},{top_z:.4f}) "
          f"half_z={half_z*1000:.1f}mm ({len(pts)} pts)", flush=True)
    return {"found": True,
            "cloud": {"points": pts.astype(np.float32)},
            "top_xyz": [float(x), float(y), float(top_z)],
            "center_xyz": [float(x), float(y), float(top_z - half_z)],
            "half_z": float(half_z)}
