"""Generic RGB-D object localization: one skill, instantiated per object.

The appendix-style perception contract: the subgraph is parameterized by a
literal ``object_query`` (the DINO noun phrase — ``"yellow tape"``, ``"gray
tape"``, ``"red tape spool"``) and emits unprefixed geometry fields the
subgraph's ``set_outputs`` renames to its own prefix (``target_*``,
``dest_*``, ...). Nothing task-specific lives here — the same node perceives
the tape, the duct, or a colour-sorted stack; ring-specific derivations
(hole/rim radii) live in the separate ``tsh-ring-geometry`` post-processor.

Pipeline (shared ``_perceive`` core, identical to the legacy
perceive_tape/perceive_duct): DINO detect -> SAM box segment -> depth
back-projection -> robust top-face slab. No ground truth, no object
dimensions, no scene constants.

``raise_if_missing=False`` turns a DINO miss into a clean ``found=False``
return instead of an error — the clean-all-items loop exit ("perception any"
pattern): route ``not_found -> done`` when repeating over items.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap_core.types import PointCloud

from ._perceive import estimate_half_thickness, perceive_top_face


class Output(TypedDict):
    found: bool
    cloud: PointCloud | None  # world-frame cloud (table/floor removed)
    top_xyz: list | None      # [x, y, top_z] — top-face centre (place targets rest here)
    center_xyz: list | None   # [x, y, top_z - half_z] — body-centroid height (grasp point)
    half_z: float | None      # derived half-thickness (m), (top - bottom) / 2


def run(ctx: NodeContext, *, object_query: str, cameras: list,
        raise_if_missing: bool = True) -> Output:
    """Localize ``object_query`` from RGB-D; emit its cloud + derived geometry.

    object_query: literal DINO noun phrase for the object (a constant per
                  subgraph instance, NOT a binding).
    cameras:      the shared observation's camera list. REQUIRED — must come
                  from an explicit ``observe`` (``robot.get_observation``) node
                  inside THIS subgraph (``Ref("observe.cameras")``).
    raise_if_missing: when False, a DINO miss returns ``found=False`` (loop
                  exit) instead of raising to ``on_error``.
    """
    got = perceive_top_face(ctx, cameras, query=object_query,
                            raise_if_missing=raise_if_missing)
    if got is None:
        print(f"[perceive_object] no '{object_query}' found (clean exit)", flush=True)
        return {"found": False, "cloud": None, "top_xyz": None,
                "center_xyz": None, "half_z": None}
    x, y, top_z, pts = got
    half_z = estimate_half_thickness(pts, top_z)
    print(f"[perceive_object] '{object_query}': top=({x:.4f},{y:.4f},{top_z:.4f}) "
          f"half_z={half_z*1000:.1f}mm ({len(pts)} pts)", flush=True)
    return {"found": True,
            "cloud": {"points": pts.astype(np.float32)},
            "top_xyz": [float(x), float(y), float(top_z)],
            "center_xyz": [float(x), float(y), float(top_z - half_z)],
            "half_z": float(half_z)}
