"""Loop control for grocery_packing's perceive_next subgraph.

Routes to "found" when a graspable grocery item is still on the table (grasp
it) or "none" when only the basket remains (the loop's clean exit).

Why a geometric check and not just ``found``: with the "grocery item" prompt
the VLM tournament prefers a real item over the basket on every comparison,
so it returns the basket ONLY when every item has already been packed and
teleported to the benchmark graveyard. We confirm that here — if the
perceived target's centroid falls inside the basket's XY footprint, there is
nothing left but the basket, so the table is clear. Authored by build_graph.py.
"""
from __future__ import annotations

import numpy as np

from gap import NodeContext


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict) -> dict:
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    center, extent = container_obb["center"], container_obb["extent"]
    # OBB ``extent`` is a half-extent; anything within the basket footprint
    # (+ a small margin) is the basket itself, so the loop is done.
    basket_r = float(max(extent["x"], extent["y"])) + 0.03
    if float(np.hypot(cx - center["x"], cy - center["y"])) < basket_r:
        return {"route": "none"}
    return {"route": "found"}
