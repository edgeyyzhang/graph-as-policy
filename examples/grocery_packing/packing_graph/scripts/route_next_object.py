"""Loop control for grocery_packing's perceive_next subgraph.

Routes to "found" when perception returned a graspable item (grasp it) or
"none" when it returned nothing (the loop's clean exit).

We do NOT geometrically reject items near the basket: a real item parked
beside the basket falls inside any reasonable basket radius and was being
dropped from delivery. Instead the basket is excluded at the *perception*
level — the perceive node's ``object_description`` tells the VLM tournament a
grocery item is a packaged product and "never the wicker basket or storage
container", so the basket is never returned while a real item remains. When
every item has been packed and teleported to the benchmark graveyard,
perception has nothing left to lock onto and returns ``found=False`` → "none".
``container_obb`` is still accepted (the subgraph wires it in) but unused here.
Authored by build_graph.py.
"""
from __future__ import annotations

import numpy as np

from gap import NodeContext


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict | None = None) -> dict:
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    return {"route": "found"}
