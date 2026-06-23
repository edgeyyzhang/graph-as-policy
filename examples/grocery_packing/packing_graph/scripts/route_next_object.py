"""Loop control for grocery_packing's perceive_next subgraph.

Termination is driven by the benchmark's own completion signal, not geometry.
The VAB packing task teleports each delivered item to the graveyard and flips
``task_completed`` once every item is packed; we read that via the
``sim.check_success`` tool and exit the loop cleanly on "none" (success) the
moment it fires — regardless of what perception still sees on the table. This
is authoritative and free of false-rejects, so the loop never spins on the
basket after the table is clear, and never stops early while items remain.

Until then we route on perception: "found" when an item was returned, "none"
when nothing was. We do NOT geometrically reject items near the basket (a real
item beside the basket falls inside any reasonable radius and was being
dropped); the perceive node's ``object_description`` keeps the tournament from
picking the basket while a real item remains. The ``sim.check_success`` call is
wrapped so non-sim connectors (real robot) fall back to perception cleanly.
``container_obb`` is still accepted (the subgraph wires it in) but unused.
Authored by build_graph.py.
"""
from __future__ import annotations

import numpy as np

from gap import NodeContext


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict | None = None) -> dict:
    # Authoritative stop: all items packed -> task_completed -> done (success).
    try:
        if ctx.tool("sim.check_success").get("task_completed"):
            return {"route": "none"}
    except Exception:
        pass  # no sim checker (real-robot etc.) -> fall back to perception
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    return {"route": "found"}
