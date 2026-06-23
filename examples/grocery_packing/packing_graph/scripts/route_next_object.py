"""Loop control for grocery_packing's perceive_next subgraph.

Termination is driven by the benchmark's own completion verdict. The VAB
packing task marks the episode complete once every item has been packed (and
teleported to the graveyard); we read that via ``sim.check_success`` and exit
the loop cleanly on "none" (success) the moment it fires — regardless of what
perception still sees on the table.

This is safe to poll every iteration because ``task_completed`` now reads the
cached success the env computed in its own ``step()`` (see
``FrankaLiberoEnv.task_completed``); it does NOT re-evaluate the stateful
``pack_all_into`` predicate, which would teleport delivered objects on each
call. The call is wrapped so non-sim connectors (real robot) fall back to
perception cleanly.

Until completion we route on perception: "found" when an item was returned,
"none" when nothing was. We do NOT geometrically reject items near the basket
(a real item beside the basket falls inside any reasonable radius and was being
dropped); the perceive node's ``object_description`` keeps the tournament from
picking the basket while a real item remains. ``container_obb`` is still
accepted (the subgraph wires it in) but unused. Authored by build_graph.py.
"""
from __future__ import annotations

import numpy as np

from gap import NodeContext


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict | None = None) -> dict:
    # Authoritative stop: env reports the task complete (all items packed).
    try:
        if ctx.tool("sim.check_success").get("task_completed"):
            return {"route": "none"}
    except Exception:
        pass  # non-sim connector -> fall back to perception
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    return {"route": "found"}
