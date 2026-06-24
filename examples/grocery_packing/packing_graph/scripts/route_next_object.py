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

import logging
import os

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

# Stop a cyclic packing episode that can no longer make progress (e.g. one
# ungraspable item left): if completion_rate has not increased for this many
# consecutive iterations, exit the loop cleanly rather than re-grasping the
# same item forever. State is per-process (one episode == one gap-run PID).
_STUCK_LIMIT = 3


def _no_progress(cr: float | None) -> bool:
    if cr is None:
        return False
    path = f"/tmp/.gap_route_progress_{os.getpid()}"
    prev, stuck = -1.0, 0.0
    try:
        with open(path) as f:
            prev, stuck = (float(x) for x in f.read().split())
    except Exception:
        pass
    stuck = 0 if cr > prev + 1e-6 else stuck + 1
    try:
        with open(path, "w") as f:
            f.write(f"{cr} {stuck}")
    except Exception:
        pass
    return stuck >= _STUCK_LIMIT


def run(ctx: NodeContext, found: bool, cloud: dict, container_obb: dict | None = None) -> dict:
    # Authoritative stop: env reports the task complete (all items packed).
    try:
        sc = ctx.tool("sim.check_success")
        cr = sc.get("completion_rate")
        logger.info(
            "[route] completion_rate=%s  task_completed=%s  perceived_item=%s",
            f"{cr:.3f}" if isinstance(cr, (int, float)) else cr,
            sc.get("task_completed"), bool(found),
        )
        if sc.get("task_completed"):
            return {"route": "none"}
        if _no_progress(cr if isinstance(cr, (int, float)) else None):
            logger.info("[route] no completion progress for %d iters -> stopping loop",
                        _STUCK_LIMIT)
            return {"route": "none"}
    except Exception:
        pass  # non-sim connector -> fall back to perception
    if not found:
        return {"route": "none"}
    pts = np.asarray(cloud["points"]) if cloud and "points" in cloud else None
    if pts is None or pts.size == 0:
        return {"route": "none"}
    return {"route": "found"}
