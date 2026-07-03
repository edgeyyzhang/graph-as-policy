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

import hashlib
import logging
import os
import tempfile

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

# Stop a cyclic packing episode that can no longer make progress (e.g. one
# ungraspable item left): if completion_rate has not increased for this many
# consecutive iterations, exit the loop cleanly rather than re-grasping the
# same item forever.
_STUCK_LIMIT = 3


def _progress_path(ctx: NodeContext | None) -> str:
    """Per-episode scratch file for the no-progress counter.

    Keyed by the workflow's trace dir (unique per trial/run), NOT just the
    PID: a benchmark worker executes many trials in one process, and a
    PID-keyed file carries the previous trial's (prev, stuck) tail into the
    next trial's first pass, ending its loop before anything is grasped.
    """
    trace = getattr(ctx, "_trace", None) if ctx is not None else None
    out_dir = getattr(trace, "_output_dir", None) if trace is not None else None
    if out_dir is not None:
        tag = hashlib.sha1(str(out_dir).encode()).hexdigest()[:12]
    else:  # no-trace run: fall back to the PID (single episode per process)
        tag = f"pid{os.getpid()}"
    return os.path.join(tempfile.gettempdir(), f".gap_route_progress_{tag}")


def _no_progress(cr: float | None, ctx: NodeContext | None = None) -> bool:
    if cr is None:
        return False
    path = _progress_path(ctx)
    prev, stuck = -1.0, 0.0
    try:
        with open(path) as f:
            prev, stuck = (float(x) for x in f.read().split())
    except Exception:
        pass
    # completion_rate is monotonic within an episode (delivered items are
    # retired) — a DROP means stale state from an earlier episode reusing
    # this key. Start fresh.
    if cr < prev - 1e-6:
        prev, stuck = -1.0, 0.0
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
        if _no_progress(cr if isinstance(cr, (int, float)) else None, ctx):
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
