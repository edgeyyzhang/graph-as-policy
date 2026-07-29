"""Re-assert the probed route AFTER the grasp, so the graph can branch on it.

``bimanual-route-arms`` decides ``direct`` vs ``needs_handover`` BEFORE the grasp, because
``pickup`` needs its ``pick_arm``. But the branch it implies happens AFTER
the grasp (both routes pick first, then diverge). This node carries the
decision across the pickup: it re-reads the already-computed ``route`` field
and exposes it as this subgraph's exit, so the top level maps two exits to two
chains exactly the way it maps every other subgraph's exits.

It also relays ``giver_held_offset`` forward as the canonical ``held_offset`` —
the shared place chain (``calculate-place-pose`` / ``transport-with-held-object`` / ``place``)
needs that name to exist, and ``pickup`` deliberately does not produce it
directly (see ``pickup``'s hard_rules), so a graph that wires the place
chain before this node has no producer to bind and fails validation instead of
silently placing the wrong holder. On the ``needs_handover`` route,
``bimanual-handover`` provides its own ``held_offset`` later, which correctly wins
at runtime since it executes after this node.

``holding_arm`` rides along with it, and must: ``held_offset`` is the object
centre in the HOLDER's TCP frame, so the offset alone is underspecified — it
means nothing without the arm whose frame it is expressed in. Every producer of
``held_offset`` emits the matching arm beside it (``pickup`` pairs
``giver_held_offset`` with ``pick_arm``; ``bimanual-handover`` pairs its
``held_offset`` with ``receiver_arm``), so consumers declare ``holding_arm``
as a required input and can never be wired to the wrong arm — or to none.
Here the giver is still the holder (the exchange, if any, happens later).

No reachability work happens here — the probing was all done by ``bimanual-route-arms``.
This is pure control flow.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Vec3


class Output(TypedDict):
    route: str  # "direct" | "needs_handover" — the conditional-edge router field
    held_offset: Vec3  # relay of giver_held_offset — the shared place chain's input
    holding_arm: int   # the arm whose TCP frame held_offset is in (still the giver)


def run(ctx: NodeContext, *, route: str, giver_held_offset: Vec3,
        pick_arm: int) -> Output:
    """Echo the probed route as this subgraph's routing field.

    route: the ``route`` output of ``bimanual-route-arms`` (auto-wired by exact name).
    giver_held_offset: ``pickup``'s grip offset, relayed as ``held_offset``.
    pick_arm: ``pickup``'s grasping arm — relayed as ``holding_arm``, the
              frame ``held_offset`` is expressed in. Nothing has changed hands
              yet at this node, so the picker is still the holder.
    """
    if route not in ("direct", "needs_handover"):
        raise RuntimeError(
            f"bimanual-dispatch-route: unknown route {route!r} — expected the "
            f"'direct' or 'needs_handover' value bimanual-route-arms emits")
    print(f"[bimanual-dispatch-route] {route} (holding_arm={pick_arm})", flush=True)
    return {"route": route, "held_offset": giver_held_offset,
            "holding_arm": int(pick_arm)}
