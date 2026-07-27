"""Re-assert the probed route AFTER the grasp, so the graph can branch on it.

``tsh-route`` decides ``direct`` vs ``needs_handover`` BEFORE the grasp, because
``tsh-pickup`` needs its ``pick_arm``. But the branch it implies happens AFTER
the grasp (both routes pick first, then diverge). This node carries the
decision across the pickup: it re-reads the already-computed ``route`` field
and exposes it as this subgraph's exit, so the top level maps two exits to two
chains exactly the way it maps every other subgraph's exits.

It also relays ``giver_held_offset`` forward as the canonical ``held_offset`` —
the shared place chain (``tsh-place-pose`` / ``tsh-transport-held`` / ``tsh-place``)
needs that name to exist, and ``tsh-pickup`` deliberately does not produce it
directly (see ``tsh-pickup``'s hard_rules), so a graph that wires the place
chain before this node has no producer to bind and fails validation instead of
silently placing the wrong holder. On the ``needs_handover`` route,
``tsh-handover`` provides its own ``held_offset`` later, which correctly wins
at runtime since it executes after this node.

No reachability work happens here — the probing was all done by ``tsh-route``.
This is pure control flow.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Vec3


class Output(TypedDict):
    route: str  # "direct" | "needs_handover" — the conditional-edge router field
    held_offset: Vec3  # relay of giver_held_offset — the shared place chain's input


def run(ctx: NodeContext, *, route: str, giver_held_offset: Vec3) -> Output:
    """Echo the probed route as this subgraph's routing field.

    route: the ``route`` output of ``tsh-route`` (auto-wired by exact name).
    giver_held_offset: ``tsh-pickup``'s grip offset, relayed as ``held_offset``.
    """
    if route not in ("direct", "needs_handover"):
        raise RuntimeError(
            f"dispatch-route: unknown route {route!r} — expected the "
            f"'direct' or 'needs_handover' value tsh-route emits")
    print(f"[dispatch-route] {route}", flush=True)
    return {"route": route, "held_offset": giver_held_offset}
