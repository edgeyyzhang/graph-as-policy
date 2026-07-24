"""Re-assert the probed route AFTER the grasp, so the graph can branch on it.

``tsh-route`` decides ``direct`` vs ``handover`` BEFORE the grasp, because
``tsh-pickup`` needs its ``pick_arm``. But the branch it implies happens AFTER
the grasp (both routes pick first, then diverge). This node carries the
decision across the pickup: it re-reads the already-computed ``route`` field
and exposes it as this subgraph's exit, so the top level maps two exits to two
chains exactly the way it maps every other subgraph's exits.

No reachability work happens here — the probing was all done by ``tsh-route``.
This is pure control flow.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext


class Output(TypedDict):
    route: str  # "direct" | "handover" — the conditional-edge router field


def run(ctx: NodeContext, *, route: str) -> Output:
    """Echo the probed route as this subgraph's routing field.

    route: the ``route`` output of ``tsh-route`` (auto-wired by exact name).
    """
    if route not in ("direct", "handover"):
        raise RuntimeError(
            f"dispatch-route: unknown route {route!r} — expected the "
            f"'direct' or 'handover' value tsh-route emits")
    print(f"[dispatch-route] {route}", flush=True)
    return {"route": route}
