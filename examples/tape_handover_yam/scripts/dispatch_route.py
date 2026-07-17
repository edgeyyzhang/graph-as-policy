"""Top-level router: after a verified grasp, dispatch on the probed route."""

from gap import NodeContext


def run(ctx: NodeContext, *, route: str) -> dict:
    if route not in ("direct", "handover"):
        raise RuntimeError(f"dispatch: unknown route {route!r}")
    return {"route": "place" if route == "direct" else "present"}
