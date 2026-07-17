"""Checkpoints for `route` — the routing decision is internally coherent."""

from gap.runtime.verify import Checkpoint


def _route_coherent(w, outputs) -> bool:
    route = outputs.get("route")
    pick = outputs.get("pick_arm")
    place = outputs.get("place_arm")
    if route not in ("direct", "handover") or pick is None or place is None:
        return False
    pick, place = int(pick), int(place)
    if pick not in (0, 1) or place not in (0, 1):
        return False
    return (pick == place) == (route == "direct")


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="route_coherent",
        subgraph="route",
        predicate=_route_coherent,
        rationale="pick_arm == place_arm exactly on the direct route; a "
                  "mismatch means the dispatch will hand the tape to an arm "
                  "that never receives it",
        validate=True,
    ),
]
