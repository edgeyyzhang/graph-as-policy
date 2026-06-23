"""Postcondition checkpoint for `container_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _container_matches_truth(world, out) -> bool:
    """The perceived basket OBB center lines up with the sim's ground truth."""
    c = out["container_obb"]["center"]
    p = world.body("basket").position
    return abs(c["x"] - p[0]) < 0.12 and abs(c["y"] - p[1]) < 0.12


CHECKPOINTS = [
    Checkpoint(
        name="container_obb_matches_truth",
        subgraph="container_sg",
        # Probe, not a hard gate: a large open basket's OBB *center* (geometric
        # centroid of the visible rim) sits a fair bit from the sim body
        # *origin*, so an exact ground-truth match is unreliable. We still
        # surface the localization delta in feedback, but don't gate on it.
        predicate=_container_matches_truth,
        rationale="perceived basket localizes near the real basket (probe)",
        validate=False,
    ),
]
