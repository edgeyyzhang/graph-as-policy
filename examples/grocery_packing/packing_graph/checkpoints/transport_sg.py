"""Probe checkpoint for `transport_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _target_in_container(world) -> bool:
    return world.body('alphabet soup').is_in(world.body("basket"))


CHECKPOINTS = [
    Checkpoint(
        name="target_in_container",
        subgraph="transport_sg",
        predicate=_target_in_container,
        rationale=(
            "probe: the canonical object settled inside the basket after "
            "release (probe, not a hard gate — see grasp_sg)"
        ),
        validate=False,
    ),
]
