"""Probe checkpoint for `grasp_sg` (authored by build_graph.py)."""

from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    return world.body('alphabet soup').is_grasped()


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "probe: when the canonical object is the one being grasped, the "
            "gripper closed ON it (multi-object loops need per-body names for "
            "a hard gate, so this stays a probe)"
        ),
        validate=False,
    ),
]
