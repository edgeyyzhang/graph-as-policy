"""Checkpoints for `ring_geometry` — derived radii bracket a plausible ring."""

from gap.runtime.verify import Checkpoint


def _radii_plausible(w, outputs) -> bool:
    hole = outputs.get("hole_radius")
    rim = outputs.get("rim_radius")
    if hole is None or rim is None:
        return False
    return 0.0 < float(hole) < float(rim) < 0.20


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="ring_radii_plausible",
        subgraph="ring_geometry",
        predicate=_radii_plausible,
        rationale="hole < rim and both centimetre-scale — the wall-midpoint "
                  "grasp math is meaningless outside this ordering",
        validate=True,
    ),
]
