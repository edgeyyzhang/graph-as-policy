"""Checkpoints for `gripper_geometry` — self-model outputs are plausible."""

from gap.runtime.verify import Checkpoint


def _offsets_plausible(w, outputs) -> bool:
    axial = outputs.get("fingertip_axial")
    gap = outputs.get("finger_half_gap")
    if axial is None or gap is None:
        return False
    return abs(float(axial)) < 0.20 and 0.005 < float(gap) < 0.20


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="gripper_offsets_plausible",
        subgraph="gripper_geometry",
        predicate=_offsets_plausible,
        rationale="FK-derived gripper offsets exist and are centimetre-scale "
                  "(a wrong-model load reads zeros or metre-scale garbage)",
        validate=True,
    ),
]
