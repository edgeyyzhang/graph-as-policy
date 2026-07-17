"""Checkpoints for `pickup` — the tape is actually held after the grasp."""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"


def _tape_grasped(w) -> bool:
    return w.body(TAPE).is_grasped()


def _tape_lifted(w) -> bool:
    tape = w.body(TAPE)
    try:
        return tape.bottom_z > w.body("table").top_z + 0.02
    except Exception:
        return tape.z > 0.80  # scene table top ~0.75


def _diag(w) -> dict:
    tape = w.body(TAPE)
    return {"tape_position": [float(v) for v in tape.position],
            "tape_contacts": sorted(tape.contacts)}


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="tape_grasped",
        subgraph="pickup",
        predicate=_tape_grasped,
        diagnostics_fn=_diag,
        rationale="the tape is in contact with a robot gripper link at "
                  "subgraph exit — the grasp's whole postcondition",
        validate=True,
    ),
    Checkpoint(
        name="tape_lifted",
        subgraph="pickup",
        predicate=_tape_lifted,
        rationale="the lift leg cleared the table (probe; a drag-along "
                  "grasp passes is_grasped but fails this)",
        validate=False,
    ),
]
