"""Checkpoints for `place` — the tape rests on the destination, released."""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"
DUCT = "duct_tape_1"


def _tape_on_dest(w) -> bool:
    tape = w.body(TAPE)
    return tape.is_on(w.body(DUCT)) and not tape.is_grasped()


def _coverage(w) -> bool:
    return w.body(TAPE).xy_coverage_over(w.body(DUCT)) > 0.5


def _diag(w) -> dict:
    tape = w.body(TAPE)
    return {"tape_position": [float(v) for v in tape.position],
            "tape_contacts": sorted(tape.contacts),
            "coverage": float(tape.xy_coverage_over(w.body(DUCT)))}


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="tape_on_dest_released",
        subgraph="place",
        predicate=_tape_on_dest,
        diagnostics_fn=_diag,
        rationale="the tape rests ON the destination and no gripper still "
                  "touches it — the task's terminal condition for one item",
        validate=True,
    ),
    Checkpoint(
        name="tape_covers_dest",
        subgraph="place",
        predicate=_coverage,
        rationale="footprint overlap above half (probe; a rim-perched "
                  "landing passes is_on but reads low here)",
        validate=False,
    ),
]
