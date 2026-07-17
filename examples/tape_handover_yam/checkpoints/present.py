"""Checkpoints for `present` — the tape survived the transport to the meet."""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"


def _still_grasped(w) -> bool:
    return w.body(TAPE).is_grasped()


def _above_table(w) -> bool:
    tape = w.body(TAPE)
    try:
        return tape.bottom_z > w.body("table").top_z + 0.05
    except Exception:
        return tape.z > 0.85


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="tape_still_grasped_at_present",
        subgraph="present",
        predicate=_still_grasped,
        rationale="the reorienting transport to the meet point is where a "
                  "marginal grip slips — the tape must still be held",
        validate=True,
    ),
    Checkpoint(
        name="tape_presented_high",
        subgraph="present",
        predicate=_above_table,
        rationale="the presented tape hangs well above the table (probe)",
        validate=False,
    ),
]
