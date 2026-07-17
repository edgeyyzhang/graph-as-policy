"""Checkpoints for `handover` — the receiver holds the tape, the giver let go.

Arm identity comes from the subgraph's own echoed outputs (route-decided),
so the predicate is direction-agnostic: arm 0 = the ``left_*`` MJCF link
namespace, arm 1 = ``right_*``.
"""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"


def _prefix(arm) -> str:
    return "left" if int(arm) == 0 else "right"


def _receiver_holds(w, outputs) -> bool:
    recv = outputs.get("receiver_arm")
    if recv is None:
        return False
    return w.body(TAPE).is_grasped_by(_prefix(recv))


def _giver_released(w, outputs) -> bool:
    giver = outputs.get("giver_arm")
    if giver is None:
        return False
    return not w.body(TAPE).is_grasped_by(_prefix(giver))


def _diag(w, outputs) -> dict:
    return {"tape_contacts": sorted(w.body(TAPE).contacts),
            "giver_arm": outputs.get("giver_arm"),
            "receiver_arm": outputs.get("receiver_arm")}


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="receiver_holds_tape",
        subgraph="handover",
        predicate=_receiver_holds,
        diagnostics_fn=_diag,
        rationale="after the exchange the RECEIVING arm's links contact the "
                  "tape — possession actually transferred",
        validate=True,
    ),
    Checkpoint(
        name="giver_released_tape",
        subgraph="handover",
        predicate=_giver_released,
        rationale="the giver retracted clear (probe; a lingering giver "
                  "contact foreshadows the known arm-arm graze)",
        validate=False,
    ),
]
