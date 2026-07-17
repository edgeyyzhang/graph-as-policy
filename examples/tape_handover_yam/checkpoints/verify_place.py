"""Checkpoints for `verify_place` — the verifier itself is verified."""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"
DUCT = "duct_tape_1"


def _verdict_agrees(w, outputs) -> bool:
    verdict = outputs.get("verdict")
    if verdict not in ("placed", "retry", "give_up"):
        return False
    return (verdict == "placed") == w.body(TAPE).is_on(w.body(DUCT))


def _diag(w, outputs) -> dict:
    return {"verdict": outputs.get("verdict"),
            "xy_error_m": outputs.get("xy_error_m"),
            "z_error_m": outputs.get("z_error_m"),
            "gt_on_dest": w.body(TAPE).is_on(w.body(DUCT))}


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="place_verdict_agrees_with_gt",
        subgraph="verify_place",
        predicate=_verdict_agrees,
        diagnostics_fn=_diag,
        rationale="the GT-free placed verdict matches privileged is_on — "
                  "meta-verification of the verifier",
        validate=True,
    ),
]
