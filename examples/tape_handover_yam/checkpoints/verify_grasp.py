"""Checkpoints for `verify_grasp` — the verifier itself is verified.

The subgraph's GT-free verdict must agree with privileged truth; a verifier
that lies (passes an empty grip, retries a good one) is worse than none.
"""

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"


def _verdict_agrees(w, outputs) -> bool:
    verdict = outputs.get("verdict")
    if verdict not in ("holding", "retry", "give_up"):
        return False
    return (verdict == "holding") == w.body(TAPE).is_grasped()


def _diag(w, outputs) -> dict:
    return {"verdict": outputs.get("verdict"),
            "fraction": outputs.get("fraction"),
            "gt_grasped": w.body(TAPE).is_grasped()}


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="grasp_verdict_agrees_with_gt",
        subgraph="verify_grasp",
        predicate=_verdict_agrees,
        diagnostics_fn=_diag,
        rationale="the GT-free holding verdict matches privileged "
                  "is_grasped — meta-verification of the verifier",
        validate=True,
    ),
]
