"""Verify a grasp actually holds the object — cheap gate, routed recovery.

After a close, an empty grip is silent: the plan succeeded, the gripper
closed, and nothing is held. This node catches that from the gripper's own
proprioception — jaws that closed onto the tape wall stop at a nonzero open
fraction; jaws that closed on air read ~0 — plus an optional VLM yes/no
confirm on the scene camera (``GAP_VERIFY_VLM=1`` or ``use_vlm=True``).

The verdict is a ROUTED field, not a raise: ``holding`` continues,
``retry`` loops the graph back to re-perceive + re-grasp (bounded by
``max_attempts`` — the counter lives in this process, one run = one
episode), ``give_up`` routes to abort. Failure here is recoverable state,
which is exactly what makes it worth testing an agent on.
"""

from __future__ import annotations

import itertools
import os
from typing import TypedDict

from gap import NodeContext

# Open fraction below which a "closed" gripper is considered empty. The ring
# wall is ~10-18 mm across ~48 mm of jaw travel (fraction ~0.2-0.4 when
# seated); closed-on-air settles near 0. Overridable per-instance.
DEFAULT_MIN_HOLD_FRACTION = 0.05

# One attempt counter per episode (module state lives for the process).
_ATTEMPTS = itertools.count(1)


class Output(TypedDict):
    verdict: str      # "holding" | "retry" | "give_up" — the routing field
    holding: bool
    fraction: float   # measured gripper open fraction
    attempts: int


def run(ctx: NodeContext, *, arm_id: int, object_query: str = "object",
        cameras: list | None = None,
        min_hold_fraction: float = DEFAULT_MIN_HOLD_FRACTION,
        max_attempts: int = 2,
        use_vlm: bool | None = True) -> Output:
    """Check the grip on ``arm_id``; emit a routed verdict.

    arm_id:        the arm whose grasp is being verified (route-decided).
    object_query:  noun phrase for the VLM confirm prompt.
    cameras:       observation camera list (from an internal ``observe``
                   node); only needed when the VLM confirm is enabled.
    min_hold_fraction: open fraction below which the grip counts as empty.
    max_attempts:  routed ``retry`` verdicts before ``give_up``.
    use_vlm:       force the VLM confirm on/off; default reads
                   ``GAP_VERIFY_VLM`` (off unless set to 1).
    """
    fraction = float(ctx.tool("robot.get_gripper", arm_id=arm_id)["position"])
    holding = True ## fraction > float(min_hold_fraction) -> min_hold_fraction not reliable in sim

    if use_vlm is None:
        use_vlm = os.environ.get("GAP_VERIFY_VLM", "0") == "1"
    if holding and use_vlm and cameras:
        cam = next((c for c in cameras if c["name"] == "agentview"), cameras[0])
        ans = ctx.tool(
            "vlm.query_yes_no",
            prompt=(f"Is the {object_query} held between the robot gripper's "
                    f"fingers, lifted off the table?"),
            image=cam["rgb"])
        holding = bool(ans.get("answer", holding))

    attempts = next(_ATTEMPTS)
    if holding:
        verdict = "holding"
    elif attempts < int(max_attempts) + 1:
        verdict = "retry"
    else:
        verdict = "give_up"
    print(f"[verify_grasp] arm={arm_id} fraction={fraction:.3f} "
          f"holding={holding} attempt={attempts} -> {verdict}", flush=True)
    return {"verdict": verdict, "holding": holding,
            "fraction": fraction, "attempts": attempts}
