"""Verify a grasp actually holds the object — a VLM look, routed recovery.

After a close, an empty grip is silent: the plan succeeded, the gripper
closed, and nothing is held. Proprioception cannot catch this reliably in
sim (the finger stop fraction barely separates closed-on-air from a seated
ring-wall grip), so this node just LOOKS: one ``vlm.query_yes_no`` on the
scene camera.

Two prompt findings, validated against sim frames (held + not-held):
  * The prompt is ARM-AGNOSTIC ("a robot gripper"), because the agentview
    faces the robots — robot-left appears on image-right, and naming a side
    makes the VLM judge the wrong arm.
  * The wrist camera is deliberately NOT attached: at home pose it looks
    down past the open fingers at the object on the table, which reads as
    "held between the fingers" and flips the empty-grip case to a false
    YES — even with a cautionary instruction. Scene-only is correct on
    both cases; the close-up actively poisons the judgment.

The verdict is a ROUTED field, not a raise: ``holding`` continues,
``retry`` loops the graph back to re-perceive + re-grasp (bounded by
``max_attempts`` — the counter lives in this process, one run = one
episode), ``give_up`` routes to abort. Failure here is recoverable state,
which is exactly what makes it worth testing an agent on.
"""

from __future__ import annotations

import itertools
from typing import TypedDict

from gap import NodeContext

# One attempt counter per episode (module state lives for the process).
_ATTEMPTS = itertools.count(1)


class Output(TypedDict):
    verdict: str      # "holding" | "retry" | "give_up" — the routing field
    holding: bool
    attempts: int


def run(ctx: NodeContext, *, arm_id: int, object_query: str = "object",
        cameras: list, max_attempts: int = 2) -> Output:
    """Ask the VLM whether the grip on ``arm_id`` holds; emit a routed verdict.

    arm_id:        the arm whose grasp is being verified (route-decided).
    object_query:  noun phrase for the VLM prompt.
    cameras:       observation camera list (from an internal ``observe``
                   node). REQUIRED — the VLM is the whole check.
    max_attempts:  routed ``retry`` verdicts before ``give_up``.
    """
    if not cameras:
        raise RuntimeError("verify_grasp: no cameras — the observe node must "
                           "feed this check (the VLM is the whole verdict)")
    scene = next((c for c in cameras if c["name"] == "agentview"), cameras[0])
    prompt = (f"Is the {object_query} currently grasped by a robot gripper "
              f"and lifted clearly OFF the table surface? Answer YES only "
              f"if a gripper is holding it up in the air. Answer NO if it "
              f"is resting on the table or no gripper is holding it.")
    ans = ctx.tool("vlm.query_yes_no", prompt=prompt, image=scene["rgb"])
    holding = bool(ans.get("answer", False))

    attempts = next(_ATTEMPTS)
    if holding:
        verdict = "holding"
    elif attempts < int(max_attempts) + 1:
        verdict = "retry"
    else:
        verdict = "give_up"
    print(f"[verify_grasp] arm={arm_id} holding={holding} "
          f"attempt={attempts} -> {verdict} | {ans.get('text', '')[:120]}",
          flush=True)
    return {"verdict": verdict, "holding": holding, "attempts": attempts}
