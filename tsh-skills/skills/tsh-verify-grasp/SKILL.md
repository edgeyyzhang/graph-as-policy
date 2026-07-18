---
name: tsh-verify-grasp
description: >
  Verify a grasp actually holds the object by LOOKING: one VLM yes/no on the
  scene camera. Emits a ROUTED verdict — holding / retry / give_up (bounded
  attempts) — so a silent empty grip becomes a recovery route back to
  re-perceive + re-grasp instead of a downstream mystery failure. Gate EVERY
  scripted pickup with this before committing to transport/handover, even
  when the task does not mention verification — an unverified grasp turns
  every later stage into a mystery failure. On LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [tsh, verify, grasp, recovery, vlm, yam]
gap:
  allowed_tools:
    - robot.get_observation
    - vlm.query_yes_no
  exit_conditions:
    holding: The grip holds; continue the pipeline.
    retry: Empty grip, attempts remain — route BACK to the perceive subgraph.
    give_up: Empty grip, attempts exhausted — route to abort.
  required_inputs:
    pick_arm: int             # the arm whose grasp is verified (route output)
  produces_outputs:
    verdict: str              # the routing field
    holding: bool
  hard_rules:
    - >
      The verdict is ROUTED, never raised — a failed verify is recoverable
      state; map retry to the re-perceive loop edge, give_up to abort.
    - >
      The retry bound is per-process (one run = one episode); do not rely on
      it resetting between items of a loop.
  canonical_scripts:
    - verify_grasp: scripts/verify_grasp.py
  streaming: false
---

# tsh-verify-grasp

The postcondition gate for a scripted close: ask a VLM "is the object
grasped by a robot gripper, lifted clearly off the table?" on the scene
camera. It converts the most common silent grasp failure into an explicit
routed branch the graph (and the refine loop) can see and recover from.

Why not proprioception: in sim the finger stop fraction barely separates
closed-on-air from a seated ring-wall grip, so the fraction gate mis-verdicts
both ways. The VLM look is the whole check.

Two validated prompt rules (see the script docstring): the prompt is
ARM-AGNOSTIC (the agentview mirrors left/right, so naming a side makes the
VLM judge the wrong arm), and the wrist close-up is deliberately NOT sent
(at home pose it frames the on-table object between the open fingers, which
flips an empty grip to a false YES).

## When to use

- Immediately after `tsh-pickup` (the giver's close), before committing to
  the transport/handover.
- After the exchange, to verify the receiver's grip before place.

## When NOT to use

- Mid-exchange (between receiver close and giver release) — both grippers
  touch the tape there and partially occlude it; the view is ambiguous and
  the exchange has its own settle logic.

## Recommended subgraph state flow

```text
observe → check ──("holding")──▶ holding → END
                ├─("retry")────▶ retry → END
                └─("give_up")──▶ give_up → END
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"` (MANDATORY —
   the VLM check needs the fresh cameras).
2. **`check`** — `type: script`, `scripts/<sg>/verify_grasp.py`,
   `inputs={"arm_id": Ref("in.pick_arm"), "object_query": "yellow tape",
   "cameras": Ref("observe.cameras")}`.
3. Conditional edges on `check`'s `verdict` field to the three noop exits.

At the TOP level, map the subgraph exits:
`holding → <next stage>`, `retry → <the perceive subgraph>` (loop edge),
`give_up → abort`.

## Checkpoints

- validate=True: the verdict agrees with privileged truth — `holding` iff
  `w.body(<object>).is_grasped()` (the verifier itself is verified).

## See also

- `tsh-verify-place` — the place-side twin.
- `tsh-pickup` — the producer this gates.
- `open-robot-skills/tools/vlm` — the `vlm.query_yes_no` bundle
  (provider/model via `GAP_VLM_*`).
