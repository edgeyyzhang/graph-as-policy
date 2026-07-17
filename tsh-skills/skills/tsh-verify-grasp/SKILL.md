---
name: tsh-verify-grasp
description: >
  Verify a grasp actually holds the object, from the gripper's own
  proprioception (jaws closed onto the tape wall stop at a nonzero open
  fraction; closed on air reads ~0), optionally confirmed by a VLM yes/no on
  the scene camera. Emits a ROUTED verdict — holding / retry / give_up
  (bounded attempts) — so a silent empty grip becomes a recovery route back
  to re-perceive + re-grasp instead of a downstream mystery failure. Use
  after any scripted close (pickup, exchange receiver) on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [tsh, verify, grasp, recovery, yam]
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_gripper
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
    fraction: float           # measured gripper open fraction
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

The cheapest useful postcondition gate: a closed-on-air gripper reads an
open fraction near 0, a seated ring-wall grip reads ~0.2-0.4. One tool call,
no perception, no sim-time cost — and it converts the most common silent
grasp failure into an explicit routed branch the graph (and the refine loop)
can see and recover from.

Enable the optional VLM confirm (`GAP_VERIFY_VLM=1`, or `use_vlm=True` on
the node) to also ask "is the object held, lifted off the table?" on the
scene camera — catches the rarer pinched-but-slipping case the fraction
alone misses.

## When to use

- Immediately after `tsh-pickup` (the giver's close), before committing to
  the transport/handover.
- After the exchange, to verify the receiver's grip before place.

## When NOT to use

- Mid-exchange (between receiver close and giver release) — both grippers
  touch the tape there; the fraction is ambiguous and the exchange has its
  own settle logic.

## Recommended subgraph state flow

```text
observe → check ──("holding")──▶ holding → END
                ├─("retry")────▶ retry → END
                └─("give_up")──▶ give_up → END
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"` (only
   needed for the VLM confirm; keep it — one capture is cheap).
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
