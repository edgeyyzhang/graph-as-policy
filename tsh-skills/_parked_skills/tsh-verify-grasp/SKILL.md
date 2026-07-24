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
      `check.py` itself never raises for `holding`/`retry`/`give_up` — it
      always returns a normal verdict string. `holding` and `retry` are
      routed via `conditional_edges` to noop exits; `give_up` is
      deliberately left OUT of that mapping so the unmatched value triggers
      the subgraph's `on_error="give_up"` exit — see "Recommended subgraph
      state flow" below. Do not add a third `give_up` node or a script that
      calls `raise` for `give_up`.
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

`give_up` is NOT a third routed node. Only TWO noop exits exist in the
subgraph body — `holding` and `retry`:

```text
observe → check ──("holding")──▶ holding → END
                └─("retry")────▶ retry   → END
```

with `on_error: "give_up"` set on the subgraph (a sibling of `exit`, not a
node). `check`'s `conditional_edges` mapping deliberately covers only
`{"holding": "holding", "retry": "retry"}` — `give_up` is left OUT of the
mapping on purpose. When `check.verdict == "give_up"` (attempts exhausted),
the router lookup fails on that unmapped value, the executor treats the
failure as the subgraph raising, and — because `on_error` is set — catches it
and exits the subgraph with status `"give_up"` instead of propagating the
error. That is the ONLY mechanism for reaching `give_up`; do not add a
`give_up` noop node or any node that explicitly raises (e.g. a
`raise_give_up` script node) — `on_error`'s value must not be a declared node
name and must not appear as a `conditional_edges` mapping target (the
validator rejects both), and an explicitly-added raise node still needs a
normal outgoing edge like any other node, which defeats the point.

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"` (MANDATORY —
   the VLM check needs the fresh cameras).
2. **`check`** — `type: script`, `scripts/<sg>/verify_grasp.py`,
   `inputs={"arm_id": Ref("in.pick_arm"), "object_query": "yellow tape",
   "cameras": Ref("observe.cameras")}`.
3. Conditional edges on `check`'s `verdict` field, mapping ONLY `holding` and
   `retry` to their noop exits. `exit.success_values` lists `["holding",
   "retry"]` — `give_up` is not in there either; it lives solely in
   `on_error`.

At the TOP level, map the subgraph exits:
`holding → <next stage>`, `retry → <the perceive subgraph>` (loop edge),
`give_up → abort` (the subgraph's `on_error` exit, wired like any other
subgraph exit value).

## Checkpoints

- validate=True: the verdict agrees with privileged truth — `holding` iff
  `w.body(<object>).is_grasped()` (the verifier itself is verified).

## See also

- `tsh-verify-place` — the place-side twin.
- `tsh-pickup` — the producer this gates.
- `open-robot-skills/tools/vlm` — the `vlm.query_yes_no` bundle
  (provider/model via `GAP_VLM_*`).
