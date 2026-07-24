---
name: tsh-verify-place
description: >
  Verify the placed object actually rests on the destination — re-perceive
  it after the retract (same classical colour+height CV core as perception;
  the view is clean again, and no model servers are needed) and check, with
  no ground truth, that its XY centre is within
  tolerance of the destination centre and its top face is at the expected
  rest height (destination top + 2 x half thickness). Emits a ROUTED
  verdict — placed / retry / give_up — so a dropped or bounced object loops
  the graph back through the full pipeline (re-perceive wherever it landed,
  re-route, re-pick) instead of silently ending "done". Finish EVERY
  tsh-place with this gate before declaring done, even when the task does
  not mention verification. On LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: verification
  tags: [tsh, verify, place, recovery, cv, yam]
gap:
  allowed_tools:
    - robot.get_observation
  exit_conditions:
    placed: Object rests on the destination; finish (or loop to the next item).
    retry: Missed/dropped, attempts remain — route BACK to the perceive subgraph.
    give_up: Missed, attempts exhausted — route to abort.
  required_inputs:
    dest_xyz: Vec3            # tsh-perceive-cv's <name>_xyz on the DESTINATION instance — that
                              # instance's output prefix MUST be exactly `dest_` (fixed wiring
                              # contract, not the task's own word for the object)
    target_half_z: float      # the object's perceived half thickness
  produces_outputs:
    verdict: str              # the routing field
    placed: bool
    xy_error_m: float
    z_error_m: float
  hard_rules:
    - >
      Re-perceive AFTER the retract, from a fresh internal observe — the
      pre-place perception is stale the moment the object is released.
    - >
      `check.py` itself never raises for `placed`/`retry`/`give_up` — it
      always returns a normal verdict string. `placed` and `retry` are
      routed via `conditional_edges` to noop exits; `give_up` is
      deliberately left OUT of that mapping so the unmatched value triggers
      the subgraph's `on_error="give_up"` exit — see "Recommended subgraph
      state flow" below. Do not add a third `give_up` node or a script that
      calls `raise` for `give_up`.
  canonical_scripts:
    - verify_place: scripts/verify_place.py
  streaming: false
---

# tsh-verify-place

Closes the loop the place skill leaves open: `tsh-place` exits "placed" when
its MOTION finished, not when the object actually rests on the destination.
This skill re-perceives the object after the gripper retracts and compares,
GT-free, against the same destination perception the place targeted:

- XY: object centre within `xy_tol` (default 6 cm) of the destination centre,
- Z: object top face within `z_tol` (default 4 cm) of
  `dest_top + 2 x half_z` — i.e. resting ON the destination, not on the
  table next to it (which passes an XY-only check when the drop bounced
  off the near edge).

For stacking loops (sorting), the expected rest height naturally grows when
the destination is re-perceived each iteration — no stack-height bookkeeping.

## When to use

- Immediately after `tsh-place`, before `done` / the next-item loop edge.

## When NOT to use

- The object is still held (nothing was released — verify the grasp
  instead).

## Recommended subgraph state flow

`give_up` is NOT a third routed node. Only TWO noop exits exist in the
subgraph body — `placed` and `retry`:

```text
observe → check ──("placed")──▶ placed → END
                └─("retry")───▶ retry  → END
```

with `on_error: "give_up"` set on the subgraph (a sibling of `exit`, not a
node). `check`'s `conditional_edges` mapping deliberately covers only
`{"placed": "placed", "retry": "retry"}` — `give_up` is left OUT of the
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

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"` — a FRESH
   capture after the retract; never reuse an earlier subgraph's cameras.
2. **`check`** — `type: script`, `scripts/<sg>/verify_place.py`,
   `inputs={"object_query": "yellow tape", "dest_xyz": Ref("in.dest_xyz"),
   "half_z": Ref("in.target_half_z"), "cameras": Ref("observe.cameras")}`.
3. Conditional edges on `check`'s `verdict` field, mapping ONLY `placed` and
   `retry` to their noop exits. `exit.success_values` lists `["placed",
   "retry"]` — `give_up` is not in there either; it lives solely in
   `on_error`.

At the TOP level: `placed → done` (or the next-item router),
`retry → <the perceive subgraph>` (loop edge), `give_up → abort` (the
subgraph's `on_error` exit, wired like any other subgraph exit value).

## Checkpoints

- validate=True: the verdict agrees with privileged truth — `placed` iff
  `w.body(<object>).is_on(w.body(<dest>))` (the verifier itself is
  verified).

## See also

- `tsh-verify-grasp` — the grasp-side twin.
- `tsh-place` — the producer this gates.
