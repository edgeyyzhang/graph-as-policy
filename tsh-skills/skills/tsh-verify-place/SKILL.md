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
  re-route, re-pick) instead of silently ending "done". Use after tsh-place
  on LIBERO-YAM.
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
    dest_xyz: Vec3            # the destination top-face centre the place used
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
      The verdict is ROUTED, never raised — map retry to the re-perceive
      loop edge, give_up to abort.
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

```text
observe → check ──("placed")──▶ placed → END
                ├─("retry")───▶ retry → END
                └─("give_up")─▶ give_up → END
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"` — a FRESH
   capture after the retract; never reuse an earlier subgraph's cameras.
2. **`check`** — `type: script`, `scripts/<sg>/verify_place.py`,
   `inputs={"object_query": "yellow tape", "dest_xyz": Ref("in.dest_xyz"),
   "half_z": Ref("in.target_half_z"), "cameras": Ref("observe.cameras")}`.
3. Conditional edges on `check`'s `verdict` field to the three noop exits.

At the TOP level: `placed → done` (or the next-item router),
`retry → <the perceive subgraph>` (loop edge), `give_up → abort`.

## Checkpoints

- validate=True: the verdict agrees with privileged truth — `placed` iff
  `w.body(<object>).is_on(w.body(<dest>))` (the verifier itself is
  verified).

## See also

- `tsh-verify-grasp` — the grasp-side twin.
- `tsh-place` — the producer this gates.
