---
name: tsh-transport-held
description: >
  Move a HELD object's centre to a world target pose — the shared
  "tape-as-EE" transport. The measured rigid held_offset (object centre in
  the holder's TCP frame, from the pickup's pre-close FK anchor or the
  exchange's grab-instant measurement) is composed into the planner's
  tcp_offset, so the target is an OBJECT-centre pose, never a bare TCP pose.
  With the perceived object cloud supplied, the move is planned with the
  object attached as a cuRobo collision body (plain tool-offset plan as
  fallback). The handover's giver PRESENT leg and the place approach are
  both this operation; use it standalone for any "hold the object at X"
  step (above a sorting stack, at a camera, at a rendezvous).
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, transport, held-object, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - curobo.plan_with_grasped_object
    - robot.get_ee_pose
  exit_conditions:
    transported: Held-object centre at the target; held_tcp bound in the outputs.
    failed: No plan for the move (raise routes to on_error).
  required_inputs:
    held_offset: Vec3            # object centre in the holder's TCP frame (measured)
    target_xyz: Vec3             # world OBJECT-centre target
    target_quat: Quaternion      # holder TCP orientation at the target, wxyz
  produces_outputs:
    transported: bool
    held_tcp: Se3Pose            # world TCP pose after the move
  hard_rules:
    - >
      The target is an OBJECT-centre pose; ALWAYS compose ``held_offset``
      into the plan — never plan the bare TCP to the target and hope the
      object lands there.
    - >
      Pass the perceived object cloud whenever the move includes a large
      reorient — the attached collision body is what keeps the held object
      out of the arm. Omit it only for small same-orientation translations.
  canonical_scripts:
    - transport_held: scripts/transport_held.py
  streaming: false
---

# tsh-transport-held

The factored-out common core of "move the thing I'm holding somewhere":
`_held.py` carries `plan_held_move` (tool-offset composition) and
`approach_with_attached` (cloud attached as a cuRobo collision body,
recentred at the live FK-tracked object centre). The handover's giver
present leg, the place hover approach, and this standalone node are all the
same operation on different targets — one implementation, so they cannot
drift apart.

## When to use

- The giver present leg of a handover (object centre → the meet point) —
  the `tsh-handover` subgraph composes this node before its exchange node
  (`skip_present=True` on the exchange).
- Any standalone held-object move: hold the tape above stack N, present an
  object to a camera, stage it at a rendezvous.

## When NOT to use

- Nothing is held (use plain motion / `robot.plan_move`).
- The final set-down legs of a place — those are constrained straight-line
  descents (`curobo_linear_move`), not free transports; use `tsh-place`.

## Recommended subgraph state flow

1 state:

```text
transport_held
```

1. **`transport_held`** — `type: script`, `scripts/<sg>/transport_held.py`.
   Inputs: `arm_id`, `held_offset=Ref("in.held_offset")`,
   `target_xyz=Ref("in.target_xyz")`, `target_quat=Ref("in.target_quat")`,
   optionally `held_cloud=Ref("in.held_cloud")`. Returns
   `{transported, held_tcp}`.

## Checkpoints

- validate=True: the held object's privileged centre is within tolerance of
  the commanded target (2-arg predicate on outputs vs world).
- probe: the object is still grasped after the move (`is_grasped()`).

## See also

- `scripts/_held.py` — the shared held-object motion core.
- `tsh-handover` — composes this as its present leg.
- `tsh-place` — uses the same `_held` core for its hover approach.
