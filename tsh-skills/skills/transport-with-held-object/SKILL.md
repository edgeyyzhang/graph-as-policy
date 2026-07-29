---
name: transport-with-held-object
description: >
  Move a HELD object's centre to a world target pose — the shared "tape-as-EE"
  transport. The measured rigid held_offset (object centre in the holder's TCP
  frame) is composed into the planner's tcp_offset, so the target is an
  OBJECT-centre pose, never a bare TCP pose. Serves the handover's giver PRESENT
  leg, the place hover approach, and any standalone "hold the object at X" step.
  Every input is identical on every instance — there is no per-instance
  renaming. holding_arm ships with held_offset from whoever last changed hands
  (bimanual-dispatch-route, then bimanual-handover), and hold_target_xyz/hold_target_quat
  carry the destination from calculate-place-pose (place legs) or from
  bimanual-route-arms's station-geometry node (present leg). Never bind a
  bare target_xyz/target_quat: that is perception's name for the picked
  object's location, and it would silently transport to the wrong pose.
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
    held_offset: Vec3
    holding_arm: int
    hold_target_xyz: Vec3
    hold_target_quat: Quaternion
  produces_outputs:
    transported: bool
    held_tcp: Se3Pose
  hard_rules:
    - >
      The target is an OBJECT-centre pose; ALWAYS compose held_offset into the
      plan — never plan the bare TCP to the target.
    - >
      The script's arm_id kwarg is ALWAYS Ref("in.holding_arm") — on every
      instance, both roles. held_offset and holding_arm are one datum (the
      object centre and the TCP frame it is expressed in), so every producer
      emits them together: bimanual-dispatch-route relays the giver's, bimanual-handover
      re-anchors them to the receiver. Never declare a bare arm_id input, and
      never pick an arm name per-instance — the holder is data that arrives
      with the offset, not a wiring choice.
    - >
      hold_target_xyz/hold_target_quat are the destination on EVERY instance —
      both roles, no per-instance renaming. calculate-place-pose emits them for the
      place legs and bimanual-route-arms's station-geometry node emits them
      (from the meet pose) for the present leg, so the latest producer on the
      path supplies the right one. Never declare a bare target_xyz/target_quat
      input: that name is perception's (the picked object's location), and
      binding to it silently transports the held object to the wrong pose.
    - >
      Do not wire held_cloud for now — leave it unset so the move takes the plain
      tool-offset plan (the attached-collision-body path is untested here).
  canonical_scripts:
    - transport_held: scripts/transport_held.py
  streaming: false
---

# transport-with-held-object

The factored-out common core of "move the thing I'm holding somewhere": the
measured `held_offset` is composed into the planner's tool offset so a "put the
object centre at X" target is one plan. The handover's giver present leg, the
place hover approach, and this standalone node are all the same operation on
different targets — one implementation, so they can't drift apart.

## When to use

- The collision-aware hover approach in the place chain (between `calculate-place-pose`
  and `place`), whose `hold_target_*` comes from `calculate-place-pose`.
- The giver PRESENT leg of a handover (between `bimanual-route-arms`'s
  station-geometry node and `bimanual-handover`), whose `hold_target_*` comes from
  that node (the meet pose).

Both roles declare the *same four inputs* — `held_offset`, `holding_arm`,
`hold_target_xyz`, `hold_target_quat`. Only the upstream producer differs, and
the latest-producer rule resolves that automatically. That uniformity is what
lets one implementation serve both roles without per-instance wiring choices.
- Any standalone held-object move (hold above a stack, present to a camera).

## When NOT to use

- Nothing is held (use plain motion).
- The final set-down legs of a place — those are straight-line descents; use
  `place`.

## Recommended subgraph state flow

```text
transport_held
```

1. **`transport_held`** — `type: script`, `scripts/<sg>/transport_held.py`.
   Inputs, **identical on every instance** (the script's kwargs keep their own
   names; only the Refs matter):
   `held_offset=Ref("in.held_offset")`, `arm_id=Ref("in.holding_arm")`,
   `target_xyz=Ref("in.hold_target_xyz")`,
   `target_quat=Ref("in.hold_target_quat")`.

   There is nothing to choose per role — the upstream producer of
   `hold_target_*` differs (station-geometry for the present leg,
   calculate-place-pose for the place legs) and the latest-producer rule binds the
   right one. Leave `held_cloud` unset. Returns `{transported, held_tcp}`.

## See also

- `scripts/_held.py` — the shared held-object motion core.
- `bimanual-route-arms` — its station-geometry node supplies
  `hold_target_xyz`/`hold_target_quat` (the meet pose) for the present instance.
- `calculate-place-pose` — supplies `hold_target_xyz`/`hold_target_quat` for the place instances.
- `bimanual-dispatch-route` — supplies `held_offset`/`holding_arm` (the giver's) after the grasp.
- `bimanual-handover` — re-anchors `held_offset`/`holding_arm` to the receiver after an exchange.
