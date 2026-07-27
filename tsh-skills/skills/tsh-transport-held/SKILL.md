---
name: tsh-transport-held
description: >
  Move a HELD object's centre to a world target pose — the shared "tape-as-EE"
  transport. The measured rigid held_offset (object centre in the holder's TCP
  frame) is composed into the planner's tcp_offset, so the target is an
  OBJECT-centre pose, never a bare TCP pose. Serves the handover's giver PRESENT
  leg, the place hover approach, and any standalone "hold the object at X" step.
  The holding arm always arrives as the holding_arm input, paired with
  held_offset by whoever last changed hands (tsh-dispatch-route, then
  tsh-handover) — data that ships with the offset, never chosen per-instance.
  The target does vary by role: the PLACE instance declares
  target_xyz/target_quat (tsh-place-pose's aliases); the PRESENT instance
  declares meet_xyz/giver_quat (tsh-route-arms-bimanual's station-geometry
  node's own names, NOT a generic target_xyz alias — it would collide with
  perception's own target_xyz), rebound to the script's target kwargs.
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
    target_xyz: Vec3
    target_quat: Quaternion
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
      emits them together: tsh-dispatch-route relays the giver's, tsh-handover
      re-anchors them to the receiver. Never declare a bare arm_id input, and
      never pick an arm name per-instance — the holder is data that arrives
      with the offset, not a wiring choice.
    - >
      Likewise target_xyz/target_quat: the place instance wires them from
      tsh-place-pose's target_xyz/target_quat aliases; the PRESENT instance
      wires them from tsh-route-arms-bimanual's station-geometry node's meet_xyz/giver_quat instead —
      declare a meet_xyz/giver_quat subgraph input on the present instance,
      never a bare target_xyz (it would collide with perception's own
      target_xyz under the latest-producer rule).
    - >
      Do not wire held_cloud for now — leave it unset so the move takes the plain
      tool-offset plan (the attached-collision-body path is untested here).
  canonical_scripts:
    - transport_held: scripts/transport_held.py
  streaming: false
---

# tsh-transport-held

The factored-out common core of "move the thing I'm holding somewhere": the
measured `held_offset` is composed into the planner's tool offset so a "put the
object centre at X" target is one plan. The handover's giver present leg, the
place hover approach, and this standalone node are all the same operation on
different targets — one implementation, so they can't drift apart.

## When to use

- The collision-aware hover approach in the place chain (between `tsh-place-pose`
  and `tsh-place`), consuming `target_xyz`/`target_quat` (from `tsh-place-pose`'s
  aliases).
- The giver PRESENT leg of a handover (between `tsh-route-arms-bimanual`'s
  station-geometry node and `tsh-handover`), consuming `meet_xyz`/`giver_quat`
  (from that node directly, rebound to the script's target_xyz/target_quat).

Both consume `held_offset`/`holding_arm` identically — that pair is what makes
one implementation serve both roles.
- Any standalone held-object move (hold above a stack, present to a camera).

## When NOT to use

- Nothing is held (use plain motion).
- The final set-down legs of a place — those are straight-line descents; use
  `tsh-place`.

## Recommended subgraph state flow

```text
transport_held
```

1. **`transport_held`** — `type: script`, `scripts/<sg>/transport_held.py`.
   Inputs: `held_offset=Ref("in.held_offset")` and
   `arm_id=Ref("in.holding_arm")` — identical on every instance. Only the move
   target differs by role:
   - place instance: `target_xyz=Ref("in.target_xyz")`,
     `target_quat=Ref("in.target_quat")` (from `tsh-place-pose`'s aliases).
   - present instance: `target_xyz=Ref("in.meet_xyz")`,
     `target_quat=Ref("in.giver_quat")` (declare `meet_xyz`/`giver_quat` as
     THIS subgraph's own inputs, from `tsh-route-arms-bimanual's station-geometry node` directly).

   Leave `held_cloud` unset. Returns `{transported, held_tcp}`.

## See also

- `scripts/_held.py` — the shared held-object motion core.
- `tsh-route-arms-bimanual` — its station-geometry node supplies
  `meet_xyz`/`giver_quat` for the present instance.
- `tsh-place-pose` — supplies `target_xyz`/`target_quat` aliases for the place instance.
- `tsh-dispatch-route` — supplies `held_offset`/`holding_arm` (the giver's) after the grasp.
- `tsh-handover` — re-anchors `held_offset`/`holding_arm` to the receiver after an exchange.
