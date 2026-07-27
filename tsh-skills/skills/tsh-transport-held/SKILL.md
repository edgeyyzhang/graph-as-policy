---
name: tsh-transport-held
description: >
  Move a HELD object's centre to a world target pose — the shared "tape-as-EE"
  transport. The measured rigid held_offset (object centre in the holder's TCP
  frame) is composed into the planner's tcp_offset, so the target is an
  OBJECT-centre pose, never a bare TCP pose. Serves the handover's giver PRESENT
  leg and the place hover approach, and any standalone "hold the object at X"
  step. The holding arm is a script kwarg wired from a declared subgraph input
  (giver_arm on the present instance, place_arm on the place instance), never a
  bare arm_id input. Likewise the target: the PLACE instance declares
  target_xyz/target_quat (tsh-place-pose's aliases); the PRESENT instance
  declares meet_xyz/giver_quat (tsh-station-geometry's own output names, NOT a
  generic target_xyz alias — that would collide with perception's own
  target_xyz), rebinding both to the script's target_xyz/target_quat kwargs.
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
      The script's arm_id kwarg must be wired from a declared subgraph input
      (giver_arm on the present instance, place_arm on the place instance) via
      Ref("in.<name>") — never a bare arm_id input (no upstream producer) and
      never a direct cross-subgraph Ref.
    - >
      Likewise target_xyz/target_quat: the place instance wires them from
      tsh-place-pose's target_xyz/target_quat aliases; the PRESENT instance
      wires them from tsh-station-geometry's meet_xyz/giver_quat instead —
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
  aliases) and `place_arm`.
- The giver PRESENT leg of a handover (between `tsh-station-geometry` and
  `tsh-handover`), consuming `meet_xyz`/`giver_quat` (from `tsh-station-geometry`
  directly, rebound to the script's target_xyz/target_quat) and `giver_arm`.
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
   Inputs: `held_offset=Ref("in.held_offset")`, and `arm_id` from a declared
   subgraph input — `giver_arm` on the present instance, `place_arm` on the
   place instance. The move target itself differs by instance:
   - place instance: `target_xyz=Ref("in.target_xyz")`,
     `target_quat=Ref("in.target_quat")` (from `tsh-place-pose`'s aliases).
   - present instance: `target_xyz=Ref("in.meet_xyz")`,
     `target_quat=Ref("in.giver_quat")` (declare `meet_xyz`/`giver_quat` as
     THIS subgraph's own inputs, from `tsh-station-geometry` directly).

   Leave `held_cloud` unset. Returns `{transported, held_tcp}`.

## See also

- `scripts/_held.py` — the shared held-object motion core.
- `tsh-station-geometry` — supplies `meet_xyz`/`giver_quat` for the present instance.
- `tsh-place-pose` — supplies `target_xyz`/`target_quat` aliases for the place instance.
- `tsh-handover` — supplies `place_arm` (aliased from `receiver_arm`) after an exchange.
