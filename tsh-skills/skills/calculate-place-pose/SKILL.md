---
name: calculate-place-pose
description: >
  Compute a reachable place pose for a held object at a destination — a
  yaw-reachability probe (plan_to_pose, execute=False), no motion. Determines
  where the held object's centre should rest (destination top face + perceived
  half-thickness) and which yaw about the vertical is plannable, sweeping
  candidate yaws until the first reachable one is found (the round tape lays flat
  identically at any yaw, so the sweep only trades reach). Feeds transport-with-held-object
  (the hover approach) and place (the descend/release) so both consume the
  same probed pose. place_arm comes from bimanual-route-arms; when the task pins the arm,
  bind a literal and omit the input.
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, place, geometry, reachability, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - curobo.plan_to_pose
  exit_conditions:
    derived: Reachable pose found; the poses bound in the outputs.
    failed: No yaw in the sweep was reachable (raise routes to on_error).
  required_inputs:
    held_offset: Vec3
    container_xyz: Vec3
    target_half_z: float
    place_arm: int
  produces_outputs:
    place_xyz: Vec3
    hover_xyz: Vec3
    place_quat: Quaternion
    hold_target_xyz: Vec3
    hold_target_quat: Quaternion
  hard_rules:
    - >
      This node never executes a trajectory — it only probes reachability
      (execute=False). The hover move is transport-with-held-object; the release is
      place.
    - >
      Rest height (place_xyz z) is DERIVED: destination top face + perceived
      half-thickness. Do not hard-code a place Z.
    - >
      On the handover route, run this AFTER bimanual-handover so held_offset is the
      receiver's grip, not the giver's.
  canonical_scripts:
    - place_pose: scripts/place_pose.py
  streaming: false
---

# calculate-place-pose

Factored out of `place` so the reachable-pose search — a pure planning probe,
no side effects — is its own node: `transport-with-held-object` (hover approach) and
`place` (descend/release) both consume its `place_xyz`/`hover_xyz`/`place_quat`
instead of re-deriving them. The target is the tape centre = perceived
destination top-face centre + tape half-thickness (so it rests flush); the tape
is round, so a yaw sweep pivots the TCP about the tape centre until the hover is
reachable.

## When to use

- The first of the three place-stage nodes: `place_pose → transport_held → place`.

## When NOT to use

- Standalone without `transport-with-held-object` + `place` following — this node only
  computes a pose, it never moves the arm.

## Recommended subgraph state flow

```text
place_pose
```

1. **`place_pose`** — `type: script`, `scripts/<sg>/place_pose.py`. Inputs:
   `held_offset=Ref("in.held_offset")`, `container_xyz=Ref("in.container_xyz")`,
   `tape_half_z=Ref("in.target_half_z")`, `arm_id=Ref("in.place_arm")` (or a
   literal). Returns `{place_xyz, hover_xyz, place_quat, hold_target_xyz, hold_target_quat}`.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Reachable pose found; route to `transport_held`. |
| `failed` | No yaw reachable; route to `abort`. |

## See also

- `perceive-tape-cv` — supplies `container_xyz`, `target_half_z`.
- `transport-with-held-object` — consumes `hover_xyz`/`place_quat` for the approach.
- `place` — consumes `place_xyz`/`place_quat` for the descend/release.
