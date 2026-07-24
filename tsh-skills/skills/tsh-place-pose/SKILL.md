---
name: tsh-place-pose
description: >
  Compute a reachable place pose for a held object at a destination — a
  yaw-reachability probe (plan_to_pose with execute=False), no motion and no
  collision-body attachment. Determines WHERE the held object's centre
  should rest (destination top face + perceived half-thickness + optional
  nudge) and WHICH yaw about the vertical is plannable, sweeping candidate
  yaws until the first reachable one is found — the round tape lays flat
  identically at any yaw, so the sweep only trades reach. Feeds
  tsh-transport-held (the collision-aware hover approach) and tsh-place (the
  descend/release/retract legs) so both consume the SAME probed pose instead
  of each re-deriving it. place_arm comes from tsh-route; when the task pins
  the arm, bind a literal int inside the subgraph and omit that input.
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, place, geometry, reachability, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - curobo.plan_to_pose
  exit_conditions:
    derived: Reachable pose found; place_xyz/hover_xyz/place_quat/target_xyz/target_quat bound in the outputs.
    failed: No yaw in the sweep was reachable (raise routes to on_error).
  required_inputs:
    held_offset: Vec3               # the LATEST holder's tool offset — pickup's tape_in_giver on the
                                     # direct route, or the exchange's receiver_offset after a handover
    container_xyz: Vec3                  # tsh-perceive-cv's <name>_xyz on the DESTINATION instance — that
                                     # instance's output prefix MUST be exactly `container_`
    target_half_z: float            # the PLACED object's half thickness — tsh-perceive-cv's <name>_half_z
                                     # (target_ prefix); rebind to the script's tape_half_z kwarg.
                                     # REQUIRED, no tuned fallback
    place_arm: int                  # the (future) holding arm from tsh-route (matches tsh-place's own
                                     # input name exactly — same arm places it). Rebind to the script's
                                     # arm_id kwarg. Task pins the arm? Pass a LITERAL in the node
                                     # inputs and OMIT this subgraph input.
  produces_outputs:
    place_xyz: Vec3                 # world OBJECT-centre rest pose (tape flush on the destination)
    hover_xyz: Vec3                 # place_xyz raised by PLACE_Z_APPROACH — the approach target
    place_quat: Quaternion          # reachable presentation orientation (wxyz), first yaw that plans
    target_xyz: Vec3                # alias of hover_xyz — matches tsh-transport-held's generic
                                     # target_xyz input so cross-subgraph auto-wire can bind it
    target_quat: Quaternion         # alias of place_quat — matches tsh-transport-held's target_quat
  hard_rules:
    - >
      This node never executes a trajectory — it only probes reachability
      (`execute=False`). The actual hover move is `tsh-transport-held`; the
      actual release is `tsh-place`.
    - >
      Rest height (`place_xyz` z-component) is DERIVED: destination top face
      + perceived `tape_half_z`. Do not hard-code a place Z.
    - >
      `target_xyz`/`target_quat` exist ONLY so `tsh-transport-held` can
      auto-wire immediately after this node in the SAME chain. Because this
      node always runs later than any perceive instance, it correctly
      shadows their `target_xyz` under the latest-producer rule — but do not
      reuse these alias names for any OTHER purpose.
  canonical_scripts:
    - place_pose: scripts/place_pose.py
  streaming: false
---

# tsh-place-pose

Factored out of `tsh-place` so the reachable-pose search — a pure planning
probe, no side effects on the sim — is its own node: `tsh-transport-held`
(collision-aware hover approach) and `tsh-place` (descend/release/retract)
both consume its `place_xyz`/`hover_xyz`/`place_quat` instead of each
re-deriving them, so the two can't drift apart on what pose they're aiming
for.

The target is the tape centre = perceived destination top-face centre + the
tape half-thickness (so it rests flush). The tape is round, so a yaw sweep
pivots the TCP about the tape centre until the hover is reachable — the
ring's own symmetry is the reach margin. The place *location* should be read
UP FRONT by the destination `tsh-perceive-cv` instance, while the
destination view is clean — at place time the held tape + gripper occlude it.

## When to use

- Always the first of the three place-stage nodes: `place_pose →
  transport_held → place`. On the `direct` route this comes right after
  `verify_grasp`; on the `needs_handover` route it MUST come AFTER
  `tsh-handover`, never before — `held_offset` must be the RECEIVER's grip
  (post-handover), not the giver's, or the yaw-reachability probe checks the
  wrong arm's tool offset.

## When NOT to use

- Standalone without `tsh-transport-held` + `tsh-place` following — this
  node only computes a pose, it never moves the arm or releases the object.

## Recommended subgraph state flow

1 state:

```text
place_pose
```

1. **`place_pose`** — `type: script`, `scripts/<sg>/place_pose.py`. Inputs:
   `held_offset=Ref("in.held_offset")`, `container_xyz=Ref("in.container_xyz")`,
   `tape_half_z=Ref("in.target_half_z")`, `arm_id=Ref("in.place_arm")` (or a
   literal, when the task pins the arm).
   Returns `{place_xyz, hover_xyz, place_quat, target_xyz, target_quat}`.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Reachable pose found; route to `transport_held`. |
| `failed` | No yaw in the sweep was reachable; route to `abort`. |

## See also

- `tsh-perceive-cv` — supplies `container_xyz`, `target_half_z`.
- `tsh-transport-held` — consumes `hover_xyz`/`place_quat` for the collision-aware approach.
- `tsh-place` — consumes `place_xyz`/`place_quat` for the final descend/release/retract.
- `tsh-route` — decides `place_arm` and whether a handover precedes the place chain.
