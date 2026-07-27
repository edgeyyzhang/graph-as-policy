---
name: tsh-place
description: >
  Release the held tape at a pre-computed pose and retract. The holder grips the
  tape off-centre (ring grasp or rim thread), so the tape is treated as the
  holder's end effector via the measured held_offset, composed once here to get
  the release TCP. Where/which-orientation come from tsh-place-pose; a preceding
  tsh-transport-held node has already carried the arm to the hover above
  place_xyz. This skill only does the straight-down set-down and straight-up
  retract (curobo_linear_move), never a free transport. place_arm comes from
  tsh-route; when the task pins the arm, bind a literal and omit the input.
compatibility: requires gap>=0.1
metadata:
  category: motion
  tags: [tsh, place, bimanual, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
    - libero-yam.execute_trajectory
    - curobo.plan_to_pose
    - curobo.plan_directed_linear
    - robot.open_gripper
  exit_conditions:
    placed: Tape released at the target and retracted; place_tcp bound in the outputs.
    failed: A set-down/retract leg had no plan (raise routes to abort).
  required_inputs:
    held_offset: Vec3
    place_xyz: Vec3
    place_quat: Quaternion
    place_arm: int
  produces_outputs:
    placed: bool
    place_tcp: Se3Pose
    place_arm: int
  hard_rules:
    - >
      Every place target is a TAPE-centre pose; compose held_offset into the
      plan — do NOT plan the bare TCP to the destination.
    - >
      This skill does NOT compute the reachable pose or do the hover approach —
      those are tsh-place-pose and tsh-transport-held. By the time this node runs
      the arm must already be at the hover above place_xyz/place_quat.
  canonical_scripts:
    - place: scripts/place.py
  streaming: false
---

# tsh-place

The holder does not hold the tape centred (a ring grasp or rim thread leaves the
tape centre several cm off the TCP), so place treats the tape as the holder's end
effector via `held_offset` — FK-measured, no GT, the pickup's grip on the direct
route or the receiver's after a handover. `place_xyz`/`place_quat` are the
reachable rest pose `tsh-place-pose` already probed; this node composes
`held_offset` against them, then descends, releases, and retracts straight up.

## When to use

- The terminal place node, always preceded by `tsh-place-pose` then
  `tsh-transport-held`.

## When NOT to use

- Objects not laid flat on a surface, or learned/policy placement.
- Standalone — it never computes its own target pose or does its own approach.

## Recommended subgraph state flow

```text
place
```

1. **`place`** — `type: script`, `scripts/<sg>/place.py`. Inputs:
   `held_offset=Ref("in.held_offset")`, `place_xyz=Ref("in.place_xyz")`,
   `place_quat=Ref("in.place_quat")`, `arm_id=Ref("in.place_arm")` (or a
   literal). Returns `{placed, place_tcp, place_arm}`.

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Tape laid and released; route to `done`. |
| `failed` | A leg had no plan; route to `abort`. |

## See also

- `tsh-place-pose` — computes `place_xyz`/`place_quat`.
- `tsh-transport-held` — carries the arm to the hover before this node.
- `tsh-dispatch-route` / `tsh-handover` — supply `held_offset` (the giver's grip relayed, or the receiver's after an exchange).
- `tsh-route` — decides `place_arm`.
