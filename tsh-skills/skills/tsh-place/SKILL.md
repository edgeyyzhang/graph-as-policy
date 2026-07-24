---
name: tsh-place
description: >
  Release the held tape at a pre-computed pose and retract. The holder grips
  the tape off-centre (ring grasp or rim thread), so the tape is treated as
  the holder's end effector via the measured held_offset, composed once here
  to get the actual release TCP. WHERE/WHICH-orientation come from
  tsh-place-pose (the yaw-reachability probe); a preceding tsh-transport-held
  node has already carried the arm to the collision-aware hover above
  place_xyz. This skill only does the straight-down set-down and straight-up
  retract (curobo_linear_move) — never a free transport. Serves both the
  direct and handed-over routes on LIBERO-YAM. place_arm comes from
  tsh-route; when the task pins the arm, bind a literal int inside the
  subgraph and omit that input.
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
    held_offset: Vec3               # the LATEST holder's tool offset — pickup's tape_in_giver on the
                                     # direct route, or the exchange's receiver_offset after a handover
                                     # (latest-producer cross-subgraph binding; both FK-measured, no GT)
    place_xyz: Vec3                 # the tape's final rest centre — from tsh-place-pose
    place_quat: Quaternion          # reachable presentation orientation, wxyz — from tsh-place-pose
    place_arm: int                  # placing arm from tsh-route (0=left, 1=right); rebind to the
                                     # script's arm_id kwarg. Task pins the arm? Pass arm_id a
                                     # LITERAL in the node inputs and OMIT this subgraph input.
  produces_outputs:
    placed: bool
    place_tcp: Se3Pose              # world TCP pose matching the RESTING tape
    place_arm: int                  # echo of the placing arm (checkpoint anchor)
  hard_rules:
    - >
      Every place target is a TAPE-centre pose; compose `held_offset` into
      the plan — do NOT plan the bare TCP to the destination.
    - >
      This skill does NOT compute the reachable pose or do the hover
      approach — those are `tsh-place-pose` and `tsh-transport-held`. By the
      time this node runs, the arm must already be at the hover above
      `place_xyz`/`place_quat`.
  canonical_scripts:
    - place: scripts/place.py
  streaming: false
---

# tsh-place

The holder does not hold the tape centred (a ring grasp or a rim thread leaves
the tape centre several cm off the TCP), so place treats the tape as the holder's
end effector via `held_offset` — FK-measured, no GT — which is the pickup's grip
on the direct route or the receiver's grip after a handover (the cross-subgraph
name rebinds to whichever ran last). `place_xyz`/`place_quat` are the reachable
rest pose `tsh-place-pose` already probed; this node composes `held_offset`
against them once to get the release TCP, then descends, releases, and
retracts straight up.

The reachable pose and the collision-aware hover approach are factored out
into `tsh-place-pose` and `tsh-transport-held` so this skill is just the
final legs — a straight-line descent and retract, never a free transport.

## When to use

- The terminal stage on BOTH routes, always preceded by `tsh-place-pose` then
  `tsh-transport-held` in the same subgraph chain: directly after
  `verify_grasp` on the `direct` route (`dispatch` → `place_pose` →
  `transport_held` → `place`), or after `tsh-handover` on the
  `needs_handover` route. `held_offset` + `place_arm` come from whichever
  holder ran last, so the same chain serves both.

## When NOT to use

- Objects that aren't laid flat on a surface, or learned/policy placement.
- Standalone — this skill never computes its own target pose or does its own
  approach; it always needs `tsh-place-pose` + `tsh-transport-held` ahead of it.

## Recommended subgraph state flow

This is THREE separate top-level subgraphs (one per skill), not three nodes
inside one subgraph — each already has its own fixed input/output contract
(see each skill's SKILL.md), connected by ordinary conditional edges and
resolved by the usual cross-subgraph exact-name auto-wire:

```text
place_pose_sg → transport_held_sg → place_sg
```

1. **`place_pose_sg`** (`tsh-place-pose`) — subgraph inputs `held_offset`,
   `container_xyz`, `target_half_z`, `place_arm` (all auto-wired from upstream by
   exact name). Produces `place_xyz`, `hover_xyz`, `place_quat`, plus
   `target_xyz`/`target_quat` (aliases of `hover_xyz`/`place_quat`, added
   specifically so the next subgraph can auto-wire).
2. **`transport_held_sg`** (`tsh-transport-held`) — subgraph inputs
   `held_offset`, `target_xyz`, `target_quat` auto-wire from
   `place_pose_sg`'s alias outputs (it is the LATEST producer of those two
   names at this point in the DAG, so it correctly wins over any earlier
   `target_xyz` producer like a perceive instance). `held_cloud`
   intentionally NOT wired for now (see `tsh-transport-held` hard_rules).
3. **`place_sg`** (`tsh-place`) — subgraph inputs `held_offset`, `place_xyz`,
   `place_quat`, `place_arm` auto-wire from `place_pose_sg`'s (unaliased,
   unambiguous) outputs. Produces `placed`, `place_tcp`, `place_arm`.

No explicit `set_outputs` glue is needed between these three — each skill's
own subgraph body already declares the outputs above; the top level just
needs conditional edges chaining `place_pose_sg → transport_held_sg →
place_sg` in that order.

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Tape laid and released; route to `done` (or a return leg). |
| `failed` | No reachable yaw (`place_pose`) / plan failed (`transport_held` or `place`); route to `abort`. |

## See also

- `tsh-place-pose` — computes `place_xyz`/`hover_xyz`/`place_quat` (the yaw-reachability probe).
- `tsh-transport-held` — carries the arm to the collision-aware hover before this node runs.
- `tsh-perceive-cv` — supplies `container_xyz`, `target_half_z` (consumed by `tsh-place-pose` now, not here).
- `tsh-pickup` — supplies `held_offset` on the direct route.
- `tsh-handover` — supplies `held_offset` (the receiver's grip) after a handover.
- `tsh-route` — decides `place_arm` and whether a handover precedes this.
