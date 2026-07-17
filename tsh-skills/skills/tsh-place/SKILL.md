---
name: tsh-place
description: >
  Lay the held tape flat on the perceived destination top face and release. The
  holder grips the tape off-centre (ring grasp or rim thread), so the tape is
  treated as the holder's end effector via the measured held_offset: every place
  target is a TAPE-centre pose, composed into the plan. Yaw-sweep the gripper
  about the tape's vertical axis for a reachable pose, approach the hover with
  the tape attached as a cuRobo collision body, set down flush (destination top +
  perceived tape half-thickness), release, and retract straight up. Serves both
  the direct and handed-over routes on LIBERO-YAM.
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
    - curobo.plan_with_grasped_object
    - robot.open_gripper
    - robot.get_ee_pose
  exit_conditions:
    placed: Tape laid on the destination and released; place_tcp bound in the outputs.
    failed: No reachable place yaw, or a place leg had no plan (raise routes to abort).
  required_inputs:
    held_offset: Vec3              # the LATEST holder's tool offset — pickup's tape_in_giver on the
                                    # direct route, or the exchange's receiver_offset after a handover
                                    # (latest-producer cross-subgraph binding; both FK-measured, no GT)
    dest_xyz: Vec3                 # from tsh-perceive (destination top-face centre; e.g. duct)
    tape_half_z: float             # from tsh-perceive (flush rest height); REQUIRED, no tuned fallback
    tape_cloud: PointCloud         # from tsh-perceive (attached collision body); REQUIRED, no tuned fallback
    place_arm: int                 # route-decided placing arm (0=left, 1=right). Default 1.
  produces_outputs:
    placed: bool
    place_tcp: Se3Pose             # world TCP pose matching the RESTING tape
    place_arm: int                 # echo of the placing arm (checkpoint anchor)
  hard_rules:
    - >
      Every place target is a TAPE-centre pose; compose `held_offset` into
      the plan (`tool_offset`) — do NOT plan the bare TCP to the destination.
    - >
      Rest height is DERIVED: destination top face + perceived `tape_half_z` (or
      derived from `tape_cloud`). Do not hard-code a place Z.
  canonical_scripts:
    - place: scripts/place.py
  streaming: false
---

# tsh-place

The holder does not hold the tape centred (a ring grasp or a rim thread leaves
the tape centre several cm off the TCP), so place treats the tape as the holder's
end effector via `held_offset` — FK-measured, no GT — which is the pickup's grip
on the direct route or the receiver's grip after a handover (the cross-subgraph
name rebinds to whichever ran last). The target is the tape centre = perceived
destination top-face centre + the tape half-thickness (so it rests flush). The
tape is round, so a yaw sweep pivots the TCP about the tape centre until the
hover is reachable — the ring's own symmetry is the reach margin. The approach is
planned with the tape attached as a cuRobo collision body so the face-on →
gripper-down reorient can't clip the arm.

The place *location* is read UP FRONT by `tsh-perceive` (`perceive_dest`), while
the destination view is clean — at place time the held tape + gripper occlude it.

## When to use

- The terminal stage on BOTH routes: directly after `verify_grasp` on the
  `direct` route (`dispatch` → `place`), or after `tsh-handover` on the
  `needs_handover` route. `held_offset` + `place_arm` come from whichever
  holder ran last, so the same subgraph serves both.

## When NOT to use

- Objects that aren't laid flat on a surface, or learned/policy placement.

## Recommended subgraph state flow

1 state:

```text
place
```

1. **`place`** — `type: script`, `scripts/<sg>/place.py`. Inputs:
   `held_offset=Ref("in.held_offset")`, `dest_xyz=Ref("in.dest_xyz")`,
   `tape_half_z=Ref("in.tape_half_z")`, `tape_cloud=Ref("in.tape_cloud")`,
   `arm_id=Ref("in.place_arm")` (route-decided placing arm).
   Returns `{placed, place_tcp, place_arm}`.

Bind the outputs (`place_tcp` feeds a return leg that re-acquires the tape):

```python
sg.set_outputs(
    place_tcp=Ref("place.place_tcp"),
    place_arm=Ref("place.place_arm"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Tape laid and released; route to `done` (or a return leg). |
| `failed` | No reachable yaw / plan failed; route to `abort`. |

## See also

- `tsh-perceive` — supplies `dest_xyz`, `tape_half_z`, `tape_cloud`.
- `tsh-pickup` — supplies `held_offset` on the direct route.
- `tsh-handover` — supplies `held_offset` (the receiver's grip) after a handover.
- `tsh-route` — decides `place_arm` and whether a handover precedes this.
