---
name: tsh-place
description: >
  Lay the held tape flat on the perceived duct top face and release. The
  receiver holds the tape off-centre (it threaded the rim), so the tape is
  treated as the receiver's end effector via the measured receiver_offset: every
  place target is a TAPE-centre pose, composed into the plan. Yaw-sweep the
  gripper about the tape's vertical axis for a reachable pose, approach the hover
  with the tape attached as a cuRobo collision body, set down flush (duct top +
  perceived tape half-thickness), release, and retract straight up. Use to place
  a handed-over tape ring on its duct target on LIBERO-YAM.
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
    placed: Tape laid on the duct and released; place_tcp bound in the outputs.
    failed: No reachable place yaw, or a place leg had no plan (raise routes to abort).
  required_inputs:
    receiver_offset: Vec3          # from tsh-handover (the receiver's tool offset)
    duct_xyz: Vec3                 # from tsh-perceive (duct top-face centre)
    tape_half_z: float             # from tsh-perceive (flush rest height); REQUIRED, no tuned fallback
    tape_cloud: PointCloud         # from tsh-perceive (attached collision body); REQUIRED, no tuned fallback
  produces_outputs:
    placed: bool
    place_tcp: Se3Pose             # world TCP pose matching the RESTING tape
  hard_rules:
    - >
      Every place target is a TAPE-centre pose; compose `receiver_offset` into
      the plan (`tool_offset`) — do NOT plan the bare TCP to the duct.
    - >
      Rest height is DERIVED: duct top face + perceived `tape_half_z` (or derived
      from `tape_cloud`). Do not hard-code a place Z.
  canonical_scripts:
    - place: scripts/place.py
  streaming: false
---

# tsh-place

The receiver does not hold the tape centred (it threaded a rim clock position),
so the tape centre sits several cm off the TCP. Place treats the tape as the
receiver's end effector via `receiver_offset` (measured by the exchange, no GT):
the target is the tape centre = perceived duct top-face centre + the tape
half-thickness (so it rests flush). The tape is round, so a yaw sweep pivots the
TCP about the tape centre until the hover is reachable — the ring's own symmetry
is the reach margin. The approach is planned with the tape attached as a cuRobo
collision body so the face-on → gripper-down reorient can't clip the arm.

The place *location* is read UP FRONT by `tsh-perceive` (perceive_duct), while
the duct view is clean — at place time the held tape + gripper occlude it.

## When to use

- After `tsh-handover`, to set the handed-over tape on its duct target.

## When NOT to use

- Objects that aren't laid flat on a surface, or learned/policy placement.

## Recommended subgraph state flow

1 state:

```text
place
```

1. **`place`** — `type: script`, `scripts/<sg>/place.py`. Inputs:
   `receiver_offset=Ref("in.receiver_offset")`, `duct_xyz=Ref("in.duct_xyz")`,
   `tape_half_z=Ref("in.tape_half_z")`, `tape_cloud=Ref("in.tape_cloud")`, plus
   `arm_id` (literal receiver arm). Returns `{placed, place_tcp}`.

Bind the outputs (`place_tcp` feeds a return leg that re-acquires the tape):

```python
sg.set_outputs(place_tcp=Ref("place.place_tcp"))
```

## Required end states

| End state | Meaning |
|---|---|
| `placed` | Tape laid and released; route to `done` (or a return leg). |
| `failed` | No reachable yaw / plan failed; route to `abort`. |

## See also

- `tsh-perceive` — supplies `duct_xyz`, `tape_half_z`, `tape_cloud`.
- `tsh-handover` — supplies `receiver_offset`.
