---
name: tsh-gripper-geometry
description: >
  Derive the gripper's grasp offsets from the robot model via forward
  kinematics — the fingertip axial offset, the finger half-gap at the open
  aperture, and the pad z-span about the TCP. Loads the arm MJCF (found through
  the curobo-config wiring), sets the gripper aperture, and reads the fingertip
  contact geoms in the grasp_site (TCP) frame. Emits ``gripper_offsets`` for the
  ring-grasp (tsh-pickup) to place its fingers on the tape wall — self-configuring
  from the URDF, so a different gripper needs no re-tuning. Use as the up-front
  self-model step before a scripted ring grasp on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: calibration
  tags: [tsh, self-model, gripper, fk, yam]
gap:
  allowed_tools: []          # loads the arm model directly (mujoco FK), no tools
  exit_conditions:
    derived: Gripper offsets computed and bound in the subgraph outputs.
    failed: Arm model could not be located/loaded (raise routes to abort).
  produces_outputs:
    fingertip_axial: float    # fingertip offset ahead of the TCP along the approach (m)
    finger_half_gap: float    # half the open-aperture finger-pair spread (m)
  canonical_scripts:
    - gripper_geometry: scripts/gripper_geometry.py
  streaming: false
---

# tsh-gripper-geometry

A **self-model** skill: it computes grasp parameters from the robot's own
kinematics instead of hand-tuned constants. The tape ring-grasp needs to know
where its fingertips sit relative to the TCP (to land one finger in the hole and
one on the rim) and how far the finger pair spreads at the open aperture. Those
are pure gripper FK — this skill reads them off the model.

It reproduces the previously hand-measured constants exactly (finger half-gap
24.1 mm, pad z-span [−14.2, +3.8] mm about the TCP) and generalises: swap the
gripper URDF and the offsets self-configure.

## When to use

- As the first (or an up-front, run-once) step before `tsh-pickup`, feeding
  `gripper_offsets` into the ring grasp.
- Any scripted grasp on LIBERO-YAM that wants model-derived finger placement
  rather than tuned constants.

## When NOT to use

- Learned/policy grasps that don't consume explicit finger offsets.
- Platforms where the arm MJCF isn't reachable through
  `GAP_CUROBO_ROBOT_CONFIGS` — the consumer falls back to its tuned constants,
  so this skill can simply be omitted.

## Recommended subgraph state flow

1 state:

```text
derive_gripper_geometry
```

1. **`derive_gripper_geometry`** — `type: script`,
   file `scripts/<sg>/gripper_geometry.py` from this bundle, `inputs: {}`.
   Returns `{fingertip_axial, finger_half_gap}`.

Bind the subgraph outputs:

```python
sg.set_outputs(
    fingertip_axial=Ref("derive_gripper_geometry.fingertip_axial"),
    finger_half_gap=Ref("derive_gripper_geometry.finger_half_gap"),
)
```

Downstream, `tsh-pickup` wires `fingertip_axial` and `finger_half_gap` via
`Ref("in.…")`.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | `fingertip_axial` + `finger_half_gap` bound; route to the grasp subgraph. |
| `failed` | Model unavailable; route to `abort` (or omit this subgraph and let the grasp use tuned constants). |

## See also

- `scripts/_gripper_geometry.py` — the FK derivation (`grip_geometry`), with a
  `__main__` that prints derived-vs-tuned validation.
- `tsh-pickup` — the consumer of `gripper_offsets`.
