---
name: tsh-gripper-geometry
description: >
  Derive the gripper's grasp offsets from the robot model via forward
  kinematics — the fingertip axial offset and the finger half-gap at the open
  aperture. Loads the arm MJCF (found through the curobo-config wiring), sets
  the gripper aperture, and reads the fingertip contact geoms in the grasp_site
  (TCP) frame. Emits ``fingertip_axial`` / ``finger_half_gap`` for the
  ring-grasp geometry — self-configuring from the URDF, so a different gripper
  needs no re-tuning. Use as the up-front self-model step before a scripted ring
  grasp on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: calibration
  tags: [tsh, self-model, gripper, fk, yam]
gap:
  allowed_tools: []
  exit_conditions:
    derived: Gripper offsets computed and bound in the subgraph outputs.
    failed: Arm model could not be located/loaded (raise routes to abort).
  produces_outputs:
    fingertip_axial: float
    finger_half_gap: float
  canonical_scripts:
    - gripper_geometry: scripts/gripper_geometry.py
  streaming: false
---

# tsh-gripper-geometry

A **self-model** skill: it computes grasp parameters from the robot's own
kinematics instead of hand-tuned constants. The tape ring-grasp needs to know
where its fingertips sit relative to the TCP (to land one finger in the hole and
one on the rim) and how far the finger pair spreads at the open aperture. Those
are pure gripper FK — this skill reads them off the model, and they self-configure
if the gripper URDF changes.

## When to use

- As an up-front (run-once) step before the ring grasp, feeding the finger
  offsets into `tsh-calculate-grasp-ring`.

## When NOT to use

- Learned/policy grasps that don't consume explicit finger offsets.
- Platforms where the arm MJCF isn't reachable through
  `GAP_CUROBO_ROBOT_CONFIGS`.

## Recommended subgraph state flow

```text
derive_gripper_geometry
```

1. **`derive_gripper_geometry`** — `type: script`,
   file `scripts/<sg>/gripper_geometry.py`, `inputs: {}`. Returns
   `{fingertip_axial, finger_half_gap}`.

```python
sg.set_outputs(
    fingertip_axial=Ref("derive_gripper_geometry.fingertip_axial"),
    finger_half_gap=Ref("derive_gripper_geometry.finger_half_gap"),
)
```

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Finger offsets bound; route to the grasp-geometry subgraph. |
| `failed` | Model unavailable; route to `abort`. |

## See also

- `scripts/_gripper_geometry.py` — the FK derivation (`grip_geometry`).
- `tsh-calculate-grasp-ring` — the consumer of the finger offsets.
