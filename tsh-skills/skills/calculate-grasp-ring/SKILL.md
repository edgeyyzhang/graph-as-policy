---
name: calculate-grasp-ring
description: >
  Ring radii + grasp poses for BOTH arms in one node. Derives the gripper's own grasp offsets
  (fingertip axial offset, finger half-gap) from the robot model via forward
  kinematics as an up-front internal step — self-configuring from the URDF, so
  a different gripper needs no re-tuning — then measures the ring's hole/rim
  radii from its perceived cloud (top-face slab) and derives the ring grasp
  from them: the TCP centres on the wall midpoint, the fingertip trails along
  the approach axis, and the grasp revolves about the vertical hole axis
  (mirrored for a −Y-side arm, read from the arm base). Emits the three grasp
  legs (hover, seat, lift) per arm as ready-to-plan world poses plus the
  predicted held-tape offset, under plain arm0_/arm1_ names — so the grasp
  geometry is computed once and consumed as data (pickup executes the
  chosen arm's legs, bimanual-route-arms probes both to choose).
  Pure geometry — it never plans or moves, and raises on a degenerate cloud.
compatibility: requires gap>=0.1
metadata:
  category: geometry
  tags: [tsh, geometry, grasp, ring, curobo, yam, self-model]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
  exit_conditions:
    derived: Gripper offsets derived, radii measured, and grasp poses computed, bound in the subgraph outputs.
    failed: Gripper self-model unavailable (arm model not loadable) OR radii outside the sanity band — both raise, both route to the single on_error.
  required_inputs:
    target_xyz: Vec3
    target_cloud: PointCloud
    target_half_z: float
  produces_outputs:
    arm0_hover_xyz: Vec3
    arm0_seat_xyz: Vec3
    arm0_lift_xyz: Vec3
    arm0_grasp_quat: Quaternion
    arm0_held_offset: Vec3
    arm1_hover_xyz: Vec3
    arm1_seat_xyz: Vec3
    arm1_lift_xyz: Vec3
    arm1_grasp_quat: Quaternion
    arm1_held_offset: Vec3
    hole_radius: float
    rim_radius: float
  hard_rules:
    - >
      ONE instance serves both arms — the script loops arm 0 and arm 1 and
      returns arm0_*/arm1_* keys directly. Do not instantiate this skill per
      arm: bimanual-route-arms consumes both arms' legs simultaneously to
      choose between them, so there is no point at which only one arm's
      geometry is wanted. hole_radius/rim_radius stay unprefixed — they
      describe the ring, not the arm.
    - >
      Radii are measured on the cloud's TOP-FACE SLAB (top_z = center_z +
      half_z), never the full cloud — from an angled view the camera sees the
      table through the hole and the low-z points collapse the hole radius.
      Outside the sanity band this RAISES (no tuned fallback).
    - >
      The three legs share one grasp_quat and differ only in z; each *_xyz +
      grasp_quat pair drives plan_tool_move(ctx, arm, xyz, quat) directly.
    - >
      Pure geometry — it never plans or moves. Its only tool is
      libero-yam.arm_base_pose (the ±90° side flip reads the arm base; the
      gripper self-model reads the arm URDF directly, no tool call).
  canonical_scripts:
    - derive_gripper_geometry: scripts/gripper_geometry.py
    - calculate_grasp_ring: scripts/calculate_grasp_ring.py
  streaming: false
---

# calculate-grasp-ring

The ring grasp — radii *and* poses — in one node, fed by its own gripper
self-model. Perception stays object-agnostic (`perceive-tape-*` emits cloud /
top face / half-thickness for any object); this skill owns everything
ring-specific: it derives the gripper's fingertip offset and finger half-gap
from the robot model (no ground truth, no hand-tuned constants), measures the
hole/rim radii off the cloud's top-face slab, then derives the grasp from
them. Computing the pose once here means `bimanual-route-arms`'s probe and
`pickup`'s grasp consume the same geometry as data and can't disagree.

For each arm it centres the TCP on the ring's wall midpoint
`(hole_r + rim_r)/2`, trails the fingertip along the approach axis, and revolves
about the vertical hole axis (plus 90° for a −Y-side arm, read from the base),
returning the three grasp legs as world poses and the predicted tape-in-TCP
offset.

## When to use

- Between perception and the grasp, to produce the ring-grasp geometry both
  arms' probes and the eventual grasp consume.

## When NOT to use

- Non-ring targets (no wall-midpoint geometry).
- When the grasp is delegated to a VLA policy (`tsh-pi05`).
- Platforms where the arm MJCF isn't reachable through
  `GAP_CUROBO_ROBOT_CONFIGS` (the gripper self-model needs FK on the real URDF).

## Recommended subgraph state flow

```text
derive_gripper_geometry → calculate_grasp_ring
```

1. **`derive_gripper_geometry`** — `type: script`,
   file `scripts/<sg>/gripper_geometry.py`, `inputs: {}`. Returns
   `{fingertip_axial, finger_half_gap}`.
2. **`calculate_grasp_ring`** — `type: script`, file
   `scripts/<sg>/calculate_grasp_ring.py`. Inputs:
   `target_xyz=Ref("in.target_xyz")`, `target_cloud=Ref("in.target_cloud")`,
   `target_half_z=Ref("in.target_half_z")`,
   `fingertip_axial=Ref("derive_gripper_geometry.fingertip_axial")`,
   `finger_half_gap=Ref("derive_gripper_geometry.finger_half_gap")`. No arm
   input — it does both. Bind its twelve outputs straight through by name.

Wire it linearly — `START → derive_gripper_geometry → calculate_grasp_ring →
derived → END` — with `set_on_error("failed")`. Both scripts signal failure by
RAISING, and a subgraph has exactly ONE `on_error` symbol, so both modes land
on `failed`. There is no second failure exit to add, no exit router, and no
guarded wrapper script: do not author one.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Gripper offsets derived, radii + grasp poses bound in the outputs. |
| `failed` | Gripper self-model unavailable, or radii outside the sanity band (either raise → abort). |

## See also

- `perceive-tape-*` — supplies `target_cloud` / `target_xyz` / `target_half_z`.
- `bimanual-route-arms` (probes these poses) · `pickup` (executes them).
- `scripts/_gripper_geometry.py` — the gripper FK derivation (`grip_geometry`).
