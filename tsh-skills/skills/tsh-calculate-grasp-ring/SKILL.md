---
name: tsh-calculate-grasp-ring
description: >
  Ring radii + grasp poses for one arm. Derives the gripper's own grasp offsets
  (fingertip axial offset, finger half-gap) from the robot model via forward
  kinematics as an up-front internal step — self-configuring from the URDF, so
  a different gripper needs no re-tuning — then measures the ring's hole/rim
  radii from its perceived cloud (top-face slab) and derives the ring grasp
  from them: the TCP centres on the wall midpoint, the fingertip trails along
  the approach axis, and the grasp revolves about the vertical hole axis
  (mirrored for a −Y-side arm, read from the arm base). Emits the three grasp
  legs (hover, seat, lift) as ready-to-plan world poses plus the predicted
  held-tape offset, so the grasp geometry is computed once and consumed as
  data (tsh-pickup executes it, tsh-route-arms-bimanual probes it).
  Parameterized by arm_id. Pure geometry — it never plans or moves, and raises
  on a degenerate cloud.
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
    "<arm0|arm1>_hover_xyz": Vec3
    "<arm0|arm1>_seat_xyz": Vec3
    "<arm0|arm1>_lift_xyz": Vec3
    "<arm0|arm1>_grasp_quat": Quaternion
    "<arm0|arm1>_held_offset": Vec3
    hole_radius: float
    rim_radius: float
  hard_rules:
    - >
      arm_id is a LITERAL (0 or 1) baked into the node's inputs, never wired via
      Ref("in.arm_id") — no upstream skill produces a bare arm_id output, so a
      subgraph input by that name never auto-wires.
    - >
      The grasp-leg outputs carry an arm0_/arm1_ PREFIX matching this instance's
      literal arm_id — arm_id=0 emits arm0_hover_xyz, arm_id=1 emits
      arm1_hover_xyz, and so on for seat/lift/grasp_quat/held_offset. The script
      returns unprefixed keys; the prefix is applied in set_outputs. Both
      instances' legs are consumed SIMULTANEOUSLY by tsh-route-arms-bimanual
      (it compares the arms to pick one), so unprefixed outputs would collide
      under the latest-producer rule and route would probe one arm's poses
      twice. hole_radius/rim_radius stay UNPREFIXED — they describe the ring,
      not the arm, and are identical from either instance.
    - >
      derive_gripper_geometry runs unconditionally, once per instance (i.e.
      once per arm_id) — unlike tsh-route-arms-bimanual's station-geometry
      node, the gripper self-model is needed on EVERY grasp, not just some
      routes, so it is wired as a plain first node rather than behind a
      conditional exit.
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

# tsh-calculate-grasp-ring

The ring grasp — radii *and* poses — in one node, fed by its own gripper
self-model. Perception stays object-agnostic (`tsh-perceive-*` emits cloud /
top face / half-thickness for any object); this skill owns everything
ring-specific: it derives the gripper's fingertip offset and finger half-gap
from the robot model (no ground truth, no hand-tuned constants), measures the
hole/rim radii off the cloud's top-face slab, then derives the grasp from
them. Computing the pose once here means `tsh-route-arms-bimanual`'s probe and
`tsh-pickup`'s grasp consume the same geometry as data and can't disagree.

For the given arm it centres the TCP on the ring's wall midpoint
`(hole_r + rim_r)/2`, trails the fingertip along the approach axis, and revolves
about the vertical hole axis (plus 90° for a −Y-side arm, read from the base),
returning the three grasp legs as world poses and the predicted tape-in-TCP
offset.

## When to use

- Between perception and the grasp, to produce the ring-grasp geometry for a
  given arm.

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
   `{fingertip_axial, finger_half_gap}`. Runs once per subgraph instance (i.e.
   once per arm_id), unconditionally — every ring grasp needs its own gripper
   offsets, so unlike station-geometry in `tsh-route-arms-bimanual` this node
   is never gated behind a conditional exit.
2. **`calculate_grasp_ring`** — `type: script`, file
   `scripts/<sg>/calculate_grasp_ring.py`. Inputs:
   `target_xyz=Ref("in.target_xyz")`, `target_cloud=Ref("in.target_cloud")`,
   `target_half_z=Ref("in.target_half_z")`, `arm_id=0` (a literal — one instance
   per arm, e.g. `arm_id=0` and `arm_id=1`),
   `fingertip_axial=Ref("derive_gripper_geometry.fingertip_axial")`,
   `finger_half_gap=Ref("derive_gripper_geometry.finger_half_gap")`. The script
   returns unprefixed `{hover_xyz, seat_xyz, lift_xyz, grasp_quat, held_offset,
   hole_radius, rim_radius}`; `set_outputs` applies this instance's arm prefix:

```python
sg.set_outputs(**{                       # instance with arm_id=0
    "arm0_hover_xyz":   Ref("calculate_grasp_ring.hover_xyz"),
    "arm0_seat_xyz":    Ref("calculate_grasp_ring.seat_xyz"),
    "arm0_lift_xyz":    Ref("calculate_grasp_ring.lift_xyz"),
    "arm0_grasp_quat":  Ref("calculate_grasp_ring.grasp_quat"),
    "arm0_held_offset": Ref("calculate_grasp_ring.held_offset"),
    "hole_radius":      Ref("calculate_grasp_ring.hole_radius"),
    "rim_radius":       Ref("calculate_grasp_ring.rim_radius"),
})
```

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

- `tsh-perceive-*` — supplies `target_cloud` / `target_xyz` / `target_half_z`.
- `tsh-route-arms-bimanual` (probes these poses) · `tsh-pickup` (executes them).
- `scripts/_gripper_geometry.py` — the gripper FK derivation (`grip_geometry`).
