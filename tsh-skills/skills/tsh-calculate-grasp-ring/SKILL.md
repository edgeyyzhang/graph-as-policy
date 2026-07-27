---
name: tsh-calculate-grasp-ring
description: >
  Ring radii + grasp poses for one arm. Measures the ring's hole/rim radii from
  its perceived cloud (top-face slab) and derives the ring grasp from them: the
  TCP centres on the wall midpoint, the fingertip trails along the approach
  axis, and the grasp revolves about the vertical hole axis (mirrored for a
  −Y-side arm, read from the arm base). Emits the three grasp legs (hover, seat,
  lift) as ready-to-plan world poses plus the predicted held-tape offset, so the
  grasp geometry is computed once and consumed as data (tsh-pickup executes it,
  tsh-route probes it). Parameterized by arm_id. Pure geometry — it never plans
  or moves, and raises on a degenerate cloud.
compatibility: requires gap>=0.1
metadata:
  category: geometry
  tags: [tsh, geometry, grasp, ring, curobo, yam]
gap:
  allowed_tools:
    - libero-yam.arm_base_pose
  exit_conditions:
    derived: Radii measured and grasp poses computed, bound in the subgraph outputs.
    degenerate: Radii outside the sanity band (raise routes to on_error).
  required_inputs:
    target_xyz: Vec3
    target_cloud: PointCloud
    target_half_z: float
    fingertip_axial: float
    finger_half_gap: float
  produces_outputs:
    hover_xyz: Vec3
    seat_xyz: Vec3
    lift_xyz: Vec3
    grasp_quat: Quaternion
    held_offset: Vec3
    hole_radius: float
    rim_radius: float
  hard_rules:
    - >
      arm_id is a LITERAL (0 or 1) baked into the node's inputs, never wired via
      Ref("in.arm_id") — no upstream skill produces a bare arm_id output, so a
      subgraph input by that name never auto-wires.
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
      libero-yam.arm_base_pose (the ±90° side flip reads the arm base).
  canonical_scripts:
    - calculate_grasp_ring: scripts/calculate_grasp_ring.py
  streaming: false
---

# tsh-calculate-grasp-ring

The ring grasp — radii *and* poses — in one node. Perception stays
object-agnostic (`tsh-perceive-*` emits cloud / top face / half-thickness for
any object); this skill owns everything ring-specific: it measures the hole/rim
radii off the cloud's top-face slab, then derives the grasp from them. Computing
the pose once here means `tsh-route`'s probe and `tsh-pickup`'s grasp consume the
same geometry as data and can't disagree.

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

## Recommended subgraph state flow

```text
calculate_grasp_ring
```

1. **`calculate_grasp_ring`** — `type: script`, file
   `scripts/<sg>/calculate_grasp_ring.py`. Inputs:
   `target_xyz=Ref("in.target_xyz")`, `target_cloud=Ref("in.target_cloud")`,
   `target_half_z=Ref("in.target_half_z")`, `arm_id=0` (a literal — one instance
   per arm, e.g. `arm_id=0` and `arm_id=1`),
   `fingertip_axial=Ref("in.fingertip_axial")`,
   `finger_half_gap=Ref("in.finger_half_gap")`. Returns `{hover_xyz, seat_xyz,
   lift_xyz, grasp_quat, held_offset, hole_radius, rim_radius}`.

Wire the exit linearly: `calculate_grasp_ring → derived → END`, with
`on_error: "degenerate"`. The script raises on a degenerate cloud, so no
conditional edges are needed.

## Required end states

| End state | Meaning |
|---|---|
| `derived` | Radii + grasp poses bound in the outputs. |
| `degenerate` | Radii outside the sanity band (raise → abort / re-perceive). |

## See also

- `tsh-perceive-*` — supplies `target_cloud` / `target_xyz` / `target_half_z`.
- `tsh-gripper-geometry` — supplies `fingertip_axial` / `finger_half_gap`.
- `tsh-route` (probes these poses) · `tsh-pickup` (executes them).
