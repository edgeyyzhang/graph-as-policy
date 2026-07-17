---
name: tsh-ring-geometry
description: >
  Derive the tape ring's hole and rim radii from its perceived point cloud —
  a pure geometry post-processor on the generic perceive-object output, with
  a sanity band that rejects degenerate clouds. Measured on the top-face slab
  so table points seen through the hole cannot collapse the estimate. Emits
  ``hole_radius`` / ``rim_radius`` for the ring grasp (tsh-pickup) and the
  reachability route (tsh-route). Use after perceiving a ring-shaped target
  (tape spool) on LIBERO-YAM; skip it for non-ring objects.
compatibility: requires gap>=0.1
metadata:
  category: geometry
  tags: [tsh, geometry, ring, yam]
gap:
  allowed_tools: []          # pure numpy post-processing, no tools
  exit_conditions:
    derived: Radii measured and bound in the subgraph outputs.
    degenerate: Radii outside the sanity band (raise routes to on_error).
  required_inputs:
    target_cloud: PointCloud       # from the perceive-object subgraph
    target_xyz: Vec3               # object centre (body-centroid height), same source
    target_half_z: float           # perceived half-thickness, same source
  produces_outputs:
    hole_radius: float             # inner hole radius (m)
    rim_radius: float              # outer rim radius (m)
  hard_rules:
    - >
      Radii are measured on the TOP-FACE SLAB (``top_z = center_z + half_z``)
      — never on the full cloud; from an angled view the camera sees the
      table through the hole and the low-z points collapse the hole radius.
    - >
      Outside the sanity band this skill RAISES (no tuned fallback) — route
      the failure to a re-perceive or abort; do not invent radii.
  canonical_scripts:
    - ring_geometry: scripts/ring_geometry.py
  streaming: false
---

# tsh-ring-geometry

The ring-specific half of what the old fused tape perception did: perception
itself is object-agnostic (`tsh-perceive`'s `perceive_object` emits cloud /
top face / half thickness for ANY object), and this skill derives the two
numbers only a *ring* consumer needs — the hole and rim radii the ring grasp
places its fingers by.

The estimator is the validated top-face-slab radial-percentile math
(2nd percentile = hole edge, 98th = rim edge; both robust to depth noise).
`scripts/_ring.py` is the single shared source for ring geometry: the radii
estimator lives next to `ring_grasp_poses`, the grasp-pose derivation that
`tsh-pickup` executes and `tsh-route` probes — the two can't drift apart.

## When to use

- After a `perceive-object` subgraph localized a ring-shaped target (tape
  spool) and a ring grasp / reachability route will consume the radii.

## When NOT to use

- Non-ring objects (the duct, boxes, a stack) — nothing downstream needs
  radii; wire perception outputs straight to the consumer.
- When the grasp is delegated to a VLA policy (`tsh-pi05`) — the policy
  needs no explicit ring geometry.

## Recommended subgraph state flow

1 state:

```text
ring_geometry
```

1. **`ring_geometry`** — `type: script`, file `scripts/<sg>/ring_geometry.py`.
   Inputs: `cloud=Ref("in.target_cloud")`, `center_xyz=Ref("in.target_xyz")`,
   `half_z=Ref("in.target_half_z")`. Returns `{hole_radius, rim_radius}`.

Wire the exit linearly: `ring_geometry → derived → END`, with
`set_on_error("degenerate")`. The script raises on a degenerate cloud, so no
conditional edges are needed.

## Checkpoints

- validate=True: the derived radii bracket a plausible ring
  (`0 < hole_radius < rim_radius`), checked against outputs.
