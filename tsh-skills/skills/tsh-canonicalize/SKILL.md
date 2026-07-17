---
name: tsh-canonicalize
description: Drive an arm to a CANONICAL, in-distribution pre-pose for the tape
  spool handover before a learned policy takes over. Perceive the tape (OBB),
  apply a fixed relative offset to get a canonical end-effector pose, and reach
  it with the connector's cuRobo-backed IK — so the frozen TSH VLA always starts
  from a pose it saw in training rather than an OOD tape position. Use when a TSH
  pickup, pre-handover, or placement segment is about to be delegated to the
  VLA and the tape may sit between training-grid positions. NOT a grasp skill —
  it positions the arm; the grasp/handover itself is the policy's job.
compatibility: requires gap>=0.1
metadata: {category: motion, tags: [tsh, canonicalize, curobo, reach, bimanual]}
gap:
  allowed_tools:
    - robot.get_observation
    - robot.get_ee_pose
    - robot.go_to_pose
  exit_conditions:
    reached: Arm moved to the canonical pre-pose.
    failed: IK/motion failed (raise propagates to on_error).
  required_inputs:
    target_obb: OrientedBoundingBox      # perceived tape, from an upstream perceive subgraph
    canonical_offset: list[float]        # [dx,dy,dz] world offset from tape center to the canonical EE pose
  produces_outputs:
    target_pose: Se3Pose
  canonical_scripts:
    - canonical_reach: scripts/canonical_reach.py
---

# tsh-canonicalize

The heart of the GaP↔TSH integration: **move the spatial variation upstream** so
the VLA never sees an out-of-distribution tape position.

1. An upstream perceive subgraph (`perceiving-objects`) localizes the tape and
   emits an `OrientedBoundingBox`.
2. `canonical_reach` turns that detection into a *canonical* end-effector pose by
   applying a fixed relative `canonical_offset`, then reaches it via
   `robot.go_to_pose` (cuRobo IK through the connector).
3. The VLA (`tsh-pi05`) starts the contact-rich grasp/handover from there.

One parameterized script serves all three segments (pickup / pre-handover /
placement) — they differ only by which OBB and which `canonical_offset` they get.

## TODO(data)

`canonical_offset` (and the canonical orientation) must be fit from data: which
relative pickup / handover / placement poses the VLA actually saw in training.
Until then the offset is a required input with no default and the orientation is
a top-down placeholder.

## When to use

- Before delegating a TSH pickup / pre-handover / placement segment to `tsh-pi05`.
