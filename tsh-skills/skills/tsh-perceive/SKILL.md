---
name: tsh-perceive
description: >
  RGB-D perception for the tape handover, with no ground truth and no object
  dimensions assumed. Grounding-DINO detect → SAM3 box segment → depth
  back-projection to a world-frame cloud, finished on the robust TOP-FACE slab.
  Two perceivers share one core: perceive_tape emits the tape grasp point, its
  ring geometry (hole/rim radii) and cloud; perceive_duct emits the duct
  top-face centre (the place target). Use to localize the tape and the place
  target for a scripted bimanual handover on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [tsh, perception, dino, sam3, depth, ring, yam]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - sam3.segment_box
  exit_conditions:
    perceived: Target localized; outputs bound in the subgraph.
    not_found: DINO found no match in the agentview (raise routes to abort).
  required_inputs: {}
    # NOTE: cameras is NOT a subgraph-level input. perceive_tape.py / perceive_duct.py
    # each REQUIRE a `cameras` argument, but it must come from an `observe`
    # (robot.get_observation) node authored INSIDE this subgraph — see "Recommended
    # subgraph state flow" below. Do not add `cameras` here; it would tell the
    # coordinator to wire it cross-subgraph, which it cannot reliably satisfy.
  produces_outputs:
    # perceive_tape:
    tape_xyz: Vec3                 # world grasp point (body-centroid height)
    tape_half_z: float             # perceived half-thickness (m)
    tape_cloud: PointCloud         # world-frame cloud (the handover's collision body)
    hole_radius: float             # inner hole radius (m) or null
    rim_radius: float              # outer rim radius (m) or null
    # perceive_duct:
    duct_xyz: Vec3                 # world duct top-face centre [x,y,top_z]
  canonical_scripts:
    - perceive_tape: scripts/perceive_tape.py
    - perceive_duct: scripts/perceive_duct.py
  streaming: false
---

# tsh-perceive

Both perceivers run the SAME pipeline (`_perceive.perceive_top_face`): DINO detect
→ SAM box segment → depth back-projection, then a robust **top-face slab** for the
centre and a 98th-percentile top face for height. They differ only in what they
derive — the tape adds ring radii + the cloud; the duct returns the top-face
centre. No ground-truth pose, no object dimensions, no scene constants.

Key robustness the core bakes in: ring radii are measured on the **top-face
slab**, not the full cloud — from an angled view the camera sees the table
*through the tape hole*, and those low-z points would otherwise collapse the
hole-radius estimate and mis-size the grasp. The duct query is colour-anchored
(`GAP_DUCT_QUERY`, default "gray tape") so the duct outscores the bright tape
ring in DINO.

## When to use

- To localize the yellow tape (grasp point + ring geometry + cloud) and the duct
  place target for `tsh-pickup` / `tsh-handover` / `tsh-place`.
- Any scripted LIBERO-YAM step needing a ring's grasp point and radii from RGB-D.

## When NOT to use

- Tasks with a learned perception front-end, or that consume an
  `OrientedBoundingBox` (use `perceiving-objects` instead — this skill emits a
  grasp point + radii, not an OBB).
- Non-ring place targets (perceive_duct assumes a flat top face).

## Recommended subgraph state flow

**HARD RULE — `observe` is a MANDATORY first node of THIS subgraph, always.**
`cameras` is deliberately absent from this skill's `required_inputs` (it is not
a cross-subgraph input the coordinator wires) — instead, `perceive_tape.py` /
`perceive_duct.py` each REQUIRE a `cameras` argument that only an internal
`observe` node can supply. Omitting `observe` is not a smaller/simpler variant
of this subgraph; the script raises immediately with a missing-argument error.

`observe` once, then either/both perceivers, ALL as internal nodes of this one
subgraph:

```text
observe → perceive_tape        (pickup path)
observe → perceive_duct        (place path; run UP FRONT, clean view)
```

State details:

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`, `inputs: {}`.
   Emits `cameras`. MUST be present — add it even if only one of the two
   perceivers below is used.
2. **`perceive_tape`** — `type: script`, `scripts/<sg>/perceive_tape.py`,
   `inputs={"cameras": Ref("observe.cameras"), "object_key": "yellow_tape_1"}`.
   Returns `{tape_xyz, tape_half_z, tape_cloud, hole_radius, rim_radius}`.
   > `object_key` is a **literal** MJCF body key (the DINO query is derived from
   > it), NOT a `Ref`.
3. **`perceive_duct`** — `type: script`, `scripts/<sg>/perceive_duct.py`,
   `inputs={"cameras": Ref("observe.cameras"), "target_key": "duct_tape_1"}`.
   Returns `{duct_xyz}`. Run this BEFORE the grasp, while the duct view is clean
   (at place time the held tape + gripper occlude it). Reads the SAME
   `observe.cameras` as `perceive_tape` when both run in this subgraph — one
   capture, not two — so both perceivers see the identical frame.

Bind the outputs the downstream skills need (`tape_xyz`, `hole_radius`,
`rim_radius`, `tape_cloud`, `tape_half_z` for the grasp/place; `duct_xyz` for
place).

## Required end states

| End state | Meaning |
|---|---|
| `perceived` | Target localized; route to the grasp / place subgraph. |
| `not_found` | DINO found no match; route to `abort`. |

## See also

- `scripts/_perceive.py` — the shared DINO+SAM+depth core (`perceive_top_face`,
  `estimate_ring_radii`, `estimate_half_thickness`).
- `perceiving-objects` (open-robot-skills) — the OBB-emitting generic perceiver.
- `tsh-pickup`, `tsh-place` — the consumers.
