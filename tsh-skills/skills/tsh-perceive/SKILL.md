---
name: tsh-perceive
description: >
  Generic RGB-D object localization, with no ground truth and no object
  dimensions assumed. Grounding-DINO detect → SAM3 box segment → depth
  back-projection to a world-frame cloud, finished on the robust TOP-FACE
  slab. One skill, instantiated per object: the subgraph is parameterized by
  a literal object query ("yellow tape", "gray tape", "red tape spool") and
  emits name-prefixed geometry (cloud, top-face centre, body-centre, half
  thickness). A raise_if_missing=False mode returns found=false cleanly for
  clean-all-items loops. Use to localize any tabletop target — the pickup
  tape, the place destination, or each item of a sorting loop — on LIBERO-YAM.
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [tsh, perception, dino, sam3, depth, yam]
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - sam3.segment_box
  exit_conditions:
    perceived: Target localized; outputs bound in the subgraph.
    not_found: DINO found no match in the agentview (raise routes to on_error;
      with raise_if_missing=false, route the found=false field instead —
      clean-all-items loop exit).
  required_inputs: {}
    # NOTE: cameras is NOT a subgraph-level input. perceive_object.py REQUIRES
    # a `cameras` argument, but it must come from an `observe`
    # (robot.get_observation) node authored INSIDE this subgraph — see
    # "Recommended subgraph state flow" below. Do not add `cameras` here; it
    # would tell the coordinator to wire it cross-subgraph, which it cannot
    # reliably satisfy.
  produces_outputs:
    # Prefix substitution: bind these under this instance's name prefix —
    # target_* for the pickup object, dest_* for the place destination, etc.
    <name>_found: bool             # false only with raise_if_missing=false
    <name>_cloud: PointCloud       # world-frame cloud (collision body, ring input)
    <name>_xyz: Vec3               # body-centroid-height centre (grasp point)
    <name>_top_xyz: Vec3           # top-face centre [x,y,top_z] (place target)
    <name>_half_z: float           # perceived half-thickness (m)
  canonical_scripts:
    - perceive_object: scripts/perceive_object.py
    - perceive_tape: scripts/perceive_tape.py     # legacy tape wrapper (adds ring radii)
    - perceive_duct: scripts/perceive_duct.py     # legacy duct wrapper (top face only)
  streaming: false
---

# tsh-perceive

One generic perceiver (`perceive_object`, the `_perceive.perceive_top_face`
core): DINO detect → SAM box segment → depth back-projection, then a robust
**top-face slab** for the centre and a 98th-percentile top face for height.
It is object-agnostic — the DINO query is a per-instance literal, and the
outputs are unprefixed geometry fields the subgraph's `set_outputs` renames
to its own prefix (`target_*`, `dest_*`, `red_stack_*`, ...). No ground-truth
pose, no object dimensions, no scene constants.

Ring-specific derivations (hole/rim radii) do NOT live here — they are the
separate `tsh-ring-geometry` post-processor on the emitted cloud. That split
is what makes this skill reusable across tasks: perceiving "the red stack"
for a sorting task is the same subgraph with a different query literal.

The legacy `perceive_tape` / `perceive_duct` wrappers remain for older graphs
(the duct query is colour-anchored via `GAP_DUCT_QUERY`, default "gray tape",
so the grey duct outscores the bright tape ring in DINO — with the generic
script simply pass the colour-anchored query as the literal).

## When to use

- To localize ANY tabletop object from RGB-D: the pickup target, the place
  destination, or each item of a clean-all-items loop
  (`raise_if_missing=false` + route on the `found` field).
- Instantiate once per object, in dependency order — perceive the place
  DESTINATION first, while its view is clean (at place time the held object
  + gripper occlude it).

## When NOT to use

- Tasks that consume an `OrientedBoundingBox` (use `perceiving-objects`
  instead — this skill emits centres + cloud, not an OBB).
- Cluttered scenes with near-identical distractors close together (the
  single DINO top-box path has no disambiguation tournament).

## Recommended subgraph state flow

**HARD RULE — `observe` is a MANDATORY first node of THIS subgraph, always.**
`cameras` is deliberately absent from this skill's `required_inputs` (it is
not a cross-subgraph input the coordinator wires) — `perceive_object.py`
REQUIRES a `cameras` argument that only an internal `observe` node can
supply.

2 states:

```text
observe → perceive
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`,
   `inputs: {}`. Emits `cameras`.
2. **`perceive`** — `type: script`, `scripts/<sg>/perceive_object.py`,
   `inputs={"cameras": Ref("observe.cameras"), "object_query": "yellow tape"}`.
   Returns `{found, cloud, top_xyz, center_xyz, half_z}`.
   > `object_query` is a **literal** noun phrase (the DINO query), NOT a
   > `Ref`. Pick a colour-anchored phrase when lookalike objects share the
   > scene ("gray tape" for the duct, not "duct tape").

Bind prefixed outputs (ALL of them — downstream skills wire by name):

```python
sg.set_outputs(
    target_found=Ref("perceive.found"),
    target_cloud=Ref("perceive.cloud"),
    target_xyz=Ref("perceive.center_xyz"),
    target_top_xyz=Ref("perceive.top_xyz"),
    target_half_z=Ref("perceive.half_z"),
)
```

(Replace `target_` with this instance's prefix — `dest_` for the place
destination.)

For the clean-all-items loop, pass `raise_if_missing=False` and add a
conditional edge on `perceive.found` (`"True" → perceived`,
`"False" → no_more_items`), both declared exits.

## Required end states

| End state | Meaning |
|---|---|
| `perceived` | Target localized; route to ring-geometry / route / grasp. |
| `not_found` | No match; route to `abort` — or to `done` in item loops. |

## Checkpoints

- validate=True: the perceived centre is within 5 cm of the privileged body
  position (2-arg predicate comparing `outputs["<name>_xyz"]` to
  `w.body(...)`).

## See also

- `scripts/_perceive.py` — the shared DINO+SAM+depth core.
- `tsh-ring-geometry` — ring radii post-processor on the emitted cloud.
- `perceiving-objects` (open-robot-skills) — the OBB-emitting generic
  perceiver (multi-camera KD-tree fusion, VLM disambiguation).
- `tsh-route`, `tsh-pickup`, `tsh-place` — the consumers.
