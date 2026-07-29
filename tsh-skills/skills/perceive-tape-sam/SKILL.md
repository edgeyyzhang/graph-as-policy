---
name: perceive-tape-sam
description: >
  Generic RGB-D object localization via Grounding-DINO detect → SAM3 box segment
  → robust depth back-projection, no ground truth and no object dimensions
  assumed. Use for targets with no nameable colour, or that need open-vocabulary
  grounding — the sibling perceive-tape-cv (colour+height CV, no model servers) is
  the DEFAULT and should be preferred whenever a colour word disambiguates the
  target. Instantiated per object, parameterized by a literal object query, and
  emits ROLE-prefixed geometry (the same output contract as perceive-tape-cv):
  target_ (object being picked) or container_ (place destination), wired by exact
  name. Leave raise_if_missing at its default: a miss RAISES onto on_error.
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
    not_found: DINO found no match — the script raises and this is the on_error symbol.
  required_inputs: {}
  produces_outputs:
    "<target|container>_found": bool
    "<target|container>_cloud": PointCloud
    "<target|container>_xyz": Vec3
    "<target|container>_top_xyz": Vec3
    "<target|container>_half_z": float
  hard_rules:
    - >
      Do NOT pass raise_if_missing=False and branch on the found flag. Leave the
      default: the script RAISES on a miss and on_error carries it to not_found,
      so the flow stays observe -> perceive with no conditional edge. Branching
      on the bool fails at RUNTIME — mapping keys 'true'/'false' never match,
      because the executor stringifies a Python bool to 'True'/'False', and
      validation checks only that mapping TARGETS are declared nodes, never that
      the KEYS match what the field emits. raise_if_missing=False exists only for
      clean-all-items loops routing found=false to `done`.
    - >
      The output prefix is EXACTLY one of two role words — target (object being
      picked) or container (place destination) — never the task's own noun for
      the object. Downstream consumers wire target_xyz / container_xyz etc. by
      exact name, so a task-vocabulary prefix (duct_tape_, bin_) breaks the wire.
    - >
      observe (robot.get_observation) is a mandatory first node INSIDE this
      subgraph — the script's cameras argument comes from it, not a
      cross-subgraph input (which is why cameras is absent from required_inputs).
  canonical_scripts:
    - perceive_object: scripts/perceive_object.py
  streaming: false
---

# perceive-tape-sam

Learned-detector object localizer: Grounding-DINO detect → SAM3 box segment →
depth back-projection, finishing on the same robust top-face core the CV sibling
uses — so the two front-ends emit identical-in-kind geometry.

The query is a per-instance literal; the outputs are unprefixed geometry the
subgraph's set_outputs renames to its role prefix (target_* or container_*). No
ground-truth pose, no object dimensions assumed. Ring radii are not derived here
— that's `calculate-grasp-ring` on the emitted cloud.

## When to use

- Targets with no nameable colour, or open-vocabulary grounding a colour+height
  segmenter can't key off. Perceive the place DESTINATION first, while its view
  is clean.

## When NOT to use

- Whenever a colour word disambiguates the target — prefer `perceive-tape-cv` (the
  DEFAULT): faster, no detector score noise, fails loudly on a lookalike.
- Tasks that consume an OrientedBoundingBox (use `perceiving-objects`).

## Recommended subgraph state flow

```text
observe → perceive
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`, `inputs: {}`.
   Emits `cameras`.
2. **`perceive`** — `type: script`, `scripts/<sg>/perceive_object.py`,
   `inputs={"cameras": Ref("observe.cameras"), "object_query": "yellow tape"}`
   (the query is a colour-anchored literal, not a Ref). Returns `{found, cloud,
   top_xyz, center_xyz, half_z}`.

Rename the outputs to the role prefix in `set_outputs` — `target_*` for the grasp
instance, `container_*` for the place-destination instance. For a clean-all-items
flow stays linear: `START -> observe -> perceive -> perceived -> END` with
`set_on_error("not_found")` and no conditional edge.

## Required end states

| End state | Meaning |
|---|---|
| `perceived` | Target localized; route to the grasp / route chain. |
| `not_found` | No match; route to `abort` — or `done` in item loops. |

## See also

- `scripts/_perceive.py` — the DINO+SAM front-end + shared top-face core.
- `perceive-tape-cv` — the DEFAULT sibling: classical colour+height CV.
- `calculate-grasp-ring` — ring radii + grasp on the emitted cloud.
- `bimanual-route-arms`, `pickup`, `place` — the consumers.
