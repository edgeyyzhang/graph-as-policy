---
name: perceive-tape-cv
description: >
  Generic RGB-D object localization by classical colour + height-above-table
  segmentation — no ground truth, no object dimensions assumed, entirely
  in-process (numpy + cv2, no model servers). The DEFAULT perceiver: faster than
  the learned perceive-tape-sam sibling and fails loudly instead of mis-detecting
  a lookalike. Instantiated per object, parameterized by a literal object query
  whose colour word selects the segmentation band ("yellow tape", "gray tape").
  Emits ROLE-prefixed geometry — target_ (object being picked) or container_
  (place destination) — that downstream skills wire by exact name.
  Leave raise_if_missing at its default: a miss RAISES and lands on on_error.
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [tsh, perception, cv, depth, yam]
gap:
  allowed_tools:
    - robot.get_observation
  exit_conditions:
    perceived: Target localized; outputs bound in the subgraph.
    not_found: No colour+height blob matched — the script raises and this is the on_error symbol.
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
      default: the script RAISES on a miss and the subgraph's on_error carries
      it to not_found, so the flow stays observe -> perceive with no conditional
      edge. Two failure modes come from doing otherwise, and both have bitten:
      routing on the bool with mapping keys 'true'/'false' never matches,
      because the executor stringifies a Python bool to 'True'/'False' and the
      lookup fails at RUNTIME (validation only checks that mapping TARGETS are
      declared nodes, never that the KEYS match the values the field emits);
      and adding a require_found guard script that re-raises is an extra node
      doing what the default already does. raise_if_missing=False exists only
      for clean-all-items loops that route found=false to `done` — not for
      single-item tasks, which is every graph here.
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
    - perceive_object_cv: scripts/perceive_object_cv.py
  streaming: false
---

# perceive-tape-cv

Classical-CV object localizer: per-pixel world height from depth, table plane
re-estimated per call, colour mask from the query's colour word, morphology +
largest interior connected component (arm blobs touch the image border and are
rejected). Entirely in-process — no model servers, fails loudly instead of
mis-detecting a lookalike. It hands its pixel mask to the same robust top-face
back half the SAM variant uses, so the two front-ends can't drift apart on the
geometry they emit.

The query is a per-instance literal; the outputs are unprefixed geometry the
subgraph's set_outputs renames to its role prefix (target_* or container_*). No
ground-truth pose, no object dimensions assumed. Ring radii are not derived here
— that's `calculate-grasp-ring` on the emitted cloud.

## When to use

- To localize any tabletop object with a nameable colour: the pickup target, the
  place destination, or each item of a clean-all-items loop. Perceive the place
  DESTINATION first, while its view is clean.
- Prefer over `perceive-tape-sam` whenever a colour word disambiguates the target.

## When NOT to use

- Targets with no nameable colour, or lookalikes sharing a colour — use
  `perceive-tape-sam`.
- Tasks that consume an OrientedBoundingBox (use `perceiving-objects`).

## Recommended subgraph state flow

```text
observe → perceive
```

1. **`observe`** — `type: tool`, `tool: "robot.get_observation"`, `inputs: {}`.
   Emits `cameras`.
2. **`perceive`** — `type: script`, `scripts/<sg>/perceive_object_cv.py`,
   `inputs={"cameras": Ref("observe.cameras"), "object_query": "yellow tape"}`
   (the query is a colour-anchored literal, not a Ref). Returns `{found, cloud,
   top_xyz, center_xyz, half_z}`.

Two nodes, one edge, no branch: `START → observe → perceive → perceived → END`
with `set_on_error("not_found")`. Rename the outputs to the role prefix in
`set_outputs` — `target_*` for the grasp instance, `container_*` for the
place-destination instance.

## Required end states

| End state | Meaning |
|---|---|
| `perceived` | Target localized; route to the grasp / route chain. |
| `not_found` | No match; the script raised — route to `abort`. |

## See also

- `scripts/_perceive_cv.py` — the colour+height segmenter.
- `scripts/_perceive.py` — the shared back-projection + top-face core.
- `perceive-tape-sam` — the DINO+SAM sibling for targets with no nameable colour.
- `calculate-grasp-ring` — ring radii + grasp on the emitted cloud.
- `bimanual-route-arms`, `pickup`, `place` — the consumers.
