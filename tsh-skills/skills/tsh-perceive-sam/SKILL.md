---
name: tsh-perceive-sam
description: >
  Generic RGB-D object localization via Grounding-DINO detect → SAM3 box
  segment → robust depth back-projection, no ground truth and no object
  dimensions assumed. Use for targets with no nameable colour, or that need
  open-vocabulary grounding — the sibling `tsh-perceive-cv` (colour+height
  CV, no model servers) is the DEFAULT and should be preferred whenever a
  colour word disambiguates the target. One skill, instantiated per object:
  parameterized by a literal object query ("yellow tape") and emits
  ROLE-prefixed geometry — the SAME output contract as `tsh-perceive-cv`.
  The prefix is EXACTLY ``target_`` (object being picked) or ``container_``
  (place destination), NEVER the object's own noun: ``target_xyz``, never
  ``yellow_tape_xyz``; downstream skills wire these by exact name.
  raise_if_missing=False returns found=false for clean-all-items loops. Also
  carries the legacy `perceive_tape` / `perceive_duct` wrappers (older
  graphs). Perceive the place DESTINATION first, while its view is clean.
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
    # Prefix substitution: <name> is the ROLE, and it is exactly one of TWO
    # values — `target` (the object being picked) or `container` (the place
    # destination). These are the SAME two role words the coordinator's own
    # base prompt hardcodes as the only legal `<name>` substitutions, so
    # matching them is what makes the auto-wire land. NEVER substitute the
    # task's noun for the object ("duct_tape", "bin", "duct") — those names
    # have no downstream consumer and cause a W8 wiring failure. This is a
    # FIXED CONTRACT: tsh-pickup / tsh-ring-geometry require an upstream
    # output literally named `target_xyz` / `target_cloud` / `target_half_z`;
    # tsh-place-pose / tsh-route require one literally named `container_xyz`.
    # Cross-subgraph wiring matches producer output names to consumer input
    # names EXACTLY (no semantic matching).
    "<target|container>_found": bool             # false only with raise_if_missing=false
    "<target|container>_cloud": PointCloud       # world-frame cloud (collision body, ring input)
    "<target|container>_xyz": Vec3               # body-centroid-height centre (grasp point)
    "<target|container>_top_xyz": Vec3           # top-face centre [x,y,top_z] (place target)
    "<target|container>_half_z": float           # perceived half-thickness (m)
  hard_rules:
    - >
      The output prefix `<name>` is a fixed wiring contract, and it is
      EXACTLY one of two role words — `target` (object being picked) or
      `container` (place destination) — never the task's own noun for the
      object. These are the two legal `<name>` values the coordinator's base
      prompt already hardcodes. The grasp instance MUST use `target_`
      (tsh-pickup / tsh-ring-geometry require `target_xyz` / `target_cloud` /
      `target_half_z` by exact name); the place-destination instance MUST use
      `container_` (tsh-place-pose / tsh-route require `container_xyz` by
      exact name). Naming the destination instance after the task's
      vocabulary (`duct_tape_`, `duct_`, `bin_`) breaks the auto-wire — a
      residual "no upstream producer declares an output named 'container_xyz'"
      W8 validation error.
  canonical_scripts:
    - perceive_object: scripts/perceive_object.py  # generic DINO+SAM entry point
    - perceive_tape: scripts/perceive_tape.py     # legacy tape wrapper (adds ring radii)
    - perceive_duct: scripts/perceive_duct.py     # legacy duct wrapper (top face only)
  streaming: false
---

# tsh-perceive-sam

Learned-detector object localizer: Grounding-DINO detect → SAM3 box segment
→ depth back-projection, finishing on the SAME robust top-face core the CV
sibling uses (`_perceive.top_face_from_mask`: depth back-projection, then a
robust **top-face slab** for the centre and a 98th-percentile top face for
height) — so the two front-ends emit identical-in-kind geometry.

The query is a per-instance literal, and the outputs are unprefixed
geometry fields the subgraph's `set_outputs` renames to its ROLE prefix —
exactly one of `target_*` (object being picked) or `container_*` (place
destination). No ground-truth pose, no object dimensions, no scene positions
assumed.

> **Why `container_` and not `dest_`:** the coordinator's own base prompt
> hardcodes the two legal destination-role names as `target` and
> `container`. Our downstream consumers therefore use `container_xyz` so the
> coordinator's substitution rule and our wiring names agree — a `dest_`
> prefix (however natural it reads) is NOT in the coordinator's legal set and
> is what caused it to fall back to the object noun.

Ring-specific derivations (hole/rim radii) do NOT live in `perceive_object`
— they are the separate `tsh-ring-geometry` post-processor on the emitted
cloud. That split is what makes `perceive_object` reusable across tasks:
perceiving "the red stack" for a sorting task is the same subgraph with a
different query literal. The legacy `perceive_tape` / `perceive_duct`
wrappers below predate that split and still derive ring radii / take a body
key inline — kept for older graphs, not the entry point for new ones.

The legacy `perceive_tape` / `perceive_duct` wrappers remain for older graphs
(the duct query is colour-anchored via `GAP_DUCT_QUERY`, default "gray tape",
so the grey duct outscores the bright tape ring in DINO — with the generic
script simply pass the colour-anchored query as the literal).

## When to use

- Targets with no nameable colour, or open-vocabulary grounding that a
  colour+height segmenter can't key off.
- To localize ANY tabletop object from RGB-D: the pickup target, the place
  destination, or each item of a clean-all-items loop
  (`raise_if_missing=false` + route on the `found` field).
- Instantiate once per object, in dependency order — perceive the place
  DESTINATION first, while its view is clean (at place time the held object
  + gripper occlude it).

## When NOT to use

- Whenever a colour word disambiguates the target — prefer `tsh-perceive-cv`
  (the DEFAULT): faster (no model servers), no detector score noise, fails
  loudly instead of mis-detecting a lookalike.
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
   > `object_query` is a **literal** noun phrase, NOT a `Ref`. Make it
   > colour-anchored where possible — it disambiguates lookalikes for DINO
   > ("gray tape" for the duct, not "duct tape").

Bind prefixed outputs (ALL of them — downstream skills wire by name). There
are only TWO legal prefixes, `target_` and `container_` (the coordinator's
own two hardcoded role words) — never the object's own name from the task
prompt, even though this subgraph NODE is itself typically named after that
object (`perceive_yellow_tape`, `perceive_duct_tape`):

```python
# CORRECT — this subgraph instance is named "perceive_duct_tape" (after the
# task's object noun), but its OUTPUTS still use the ROLE prefix "container_",
# because it is perceiving the place destination:
sg.set_outputs(
    container_found=Ref("perceive.found"),
    container_cloud=Ref("perceive.cloud"),
    container_xyz=Ref("perceive.center_xyz"),
    container_top_xyz=Ref("perceive.top_xyz"),
    container_half_z=Ref("perceive.half_z"),
)
# WRONG — do NOT do this, even though the subgraph is literally named
# perceive_duct_tape: naming outputs duct_tape_found/duct_tape_xyz/... breaks
# every downstream auto-wire (tsh-place-pose, tsh-route, ... all require
# container_xyz by EXACT name, not duct_tape_xyz):
#   sg.set_outputs(duct_tape_found=..., duct_tape_xyz=..., ...)  # NO
```

For the pickup/grasp instance (typically named after the target object,
e.g. `perceive_yellow_tape`), use `target_` in exactly the same way —
`target_found`, `target_cloud`, `target_xyz`, `target_top_xyz`,
`target_half_z` — never `yellow_tape_found` / `yellow_tape_xyz` / etc.

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

- `scripts/_perceive.py` — the DINO+SAM front-end (`perceive_top_face`) and
  the shared back-projection + robust top-face core (`top_face_from_mask`,
  `estimate_half_thickness`, `estimate_ring_radii`).
- `tsh-perceive-cv` — the DEFAULT sibling: classical colour+height CV,
  in-process, no model servers. Prefer it whenever a colour word
  disambiguates the target.
- `tsh-ring-geometry` — ring radii post-processor on the emitted cloud
  (the modern replacement for `perceive_tape`'s inline ring derivation).
- `perceiving-objects` (open-robot-skills) — the OBB-emitting generic
  perceiver (multi-camera KD-tree fusion, VLM disambiguation).
- `tsh-route`, `tsh-pickup`, `tsh-place` — the consumers.
