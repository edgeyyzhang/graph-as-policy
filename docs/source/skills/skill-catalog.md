# Skill Catalog

The [open-robot-skills](gh-skills:.) registry ships ten skill bundles
under `skills/`. Each bundle owns one **subgraph** in a generated
workflow: its `SKILL.md` frontmatter declares the tools it may call
(`allowed_tools`), the exit conditions, the typed inputs and outputs it
exchanges with neighboring subgraphs, and the canonical scripts that are
materialized into the workflow directory. The body of `SKILL.md` is the
guidance the subgraph-authoring agent follows.

Every bundle declares a pip extra named after itself; meta-extras bundle
common sets (`quickstart = sam3 + grounding-dino + geometry`,
`grocery = quickstart + curobo`, and `all`). See
[Skill registries](registries.md) for discovery and `gap check`, and the
[tool catalog](tool-catalog.md) for the model-backed tools these skills
call.

:::{note} Requirements
Skill bundles carry no model weights themselves — their requirements come
from the tool bundles they call. The perception and tracking skills need
the GPU-backed `sam3`/`grounding-dino` bundles plus a hosted VLM API key;
the planner-based grasp skills need the `curobo` bundle (CUDA). Run
`gap check` to see which skills are runnable in your environment.
:::

## Choosing a perception skill

Four perception tiers trade latency, robustness, and loop semantics:

| Skill | Cost | Method | "Not found" behavior | Choose when |
|---|---|---|---|---|
| [perceiving-objects](#perceiving-objects) | ~2–4 s | DINO + pairwise VLM crop tournament + SAM3 | raises → `not_found` | Default single-target perception |
| [perceiving-objects-oneshot](#perceiving-objects-oneshot) | One VLM call | DINO + one set-of-marks letter pick + SAM3 | Clean `found: False` (no exception) | Clean-all-items loops needing a terminator |
| [perceiving-objects-multiview](#perceiving-objects-multiview) | ~5–8 s | Three detectors + VLM disambiguation merge | raises → `not_found` | Cluttered scenes, maximum robustness |
| [perceiving-object-parts](#perceiving-object-parts) | Two detection passes | DINO parent crop → DINO + SAM3 subpart | raises → `not_found` | The affordance is a subpart (handle, rim) |

## Choosing a grasp strategy

Three grasp strategies, by planner availability and target shape:

| Skill | Planner | Grasp pose | Choose when | Avoid when |
|---|---|---|---|---|
| [grasping-direct-ik](#grasping-direct-ik) | None | Candidate fan, align-then-descend | cuRobo not deployed; uncluttered scene | Clutter the arm must thread through |
| [grasping-with-planner](#grasping-with-planner) | cuRobo | Sampled candidate fan as a goalset | Default whenever cuRobo is deployed | Off-centroid affordances (bowl rims, handles) |
| [grasping-short-axis](#grasping-short-axis) | cuRobo | One deterministic short-axis pose | Elongated targets: pan/pot handles, bottles, tools | Symmetric objects where any yaw works |

The planner skills' candidate fans are centered on the OBB XY — they
slip on hollow centers and miss handles. For elongated targets a
sampled pose often holds at close but shears out under the lift; the
short-axis pose grips the bar squarely so it survives lift + transport.

## Perception skills

### perceiving-objects

Fast single-path 3D perception: Grounding-DINO broad detect → pairwise
VLM crop tournament → SAM3 box segmentation → depth back-projection →
`geometry.filter_and_compute_obb`. The pairwise "A or B?" tournament on
upscaled crops took object identification from ~30% to 97% on the
LIBERO-PosVar study versus a one-shot set-of-marks pick.

| | |
|---|---|
| Bundle | [skills/perceiving-objects](gh-skills:skills/perceiving-objects) |
| States | `observe → perceive → filter_obb` |
| Exits | `found` / `not_found` (via `on_error`) |
| Inputs | none (captures its own observation) |
| Outputs | `<name>_obb`, `<name>_mask`, `<name>_cloud` — all three required |
| Tool bundles | grounding-dino, vlm, sam3, geometry |
| Install | `quickstart` meta-extra + a VLM API key |

Multi-camera rigs get a verified **wrist-fallback gate**: identification
defaults to the exterior view, and the wrist view is consulted only when
the exterior pick fails its own close-up verify while the wrist pick
passes — the only zero-regression policy on the 4-suite regression study
(+2.5% net, 94.5% → 97.0%, 0/189 frames regressed). Wrist clouds still
fuse into the
output geometry, so the OBB recovers the faces a single front view
misses.

Results are cached per observation. Set `GAP_PERCEPTION_CACHE=0` to
disable, or `GAP_PERCEPTION_CACHE_DIR` to relocate the cache (default:
`.llm_cache/perceiving-objects/` inside the registry checkout).

### perceiving-objects-oneshot

Lightweight one-shot perception: DINO broad detect, a **single** VLM
set-of-marks letter pick over labeled boxes, then SAM3 box segmentation.
When the VLM answers "none" or DINO emits no detections, the script
returns a clean `found: False` instead of raising — the loop terminator
for clean-all-items tasks (`target.not_found → done`).

| | |
|---|---|
| Bundle | [skills/perceiving-objects-oneshot](gh-skills:skills/perceiving-objects-oneshot) |
| States | `observe → perceive → filter_obb` |
| Exits | `found` / `not_found` (via `on_error`) |
| Inputs | none |
| Outputs | `<name>_obb`, `<name>_mask`, `<name>_cloud` |
| Tool bundles | grounding-dino, vlm, sam3, geometry |
| Install | `quickstart` meta-extra + a VLM API key |

Choose it for generic target descriptions ("the next remaining grocery
item") and multi-item loops. Avoid it for small targets (< 40 px wide) —
the pairwise tournament in `perceiving-objects` is far more reliable in
that regime.

### perceiving-objects-multiview

Robust three-method perception: a Molmo point + DINO box pipeline, a
Molmo point + SAM3 point pipeline, and a DINO + VLM box-selection
pipeline all run on the same observation. A `merge` script drops
geometrically implausible masks, disambiguates survivors with a VLM
side-by-side panel, and emits an already-filtered OBB.

| | |
|---|---|
| Bundle | [skills/perceiving-objects-multiview](gh-skills:skills/perceiving-objects-multiview) |
| States | `observe → perceive_dino → perceive_point → perceive_dino_vlm → merge` |
| Exits | `found` / `not_found` (via `on_error`) |
| Inputs | none |
| Outputs | `<name>_obb`, `<name>_mask` (the declared set; `merge` also returns a `cloud` you can bind when a downstream skill needs it) |
| Tool bundles | molmo (optional), grounding-dino, vlm, sam3, geometry |
| Install | `quickstart` meta-extra + a VLM API key; optional Molmo endpoint |

The Molmo paths need a self-hosted vLLM endpoint via
`GAP_MOLMO_BASE_URL`; without one the scripts catch the error and fall
back, so the skill keeps working with two detectors. On API-only
platforms, `gemini-er.detect` is the hosted, zero-GPU detection swap-in.
Note that `merge` returns only `{cloud, mask, obb}` — there is no
`found` field, so the subgraph uses linear edges plus `on_error`, and
this skill has no clean loop-terminating signal (use
`perceiving-objects-oneshot` for loops).

### perceiving-object-parts

Hierarchical zoom-in perception for subparts. A frypan handle is ~3% of
image pixels — bad signal for text segmentation; cropping to the parent
first brings it to ~30% of the cropped image. Pipeline: DINO detects the
parent → crop with `padding_px` (default 30) → DINO + SAM3 segment the
subpart inside the crop → uncrop → world cloud →
`geometry.filter_noise` + `geometry.compute_obb` with an
unfiltered-cloud fallback.

| | |
|---|---|
| Bundle | [skills/perceiving-object-parts](gh-skills:skills/perceiving-object-parts) |
| States | `observe → perceive_handle → found` (single canonical script) |
| Exits | `found` / `not_found` (via `on_error`) |
| Inputs | none (`parent_prompt` / `subpart_prompt` are literal strings on the node) |
| Outputs | `<name>_obb`, `<name>_mask`, `<name>_cloud`, plus `<name>_parent_obb`, `<name>_parent_cloud` |
| Tool bundles | grounding-dino, vlm, sam3, geometry |
| Install | `quickstart` meta-extra + a VLM API key |

The `mask` output is the **parent's** mask (used for collision isolation
downstream); the `obb` and `cloud` are the subpart's, and the subpart's
own mask rides along as `subpart_mask`. The parent OBB/cloud feed the
transport skill's `drop_offset` node so the parent body — not the grasp
point — lands at the drop XY.

:::{warning}
Do not append a `geometry.filter_and_compute_obb` node after this skill.
The script already returns a noise-filtered OBB with an unfiltered-cloud
fallback; re-running DBSCAN on a thin handle shell routinely collapses
the OBB. Bind `target_obb` directly to the script's `obb` output.
:::

## Grasping skills

### grasping-direct-ik

No-planner grasp: pre-rotate the gripper to the grasp orientation at a
safe height above the target, descend straight down, close. The
align-then-descend split avoids the corkscrew failure of a single
`go_to_pose` that blends rotation and descent and drags across the
target.

| | |
|---|---|
| Bundle | [skills/grasping-direct-ik](gh-skills:skills/grasping-direct-ik) |
| States | `open → compute_grasp → compute_align → rotate_align → descend → close` |
| Exits | `grasped` / `failed` (`on_error` only — never declare a `failed` node) |
| Inputs | `target_obb: OrientedBoundingBox` |
| Outputs | none declared |
| Tool bundles | geometry (+ connector `robot.*` tools) |
| Install | `geometry` extra |

The align pose sits 0.15 m above the OBB top
(`compute_align_pose.py`). Grasp success is verified by a
`validate=True` checkpoint (`target_held`), not by a verify node — no
`verify_gripper_grasp` skill exists. Lifting to carry height belongs to
the following `transporting-objects` subgraph.

### grasping-with-planner

Collision-aware grasping with cuRobo — the default whenever the `curobo`
tool bundle is deployed. The subgraph builds a per-observation collision
world with the target excluded, plans to one of several candidate
top-down grasps (a goalset), executes, and closes.

| | |
|---|---|
| Bundle | [skills/grasping-with-planner](gh-skills:skills/grasping-with-planner) |
| States | `open → compute_grasp → approach → observe → build_world → plan → execute → close` |
| Exits | `grasped` / `failed` (`on_error` only) |
| Inputs | `target_obb`, `target_mask` |
| Outputs | `ee_pose_at_grasp` (= `Ref("observe.arms.0.ee_pose")`), `grasp_pose` (= `Ref("compute_grasp.candidates.poses.0")`) |
| Tool bundles | curobo, geometry |
| Install | `grocery` meta-extra (or `curobo` + `geometry`); CUDA GPU, `CUDA_HOME`, `--no-build-isolation` |

`ee_pose_at_grasp` is the live TCP pose at the `observe` step;
`grasp_pose` is the computed pose the planner targets. Downstream
`transporting-objects` consumes `ee_pose_at_grasp` to compute drop
height, and `grasp_pose` supports output-anchored checkpoint verifiers.
The canonical `plan_grasp.py` deliberately sets
`use_grasp_approach=False`: cuRobo's built-in approach flag downweights
Z and lets the optimizer "complete" at the pre-grasp without descending,
so the skill pre-rotates in its own `approach` state instead. An
optional `select_short_axis` script state reorders the candidate fan for
elongated targets without changing the poses.

Not for off-centroid affordances (bowl rims, mug/pan handles): the
candidates are OBB-XY-centered. Use `grasping-short-axis` for those.

### grasping-short-axis

Deterministic, geometry-locked grasp for elongated targets. The single
grasp pose is computed (not sampled): straight-down approach with the
finger-opening axis snapped to the OBB's *shorter* horizontal axis, so
the jaws close across the narrow dimension of the bar.

| | |
|---|---|
| Bundle | [skills/grasping-short-axis](gh-skills:skills/grasping-short-axis) |
| States | `open → compute_grasp → offset_from_base → approach → observe → build_world → plan → execute → finalize → close` |
| Exits | `grasped` / `failed` (`on_error` only) |
| Inputs | `target_obb`, `target_mask`; optional `base_obb` (deliberately not in `required_inputs` so validation does not force it) |
| Outputs | `ee_pose_at_grasp`, `grasp_pose` (= `Ref("offset_from_base.adjusted_grasp")`) |
| Tool bundles | curobo, geometry |
| Install | `curobo` + `geometry` extras; CUDA GPU |

Key levers:

- `short_axis_grasp_pose.py` defaults to `z_offset=-0.04` (fingertip
  4 cm into the OBB top), with grasp Z clamped to −0.05 m.
- `offset_from_base` slides the grasp outward along the handle, clear of
  a heavier attached body, when a second perception subgraph supplies
  `base_obb`; without `base_obb` it safely no-ops.
- `finalize_trajectory.py` between `execute` and `close` is
  **mandatory**: with `subsample=4` the PD controller is still driving
  when control returns, and a `close_gripper` fired mid-motion is
  silently dropped by the simulator — fingers never close, no error.
- The canonical `plan_grasp.py` keeps the planner deliberately lean
  (`topk=8`, `retries=1`, `num_ik_seeds=128`): try the 8 best-ranked
  candidates once each and fail fast. The previously inflated values
  (25/3/256) made every grasp-poor attempt grind through up to 75
  silent `plan_to_grasp_poses` calls (~6 min) without improving success.
- Escalation: when the standard plan reports "0/N feasible" on a
  far/low/occluded handle, `plan_grasp_motion.py` (cuRobo v0.8
  `curobo.plan_grasp_motion`) decomposes the grasp into approach, grasp,
  and lift trajectories with the gripper close interleaved between the
  grasp and lift legs.

## Transport

### transporting-objects

Move the currently held object above a destination container and
release. Enters holding; exits with the object placed.

| | |
|---|---|
| Bundle | [skills/transporting-objects](gh-skills:skills/transporting-objects) |
| States | `compute_drop → move_above → release` (3, default) or `perceive_zone → compute_drop → move_above → release` (4); optional `drop_offset` after `compute_drop` |
| Exits | `placed` / `blocked` |
| Inputs | `container_obb`, `container_mask`, `target_obb`, `target_mask`, `ee_pose_at_grasp` |
| Outputs | none declared |
| Tool bundles | geometry; optionally curobo, grounding-dino, vlm, sam3 (zone + carve variants) |
| Install | `geometry` extra (plus `curobo` for the carve/linear variants) |

`compute_drop_pose.py` accepts the held object as `held_obb`
(defaults: `drop_clearance=0.05`, `approach_height=0.20`, panda
hand-to-TCP offset 0.1029 m). Binding it as `target_obb` is silently
discarded and the drop lands wrong by the held object's full height.
Optional states and variants:

- **`perceive_zone`** — VLM-grounded sub-region localization ("the left
  compartment of the caddy"); returns `placement_zone_obb` or `None`
  (automatic bare-container fallback). No other container re-perception
  states are allowed.
- **`drop_offset`** — required whenever the grasp was on a subpart
  (upstream is `perceiving-object-parts`): shifts the drop XY so the
  *parent* centroid lands at the zone centroid. The single most common
  omission in handle-grasp place-on failures.
- **`waypoint_move_carve`** — collision-aware variant of `move_above`
  that routes through `curobo.plan_with_grasped_object`; use when an
  obstacle sits on the transport path. The default `waypoint_move`
  lifts to 0.45 m and translates with no planner — tuned for LIBERO
  tabletops, unsafe under shelves or overhead obstacles.
- **`descend_release_linear`** — straight-line Cartesian descent via
  `curobo.plan_linear` with a `robot.go_to_pose` fallback; the cleanest
  release dynamics for subpart-grasp place-on tasks.

## Policy and tracking skills

These two are **class-based, long-running skills**: a workflow invokes
them as a single stateful node fed by the graph-scoped
`observation_stream`, rather than as a multi-state subgraph. Both are
also exposed as **flat tools** (`running-policies.run`,
`tracking-objects.track`) for callers that want a one-shot invocation.

### running-policies

Run a learned VLA policy in closed loop until a termination signal
fires. Each window the loop reads `observation_stream.latest()`, asks an
openpi websocket policy server for an action chunk, and forwards the
first `replan_every` rows to the robot via `sim.apply_policy_action`.

| | |
|---|---|
| Bundle | [skills/running-policies](gh-skills:skills/running-policies) |
| Form | Class-based skill + flat tool `running-policies.run` |
| Key inputs | `observation_stream`, `policy_id`, `prompt`; knobs `termination_prompt=""`, `max_windows=20`, `replan_every=5`, `term_period=2`, `arm_id=0`, `vlm_camera=0`, `settle_steps=10`, `gripper_cycle_termination=False` |
| Returns | `{status, num_windows, num_steps}` with `status` ∈ `completed_by_vlm` \| `gripper_cycle` \| `max_windows` |
| Install | `running-policies` extra (`openpi-client`); a policy server |
| Env vars | `GAP_OPENPI_DIR` (used by the shipped presets) |

Three additive terminators: the `max_windows` backstop, a VLM yes/no
`termination_prompt` checked every `term_period` windows, and
gripper-cycle termination (one open → close → open cycle, with the close
debounced to ≥ 3 windows) — the per-item terminator for clean-all-items
loops. Policies are registered in the task config's `policies:` block as
an external `url:`, a managed `start_cmd:`, or a `preset:`
(`pi05-libero`, `molmoact-libero`); `gap policy serve pi05-libero` is
the one-command path. The websocket client is cached per `policy_id` via
the executor's `PolicyExecutor`. Connectors without a VLA passthrough
do not register `sim.apply_policy_action`, so the call fails loudly
rather than driving the robot with the wrong action space. See
[Benchmark policies](../benchmarks/policies.md) for the policy-serving
workflow.

### tracking-objects

The registry's only `streaming: true` skill: a SAM3 tracker loop over
the observation stream. Seeds the tracker with a text prompt on the
first frame, then polls at `update_hz` and advances via
`sam3.tracker_update`, publishing a per-tick snapshot
`{mask, box, confidence, object_present, n_updates}` via `ctx.publish`
so downstream consumers see live state while the loop runs.

| | |
|---|---|
| Bundle | [skills/tracking-objects](gh-skills:skills/tracking-objects) |
| Form | Class-based streaming skill + flat tool `tracking-objects.track` |
| Key inputs | `observation_stream`, `target_prompt`; knobs `camera_index=0`, `update_hz=5.0`, `max_updates=200`, `allow_lost_frames=10`, `close_on_exit=True` (skill form only) |
| Returns | `{final_mask, final_box, final_confidence, object_present, n_updates}` |
| Tool bundles | sam3 |
| Install | `sam3` extra; CUDA GPU |

The loop checks `ctx.cancel_token` each tick (so it can be wrapped under
a `parallel` state with a `join_policy`) and closes the tracker session
in a `finally` block. Pass `close_on_exit=False` in the skill form to
resume the session across repeated visits to the same state; the flat
`tracking-objects.track` tool always runs one fresh, self-closing
session. A failed seed returns immediately with `object_present=False`.

## See also

- [Tool catalog](tool-catalog.md) — the model-backed tools these skills
  call, with exact signatures.
- [Skill registries](registries.md) — discovery, precedence, `gap check`.
- [Authoring bundles](authoring-bundles.md) — the `SKILL.md` contract.
- [Generation](../authoring/generation.md) — how skills become subgraphs
  in generated workflows.
