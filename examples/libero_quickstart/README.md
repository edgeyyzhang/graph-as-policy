# libero_quickstart — pick the soup can into the basket

> **What:** The end-to-end hero: real vision → OBB grasp → transport, ground-truth verified · **Needs:** `quickstart` + GPU + LLM key · **Time:** ~25–55 s/trial · **Measured:** 9/10 grasp · 7/10 task (10 seeds)

The end-to-end quickstart for GaP: a static, fully-authored workflow graph
that perceives a target object and a container with real vision models
(Grounding DINO + SAM3 + a hosted VLM), derives a top-down grasp from the
fused 3D oriented bounding box, picks the object with a direct-IK
align-then-descend grasp, and places it in the basket — verified by the
sim's own success predicate and a ground-truth `target_held` checkpoint.

## Run it

From the GaP checkout (open-robot-skills cloned next to it):

```bash
uv sync --extra quickstart            # engine + LIBERO sim + perception models
uv run gap skills check --download    # verify bundles + prefetch weights (~3.5 GB)
export ANTHROPIC_API_KEY=...          # VLM provider — alternatives below

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                        # browse the recorded trial
```

- The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
  checkout next to the graph-as-policy checkout); `--skills PATH` overrides.
- `--validate-only` checks the graph without executing.
- `--checkpoints warn` (default) evaluates the ground-truth postcondition
  checkpoints in `graph/checkpoints/` after each subgraph (sim only) —
  `grasp_sg.target_held` verifies via privileged contacts that the gripper
  actually holds the can after `close`.
- Traces (per-node inputs/outputs, every model call, rendered frames) land
  in `outputs/run_<timestamp>/`.

## Two variants

| Graph | Grasp strategy | Needs |
|---|---|---|
| `graph/` (**default**) | direct-IK align-then-descend (`grasping-direct-ik` skill recipe): pre-rotate to the grasp yaw at a safe height above the OBB, descend straight down, close | `--extra quickstart` |
| `graph_planner/` | CuRobo goal-set planning over the same grasp candidates (`grasping-with-planner`): reconstruct a collision world from RGB-D, plan a collision-free joint trajectory to the best reachable candidate | `--extra grocery` (CUDA build, see below) |

`graph_planner/` is the faithful port of the dev tree's
`libero_pro/graph_cartesian_obb` example. The default `graph/` replaces only
its grasp subgraph with the planner-free recipe so the quickstart needs no
CuRobo install; perception and transport are identical.

`--extra quickstart` pulls torch, torchvision, SAM3 (pinned git SHA),
Grounding DINO (transformers), and the CPU geometry deps — pinned by the
committed `uv.lock`. Model weights (`facebook/sam3`,
`IDEA-Research/grounding-dino-base`) download from HuggingFace on first
call. (pip equivalent: see the [main README](../../README.md#installation-details).)

## Results (measured live)

10 trials, seeds 1-10, LIBERO `libero_object_all_variance/0`
("pick up the alphabet soup and place it in the basket"), Gemini
`gemini-3.1-flash-lite-preview` as the VLM, A100 GPU:

| Seed | Grasp (`target_held`) | Task success | Failure mode |
|---|---|---|---|
| 1 | pass | **yes** | — |
| 2 | fail | no | target perception mis-ID (grasped at a wrong-object location) |
| 3 | pass | no | container perception (degenerate basket OBB) → place miss |
| 4 | pass | **yes** | — |
| 5 | pass | **yes** | — |
| 6 | pass | **yes** | — |
| 7 | pass | **yes** | — |
| 8 | pass | **yes** | — |
| 9 | pass | no | container perception (oversized basket OBB) → place miss |
| 10 | pass | **yes** | — |

- **Grasp success: 9/10 (90%)** — the single grasp failure was a target
  mis-identification upstream, not a grasp-mechanics failure; every
  correctly-perceived target was grasped.
- **End-to-end task success: 7/10 (70%)** — both place misses trace to
  basket-perception errors, which the planner variant shares (same
  perception subgraphs), so CuRobo would not have recovered them.
- Wall-clock ~25-55 s per trial on one A100 (models stay resident across
  trials in one process; first trial pays model load).

Per the release plan's decision rule (adopt the planner variant as the
default if direct-IK grasp success < 80%), direct-IK at 90% stays the
quickstart default.

## VLM provider

The perception pipeline disambiguates DINO detections with a hosted VLM
(`vlm.query`). The default provider is Anthropic:

```bash
export ANTHROPIC_API_KEY=...           # default provider; model override:
export GAP_VLM_MODEL=claude-opus-4-8   # optional
```

Alternative — Vertex AI with application-default credentials (what the
results above were measured with):

```bash
gcloud auth application-default login
export GAP_VLM_PROVIDER=vertex
export GAP_VLM_PROJECT_ID=<your-project>
export GAP_VLM_REGION=global
export GAP_VLM_MODEL=gemini-3.1-flash-lite-preview
```

Any OpenAI-compatible endpoint also works (`GAP_VLM_PROVIDER=openai` +
`GAP_VLM_BASE_URL` + `GAP_VLM_MODEL`).

## The planner variant

To run `graph_planner/`, install CuRobo first (CUDA build), then point
`gap run` at it:

```bash
CUDA_HOME=/usr/local/cuda uv sync --extra grocery
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph_planner \
  --sim libero_object_all_variance/0
```

## Graph anatomy

```
START → target → container → grasp → transport → done
            ↘ not_found      ↘ failed   ↘ blocked → abort (open gripper, go home)
```

- **target_sg** (`perceiving-objects-multiview`): three perception paths on
  the same observation — DINO-box+SAM3, point+SAM3, and DINO+VLM-pairwise
  selection — merged by KD-tree multiview fusion, VLM-arbitrated, then
  `geometry.filter_and_compute_obb` fits the target's oriented bounding box.
- **container_sg** (same bundle): DINO+VLM perception of the basket →
  container OBB.
- **grasp_sg** (`grasping-direct-ik` in `graph/`): `robot.open_gripper` →
  `geometry.top_down_grasp_candidates(obb)` → compute the align pose 15 cm
  above the OBB top → `robot.go_to_pose` (rotate at altitude) →
  `robot.go_to_pose` straight down to `candidates.poses[0]` →
  `robot.close_gripper`.
  In `graph_planner/` this is instead: approach above the target →
  re-observe → `geometry.build_world_config` (collision scene from RGB-D,
  target carved out by its SAM3 mask) → `curobo.plan_to_grasp_poses` over
  the full candidate set → `robot.execute_trajectory`.
- **transport_sg** (`transporting-objects`): drop pose from the container
  OBB → lift + lateral `robot.go_to_pose` waypoints → descend, open,
  `robot.go_home`.

Scripts under `graph*/scripts/` are per-workflow copies of the canonical
bundle scripts in open-robot-skills (`skills/<bundle>/scripts/`); the subgraph's
`skill:` field names the owning bundle so scripts import under its package
(prompt loading, sibling imports).

## Next steps

- [build_a_graph](../build_a_graph/) — author this same graph in Python
  with `gap.builder`.
- [generate_a_graph](../generate_a_graph/) — let the LLM pipeline author it
  from one instruction.
- [grocery_fulfillment](../grocery_fulfillment/) — the generated-graph
  acceptance benchmark built on the same skills.
