# LIBERO Quickstart

:::{note}
**Requirements:** a GPU, the `quickstart` extra, and an API key for a hosted
VLM (OpenRouter by default — [alternatives below](#vlm-provider)). Each trial
takes ~25–55 s on an A100.
:::

This is GaP's end-to-end hero example: a static, fully-authored workflow
graph that perceives a target object and a container with real vision models
(Grounding DINO + SAM3 + a hosted VLM), derives a top-down grasp from the
fused 3D oriented bounding box, picks the object with a direct-IK
align-then-descend grasp, and places it in the basket — verified by the
sim's own success predicate and a ground-truth `target_held` checkpoint.

```{image} ../_static/quickstart_rollout.gif
:alt: LIBERO rollout — the arm picks the alphabet soup can and places it in the basket
:width: 640px
:align: center
```

The graph lives at
[examples/libero_quickstart](gh-engine:examples/libero_quickstart) in the
engine repo. Nothing in it is generated at run time: every node, edge, and
script is on disk, so you can read exactly what the robot will do before it
does it.

## Run it

From the GaP checkout, with [open-robot-skills](gh-skills:.) cloned next to
it (see [Installation](../getting-started/installation.md)):

```bash
uv sync --extra quickstart            # engine + LIBERO sim + perception deps
uv run gap skills check --download    # validate every bundle (PASS/WARN/FAIL)
export OPENROUTER_API_KEY=...         # VLM provider — alternatives below

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                        # browse the recorded trial
```

A few things to know:

- `MUJOCO_GL=egl` is required on every sim run command (headless rendering).
- The open-robot-skills checkout is auto-discovered: `$GAP_SKILLS_PATH` or
  the checkout next to the graph-as-policy checkout. `--skills PATH`
  overrides; see [Registries](../skills/registries.md).
- `--validate-only` checks the graph against the skill registry without
  executing anything.
- `--checkpoints warn` (the default) evaluates the ground-truth postcondition
  checkpoints in `graph/checkpoints/` after each subgraph — sim only. The
  `grasp_sg.target_held` checkpoint verifies via privileged contacts that
  the gripper actually holds the can after `close`, not an empty grip.
- Traces (per-node inputs and outputs, every model call, rendered frames)
  land in `outputs/run_<timestamp>/` — see
  [Browse the trace](#browse-the-trace).

Model weights (`facebook/sam3`, `IDEA-Research/grounding-dino-base`)
download from HuggingFace on the first model call; the `--download` flag
runs each bundle's optional `prefetch()` hook for bundles that define one.
The `quickstart` extra pulls torch, torchvision, SAM3 (pinned to a git SHA),
Grounding DINO (transformers), and the CPU geometry deps — all pinned by the
committed `uv.lock`.

## Graph anatomy

```text
START → target → container → grasp → transport → done
            ↘ not_found      ↘ failed   ↘ blocked → abort (open gripper, go home)
```

```{image} ../_static/quickstart_graph.png
:alt: Rendered quickstart workflow graph — four subgraphs routed to done or abort
:width: 680px
:align: center
```

Four subgraphs, each implementing a skill recipe from open-robot-skills:

- **target_sg**
  ([perceiving-objects-multiview](gh-skills:skills/perceiving-objects-multiview)):
  three perception paths run on the same observation — DINO-box+SAM3,
  point+SAM3, and DINO+VLM-pairwise selection — merged by KD-tree multiview
  fusion with VLM arbitration, producing the target's oriented bounding box
  and mask.
- **container_sg** (same bundle): DINO+VLM perception of the basket, then
  `geometry.filter_and_compute_obb` fits the container OBB.
- **grasp_sg**
  ([grasping-direct-ik](gh-skills:skills/grasping-direct-ik)):
  `robot.open_gripper` → `geometry.top_down_grasp_candidates(obb)` →
  compute the align pose 15 cm above the OBB top → `robot.go_to_pose`
  (rotate to the grasp yaw at altitude) → `robot.go_to_pose` straight down
  to `candidates.poses[0]` → `robot.close_gripper`.
- **transport_sg**
  ([transporting-objects](gh-skills:skills/transporting-objects)): drop pose
  from the container OBB → lift to 0.45 m and move laterally with
  `robot.go_to_pose` waypoints → descend, open, `robot.go_home`.

Dataflow between subgraphs is bound **by name**: `grasp_sg` declares an
input `target_obb`, and the executor binds it to the upstream output of the
same name from `target_sg` — no explicit wiring. Each subgraph declares an
`on_error` exit (`not_found`, `failed`, `blocked`), so any raised exception
routes to the `abort` end node, which runs recovery actions
(`robot.open_gripper`, then `robot.go_home`) before exiting with failure.

Scripts under `graph*/scripts/` are per-workflow copies of the canonical
bundle scripts in open-robot-skills (`skills/<bundle>/scripts/`); the
subgraph's `skill:` field names the owning bundle so scripts import under
its package. See [Workflow schema](../reference/workflow-schema.md) for the
JSON format and [Patterns](../authoring/patterns.md) for the recipes used
here.

:::{tip}
`time.sleep()` does **not** advance the LIBERO sim. The graph instead passes
`settle_steps` to the gripper tools (40 on `open`, 60 on `close`) so physics
settles before the next node runs.
:::

## Two variants

| Graph | Grasp strategy | Needs |
|---|---|---|
| `graph/` (**default**) | direct-IK align-then-descend ([grasping-direct-ik](gh-skills:skills/grasping-direct-ik)): pre-rotate to the grasp yaw at a safe height above the OBB, descend straight down, close | `--extra quickstart` |
| `graph_planner/` | CuRobo goal-set planning over the same grasp candidates ([grasping-with-planner](gh-skills:skills/grasping-with-planner)): reconstruct a collision world from RGB-D, plan a collision-free joint trajectory to the best reachable candidate | `--extra grocery` (CUDA build, see below) |

The two graphs differ **only** in the grasp subgraph; perception and
transport are identical. In `graph_planner/` the grasp becomes: approach
above the target → re-observe → `geometry.build_world_config` (collision
scene from RGB-D, with the target carved out by its SAM3 mask) →
`curobo.plan_to_grasp_poses` over the full candidate set →
`robot.execute_trajectory`. The default `graph/` is deliberately
planner-free so the quickstart needs no CuRobo install.

To run the planner variant, install CuRobo first (CUDA build), then point
`gap run` at it:

```bash
CUDA_HOME=/usr/local/cuda uv sync --extra grocery
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph_planner \
  --sim libero_object_all_variance/0
```

## VLM provider

The perception pipeline disambiguates DINO detections with a hosted VLM (the
`vlm.query` tool). The default provider is OpenRouter:

```bash
export OPENROUTER_API_KEY=...                       # default provider; model override:
export GAP_VLM_MODEL=gemini-3.1-flash-lite-preview  # optional (this is the default)
```

Alternative — Vertex AI with application-default credentials (what the
measured results below used):

```bash
uv sync --inexact --extra vertex       # the vertex SDK (google-genai)
gcloud auth application-default login
export GAP_VLM_PROVIDER=vertex
export GAP_VLM_PROJECT_ID=<your-project>
export GAP_VLM_REGION=global
export GAP_VLM_MODEL=gemini-3.1-flash-lite-preview
```

The default `openrouter` provider is OpenAI-compatible, so pointing at any
other such endpoint (e.g. a local vLLM server) just needs a base-URL override:

```bash
export GAP_VLM_BASE_URL=...            # your endpoint (e.g. local vLLM)
export GAP_VLM_MODEL=...               # model served at that endpoint
export GAP_VLM_API_KEY=...             # only if the endpoint requires one
```

The full variable list is in
[Environment variables](../reference/environment-variables.md); provider
setup for graph *generation* (a separate LLM call path) is covered in
[LLM providers](../authoring/llm-providers.md).

## Measured results

10 trials, seeds 1–10, LIBERO `libero_object_all_variance/0` ("pick up the
alphabet soup and place it in the basket"), Gemini
`gemini-3.1-flash-lite-preview` as the VLM, one A100 GPU:

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
- Wall-clock ~25–55 s per trial on one A100. Models stay resident across
  trials within one process, so the first trial pays the model-load cost.

Per the release plan's decision rule (adopt the planner variant as the
default if direct-IK grasp success < 80%), direct-IK at 90% stays the
quickstart default.

### Seed semantics

Each seed deterministically selects one of the task's baked initial object
layouts (`(seed - 1) % n_inits` over the suite's 50 baked init states), so
seeds 1–10 cover ten distinct scenes — same convention the
[benchmark harness](../benchmarks/benchmarking.md) uses for its
`n_seeds` trials. From Python you can pin a seed via the connector:

```python
conn = gap.connector.sim("libero", task="libero_object_all_variance/0", seed=3)
```

:::{warning}
A single green run — or ten — is not a success-rate claim. Claims need
`gap benchmark <yaml> --gate`; see [Benchmarking](../benchmarks/benchmarking.md).
:::

## Browse the trace

Every run writes a trace directory (default `outputs/run_<timestamp>/`;
override with `--trace-dir`). Inside:

- `dag_trace.json` — the full execution record: node order, routing
  decisions, timings, checkpoint results.
- `node_data/<subgraph>.<node>/` — per-node `resolved_inputs.json`,
  `output.json`, tool `calls/`, and `assets/` (camera frames, perception
  masks rendered as PNGs).
- `workflow.json` and `scripts/` — a copy of exactly what was executed.

Launch the trace browser:

```bash
uv run gap viz                         # serves http://127.0.0.1:9432
```

It scans `outputs/` recursively for trials and shows the graph swimlane,
per-node inputs/outputs, and image assets (camera frames, masks). Add
`--record-video` to the run command to also save
`<trace-dir>/run_video.mp4`, which you watch from disk with any video
player. To compare two runs node-by-node, use `gap trace-diff <a> <b>`.
More in [Traces](../running/traces.md).

## Next steps

- [Build a Graph](build-a-graph.md) — author this same graph in Python with
  `gap.builder`.
- [Generate a Graph](generate-a-graph.md) — let the LLM pipeline author it
  from one instruction.
- [Agent Quickstart](agent-quickstart.md) — have Claude Code drive the whole
  loop for you.
- [Grocery Fulfillment](grocery-fulfillment.md) — the generated-graph
  acceptance benchmark built on the same skills.
- [Checkpoints](../running/checkpoints.md) — how ground-truth postcondition
  verification works.
