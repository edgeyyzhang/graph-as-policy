# zero_shot_prompts — the grocery_fulfillment zero-shot prompt matrix

> **What:** the six zero-shot tester prompts, fed verbatim to
> `gap.agent.generate`, each judged by a programmatic **object-state
> criterion** over simulator ground truth · **Needs:** `grocery` (CUDA)
> + LLM credential (generation and the perception VLM) · **Time:**
> ~10–30 min/case (one generation + 1–3 executions).

The prompts and their **baseline** column come from a manual test pass
(observed outcomes and suspected causes, quoted in `cases.py`). This
suite replaces eyeballed verdicts with measured ones: it records the
ground-truth world state around **every tool call** during execution and
scores each prompt on what physically happened — which objects were
actually grasped and lifted, what ended up inside the basket, how far
and in which direction things moved, whether the can is still upright —
never on the model's own success claims and never on the env's built-in
reward (that measures a different task; it is recorded as advisory
metadata only).

## The matrix

| case | prompt | tester baseline | measured criterion (abridged) |
|---|---|---|---|
| `soup_basket` | "Pick up the alphabet soup can and place it in the basket" | Success, robust to position variations | per seed: soup **grasped-and-lifted** ∧ finishes settled in the basket ∧ nothing else packed; runs on **3 seeds** (layouts genuinely differ) and passes only if all pass |
| `all_objects` | "Pick up all objects and place them in the basket" | Failure: only the soup; blamed on "GaP is DAG-only, task needs cycles" | 6/6 settled in basket (score = fraction); per-object grasp events; **diagnostics: does the generated graph contain cycles + did the executor actually revisit nodes at runtime** |
| `soup_then_milk` | "…soup in the basket, then …milk in the basket" | Failure: hangs at milk grasp (mesh loading suspected) | two distinct grasp events ∧ both packed ∧ soup enters basket strictly first; a wall-clock **hang is a first-class outcome** — the killed run is scored from its flushed partial trace and the last in-flight tool call is reported as the hang site |
| `two_inches_right` | "…place it down two inches to the right" | Failure: picks up, then knocks it down | grasped ∧ final Δy = 5.1 ± 3 cm (either y direction: **+y is image-right in the agentview video — measured from the camera frame**; −y is the robot's right; direction logged) ∧ Δx ≤ 5 cm ∧ **still upright** (initially-up body axis ≤ 30° from vertical; knocked-over ≈ 90°) ∧ released on the table |
| `cream_cheese` | "Pick up the cream cheese and put it in the basket" | Failure: CuRobo can't find a collision-free plan; oversized bounding box from `geometry.top_down_grasp_candidates` | grasped-and-lifted ∧ settled in basket ∧ nothing else packed; **diagnostics: planner tool errors + every `geometry.*` numeric output** for comparison against the ground-truth extents (measured: 8.3 × 4.3 × 2.0 cm — it really is a low object) |
| `move_basket_ranch` | "Move the basket to the opposite side, then pick up the ranch and place it in the basket" | (no outcome recorded) | basket displaced ≥ 20 cm with a lateral side-flip (or ≥ 30 cm in x) ∧ ranch (`salad_dressing`) grasped ∧ settled inside the **relocated** basket (containment follows the basket's live pose); runs on seed 1 where the basket starts clearly on one side |

### Extended matrix (new capability probes + more VAB tasks/suites)

Seven further cases probe axes the tester matrix didn't — all within
the Variational-Automation-Benchmark's own grocery task families
(`third_party/Variational-Automation-Benchmark/tasks/`), whose
vocabulary spans 10 items (alphabet_soup, bbq_sauce, butter,
chocolate_pudding, cream_cheese, ketchup, milk, orange_juice,
salad_dressing, tomato_sauce): each variance task carries a different
6-item subset + basket, and the basket-swap suite relocates the basket
across inits. Every scene below was probed live before its criterion
was written:

| case | VAB task | prompt | probes | criterion (abridged) |
|---|---|---|---|---|
| `stack_butter_on_cream_cheese` | all_variance/0 | "Stack the butter on top of the cream cheese" | precision placement | butter resting ON the cream cheese (`is_on_strict`-style window, centroid inside base), base not dragged |
| `all_except_milk` | all_variance/0 | "Put all the objects except the milk into the basket" | negation | the 5 non-milk objects in the basket ∧ **milk stays out** |
| `soup_left_of_milk` | all_variance/0 | "Place the alphabet soup to the left of the milk" | spatial relation | soup settled 6–30 cm from the milk along y (either reading of "left"; side logged), ≤ 15 cm in x, on the table |
| `both_boxes` | all_variance/0 | "Put both boxes in the basket" | category grounding | exactly the two box-shaped items (butter, cream_cheese) in the basket; cans/bottles/cartons stay out |
| `pudding_basket` | all_variance/3 | "Pick up the chocolate pudding and place it in the basket" | different task scene + new object | chocolate_pudding picked and settled in the basket (scene: pudding, OJ, bbq sauce, ketchup, dressing, soup) |
| `oj_swap_scene` | target_basket_swap_variance/5 | "Pick up the orange juice and place it in the basket" | basket-swap suite, 2 seeds | orange_juice picked and settled in the basket across two basket placements |
| `pack_scene_all` | libero_object_packing/0 | "Pick all the objects and place them in the basket" *(the suite's own task language, verbatim)* | native packing suite | all 6 movables settled in the basket; the suite's `pack_all_into` reward recorded as an external cross-check |

Seed coverage in the extended run: `soup_basket` and `cream_cheese` at
3 seeds, `two_inches_right`, `soup_then_milk`, and `oj_swap_scene` at
2, the rest at 1 (the basket-move case stays on seed 1 where "opposite
side" is well-defined).

Every executed seed also passes through `scene_sane`: no object may ever
leave the workspace envelope (flung off the table / through the floor);
a violation fails the seed regardless of task progress.

## Run it

```bash
# criteria self-test on synthetic traces (no sim, no LLM, ~1 s)
uv run python examples/zero_shot_prompts/selftest.py

# one case
MUJOCO_GL=egl uv run python \
    examples/zero_shot_prompts/run_prompt.py \
    --case soup_basket --out outputs/zero_shot_prompts/dev

# the whole matrix (one subprocess per case, per-case timeout,
# resumable: re-run with the same --out to skip finished cases)
MUJOCO_GL=egl uv run python \
    examples/zero_shot_prompts/run_all.py \
    --out outputs/zero_shot_prompts/run1
```

Defaults: provider `vertex`, model `gemini-3.1-pro-preview` (override
with `--provider/--model` or `GAP_LLM_PROVIDER`/`GAP_LLM_MODEL`). If
`GOOGLE_CLOUD_PROJECT` is unset, the runner uses the ADC quota project
from `~/.config/gcloud/application_default_credentials.json`. The runner
also moves `TMPDIR` to the home volume (video frame buffers) and keeps
`GAP_DECIDE_TRUST_ENV` unset — that flag opts a generated decide-loop
into trusting the env's *own* task reward, which is not the prompt's
task.

## How a verdict is produced

```
run_all.py ── subprocess + timeout, killpg on hang ──▶ run_prompt.py
  1. generate      gap.agent.generate_sync(prompt)      → <case>/gen/task_00
  2. per seed      fresh gap.connector.sim(...); StateRecorder wraps
                   conn.tool_registry.invoke (the single dispatch point
                   for every script/tool node call);
                   gap.execute(graph, conn, checkpoints="warn")
                     → seed<k>/events.jsonl   (call/return/error, flushed)
                     → seed<k>/state.jsonl    (ground-truth snapshot per call)
                     → seed<k>/run_video.mp4  (+ per-camera extras)
  3. score         criteria.evaluate_seed(...)          → seed<k>/result_seed.json
                   aggregate (PASS = all seeds pass)    → <case>/result.json
```

- **Hangs**: the `call` event is written *before* dispatch, so a hung
  tool leaves an unmatched call as the last flushed line. `run_all.py`
  kills the process group at the timeout, drops `timeout_marker.json`,
  and re-scores the partial trace (`run_prompt.py --re-eval`) — the
  verdict is `TIMEOUT` with the hang site (tool, node, kwargs) attached.
- **Generation refusals/errors** are their own verdicts (`GEN_REFUSED` /
  `GEN_ERROR`), never silently converted to task failures.
- **Anti-gaming**: "picked up" requires the engine's `held_body()` to
  report the object with the gripper closed *and* the object ≥ 3 cm
  above its initial height — a nudge doesn't count, and an object that
  appears in the basket without ever being carried fails the grasp
  check. Containment mirrors `Body.contains` (±2 cm on the outer AABB —
  the basket registers no interior cavity on LIBERO, probed).
- **Re-scoring**: criteria iterate cheaply over saved traces —
  `run_prompt.py --case X --out DIR --re-eval`. A hand-built graph can
  be baselined without generation via `--graph <workflow_dir>`.

## Measured conventions (probed live on `libero_object_all_variance/0`)

- Snapshots are in the **robot-base frame**, which in this env equals
  the MuJoCo world frame (robot0_base at the origin, identity rotation).
  Tabletop surface ≈ z 0. Movable bodies: alphabet_soup, butter,
  cream_cheese, milk, salad_dressing, tomato_sauce (+ basket).
- **Agentview image-right = +y** (from the camera's frame axes: the
  camera sits at world (0.90, 0, 0.65) looking back toward the robot).
  The robot's right is −y. `two_inches_right` accepts either reading
  of "right" and logs the direction taken.
- The soup can's local z-axis is **horizontal** at rest (asset
  modeling), so uprightness is measured against the body direction that
  initially pointed up, not against local z.
- Initial layouts vary per seed (e.g. the basket spawns at (0.75, 0.03)
  on seeds 0/2 and (0.71, −0.23) on seed 1) — that variation is what
  `soup_basket`'s three-seed robustness gate exercises.
- The executor's own node-visit cap (`GAP_ITERATION_CAP`, default
  10000) exists precisely to bound *repeated node visits*; whether a
  generated graph actually contains cycles, and whether nodes were
  revisited at runtime, are reported per run from the workflow's edge
  list and the executor's `node_data/<node>/iters/` layout.

## Outputs

`<out>/report.md` (rebuild with `run_all.py --out DIR --report-only`)
holds the verdict table plus per-seed checks and diagnostics. Every
claim is auditable: `state.jsonl` (ground truth per tool call),
`events.jsonl` (every call with kwargs/result summaries), `graph.txt` +
`gen/task_00/` (the generated graph), `trace/` (executor trace),
`run_video.mp4` (what a human would have watched).
