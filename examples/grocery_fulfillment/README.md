# grocery_fulfillment — the acceptance benchmark

> **What:** The flagship acceptance family: graphs LLM-generated per task, nothing hand-written · **Needs:** `grocery` (CUDA) + LLM key · **Time:** minutes (smoke) · **Measured:** 10/10 dev gate (2026-06-11)

The flagship task family: pick a described grocery item and place it in the
basket, under the variational-automation benchmark's baked pose /
permutation / basket-swap variations. The graphs are **generated per task**
by `gap.agent.generate` (`llm_generation` mode); nothing is hand-written.
This is the release gate: a release must clear **≥90% success** on the full
config.

## Run it

```bash
CUDA_HOME=/usr/local/cuda uv sync --extra grocery   # quickstart set + CuRobo planning
uv run gap skills check --download

# the 20-trial smoke (tasks 0-1)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance_smoke.yaml --gate

# the full 500-trial acceptance gate (interruptible; --resume continues)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

You also need an LLM credential for graph generation — the reference runs
used Vertex (`gemini-3.1-flash-lite-preview`, as the original); `anthropic`
works by switching the config's `llm:` block.

Measured: **10/10** on the 10-task × 1-seed development gate (2026-06-11);
the full 500-trial gate is the release procedure.

## The recipe (ported verbatim)

[`examples/benchmark/grocery_acceptance.yaml`](../benchmark/grocery_acceptance.yaml):
10 tasks × 50 trials, per-task curated object descriptors
(`target` / `expected_label` / `shape_hint` — these hints are appended to
the codegen prompt and are load-bearing), 600 s trial timeout, video on.

## The sample graph — direct top-down grasp

[`sample_generated_graph/`](sample_generated_graph) began as a real
`gap.agent.generate` output for task 0 ("Pick the blue and yellow alphabet
soup can and place it in the basket"): the coordinator decomposed the task
into `perceiving-objects` (target + basket) → grasp → `transporting-objects`,
the subgraph agents authored the inner state machines + scripts, and the
checkpoint agent attached `validate=True` postconditions (held-after-grasp,
placed-in-basket) that `gap.execute(checkpoints="warn"|"raise")` enforces
against simulator ground truth at every subgraph exit.

Its grasp subgraph (`grasp_sg`) has since been **hand-modified to a direct
top-down grasp**. The CuRobo planner stack — `build_world` (collision model)
→ `plan` (trajectory optimization) → `execute_trajectory` — is replaced by a
single `robot.go_to_pose` to the top-down pose from
`geometry.top_down_grasp_candidates`:

```
open → compute_grasp → go_to_pose(top-down, z_approach=0.10) → observe → close
```

The subgraph's inputs/outputs (`ee_pose_at_grasp`, `grasp_pose`) and its
`grasp_sg` checkpoints are untouched, so perception, transport, and the
postconditions still apply. Run it directly — no generation needed (still
needs the perception VLM credential):

```bash
MUJOCO_GL=egl uv run gap run examples/grocery_fulfillment/sample_generated_graph \
  --sim libero_object_all_variance/0 --checkpoints warn
```

**Why direct?** On the can it matches the planner (reward 1.0) and runs
~20 s faster (no world-build + trajectory opt). On a low-profile object —
e.g. the cream-cheese box (task 1) — the planner can fail outright: CuRobo
rejects every grasp candidate because the box's top-down pose sits at/below
the table (read as a "table collision"), whereas the direct move just goes
there and the gripper pinches the box. The trade-off is no collision
avoidance, so it relies on an uncluttered straight-line approach.

## Next steps

- [benchmark](../benchmark/) — all benchmark configs, the gate semantics,
  and resume behavior.
- [generate_a_graph](../generate_a_graph/) — the generation pipeline this
  benchmark drives at grid scale.
