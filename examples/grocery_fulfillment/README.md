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

## What gets generated

[`sample_generated_graph/`](sample_generated_graph) is a real, unedited
output of `gap.agent.generate` for task 0 ("Pick the blue and yellow
alphabet soup can and place it in the basket"): the coordinator decomposed
the task into `perceiving-objects` (target + basket) → `grasping-with-planner`
→ `transporting-objects`, the subgraph agents authored the inner state
machines + scripts, and the checkpoint agent attached `validate=True`
postconditions (held-after-grasp, placed-in-basket) that
`gap.execute(checkpoints="warn"|"raise")` enforces against simulator ground
truth at every subgraph exit.

## Next steps

- [benchmark](../benchmark/) — all benchmark configs, the gate semantics,
  and resume behavior.
- [generate_a_graph](../generate_a_graph/) — the generation pipeline this
  benchmark drives at grid scale.
