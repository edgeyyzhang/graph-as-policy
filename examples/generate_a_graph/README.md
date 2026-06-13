# generate_a_graph — instruction → validated workflow

> **What:** Instruction → validated workflow dir; CLI + Python, all providers · **Needs:** `uv sync` + LLM key · **Time:** minutes

The canonical generation flows: one natural-language instruction goes
through the LLM agent pipeline (coordinator → per-subgraph agents →
checkpoint agent → validation + script-fix loop) and comes out as a typed,
verified workflow directory — the **same artifact**
[gap.builder](../build_a_graph/) authors by hand and `gap run` executes.

## CLI

Generation needs only the engine (`uv sync`), the open-robot-skills checkout next
to this repo, and an LLM credential — no GPU, no sim:

```bash
export ANTHROPIC_API_KEY=...    # default provider; see "Providers" below

uv run gap generate "pick up the alphabet soup can and place it in the basket" --out my_graph
```

The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
checkout next to the graph-as-policy checkout); pass `--skills /path/to/open-robot-skills` to
override. `--provider`, `--model`, and `--config <yaml>` tune the
pipeline.

## Python

```python
import gap

graph = gap.agent.generate_sync(
    "pick up the alphabet soup can and place it in the basket",
    out_dir="my_graph",          # skills= omitted -> auto-discovered
)
print(graph)                     # the graph, as terminal text (below)
print(graph.path)                # the written workflow folder
print(sorted(graph.code))        # every generated source file
```

`print(graph)` renders the workflow as box-drawing text — the same output
`gap generate` prints. On the checked-in
[sample_generated_graph](../grocery_fulfillment/sample_generated_graph):

```text
task_00
Pick the blue and yellow alphabet soup can and place it in the basket.

START
  │
  ▼
┌─ target_sg ───────────────────────────────────────── perceiving-objects ─┐
│ observe ─▶ perceive ─▶ filter_obb                                        │
└──────────────────────────────────────────────────────────────────────────┘
  │ found                                                    abort ▶ ✗ abort
  ▼
┌─ container_sg ────────────────────────────────────── perceiving-objects ─┐
│ observe ─▶ perceive ─▶ filter_obb                                        │
└──────────────────────────────────────────────────────────────────────────┘
  │ found                                                    abort ▶ ✗ abort
  ▼
┌─ grasp_sg ─────────────────────────────────────── grasping-with-planner ─┐
│ open ─▶ compute_grasp ─▶ approach ─▶ observe ─▶ build_world ─▶ plan      │
│   ─▶ execute ─▶ close                                                    │
└──────────────────────────────────────────────────────────────────────────┘
  │ grasped                                                  abort ▶ ✗ abort
  ▼
┌─ transport_sg ──────────────────────────────────── transporting-objects ─┐
│ compute_drop ─▶ move_above ─▶ release                                    │
└──────────────────────────────────────────────────────────────────────────┘
  │ placed ▶ ✓ done                                          abort ▶ ✗ abort

✓ done (success)   ✗ abort (failure, recovery: open_gripper, go_home)
```

(`gap.viz.to_text(workflow)` renders any workflow dict or directory the
same way; `gap.agent.generate` is the async variant.)
[generate.py](generate.py) wraps this in a small CLI:

```bash
uv run python examples/generate_a_graph/generate.py "pick up the milk and put it in the basket"
```

## What the output directory contains

```
my_graph/task_00/           # = graph.path
├── workflow.json           # the v3 graph: top-level DAG + one entry per subgraph
├── scripts/                # agent-authored scripts referenced by type:script nodes
│   ├── perceive_dino_vlm.py
│   ├── plan_grasp.py
│   └── ...
├── checkpoints/            # LLM-authored postcondition sidecars per subgraph
│   ├── grasp_sg.py         #   (validate=True predicates, enforced against
│   └── transport_sg.py     #    sim ground truth at subgraph exit)
├── agent_traces/           # per-agent LLM transcripts for debugging
└── multi_agent_meta.json   # pipeline provenance (models, retries, timings)
```

`gap.agent.generate_sync` returns the exact folder as `graph.path`.
[grocery_fulfillment/sample_generated_graph](../grocery_fulfillment/sample_generated_graph)
is a checked-in, unedited example of one.

## Validate + run the result

```bash
uv run gap run my_graph/task_00 --validate-only           # structural + skill checks
MUJOCO_GL=egl uv run gap run my_graph/task_00 \
  --sim libero_object_all_variance/0                      # execute on LIBERO
uv run gap viz                                            # browse the trace
```

Or in Python: `gap.execute(graph.path, gap.connector.sim("libero", task=...))`.

## Providers

| Provider | Setup |
|---|---|
| `anthropic` (default) | `export ANTHROPIC_API_KEY=...`; `--model claude-opus-4-8` is the default |
| `openai` (incl. OpenRouter / vLLM) | `export OPENAI_API_KEY=...`; custom endpoints via a `--config` YAML with `llm: {provider: openai, endpoint: ...}` |
| `vertex` | `gcloud auth application-default login`, `export GOOGLE_CLOUD_PROJECT=...` (`GOOGLE_CLOUD_REGION` optional, default `global`); pick per call (`--provider vertex --model <m>` — no provider default) or once per shell (`export GAP_LLM_PROVIDER=vertex GAP_LLM_MODEL=<m>`), and include the SDK: `uv run --extra vertex gap generate ...`; claude-* and gemini-* models both route correctly |

Pick per call with `--provider/--model`, or pin everything (temperature,
retries, per-agent models) in a config YAML passed via `--config`.

## Benchmark-scale generation

Generating one graph is the unit; the benchmark harness drives the same
pipeline over task × seed grids. See
[grocery_fulfillment](../grocery_fulfillment/) for the flagship recipe —
`gap generate` on grocery instructions under pose / permutation /
basket-swap variations, gated at ≥90% success over 500 trials.
