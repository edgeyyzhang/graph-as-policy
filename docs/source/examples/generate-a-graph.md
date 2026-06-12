# Generate a Graph

One natural-language instruction goes through the LLM agent pipeline
(coordinator → per-subgraph agents → checkpoint agent → validation +
script-fix loop) and comes out as a typed, verified workflow directory —
the **same artifact** [gap.builder](build-a-graph.md) authors by hand and
`gap run` executes.

:::{note} Requirements
An LLM credential (`ANTHROPIC_API_KEY` by default — see
[Providers](#providers)) plus the engine install (`uv sync`) and the
open-robot-skills checkout. No GPU and no simulator are needed to
*generate*; running the result in sim has the usual
[quickstart requirements](../getting-started/quickstart.md).
:::

Source: [examples/generate_a_graph](gh-engine:examples/generate_a_graph).

## CLI

```bash
export ANTHROPIC_API_KEY=...    # default provider; see "Providers" below

uv run gap generate "pick up the alphabet soup can and place it in the basket" --out my_graph
```

The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or
the checkout next to the graph-as-policy checkout); pass
`--skills /path/to/open-robot-skills` to override (repeatable,
precedence-ordered). Other flags:

| Flag | Default | Meaning |
|---|---|---|
| `--skills PATH` | resolved registry set | Skill registry root(s); repeatable |
| `--provider` | `anthropic` | `anthropic` \| `openai` \| `vertex` |
| `--model` | provider default | LLM model override |
| `--out DIR` | `outputs/generated_<timestamp>` | Output directory |
| `--config YAML` | — | Pipeline config (llm/composition/skills knobs) |
| `-v, --verbose` | off | Debug logging |

:::{warning}
The generated workflow lands in `<out>/task_00/`, not in `--out` itself —
point `gap run` at the task subdirectory (`my_graph/task_00`).
:::

## Python

```python
import gap

graph = gap.agent.generate_sync(
    "pick up the alphabet soup can and place it in the basket",
    out_dir="my_graph",          # skills= omitted -> auto-discovered
)
print(graph)                     # the graph, as terminal text (below)
print(graph.path)                # the written workflow folder (my_graph/task_00)
print(sorted(graph.code))        # every generated source file
```

`print(graph)` renders the workflow as box-drawing text — the same output
`gap generate` prints. On the checked-in
[sample_generated_graph](gh-engine:examples/grocery_fulfillment/sample_generated_graph):

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

`gap.agent.generate` is the async variant with the same signature;
`gap.viz.to_text(workflow)` renders any workflow dict or directory the
same way.

[generate.py](gh-engine:examples/generate_a_graph/generate.py) wraps this
in a small CLI:

```bash
uv run python examples/generate_a_graph/generate.py "pick up the milk and put it in the basket"
```

## What the output directory contains

```text
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

[grocery_fulfillment/sample_generated_graph](gh-engine:examples/grocery_fulfillment/sample_generated_graph)
is a checked-in, unedited example of one (its `workflow.json`, `scripts/`,
and `checkpoints/` — the `agent_traces/` and `multi_agent_meta.json` debug
artifacts are not committed).

## Validate and run the result

```bash
uv run gap run my_graph/task_00 --validate-only           # structural + skill checks
MUJOCO_GL=egl uv run gap run my_graph/task_00 \
  --sim libero_object_all_variance/0                      # execute on LIBERO
uv run gap viz                                            # browse the trace
```

Or in Python:
`gap.execute(graph.path, gap.connector.sim("libero", task=...))`. See
[Execution](../running/execution.md) and [Traces](../running/traces.md).

## Providers

| Provider | Setup |
|---|---|
| `anthropic` (default) | `export ANTHROPIC_API_KEY=...`; the default model is `claude-opus-4-8` |
| `openai` (incl. OpenRouter / vLLM) | `export OPENAI_API_KEY=...` and pass `--model`; custom endpoints via a `--config` YAML with `llm: {provider: openai, endpoint: ...}` |
| `vertex` | `gcloud auth application-default login`, the `vertex` extra, and a `--config` YAML setting `llm: {provider: vertex, project_id: ..., region: ...}`; claude-* and gemini-* models both route correctly |

Only `anthropic` has a default model — for `openai` and `vertex` the
endpoints serve arbitrary models, so set `--model` (or `llm: {model: ...}`
in the config) explicitly.

Pick per call with `--provider`/`--model`, or pin everything (endpoint,
temperature, max tokens, concurrency, per-agent models) in a config YAML
passed via `--config`:

```yaml
llm:
  provider: openai
  model: my-model
  endpoint: http://localhost:8000/v1
```

See [LLM providers](../authoring/llm-providers.md) for the full config
reference, and [Generation](../authoring/generation.md) for how the
multi-agent pipeline works.

## Benchmark-scale generation

Generating one graph is the unit; the benchmark harness drives the same
pipeline over task × seed grids. See
[grocery_fulfillment](grocery-fulfillment.md) for the flagship recipe —
`gap generate` on grocery instructions under pose / permutation /
basket-swap variations, gated at ≥90% success over 500 trials — and
[Benchmarking](../benchmarks/benchmarking.md) for the harness. A single
green run is not a success-rate claim: gate it with
`gap benchmark <yaml> --gate`.
