# gap — graph as policy

**The policy is the graph.** gap compiles a natural-language task into a
typed, verified execution graph of robot skills — and runs the graph, not a
black-box policy, on simulators and real robots.

```{image} ../_static/quickstart_rollout.gif
:alt: quickstart rollout on LIBERO — perceive, grasp, transport
:width: 49%
```
```{image} ../_static/quickstart_graph.png
:alt: the executed workflow graph
:width: 49%
```

*"pick up the soup can and put it in the basket" — the live rollout, and the
graph that ran it.*

```python
import gap

conn = gap.connector.sim("libero", task="libero_object/0")        # one process, no extra terminals
result = gap.execute("examples/libero_quickstart/graph", conn)    # skills auto-discovered
graph = gap.agent.generate_sync("pick up the soup can and put it in the basket")
print(graph)                                                      # the policy, as a graph
gap.viz.serve("outputs")                                          # browse the trial trace
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 🚀 Installation
:link: ../getting-started/installation
:link-type: doc
One `uv sync` per setup — engine-only on any laptop, sim + perception on a GPU box.
:::

:::{grid-item-card} ⏱️ The 15-minute tour
:link: ../getting-started/quickstart
:link-type: doc
Zero to a verified rollout on LIBERO, with the recorded trace open.
:::

:::{grid-item-card} 🧪 Examples
:link: ../examples/index
:link-type: doc
Ten examples, from a CPU-only hello-world to the release gate and real robots.
:::

:::{grid-item-card} 🧩 Skill catalog
:link: ../skills/skill-catalog
:link-type: doc
Contributable perception, grasping, transport, tracking, and policy skills.
:::

:::{grid-item-card} ⌨️ CLI reference
:link: ../reference/cli
:link-type: doc
`gap run`, `generate`, `check`, `registry`, `skills`, `benchmark`, `viz`, and more.
:::

:::{grid-item-card} 🤖 Use with Claude Code
:link: ../agents/claude-code
:link-type: doc
One plugin install teaches AI coding agents to drive the whole workflow.
:::
::::

## Why gap

- 🧭 **Language → typed graph.** A coordinator → subgraph-agents →
  checkpoint-agent pipeline compiles one instruction into a validated
  workflow of skills, with a script-fix loop on validation errors.
- ✅ **Verified, not hoped.** LLM-authored postcondition checkpoints are
  enforced against simulator ground truth at every subgraph exit.
- 🧩 **Skills are contributable.** Strategies and model tools live in
  [open-robot-skills](https://github.com/graph-robots/open-robot-skills) as
  Agent Skills bundles — one directory, one PR. The LLM composes them; you
  can too.
- ⚡ **One process, no servers.** Env + vision models + IK in-process; plain
  numpy + TypedDicts as the data contract. Ray is an opt-in extra, never a
  prerequisite.
- 🔍 **The trace is the product.** Every run records `workflow.json`,
  `dag_trace.json`, per-node I/O and assets; `gap viz` browses them,
  `gap trace-diff` compares them.
- 📊 **Benchmarked with a gate.** A grid harness with `--gate`: the
  grocery-fulfillment acceptance config must clear **≥90% success** for a
  release.

## Measured results

Every number is measured on the engine repo at the committed `uv.lock`:

| Benchmark | Result |
|---|---|
| [Quickstart](../examples/libero-quickstart.md) — 10 seeded LIBERO trials | **9/10 grasp**, 7/10 end-to-end (~25–55 s/trial, one A100) |
| [Acceptance](../examples/grocery-fulfillment.md) — 10-task development gate | **10/10**, graphs LLM-generated per task |
| [Release gate](../examples/benchmark.md) — 10 tasks × 50 trials | must clear **≥90%** before any release |

## Architecture

```text
┌────────────────────────── gap (engine) ──────────────────────────┐
│ agent/      instruction ─► coordinator ─► subgraph agents        │
│             ─► checkpoint agent ─► validate / fix ─► graph       │
│ runtime/    executor (super-steps, streaming, Send), validator,  │
│             tracing, policy loop, verify/ (World, checkpoints)   │
│ tools/      ToolRegistry: typed schemas, tags→guards, dispatch   │
│ connector/  sim()/real(), env registry, in-process pyroki IK,    │
│             rr_launcher, data collector                          │
│ envs/       libero (+perturbed), franka_real, ur_zed, msgpack    │
│ benchmark/  grid harness (modes × families × seeds), --gate      │
│ viz/        FastAPI+React trial browser, replay3d, PDF render    │
└──────────────┬───────────────────────────────▲───────────────────┘
               │ discovers (by path)           │ registers tools
        ┌──────▼───────────────────────────────┴──────┐
        │ open-robot-skills (Agent Skills format)     │
        │ tools/   sam3, grounding-dino, gemini-er,   │
        │          molmo, vlm, curobo, geometry       │
        │ skills/  perceiving-*, grasping-*,          │
        │          transporting-objects,              │
        │          tracking-objects, pi05-libero,     │
        │          molmoact-libero                    │
        └─────────────────────────────────────────────┘
```

Read more in [Overview](../getting-started/overview.md), or jump straight to
[Installation](../getting-started/installation.md).

```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 1

../getting-started/overview
../getting-started/installation
../getting-started/quickstart-cpu
../getting-started/quickstart
../getting-started/concepts
```

```{toctree}
:hidden:
:caption: Examples
:maxdepth: 1

../examples/index
```

```{toctree}
:hidden:
:caption: Running Graphs
:maxdepth: 1

../running/execution
../running/environments
../running/checkpoints
../running/traces
```

```{toctree}
:hidden:
:caption: Generating & Authoring
:maxdepth: 1

../authoring/generation
../authoring/llm-providers
../authoring/builder
../authoring/patterns
```

```{toctree}
:hidden:
:caption: Skills & Tools
:maxdepth: 1

../skills/registries
../skills/skill-catalog
../skills/tool-catalog
../skills/authoring-bundles
../skills/testing-bundles
```

```{toctree}
:hidden:
:caption: Benchmarks & Policies
:maxdepth: 1

../benchmarks/benchmarking
../benchmarks/policies
```

```{toctree}
:hidden:
:caption: Real Robots
:maxdepth: 1

../real-robots/connectors
../real-robots/safety
```

```{toctree}
:hidden:
:caption: AI Agents
:maxdepth: 1

../agents/claude-code
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 1

../reference/cli
../reference/api
../reference/workflow-schema
../reference/executor
../reference/connector-tools
../reference/environment-variables
../reference/benchmark-config
../reference/faq
```

```{toctree}
:hidden:
:caption: Developers
:maxdepth: 1

../developers/architecture
../developers/contributing
../developers/roadmap
```
