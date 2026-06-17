# graph-as-policy

**The policy is the graph.** GaP compiles a natural-language task into a
typed, verified execution graph of robot skills — and runs the graph, not a
black-box policy, on simulators and real robots.

<div class="gap-real-grid">
  <figure>
    <video src="../_static/grocery_packing_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Pack grocery items — Franka, 10×.</figcaption>
  </figure>
  <figure>
    <video src="../_static/popcorn_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Make popcorn — long-horizon stove manipulation, 16×.</figcaption>
  </figure>
  <figure>
    <video src="../_static/tool_packing_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Pack tools &amp; chargers into tagged bins.</figcaption>
  </figure>
  <figure>
    <video src="../_static/usb_insertion_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>USB-C cable insertion — UR5 with force feedback.</figcaption>
  </figure>
</div>

*Real-robot rollouts of graphs generated from one-sentence task descriptions.*

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
`uv sync` + `gap skills install --all` — engine, sim, perception, motion planning, each in its own venv.
:::

:::{grid-item-card} ⏱️ The 15-minute tour
:link: ../getting-started/quickstart
:link-type: doc
Zero to a verified rollout on LIBERO, with the recorded trace open.
:::

:::{grid-item-card} 🧪 Examples
:link: ../examples/index
:link-type: doc
Ten examples, from the end-to-end quickstart to the release gate and real robots.
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

## Why GaP

- 🧭 **Language → typed graph.** A coordinator → subgraph-agents →
  checkpoint-agent pipeline compiles one instruction into a validated
  workflow of skills, with a script-fix loop on validation errors.
- ✅ **Verified, not hoped.** LLM-authored postcondition checkpoints are
  enforced against simulator ground truth at every subgraph exit.
- 🧩 **Skills are contributable.** Strategies and model tools live in
  [open-robot-skills](https://github.com/graph-robots/open-robot-skills) as
  Agent Skills bundles — one directory, one PR. The LLM composes them; you
  can too.
- 📦 **Per-bundle isolation, no monolith venv.** Each tool bundle (sam3,
  cuRobo, vlm, openpi, …) runs in its own venv via stdio-msgpack RPC; the
  engine stays ~150 MB. Heavy ML stacks don't fight each other.
- 🔍 **The trace is the product.** Every run records `workflow.json`,
  `dag_trace.json`, per-node I/O and assets; `gap viz` browses them,
  `gap trace-diff` compares them.
- 📊 **Benchmarked with a gate.** A grid harness with `--gate`: the
  grocery-fulfillment acceptance config must clear **≥90% success** for a
  release.

## Two ways in

GaP runs the same workflow artifact however you produce it — by hand or
from language:

- **[Hand-curated quickstart](../examples/libero-quickstart.md)** — the
  reference graph for `libero_object_all_variance/0`. Open a sim,
  perceive, grasp with in-process IK, transport, verify against
  ground truth.
- **[Generate from language](../authoring/generation.md)** — one
  sentence to a validated graph. The coordinator → subgraph-agents →
  checkpoint-agent pipeline picks `perceiving-objects`,
  `grasping-with-planner`, and `transporting-objects` from the skill
  registry; the result runs end-to-end in sim with LLM-authored
  postcondition checkpoints enforced at every subgraph exit.
- **[Release gate](../examples/benchmark.md)** — a grid harness with
  `--gate` for batch evaluation across modes × families × seeds.

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
