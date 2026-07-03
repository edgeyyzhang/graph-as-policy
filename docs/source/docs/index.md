# GaP — Graph-as-Policy

:::{admonition} 🧪 GaP Beta Code release (1 July 2026)
:class: important

**GaP is under active development and now in beta testing.** Please send
comments and suggestions to [kych@berkeley.edu](mailto:kych@berkeley.edu) — we
plan to release an updated version by **1 Aug 2026**. Expect rough edges: APIs,
the workflow schema, and skill interfaces may change without notice between
releases.
:::

**The graph is the policy.** GaP targets **Variational Automation (VA)** —
tasks a robot must perform *persistently and reliably across many varying
instances* (objects vary in geometry and pose), not just solve **once**. GaP is
a multi-agent coding harness that compiles a natural-language task into a typed,
verified computation graph of modular skills, self-improves it in simulation,
and runs the graph — not a black-box policy — on simulators and real robots.

<p><em>GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation</em></p>

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

## Abstract

GaP (Graph-as-Policy) targets **Variational Automation (VA)** — tasks a robot
must perform persistently and reliably across many varying instances (objects
vary in geometry and pose), not just solve once. Model-free policies struggle to
close this **reliability gap**. GaP is a multi-agent coding harness that turns a
natural-language task into a **directed computation graph of modular skills**
(MORSL), self-improves it through simulation rehearsal, and ships it to an edge
device for persistent execution — evaluated across **8 Variational Automation
benchmarks (4 sim + 4 real)**; results are on the
[project page](https://graph-robots.github.io/graph-as-policy-anonymous/).

**Highlights**

- We identify **Variational Automation (VA)** — a class of tasks, between fixed
  automation and full generalist robotics, where a robot persistently performs
  varying instances of a task with non-trivial variation in object geometry and
  pose (e.g., sort packages, make coffee, build sandwiches).
- **Graph-as-Policy (GaP)** represents a robot policy as a directed computation
  graph of modular perception, planning, and control nodes — harnessing the
  open-world adaptivity of LLM coding agents while preserving an interpretable,
  reliable structure.
- A hierarchical multi-agent harness decomposes a natural-language task, has
  Skill Agents synthesize localized subgraphs from the **Modular Open Robot
  Skill Library (MORSL, 51 initial skills)**, and wires them into a type-checked
  executable graph.
- GaP rehearses graphs in an internal **Isaac simulation** across parallel
  sampled task instances, uses physical contact and state feedback to localize
  failures to specific nodes, and iteratively refines graph topology and
  parameters before deployment.
- GaP is evaluated on **8 new open VA benchmarks (4 sim + 4 real)** against
  VLA, TAMP, and single-agent code-as-policy baselines — see the
  [project page](https://graph-robots.github.io/graph-as-policy-anonymous/)
  for the results.

## What is Variational Automation — and why GaP is different

Many agentic-coding papers solve a task **once**. GaP targets **Variational
Automation (VA)**: a robot must perform *varying instances* of a task —
persistently and reliably — with non-trivial variation in the geometry and pose
of objects, inside a known, bounded workcell. VA sits **between fixed automation
and full generalist robotics**:

- **Fixed Automation (FA)** — persistently repeats *identical* motions (spot
  welding, spray painting). High reliability, zero adaptivity.
- **Variational Automation (VA)** — persistently performs *varying* instances
  (different SKUs, poses, arrangements) within a known workcell. **This is GaP's
  target.**
- **Generalist Robotics (GR)** — open-ended tasks via model-free end-to-end VLA
  policies. Flexible, but not yet at commercial / industrial reliability.

:::{admonition} Why not just single-agent Code-as-Policy?
:class: note

Single-agent Code-as-Policy (CaP) prompts a coding agent to emit free-form
Python. For persistent, repeated execution this is unstructured: the context
window grows, constraints are hard to obey, and agents are prone to
hallucination and "cheating" on success metrics. GaP's directed computation
graph revives the structure proven in the Robot Operating System (ROS) — routing
data through an explicit graph to manage dependencies and ensure reliable
execution — and layers the open-world adaptivity of pretrained LLM coding agents
on top, preserving an interpretable policy that stays reliable across a VA
task's many instances.
:::

## The grocery benchmark family, runnable here

The two simulation benchmarks from the paper's grocery family ship in this
repo as runnable examples — the same graphs, suites, and configs:

- **[Grocery Fulfillment](../examples/grocery-fulfillment.md)** — pick a
  *described* grocery item into the basket under pose / permutation /
  basket-swap variations; every graph is LLM-generated per task, and the
  full config is the release gate.
- **[Grocery Packing](../examples/grocery-packing.md)** — pack *every*
  item with a loop: a static graph with a real backward edge and
  unprivileged VLM termination, plus the `gap generate` recipe that
  reproduces it from one sentence.

Quantitative comparisons against VLA, TAMP, and code-as-policy baselines
live on the
[project page](https://graph-robots.github.io/graph-as-policy-anonymous/);
this documentation stays with what you can run and inspect.

## What you need to try GaP

| Requirement | Details |
|---|---|
| **GPU** | 1× NVIDIA RTX 4090-class GPU (**≥ 24 GB VRAM**), Linux + EGL. |
| **LLM** | An API key for a coding LLM — **OpenRouter** (default, OpenAI-compatible) or Vertex — drives the multi-agent codegen harness. |
| **VLM** | A vision-language model for perception (object identification & grounding), plus local **SAM3 + Grounding DINO** weights. A free [HuggingFace token](https://huggingface.co/settings/tokens) is needed for the gated SAM3 weights. |
| **Tooling** | [uv](https://docs.astral.sh/uv/); the first run downloads ~3.5 GB of model weights. |

## Hello world — your first run

The fastest path is the **[15-minute tour](../getting-started/quickstart.md)**:
clone the two repos side by side, sync, install the skill bundles, set one LLM
key, and run the hero LIBERO graph end-to-end with the trace open.

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # sibling, auto-discovered
cd graph-as-policy
uv sync                                   # engine + LIBERO sim
uv run gap skills install --all           # per-bundle venvs (sam3, cuRobo, vlm, …)
export HF_TOKEN=...                        # gated SAM3 weights
export OPENROUTER_API_KEY=...             # codegen + in-graph VLM (openrouter is the default)
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph --sim libero_object_all_variance/0
uv run gap viz                            # browse the trial at localhost:9432
```

…or drive the whole workflow from Python:

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
Eleven examples, from the end-to-end quickstart to the release gate and real robots.
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
