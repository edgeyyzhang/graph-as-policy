# Overview

**The graph is the policy.** GaP compiles a natural-language task into a
typed, verified execution graph of robot skills — and runs the graph, not a
black-box policy, on simulators and real robots.

```python
import gap

conn = gap.connector.sim("libero", task="libero_object/0")        # one process, no extra terminals
result = gap.execute("examples/libero_quickstart/graph", conn)    # skills auto-discovered
graph = gap.agent.generate_sync("pick up the soup can and put it in the basket")
print(graph)                                                      # the policy, as a graph
gap.viz.serve("outputs")                                          # browse the trial trace
```

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

## The policy is the graph

Most robot stacks hide the policy: it is a neural network's weights, or an
LLM agent's transient chain of tool calls. In GaP the policy is a durable,
inspectable artifact — a directory containing `workflow.json` (a version-3
JSON graph), `scripts/` (typed Python step bodies), and `checkpoints/`
(postcondition predicates). You can read it, diff it, validate it, version
it, and execute it deterministically.

Three authoring surfaces produce the *identical* artifact, all gated by the
same strict parser and structural validator:

1. **`gap generate`** — an LLM pipeline (coordinator → per-subgraph agents →
   checkpoint agent → validate/fix loop) compiles one instruction into a
   workflow directory. See [Generating graphs](../authoring/generation.md).
2. **`gap.builder`** — a LangGraph-style Python API
   (`Workflow`/`Subgraph`, `add_node`/`add_edge`/`add_exit`/`save`).
   See [The builder API](../authoring/builder.md).
3. **Hand-written JSON** — anything that passes validation. See the
   [workflow schema reference](../reference/workflow-schema.md).

Because the artifact is the contract, execution and verification compose
around it: the executor records a full trace of every run, and
LLM-authored checkpoints are evaluated against simulator ground truth at
every subgraph exit.

:::{figure} ../_static/quickstart_graph.png
:alt: The executed quickstart workflow graph
:width: 75%

The graph behind the quickstart: perceive → grasp → transport, with exit
routing and recovery. Rendered from
[examples/libero_quickstart](gh-engine:examples/libero_quickstart).
:::

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

## Two repos, discovered by path

GaP is split into an engine and a skills library:

- **[graph-as-policy](gh-engine:.)** (import name `gap`) — the engine:
  runtime, validator, agent pipeline, connectors, environments, benchmark
  harness, trace viewer.
- **[open-robot-skills](gh-skills:.)** — the canonical public registry of
  skill bundles in the Anthropic Agent Skills format: one directory per
  bundle, contributable with a single PR.

Clone them side by side and every command finds the skills automatically —
discovery is **by path**, never by pip entry points:

```text
your-workspace/
├── graph-as-policy/       # the engine
└── open-robot-skills/     # skill bundles — auto-discovered, no flags
```

The sibling checkout is only the last layer of a five-level registry
resolution chain (`--skills` flags, `$GAP_SKILLS_PATH`, project
`pyproject.toml`, user config, auto-discovery), so a lab can layer a
private registry that shadows a single public bundle instead of forking
the whole repo. See [Skill registries](../skills/registries.md).

## Tools vs skills

The repo layout mirrors a first-class conceptual split:

- **Tools** (`open-robot-skills/tools/<bundle>/`) are *what the robot can
  compute*: model-backed typed callables with no task strategy. Tool
  bundles are named after the model — `sam3`, `grounding-dino`,
  `gemini-er`, `molmo`, `vlm`, `curobo`, `geometry` — and expose functions
  like `sam3.segment_text` or `curobo.plan_to_pose`.
- **Skills** (`open-robot-skills/skills/<bundle>/`) are *what the robot
  can do*: manipulation strategies that own subgraphs in generated graphs —
  `perceiving-objects`, `grasping-with-planner`, `transporting-objects`,
  `tracking-objects`, the per-checkpoint learned-policy skills
  `pi05-libero` / `molmoact-libero`, and friends. The LLM composes them;
  so can you.

A third tool source ships with the engine itself: **connector tools**
(`robot.*` / `sim.*`), the embodiment surface registered by whichever
simulator or real-robot connector you open. The full taxonomy is in
[Core concepts](concepts.md) and the catalogs in
[Skill catalog](../skills/skill-catalog.md) and
[Tool catalog](../skills/tool-catalog.md).

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
        │ open-robot-skills (Agent Skills format, contributable)
        │ tools/   sam3, grounding-dino, gemini-er,    │
        │          molmo, vlm, curobo, geometry        │
        │ skills/  perceiving-objects(-oneshot/        │
        │          -multiview), perceiving-object-     │
        │          parts, grasping-*, transporting-    │
        │          objects, tracking-objects,          │
        │          pi05-libero, molmoact-libero        │
        └──────────────────────────────────────────────┘
```

Everything runs **in one process**: environment, vision models, and IK.
No gRPC, no protobufs, no self-hosted model servers — the data contract is
plain numpy arrays and TypedDicts. A deeper tour of the packages is in
[Architecture](../developers/architecture.md).

## Why GaP

- **Language → typed graph.** A coordinator → subgraph-agents →
  checkpoint-agent pipeline compiles one instruction into a validated
  workflow of skills, with a script-fix loop on validation errors.
- **Verified, not hoped.** LLM-authored postcondition checkpoints are
  enforced against simulator ground truth at every subgraph exit
  (`--checkpoints warn|raise`).
- **Skills are contributable.** Strategies and model tools live in
  open-robot-skills as Agent Skills bundles — one directory, one PR.
- **One process, no servers.** Env + vision models + IK in-process; Ray is
  an opt-in extra, never a prerequisite.
- **The trace is the product.** Every run records `workflow.json`,
  `dag_trace.json`, and per-node I/O and assets; `gap viz` browses them and
  `gap trace-diff` compares them. See [Traces](../running/traces.md).
- **Benchmarked with a gate.** A grid harness (modes × families × seeds)
  with `--gate`: the grocery-fulfillment acceptance config must clear its
  configured `gate_threshold` for a release. See
  [Benchmarking](../benchmarks/benchmarking.md).

## What you need to try GaP

| Requirement | Details |
|---|---|
| **GPU** | 1× NVIDIA RTX 4090-class GPU (**≥ 24 GB VRAM**), Linux + EGL. |
| **LLM** | An API key for a coding LLM — **OpenRouter** (default, OpenAI-compatible) or Vertex — drives the multi-agent codegen harness. |
| **VLM** | A vision-language model for perception (object identification & grounding), plus local **SAM3 + Grounding DINO** weights. A free [HuggingFace token](https://huggingface.co/settings/tokens) is needed for the gated SAM3 weights. |
| **Tooling** | [uv](https://docs.astral.sh/uv/); the first run downloads ~3.5 GB of model weights. |

## Scope of v1

v1 draws the line at **language → verified pick-and-place graphs on LIBERO
and real Franka/UR**. Cut for scope and planned to return after v1:
execution-feedback graph repair, learned grasp planners, a remote
model-serving tier, bimanual support, Isaac-based rehearsal, non-pick-place
domains (articulated, contact-rich, long-horizon), and self-hosted pointing
VLMs. Details in the [roadmap](../developers/roadmap.md).

## Where to go next

| If you want to… | Go to |
|---|---|
| Install the engine and skills | [Installation](installation.md) |
| Run the sim quickstart | [Quickstart](quickstart.md) |
| Learn the vocabulary (workflow, subgraph, tool, skill, …) | [Core Concepts](concepts.md) |
| Browse runnable examples, hello-world to real robots | [Examples](../examples/index.md) |
| Generate graphs from language | [Generation](../authoring/generation.md) |
| Author graphs in Python | [Builder](../authoring/builder.md) |
| Execute graphs and read traces | [Execution](../running/execution.md) · [Traces](../running/traces.md) |
| Verify runs with checkpoints | [Checkpoints](../running/checkpoints.md) |
| Manage registries and write skill bundles | [Registries](../skills/registries.md) · [Authoring bundles](../skills/authoring-bundles.md) |
| Run benchmark grids and gates | [Benchmarking](../benchmarks/benchmarking.md) |
| Go to real hardware | [Connectors](../real-robots/connectors.md) · [Safety](../real-robots/safety.md) |
| Drive GaP from Claude Code | [Claude Code](../agents/claude-code.md) |
| Look something up | [CLI](../reference/cli.md) · [Python API](../reference/api.md) · [Workflow schema](../reference/workflow-schema.md) |
