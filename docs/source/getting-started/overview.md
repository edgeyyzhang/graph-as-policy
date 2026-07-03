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
