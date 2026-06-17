# Core Concepts

This page defines the vocabulary the rest of the documentation uses. Skim
the table, then read the sections for the concepts you will touch first.
The normative specifications live in the reference section:
[workflow schema](../reference/workflow-schema.md) and
[executor semantics](../reference/executor.md).

| Concept | One-line definition |
|---|---|
| **Workflow / graph** | The v3 JSON policy artifact: a top-level DAG of nodes plus named subgraphs. |
| **Node** | One unit in a graph: `tool` \| `script` \| `router` \| `subgraph` \| `noop` \| `end`. |
| **Subgraph** | A self-contained inner graph owned by one skill, with typed inputs/outputs, exit declarations, and optional `on_error`. |
| **Tool** | The unit of *execution*: one typed callable dispatched by flat name through the ToolRegistry. |
| **Skill** | The unit of *packaging, discovery, and LLM context*: an Agent Skills bundle (`SKILL.md` + scripts/prompts/references). |
| **Connector** | The embodiment: owns a simulator or robot link and registers the `robot.*`/`sim.*` tools. |
| **Checkpoint** | An authored postcondition predicate over a world snapshot, evaluated when its subgraph exits. |
| **Trial / trace** | One execution of a graph, and the on-disk record it produces. |
| **Registry** | A local directory of skill bundles; multiple registries layer by precedence. |

## Workflow (graph)

A **workflow** is the policy artifact: a directory containing

```text
workflow.json      # the v3 graph: nodes, edges, conditional_edges, subgraphs
scripts/           # Python bodies for script nodes (paths relative to the dir)
checkpoints/       # one optional sidecar module per subgraph (see below)
```

`workflow.json` has exactly these top-level keys: `version` (must be `3`),
`meta` (`name`, `description`), `nodes`, `edges`, `conditional_edges`, and
`subgraphs`. Parsing is **strict**: any unknown key anywhere is a hard
error, and legacy v2 constructs are rejected with targeted migration
messages rather than silently misread.

Control flow is explicit. `START` and `END` are virtual node names —
`["START", "x"]` edges seed execution, an edge to `END` marks a scope's
terminal node, and neither is ever declared. Multiple outgoing edges from
one node run their targets concurrently in the same scheduler *super-step*;
data flows per input via `{"$ref": "node.field.subfield"}` references
rather than shared channels.

The top level of a typical generated workflow is small — subgraph nodes,
end nodes, and conditional edges that route on each subgraph's exit value:

```json
{
  "version": 3,
  "meta": {"name": "pick_and_place"},
  "nodes": {
    "grasp": {"type": "subgraph", "ref": "grasp_sg"},
    "done":  {"type": "end", "status": "success"},
    "abort": {"type": "end", "status": "failure",
              "recovery": [{"tool": "robot.open_gripper"},
                           {"tool": "robot.go_home"}]}
  },
  "edges": [["START", "grasp"]],
  "conditional_edges": {
    "grasp": {"router_field": "exit",
              "mapping": {"grasped": "done", "failed": "abort"}}
  },
  "subgraphs": {"grasp_sg": {"...": "..."}}
}
```

## Node types

| Type | Requires | What it does |
|---|---|---|
| `tool` | `tool` | Calls one registered tool by flat name (`robot.close_gripper`, `sam3.segment_text`, …). |
| `script` | `script` | Runs a Python module from `scripts/` defining a typed `run(ctx, ...)` that returns a TypedDict (or `None`). |
| `router` | `script` | Like `script`, but it returns `{"route": ...}` and the route steers control flow: a string mapped through `conditional_edges`, or a list of `{"to", "inputs"}` *Send* dicts for dynamic fan-out. |
| `subgraph` | `ref` | Enters a named subgraph from the top-level `subgraphs` map. Top level only. |
| `noop` | — | Passthrough that returns `{}`; used as a named success terminal inside subgraphs. |
| `end` | `status` | Terminates the workflow with `"success"` or `"failure"`; may carry best-effort `recovery` tool calls (open gripper, go home). Top level only. |

`tool` and `script` nodes may set `streaming: true`: the node is spawned
detached, never blocks readiness, must have no outgoing edges, and
publishes snapshots that consumers read transparently via `$ref` — that is
how a camera feed coexists with a step-by-step graph.

## Subgraphs, exits, and `on_error`

A **subgraph** is an inner graph owned by one skill. Its declaration
carries:

- `skill` — the owning skill bundle (e.g. `perceiving-objects`); this is
  what places the bundle's SKILL.md in front of the generating LLM.
- `inputs` — typed declarations (`{"target_name": "str"}`); nodes inside
  reference them as `{"$ref": "in.target_name"}`. Type names resolve
  through the `gap.schema` registry (`Observation`, `Se3Pose`,
  `OrientedBoundingBox`, scalars, …).
- `outputs` — named `$ref` bindings into inner-node results. Dataflow
  *between* subgraphs is by name matching: a subgraph input is satisfied by
  any upstream subgraph declaring a same-named output.
- `exit` — how the subgraph reports what happened, and `on_error` — what it
  reports when something raises.

The exit declaration has two modes:

- `"router_field": null` — the exit value is the *name* of the terminal
  node reached; `success_values` must name declared `noop` nodes
  (the common case: `add_exit("found")` in the builder).
- `"router_field": "some_field"` — the exit value is read from that field
  of the terminal node's output; success values must *not* be node names.

`on_error` is a symbol, never a node: any exception inside the subgraph is
caught and surfaced as that exit value, so the top-level graph can route
failures to a recovery `end` node. Without `on_error`, an exception crashes
the whole workflow. The resolved exit value must be in
`success_values ∪ {on_error}`. At the top level, conditional edges on a
subgraph node route on `"router_field": "exit"`.

Subgraphs do not nest, and `$ref`s cannot cross a subgraph boundary —
all coupling goes through declared inputs and outputs.

## Tools

A **tool** is the unit of execution: one typed callable dispatched by flat
name through the engine's `ToolRegistry`, either as a `type: tool` node or
from inside a script via `ctx.tool(name, **kwargs)`. Inputs and outputs are
introspected from type hints into a schema (`gap tools show <name>` prints
it live). Tools come from three sources sharing one namespace:

- **Connector tools** — `robot.*` and `sim.*`. These prefixes are reserved:
  only a connector can claim them, so a graph can never invent motion. They
  are the embodiment surface (`robot.get_observation`, `robot.go_to_pose`,
  `sim.step`, …) shipped by the engine, not by skill bundles. See the
  [connector tool reference](../reference/connector-tools.md).
- **Bundle tools** — `<bundle>.<func>`, e.g. `sam3.segment_text`,
  `curobo.plan_to_pose`, `geometry.iou`. Defined with the `@tool` decorator
  in a tool bundle's `tools.py`; the bundle name is the model name.
- **Plugin tools** — your own `@tool` functions, registered the same way.

Tools can carry tags (`perception`, `planning`, `sim_step`) that subject
them to call-count guards (`GAP_MAX_PERCEPTION_CALLS` and friends) — a
runaway graph hits a limit instead of burning API quota or sim time.

Graph `script` nodes are *neither* tools nor skills: they are per-graph
generated code that calls tools.

## Skills

A **skill** is the unit of packaging, discovery, and LLM context: a bundle
directory in the Agent Skills format. Its `SKILL.md` carries YAML
frontmatter (`name` must equal the directory name; `description` says when
to use it) plus gap extensions under a single `gap:` key — allowed tools,
exit conditions, typed `required_inputs`/`produces_outputs`, canonical
scripts, prompts, and runtime requirements. The markdown body is guidance
the generating LLM reads verbatim.

The bundle's *kind* is determined by its folder, never by frontmatter:

- `tools/<bundle>/` — a **tool bundle**: model-backed `@tool` functions.
  Tool bundles never own subgraphs.
- `skills/<bundle>/` — a **skill bundle**: a strategy that owns subgraphs
  in generated graphs, bundling canonical scripts (emitted as
  `type: script` nodes) and LLM guidance; some are also callable as a
  single unit.

So "tool vs skill" is *what the robot can compute* vs *what the robot can
do*. See [Authoring bundles](../skills/authoring-bundles.md) for the full
format.

## Connectors

A **connector** is the embodiment object: it owns a simulator environment
or a real-robot link and registers the `robot.*`/`sim.*` tools that graphs
execute against. You create one and pass it to `gap.execute`:

```python
conn = gap.connector.sim("libero", task="libero_object/0")   # simulator
conn = gap.connector.real("franka")                          # real hardware
```

The same graph runs on either — but capabilities differ by design:
simulator connectors expose `sim.*` tools and a privileged world snapshot
(which is what checkpoints evaluate against); real connectors register no
`sim.*` tools, and a perception-only connector like `ur_zed` registers no
motion tools at all, making unintended motion structurally impossible. See
[Connectors](../real-robots/connectors.md) and
[Safety](../real-robots/safety.md).

## Checkpoints

A **checkpoint** is an authored postcondition: a Python predicate over a
privileged `World` snapshot (optionally also the subgraph's outputs),
evaluated when its subgraph exits. Checkpoints live in sidecar modules —
`checkpoints/<subgraph_name>.py` exporting a `CHECKPOINTS` list — and are
never referenced from `workflow.json`; the executor auto-loads them by
path.

Two flavors:

- **Hard checkpoints** (`validate=True`, the default) are enforced
  according to the checkpoint mode: `off`, `warn` (default, log and
  continue), or `raise` (fail the run with `VerificationFailed`). Select
  the mode with `gap run --checkpoints …` or
  `gap.execute(..., checkpoints=...)`.
- **Probes** (`validate=False`) surface in feedback but never gate
  anything; the executor skips them entirely at enforcement time.

Checkpoints are never evaluated on an `on_error` exit (the subgraph already
declared failure), and when the connector exposes no world snapshot — real
robots — enforcement degrades to a single warning and the run proceeds.
See [Checkpoints](../running/checkpoints.md).

## Trials and traces

A **trial** is one execution of a graph; a **trace** is its on-disk record.
Tracing is on by default — `gap run` writes `outputs/run_<timestamp>/`:

```text
dag_trace.json     # enriched node metadata: timing, status, edges, events
workflow.json      # copy of the executed workflow (plus scripts/)
node_data/<id>/    # per-node resolved inputs, outputs, ctx.tool sub-calls,
                   # stream reads, and extracted assets (PNG masks/images,
                   # NPY depth, NPZ point clouds)
```

This layout is a stability guarantee consumed by `gap viz` (the trial
browser) and `gap trace-diff` (structural comparison of two trials).
Benchmarks aggregate success over many seeded trials. See
[Traces](../running/traces.md).

## Registries

A **registry** is a local directory containing `tools/` and/or `skills/`
bundle roots — [open-robot-skills](gh-skills:.) is the canonical public
one. Several registries can be active at once, resolved by precedence:

1. `--skills PATH` flags (repeatable) — full override
2. `$GAP_SKILLS_PATH` (an OS-pathsep-separated list) — full override
3. the nearest `pyproject.toml` with `[tool.gap].registries`
4. `~/.config/gap/registries.toml`, managed by `gap registry add/remove`
5. an `open-robot-skills` checkout next to the engine (auto-discovered)

Layers 3–5 merge; the first two replace everything. When two registries
ship a bundle with the same name, the higher-precedence one wins with a
loud warning — so a lab registry can shadow a single public bundle instead
of forking the repo. See [Registries](../skills/registries.md).

## How the pieces fit

One sentence end to end: `gap generate` asks an LLM to compose **skills**
into a **workflow** of **subgraphs** whose nodes call **tools**; you
execute it against a **connector** with `gap run`, the executor enforces
**checkpoints** at every subgraph exit, and the **trial trace** is what you
debug with `gap viz`. Try it in the [15-minute tour](quickstart.md).
