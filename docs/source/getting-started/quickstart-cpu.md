# gap in 2 Minutes (No GPU)

The gap engine runs anywhere. In two commands you will build, validate,
and render a real workflow graph on a laptop — CPU only, no API key, no
simulator — and meet the four ideas everything else in gap is built on.

:::{note} Requirements
The [two repos cloned side by side](installation.md#clone-the-two-repos)
(`graph-as-policy` with `--recurse-submodules`, plus `open-robot-skills`),
and uv. Nothing else: no GPU, no LLM key, no sim.
:::

## Run it

From inside `graph-as-policy/`:

```bash
uv sync                                        # engine only, any OS
uv run python examples/hello_graph/hello.py    # → outputs/hello_graph/graph.png
```

:::{warning}
`uv sync` is exact — if you previously synced an extras set (say
`--extra quickstart`), a bare `uv sync` removes those extras again. On a
machine that already has a fuller install, just skip the first command.
:::

In those two commands, gap:

- assembled a perceive-then-grasp workflow with `gap.builder` — the same
  artifact (`workflow.json` + `scripts/`) the LLM pipeline emits;
- ran it through the structural and skill-registry validation that gates
  every generated graph (the same checks as
  `gap run <graph> --validate-only`);
- rendered the typed graph to a PNG.

## What you are looking at

Open `outputs/hello_graph/graph.png`:

:::{image} ../_static/hello_graph.png
:alt: rendered hello_graph workflow — perceive subgraph routing to a grasp subgraph
:width: 560px
:align: center
:::

This one picture is the four ideas gap is built on:

- **Workflow** — the whole picture: a typed DAG, stored as
  `workflow.json`. This *is* the policy.
- **Subgraph** — each grey box: the `perceive` box runs the `perceive_sg`
  subgraph, owned by the `perceiving-objects` **skill** (a strategy bundle
  from [open-robot-skills](gh-skills:skills/perceiving-objects)); the
  `grasp` box runs `grasp_sg`, owned by `grasping-direct-ik`. Subgraph IDs
  and owners live in `workflow.json`; subgraphs declare typed
  inputs/outputs and named exits.
- **Node** — the colored boxes: blue `tool` calls dispatch typed functions
  (`robot.open_gripper`, `geometry.top_down_grasp_candidates`); green
  `script` nodes run per-graph code that calls tools.
- **Routes** — green/red arrows: the executor follows exit values
  (`found` → grasp, `not_found` → abort). Failure handling is graph
  structure, not exception spaghetti.

## The artifact on disk

What [hello_graph](gh-engine:examples/hello_graph) wrote is exactly what
the LLM pipeline emits:

```text
outputs/hello_graph/
├── workflow.json    # version: 3, meta, nodes, edges, conditional_edges, subgraphs
├── scripts/         # per-graph script nodes (copied canonical bundle scripts)
└── graph.png        # the render — your copy, not part of the artifact
```

`workflow.json` opens with the graph's identity and is fully diffable —
the `nodes`, `edges`, `conditional_edges`, and `subgraphs` keys follow:

```json
{
  "version": 3,
  "meta": {
    "name": "hello_graph",
    "description": "Perceive the blue and yellow alphabet soup can, then grasp it."
  }
}
```

The script under `scripts/` is a canonical script copied verbatim from the
`perceiving-objects` bundle — graphs carry their own copies of the code
they run, so the artifact is self-contained and reproducible.

`hello.py` takes two flags if you want to play: `--out` (default
`outputs/hello_graph`) and `--skills` (default: the auto-discovered
`open-robot-skills` checkout next to the engine repo).

## Where next

The graph you just rendered never executed — there is no robot, no sim,
and no model on this path. But it is the same kind of artifact that runs
for real:

- **Got a GPU?** The [15-minute tour](quickstart.md) runs the quickstart
  graph on the LIBERO simulator — real vision models, IK, and a recorded
  trace — and then generates a graph from one sentence.
- **Want the vocabulary first?** [Concepts](concepts.md) defines
  workflows, subgraphs, nodes, routes, skills, and tools precisely.
- **Want to author graphs like this yourself?**
  [Build a graph](../examples/build-a-graph.md) is the complete
  `gap.builder` walkthrough — checkpoints, recovery actions, and
  execution included; the full schema lives in the
  [workflow schema reference](../reference/workflow-schema.md).
