# The 15-minute tour

Zero to a verified rollout, with the trace open. You will:

- build and **picture** a graph on any machine (CPU, no API key);
- run the checked-in quickstart graph on the LIBERO sim (GPU);
- read the recorded trace — the artifact everything else revolves around;
- browse it in `gap viz`;
- generate your own graph from one sentence.

Install per [README › Get started](../README.md#get-started) — any of the
`uv sync` sets works for the first stop; from the GPU step on you need the
[hardware floor](../README.md#get-started) and an LLM key.

## Minute 0–2: hello_graph (CPU)

```bash
uv run python examples/hello_graph/hello.py    # → outputs/hello_graph/graph.png
```

(Fresh no-GPU machine? Plain `uv sync` is enough — but don't re-run it bare
after syncing an extra set: uv syncs *exactly*, so a bare `uv sync` removes
the extras again.)

Open `outputs/hello_graph/graph.png` and you are looking at the four ideas
gap is built on:

- **Workflow** — the whole picture: a typed DAG, stored as
  `workflow.json`. This *is* the policy.
- **Subgraph** — each grey box: the `perceive` box runs the `perceive_sg`
  subgraph, owned by the `perceiving-objects` **skill** (a strategy bundle
  from [open-robot-skills](https://github.com/graph-robots/open-robot-skills)); the `grasp` box runs `grasp_sg`, owned by
  `grasping-direct-ik`. Subgraph IDs and owners live in `workflow.json`;
  subgraphs declare typed inputs/outputs and named exits.
- **Node** — the colored boxes: blue `tool` calls dispatch typed functions
  (`robot.open_gripper`, `geometry.top_down_grasp_candidates`); green
  `script` nodes run per-graph code that calls tools.
- **Routes** — green/red arrows: the executor follows exit values
  (`found` → grasp, `not_found` → abort). Failure handling is graph
  structure, not exception spaghetti.

The artifact on disk is exactly what the LLM pipeline emits:

```
outputs/hello_graph/
├── workflow.json    # version: 3, meta, nodes, edges, conditional_edges, subgraphs
├── scripts/         # per-graph script nodes (copied canonical bundle scripts)
└── graph.png        # the render — your copy, not part of the artifact
```

`workflow.json` opens with the graph's identity and is fully diffable:

```json
{
  "version": 3,
  "meta": {
    "name": "hello_graph",
    "description": "Perceive the blue and yellow alphabet soup can, then grasp it."
  },
  ...
}
```

The full schema and executor semantics live in [runtime.md](runtime.md);
the [hello_graph README](../examples/hello_graph/README.md) shows the
builder calls that produced it.

## Minute 2–8: run the quickstart graph (GPU)

The same kind of artifact, executing for real — vision models, IK, and the
sim in one process:

```bash
export ANTHROPIC_API_KEY=...          # or another provider, see "LLM providers"

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
```

While it runs (~25–55 s), what you are watching in the log: perception
(Grounding DINO proposes boxes, a hosted VLM picks the right one, SAM3
segments it), geometry (mask + depth → oriented bounding box → top-down
grasp candidates), then motion (align, descend, close, transport).

When it finishes, the **trace** is in `outputs/run_<timestamp>/`:

```
outputs/run_<timestamp>/
├── workflow.json        # the graph that ran (self-contained copy)
├── dag_trace.json       # every node visit: timings, exits, errors, routing
├── node_data/<node>/    # per-node inputs/outputs + extracted assets
└── scripts/             # the script nodes, as executed
```

`node_data/` is where debugging lives: the perception node's directory
holds the camera frames it saw (PNG), the masks and point clouds it
produced (NPZ), and the VLM exchange; the grasp nodes record the poses
they targeted. If a run fails, the failing node's directory shows exactly
what it was looking at when it failed.

Two flags worth knowing now: `--validate-only` checks a graph (structure +
skill registry) without touching a sim, and `--checkpoints warn|raise`
controls how ground-truth postcondition checkpoints are enforced at
subgraph exits.

## Minute 8–10: browse it

```bash
uv run gap viz                        # browse the recorded trial at localhost:9432
```

Click into the trial: the graph renders as swimlanes (one per subgraph)
with the executed route highlighted; clicking a node shows its inputs,
outputs, timings, and assets — the same `node_data/` you just saw, with
images inline (benchmark runs additionally collate rollout videos next to
their summaries). When two runs disagree,
`gap trace-diff <trial_a> <trial_b>` diffs them structurally.

## Minute 10–15: generate your own

```bash
uv run gap generate "pick up the alphabet soup can and place it in the basket"
```

The agent pipeline — coordinator → per-subgraph agents → checkpoint
agent — picks skills from the open-robot-skills catalog, writes the
subgraphs and their scripts, attaches ground-truth checkpoints, and
validates the result; on validation errors a script-fix loop repairs its
own output (you'll see those attempts in the log). The artifact lands in
`outputs/generated_<timestamp>/task_00/` — the same shape as hello_graph's,
so you already know how to read it. Run it:

```bash
MUJOCO_GL=egl uv run gap run outputs/generated_<timestamp>/task_00 \
  --sim libero_object_all_variance/0
```

## Where next

| You want to | Go to |
|---|---|
| See everything gap can do | [examples gallery](../examples/README.md) |
| Author graphs in Python | [examples/build_a_graph](../examples/build_a_graph/) |
| Write or contribute a skill | [skills.md](skills.md) + [open-robot-skills](https://github.com/graph-robots/open-robot-skills) |
| Run the benchmark gate | [examples/benchmark](../examples/benchmark/) |
| Move a real robot | [safety.md](safety.md) **first**, then [examples/real_franka_pick_place](../examples/real_franka_pick_place/) |
