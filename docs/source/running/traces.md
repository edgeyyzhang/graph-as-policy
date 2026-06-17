# Traces and the Trial Browser

Every GaP run records a full trace by default: the executed workflow, every
node's resolved inputs and outputs, every tool sub-call, and auto-extracted
visual assets (camera frames, masks, point clouds). The on-disk layout is a
stability contract consumed by two tools — the `gap viz` browser and
`gap trace-diff` — so you can debug a rollout long after the process exited.

Where traces go:

- `gap run` writes to `./outputs/run_<timestamp>` unless you pass
  `--trace-dir PATH`; `--no-trace` disables tracing entirely.
- Programmatically, the precedence is the `trace_dir` argument >
  the `GAP_TRACE_DIR` environment variable > the workflow directory itself.

## Trial directory anatomy

A *trial* is one executed run. Its directory looks like this:

```text
outputs/run_20260612_141503/
├── workflow.json            # snapshot copy of the executed workflow
├── scripts/                 # snapshot copy of its script files
├── dag_trace.json           # node timings, statuses, edges, event log
├── run_video.mp4            # sim only, with `gap run --record-video`
└── node_data/
    ├── target_sg.perceive/  # one dir per node, id = <subgraph>.<node>
    │   ├── resolved_inputs.json
    │   ├── output.json
    │   ├── request.json         # recorded tool request (tool nodes)
    │   ├── request.meta.json    # {"tool": "<name>"} — replay metadata
    │   ├── assets/              # auto-extracted visual data
    │   │   ├── output_rgb.png
    │   │   ├── output_mask.png
    │   │   └── output_cloud.npz
    │   ├── calls/               # ctx.tool(...) sub-calls from scripts
    │   │   └── 000_query/       # <seq:03d>_<short tool name>
    │   │       ├── request.json
    │   │       ├── request.meta.json
    │   │       ├── response.json
    │   │       └── assets/
    │   └── stream_reads/        # observation-stream .latest() reads
    │       └── 000_observation_stream/
    │           ├── value.json
    │           └── meta.json
    └── grasp_sg.close/
        └── ...
```

The pieces:

- **`workflow.json` + `scripts/`** — a copy-once snapshot of what actually
  ran, so the trace stays interpretable after you edit the source graph.
- **`dag_trace.json`** — the structured trace: a `nodes[]` list (name, node
  type, start/finish timestamps, `duration_ms`, status
  `pending|running|ok|error|skipped`, condition results, error messages,
  asset names), an `edges[]` list, and a sequenced `events[]` log
  (`node_started`, `node_finished`, `inputs_recorded`, `output_recorded`,
  `subcall_recorded`, `stream_read`, `condition_evaluated`,
  `error_recorded`, ...).
- **`node_data/<id>/`** — per-node payloads. Node ids inside subgraphs are
  fully qualified as `<subgraph>.<node>`. `request.json` +
  `request.meta.json` capture the exact tool request for replay; each
  `ctx.tool(...)` call made by a script gets its own numbered directory
  under `calls/`.
- **Assets** — image-, mask-, depth- and pointcloud-shaped numpy values are
  extracted structurally to files: `*_rgb.png` (uint8 HxWx3), `*_mask.png`,
  `*_depth.npy` (float32), `*_cloud.npz` (positions + optional colors).

:::{warning}
`dag_trace.json` is flushed when the run reaches an **end node**. On a
crash path (an exception that never reaches an end node), `node_data/`
contents are present but `dag_trace.json` is not written — `gap viz` will
show the trial with an empty trace, falling back to the static graph.
:::

### Large-array summarization

Trace JSON stays human-sized: any ndarray with **100 or more elements** is
collapsed to the string `"<ndarray shape=(480, 640, 3) dtype=uint8>"`, and
`bytes` values to `"<bytes len=N>"`. Real pixel and point data lives only in
the extracted `assets/` files. Arrays under 100 elements (poses, joint
vectors, OBB corners) are inlined as lists.

For high-rate observation streams, `GAP_TRACE_STREAM_SAMPLE_N` (default `1`)
persists every Nth `.latest()` read; `0` disables stream-read recording. See
[Environment variables](../reference/environment-variables.md).

## The trial browser: gap viz

```bash
gap viz                      # scan ./outputs recursively, serve on :9432
gap viz --root path/to/runs --port 9432 --no-browser
gap viz --services ../open-robot-skills   # enrich port schemas (see below)
```

`gap viz` starts a local FastAPI server (default `127.0.0.1:9432`) and opens
your browser. Anything under `--root` that contains a `dag_trace.json` (an
executed trial) or a `workflow.json` (a static graph with an empty trace) is
discovered as a trial; copies under `codegen` directories are skipped.

### Three views

::::{grid} 1 1 3 3

:::{grid-item-card} State Machine
The workflow graph: subgraphs expand and collapse on click, with control
edges, data edges, and typed ports. Best for "what is this graph?"
:::

:::{grid-item-card} Execution
Swimlanes — one column per subgraph, sequence-ordered step cards with status
dots and transition arcs. Best for "what happened, in what order?"
:::

:::{grid-item-card} 3D Scene
An embedded viser replay of the rollout: animated arm(s), camera frusta,
point clouds, OBB wireframes. Requires the trial to contain `scene_log/`
data.
:::

::::

The browser opens on the Execution view when the trial has recorded steps,
otherwise on the State Machine. If a trace has no event stream (e.g. a
crashed run), a degraded banner appears and step ordering falls back to
coarse timestamps.

### Per-node inspection

Clicking a node or step card opens the inspector with tabs for:

- **Summary** — sequence, duration, asset counts, plus clickable data
  provenance ("inputs sourced from" / "outputs consumed by" the producing
  and consuming steps).
- **Inputs / Output** — the recorded `resolved_inputs.json` and
  `output.json`; static port schemas are shown when no runtime JSON exists.
- **Script** — the script or router source as it ran.
- **Calls** — expandable cards per `ctx.tool` sub-call with request/response
  JSON and inline image assets (zoomable lightbox).

:::{note}
Videos are not played in the browser. `gap run --record-video` saves
`<trace-dir>/run_video.mp4`, but the viz UI renders PNG/JPEG assets only —
open the mp4 from disk with your video player.
:::

Node replay (`POST /api/node/{id}/replay`) re-issues a node's saved tool
request through the live tool registry. It is best-effort: requests whose
large arrays were summarized in the trace are replayed with placeholder
strings, so replay works best for tools with compact inputs (poses, names,
parameters). The 3D Scene replay runs as a separate viser server on port
8890; only one replay server runs at a time.

### Multiple trials

When the root contains more than one trial, a sidebar rail lists them and
the browser loads the first. If you call the REST API directly
(`/api/trial`, `/api/workflow`, ...), you must disambiguate with
`?trial=<path>` — with multiple trials and no `trial` parameter the API
returns HTTP 400 ("Multiple trials available — specify ?trial=<path>").
Benchmark output trees can surface many trials this way, since every
directory with a `dag_trace.json` counts.

### Port schemas with --services

Node tooltips and the inputs/output fallback views show typed port schemas
resolved from the tool registry. Without `--services`, only built-in `@tool`
plugins are registered — connector tools (`robot.*`, `sim.*`) and bundle
tools show no schemas, and replaying their nodes fails with a tool-not-found
error. Point `--services` at your
[open-robot-skills](../skills/registries.md) checkout to register bundle
tools and get full schemas.

## Generation traces

[`gap generate`](../authoring/generation.md) records its own trace inside
the emitted workflow directory, under `agent_traces/` — every prompt and raw
LLM response from the pipeline:

```text
<workflow_dir>/agent_traces/
├── _coordinator/
│   ├── system_prompt.md
│   └── llm_response_attempt_0.md      # one file per retry attempt
├── <subgraph_name>/                   # one dir per generated subgraph
│   ├── system_prompt.md
│   ├── subgraph_spec.json
│   └── llm_response_attempt_0.md
├── _checkpoint_agent/
│   ├── system_prompt.md
│   ├── user_prompt.md
│   └── llm_response_attempt_0.md
└── _validation_fix/
    └── attempt_0/                     # per validation-fix round
        ├── <script>_prompt.txt
        ├── <script>_response.txt
        └── <script>_fixed.py
```

When a generated graph misbehaves, read these alongside the execution trace:
the execution trace tells you *what* the graph did; `agent_traces/` tells
you *why the LLM authored it that way* (and what the validator made it fix).

## Comparing runs: gap trace-diff

`gap trace-diff` aligns two `dag_trace.json` files by **node name** and
reports where they agree — the backbone of the rehearse-in-sim,
run-on-hardware workflow:

```console
$ gap trace-diff outputs/rehearsal outputs/real_run --out diff.json
$ echo $?
1
```

Both arguments may be trace directories or the `dag_trace.json` files
themselves. Per matched node it reports:

- **verdict agreement** — both `ok`, or both not-`ok`;
- **status agreement** — exact status string match;
- **condition agreement** — same `condition_result.met` outcome (or both
  absent).

The summary includes the aggregate agreement rates, the **first divergence**
(in trace A's DAG order), and any nodes present in only one trace
(rehearsal-only / real-only — reported, never matched). Because alignment is
by node name, both runs must come from the same `workflow.json`; comparing
different graphs yields zero matched nodes.

Exit codes are CI-friendly:

| Code | Meaning |
|---|---|
| `0` | At least one matched node and verdict agreement is 1.0 |
| `1` | Some matched node disagreed on verdict |
| `2` | Zero matched nodes (different graphs; a missing `dag_trace.json` raises an error instead) |

`--out PATH` writes the structured diff as JSON; `--quiet` suppresses the
human-readable summary. The same machinery is available programmatically as
`gap.runtime.trace_diff.diff_trace_dirs(a, b)` with `format_summary()` /
`to_json_dict()` renderers.

## Next steps

- [Executing graphs](execution.md) — how runs produce these traces.
- [Checkpoints](checkpoints.md) — ground-truth verification recorded alongside the trace.
- [CLI reference](../reference/cli.md) — full flag listings for `gap run`, `gap viz`, and `gap trace-diff`.
