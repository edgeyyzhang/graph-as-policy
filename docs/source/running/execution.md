# Executing Graphs

A gap graph is a directory: a `workflow.json` plus optional `scripts/` (script-node
bodies) and `checkpoints/` (postcondition sidecars). You execute one of two ways —
`gap run` from the shell, or `gap.execute()` from Python. Both drive the same
executor; the CLI is a thin wrapper that builds a connector, parses inputs, and
picks a trace directory for you.

```bash
# Tools-only (no robot, runs anywhere):
gap run my_graph

# Against a LIBERO sim task:
MUJOCO_GL=egl gap run my_graph --sim libero_object/0

# Validate without executing:
gap run my_graph --validate-only
```

## Run with the CLI

### The graph argument

`gap run GRAPH` accepts either a workflow directory or a direct path to a
`workflow.json`. A directory has `workflow.json` appended automatically:

```bash
gap run examples/libero_quickstart/graph                 # directory
gap run examples/libero_quickstart/graph/workflow.json   # same thing
```

Script-node paths inside the workflow resolve relative to the workflow directory,
so always keep `workflow.json` next to its `scripts/`.

### Choosing a connector

By default `gap run` executes **tools-only**: no robot, no sim — nodes run against
the resolved skill registries and gap's core tools. This is how you run pure
codegen/test graphs like [build_a_graph](../examples/build-a-graph.md) on any machine.

`--sim SUITE/TASK` builds a LIBERO simulation connector, e.g.
`--sim libero_object/0` or `--sim libero_object_all_variance/3`. The value is a
suite name plus a task index; see [Simulators and Environments](environments.md)
for every suite and the task-numbering rules.

:::{note} Requirements
`--sim` needs the `[libero]` extra (Linux + NVIDIA GPU + EGL) and `MUJOCO_GL=egl`
for headless rendering — see [Installation](../getting-started/installation.md).
:::

`--real {franka,ur_zed}` builds a real-hardware connector instead:

- `franka` — full motion control over the robots_realtime msgpack bridge. By
  default gap spawns the `rr-session` client itself; `--no-rr-autostart` restores
  the two-terminal debug flow, and `--rr-config YAML` picks an rr-session config
  (relative to `third_party/robots_realtime`; default
  `configs/franka/franka_robotiq_client.yaml`).
- `ur_zed` — perception-only UR + ZED. No motion tools are ever registered.

:::{warning}
Read the [safety notes](../real-robots/safety.md) before driving hardware.
`--sim` and `--real` are mutually exclusive — passing both aborts the command.
:::

See [Connectors](../real-robots/connectors.md) for the hardware setup behind both
real backends.

### Passing inputs

`--inputs` takes `K=V` pairs and seeds the workflow's initial inputs (the values
nodes reference as `{"$ref": "in.<name>"}`). Each value is parsed as JSON when
possible and kept as a plain string otherwise:

```bash
gap run my_graph --sim libero_object/0 \
  --inputs target_label="alphabet soup" max_attempts=3 use_depth=true offsets='[0.0, 0.02]'
```

Here `max_attempts` arrives as an `int`, `use_depth` as a `bool`, `offsets` as a
list, and `target_label` as a string (it is not valid JSON, so it falls back).
An entry without `=` aborts the command with an error.

### Checkpoint enforcement

`--checkpoints {off,warn,raise}` (default `warn`) controls how postcondition
checkpoints in `checkpoints/<subgraph>.py` are enforced after each subgraph:

- `off` — skip evaluation entirely.
- `warn` — evaluate, log failures, keep running.
- `raise` — abort the run at the first failed `validate=True` checkpoint.

Checkpoints need ground-truth world snapshots, which only sim connectors provide;
on tools-only and real runs enforcement degrades to a one-shot warning and the run
proceeds. See [Checkpoints and Verification](checkpoints.md).

### Traces and video

Tracing is **on by default**: every run writes per-node inputs/outputs, model
calls, and rendered frames into `./outputs/run_<timestamp>` (e.g.
`outputs/run_20260612_153000`). `--trace-dir DIR` redirects it; `--no-trace` turns
the timestamped directory off. Browse recorded runs with `gap viz` — see
[Traces and the Trial Browser](traces.md).

:::{note}
The executor always records a trace. With `--no-trace`, it falls back to
`$GAP_TRACE_DIR` if set, otherwise the workflow directory itself — so the trace
artifacts land next to your `workflow.json` rather than disappearing.
:::

`--record-video` (sim only) saves the run as `<trace-dir>/run_video.mp4`, plus one
`run_video_<camera>.mp4` per camera when the env buffers per-camera frames. It
requires both a sim connector and tracing — `--record-video` without `--sim`, or
combined with `--no-trace`, exits with code 2. On success the command prints
`video: <path> (<n> frames)`. Videos are watched from disk; the `gap viz` UI
renders the trace's image assets, not the mp4.

### Skill registries

`--skills PATH` (repeatable, precedence-ordered) overrides the resolved registry
set entirely. Without it, gap resolves registries via `$GAP_SKILLS_PATH`, the
project's `[tool.gap]` table, the user config, or an `open-robot-skills` checkout
next to the gap checkout — see [Registries](../skills/registries.md).

### Flag reference

| Flag | Default | Meaning |
|---|---|---|
| `GRAPH` (positional) | — | Workflow directory or `workflow.json` path |
| `--sim SUITE/TASK` | off (tools-only) | LIBERO sim connector, e.g. `libero_object/0` |
| `--real {franka,ur_zed}` | off | Real-hardware connector |
| `--rr-config YAML` | `configs/franka/franka_robotiq_client.yaml` | franka only: rr-session config, relative to `third_party/robots_realtime` |
| `--no-rr-autostart` | autostart | franka only: don't spawn rr-session; run it yourself |
| `--skills PATH` | resolved registry set | Skill registry root(s); repeatable, full override |
| `--validate-only` | off | Validate the graph without executing |
| `--no-trace` | tracing on | Don't create the timestamped trace directory |
| `--trace-dir DIR` | `outputs/run_<timestamp>` | Trace output directory |
| `--record-video` | off | Sim only: save `<trace-dir>/run_video.mp4` |
| `--checkpoints {off,warn,raise}` | `warn` | Checkpoint enforcement mode |
| `--inputs K=V ...` | none | Initial workflow inputs, JSON-typed |
| `-v, --verbose` | off | Debug logging |

### Output and exit codes

A run prints a status line, the trace path, one line per checkpoint result, and
the error when the workflow failed:

```console
$ MUJOCO_GL=egl gap run examples/libero_quickstart/graph --sim libero_object_all_variance/0
SUCCESS (exit=success, 42.3s)
trace: outputs/run_20260612_153000
checkpoint: CheckpointResult(name='target_held', subgraph='grasp_sg', passed=True, ...)
```

| Exit code | Meaning |
|---|---|
| `0` | Run succeeded (or `--validate-only` found no errors) |
| `1` | Run failed, validation found error-severity issues, or bad usage (`--sim` with `--real`, malformed `--inputs`) |
| `2` | `--record-video` without `--sim`, or with `--no-trace` |

This matches the CLI-wide convention (0 success, 1 operation failed, 2 usage
error) documented in the [CLI reference](../reference/cli.md).

## Validate without executing

`--validate-only` loads the workflow, resolves the active skill registries, and
runs the structural validator — no connector, no execution, no trace:

```console
$ gap run my_graph --validate-only
OK: 0 errors, 2 warning(s)
```

Every issue prints on its own line; the command exits 0 when there are no
error-severity issues and 1 otherwise (including when `workflow.json` fails to
load). Tools that aren't in any resolved registry degrade to warnings — connector
`robot.*`/`sim.*` tools only exist at run time, so a graph that uses them
validates clean with warnings. Validation rules are listed in the
[workflow schema reference](../reference/workflow-schema.md).

Run `--validate-only` after every hand edit and in CI; it catches dangling edges,
unreachable nodes, bad `$ref` targets, and type mismatches before you spend
sim time.

## Run from Python

`gap.execute()` is the one-call facade behind `gap run`:

```python
import gap

conn = gap.connector.sim("libero", task="libero_object_all_variance/0", seed=1)
try:
    result = gap.execute(
        "examples/libero_quickstart/graph",
        conn,
        trace_dir="outputs/my_run",
        checkpoints="warn",
    )
finally:
    conn.close()

print(result.success, result.exit_status, result.trace_path)
```

The full signature:

```python
gap.execute(
    graph,                  # dir / workflow.json path / dict / builder Workflow
    connector=None,         # None = tools-only
    *,
    skills=None,            # registry root(s); default: resolved registry set
    inputs=None,            # initial inputs, addressable as {"$ref": "in.<name>"}
    trace_dir=None,         # default: $GAP_TRACE_DIR, else the workflow dir
    checkpoints="warn",     # "off" | "warn" | "raise"
    max_node_workers=8,     # thread budget per parallel super-step
) -> ExecutionResult
```

### Accepted graph forms

`graph` accepts four forms:

- a `str`/`Path` to a workflow **directory**,
- a `str`/`Path` to a **`workflow.json`** file,
- a materialized workflow **dict**,
- any object with a `to_dict()` method — i.e. a
  [`gap.builder.Workflow`](../authoring/builder.md).

Dicts and builder objects are written to a temporary directory before execution.
Only `workflow.json` is materialized there, so graphs containing **script nodes
must be passed as a directory path** (or saved with `Workflow.save()` first) so
their `scripts/` resolve.

### Connector, skills, and tracing

`connector=None` runs tools-only against the default `@tool` registry — useful for
tests and dry runs (see [Testing bundles](../skills/testing-bundles.md)). The
connector argument is duck-typed: anything exposing `tool_registry`,
`get_observation`, and optionally `world_snapshot`/`capabilities` works.

`skills` takes one path or a precedence-ordered sequence of registry roots. When
omitted, the same resolution chain as the CLI applies (`$GAP_SKILLS_PATH` →
project `[tool.gap]` → user config → sibling `open-robot-skills` checkout); when
nothing resolves, execution proceeds without a skill registry.

Unlike the CLI, `gap.execute()` does **not** invent a timestamped directory: with
`trace_dir=None` the trace goes to `$GAP_TRACE_DIR` or the workflow directory
itself (for dict/builder graphs, that is the temp dir). Pass `trace_dir=` when you
want the CLI-style layout.

### ExecutionResult

`gap.execute()` **never raises on workflow failure** — exceptions are caught and
returned in the result:

| Field | Type | Meaning |
|---|---|---|
| `success` | `bool` | True iff no exception and the run reached a `success` end node |
| `exit_status` | `str \| None` | `"success"`, `"failure"`, or `None` when no end node was reached |
| `outputs` | `dict` | `{subgraph_name: bound outputs}` |
| `trace_path` | `Path \| None` | Directory the trace was written to |
| `checkpoint_results` | `list` | `CheckpointResult` per evaluated checkpoint |
| `error` | `Exception \| None` | The exception when the workflow crashed |
| `duration_s` | `float` | Wall-clock execution time |

With `checkpoints="raise"`, a failed `validate=True` checkpoint raises
`VerificationFailed` *inside* the run — it aborts the workflow and lands in
`result.error` like any other failure. Safety guards are the one exception to
"never raises": `GuardLimitExceeded` inherits from `BaseException` and propagates
out of `gap.execute()` so nothing can swallow it.

See the [executor reference](../reference/executor.md) for what happens between
START and END.

## Next steps

- [Simulators and Environments](environments.md) — suites, seeds, cameras, video
- [Traces and the Trial Browser](traces.md) — what lands in the trace dir
- [Checkpoints and Verification](checkpoints.md) — writing postcondition sidecars
- [Build a graph in Python](../authoring/builder.md) — author what you just ran
