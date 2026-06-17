# Executor Semantics

This page specifies how the GaP runtime executes a loaded v3 workflow. The
implementation is `WorkflowExecutor` in
[gap/runtime/executor.py](gh-engine:gap/runtime/executor.py); the schema it
consumes is specified in [Workflow JSON Schema](workflow-schema.md), and the
practical how-to-run guide is [Running Graphs](../running/execution.md).

The executor is a **frontier-based super-step scheduler**: each tick it runs
every ready node concurrently, folds the outputs back into the scope's
local-outputs dict, then advances the frontier along edges and conditional
edges. Subgraph nodes recurse into the same scheduler. Streaming nodes are
detached producers consumed by snapshot; Send fan-out spawns dynamic copies
of a target node.

## Entry

`WorkflowExecutor.execute()` (or the `gap.execute(...)` facade — see
[Python API](api.md)):

1. resets guard counters and checkpoint results;
2. runs `validate_workflow` — error-severity issues raise
   `GraphValidationError` before anything executes;
3. initializes the trace (graph topology + a copy of the workflow dir);
4. starts the observation stream when the connector provides a poll fn;
5. runs the top-level scope; the terminal node must be `type: end`.
   Its `status` is recorded as `exit_status`, recovery runs, the trace
   flushes, and a `failure` status raises `PipelineError`.

:::{note}
`WorkflowExecutor.execute()` raises on failure; the `gap.execute(...)`
facade never does — it converts the exception into
`ExecutionResult(success=False, error=...)`. The two entry points also
differ on the checkpoints default: the facade defaults to
`checkpoints="warn"`, the raw executor to `"off"`.
:::

## Scope scheduling (super-steps)

Each scope (the top-level workflow or one subgraph visit) runs the same
loop:

```text
frontier := targets of START edges
loop while frontier nonempty:
    spawn any streaming nodes in frontier (detached; removed from frontier)
    run all remaining ready nodes concurrently           # one super-step
    fold outputs into local_outputs; mark completed
    for each completed node:
        follow static edges  (dst == END ⇒ record node as scope terminal)
        resolve conditional edge (value must be in mapping)
    frontier := the newly-discovered targets
```

- All nodes of one super-step run in a `ThreadPoolExecutor`
  (`max_node_workers`, default 8); a single ready node takes a no-thread
  hot path. The first exception cancels not-yet-started siblings and
  re-raises.
- A node that already completed in this scope is never re-executed —
  re-entry attempts are silently skipped. **Looping is done across
  subgraphs** (each subgraph visit gets a fresh scope), not within one.
- Each scope counts super-steps against `node_visit_cap` (constructor arg,
  else `GAP_ITERATION_CAP`, default 10000) as a runaway-loop guard.
  Termination is otherwise a property of the graph's exit-condition wiring
  — there is no `max_retries` anywhere in the schema.

## Conditional dispatch

After a node completes, its conditional edge (if any) resolves per the
[schema rules](workflow-schema.md#conditional-edges). For subgraph nodes
the canonical `router_field: "exit"` reads the reserved `_exit` key of the
subgraph node's result. The resolved value must be a key in the edge's
`mapping`; anything else is a runtime `PipelineError`. A resolved target of
`END` marks the source as the scope terminal.

## Cross-subgraph data

The executor maintains `cross_subgraph_outputs: {sg_name: {output: value}}`
— **most recent run wins** (a loop's later visit overwrites the earlier
one). At subgraph entry each declared input is bound from the latest
producer, falling back to the facade-supplied initial inputs
(`gap.execute(..., inputs={...})`, also addressable at top level as
`in.<name>`). A missing producer raises (the validator's W8 catches this
statically for name-level wiring).

:::{warning}
Subgraph inputs resolve by *name only* across subgraphs, and the latest
producer wins — two upstream subgraphs declaring an output of the same
name silently shadow each other.
:::

## Subgraph node execution

A top-level `{"type": "subgraph", "ref": ...}` node runs as:

1. Bind declared inputs (see above) into the reserved `in` pseudostate.
2. Run the inner scope with the same super-step loop.
3. Compute the exit value
   ([exit declaration](workflow-schema.md#subgraph-exit-declaration) /
   [on_error](workflow-schema.md#the-on_error-failure-exit)) and check it
   against `success_values ∪ {on_error}`.
4. Bind declared outputs by resolving their `$ref`s against the inner
   scope's outputs. Outputs unresolvable on this exit path are dropped with
   a debug log — downstream consumers simply see no value.
5. Enforce the subgraph's `validate=True` checkpoints — normal exits only,
   never on the `on_error` path
   (see [Checkpoints](../running/checkpoints.md)).
6. Fire the `subgraph_exit_hook` (if installed) with a `SubgraphExitEvent`
   `{sg_name, visit_index, bound_outputs, exit_value, error_path,
   elapsed_s}`. Harnesses use this seam for world snapshots; production
   runs leave it `None` (zero overhead). Hook exceptions are logged, never
   propagated.
7. The subgraph node's result is `{**bound_outputs, "_exit": exit_value}` —
   the reserved `_exit` key is what the top-level conditional edge
   dispatches on.

## Streaming nodes

A `streaming: true` node is spawned into a detached single-thread future the
moment the frontier reaches it; it never blocks downstream readiness and is
removed from the frontier immediately.

- Its `StreamSlot` is placed in `local_outputs` **before** spawning, so
  same-super-step consumers resolving `$ref` to its name get the slot;
  `latest()` blocks (default 60 s) until the first `ctx.publish`, then
  returns the newest value. If nothing is published within the timeout, the
  consumer's read raises `PipelineError`.
- The skill convention: loop, `ctx.publish(snapshot)` each iteration, check
  `ctx.cancel_token.raise_if_set()`.
- On scope exit (success or error) the runtime fires each streaming node's
  `CancelToken`, grace-waits `GAP_PARALLEL_CANCEL_GRACE_S` (default 2.0 s)
  for cooperative shutdown, then force-cancels stragglers.
- Streaming spawn is idempotent per scope, and validation guarantees
  streaming nodes are pure sources (rule S3) whose skill contract matches
  (rule S4).

## Send (dynamic fan-out)

A `router` node's script returns `{"route": ...}`:

- **string** — static dispatch through the node's conditional-edges mapping.
- **list of `{"to": <node>, "inputs": {...}}`** — one copy of the target
  node is dispatched per entry, concurrently, with the entry's inputs merged
  over the target's declared inputs. Copies are traced as `<target>#<i>`,
  and the **results collect into a list stored under the router node's
  name**, so downstream `$ref`s to the router see the gathered outputs.

## End nodes and recovery

Reaching an `end` node terminates the top-level scope. Its
[recovery tool calls](workflow-schema.md#recovery-toolcall) then run
sequentially, best-effort — each failure is logged and the remaining
actions still run. The workflow's exit status is the end node's `status`;
`"failure"` makes `execute()` raise after recovery completes. Exiting a
scope without reaching an end node, or terminating on a node that is not
`type: end`, is also a `PipelineError`.

## Observation stream

When the connector supplies a poll fn, the executor runs a background
thread polling `get_observation()` at a configured rate. The rate resolves
in priority order:

1. workflow `meta.observation_stream_hz`
2. the `GAP_OBSERVATION_STREAM_HZ` environment variable
3. the constructor's `observation_hz` (default 10 Hz)

The stream is injected as the reserved bound input
`in.observation_stream`; node bodies receive a per-node handle whose
`.latest(timeout)` / `.latest_with_age(timeout)` reads are recorded into the
trace as addressable `(node_id, seq)` events for deterministic replay
(default `timeout` 2.0 s). The stream stops in `execute()`'s `finally`.

`.latest()` raises `StreamUnavailable` until the first successful poll;
after that it is **stale-tolerant** — it returns the cached last-good
observation even if later polls fail. Poll errors back off from 0.1 s up to
a maximum of 1.0 s, doubling each time.

## NodeContext

Every script/tool node body receives a `NodeContext`
([gap/runtime/context.py](gh-engine:gap/runtime/context.py)); a script's
resolved workflow inputs arrive as keyword arguments to its
`run(ctx, ...)` function, not through the context.

- **`ctx.tool(name, **kwargs)`** — the sole dispatch call. Applies guard
  enforcement, invokes through the `ToolRegistry` (connector
  `robot.*`/`sim.*` tools, bundle tools, `@tool` plugins), and records the
  sub-call into the trace. Dispatch filters kwargs to the tool function's
  signature and injects `ctx` iff the function declares it.
- **`ctx.publish(value)`** — streaming nodes only: publish a snapshot to
  the node's `StreamSlot`. Calling it from a non-streaming node raises
  `RuntimeError`.
- **`ctx.cancel_token.raise_if_set()`** — cooperative cancellation; long
  loops must call this once per iteration and let `TaskCancelled`
  propagate.
- **`ctx.policy_executor`** — shared websocket-client cache for
  learned-policy skills.

## Guards (call-count safety limits)

Tools are classified by registry **tags**: `perception`, `planning`,
`sim_step`. At every `ctx.tool` dispatch the matching counter is
incremented and checked. Limits resolve in priority order:

1. per-workflow `gap.tools.guards.set_limits(...)` overrides
2. the `GAP_MAX_PERCEPTION_CALLS` / `GAP_MAX_PLANNING_CALLS` /
   `GAP_MAX_SIM_STEPS` environment variables
3. unlimited

Counters reset at the start of every `execute()`.

:::{warning}
`GuardLimitExceeded` subclasses **`BaseException`**, deliberately: a
skill's bare `except Exception` cannot swallow a tripped safety guard. See
[Safety](../real-robots/safety.md) for the wider safety model.
:::

Skills signal failures by raising `gap.errors.PipelineError` subclasses
(`PerceptionFailed`, `PlanningFailed`, `GraspFailed`, `ValidationFailed`,
`ToolError`, …). The node executors wrap any node-body exception into
`NodeExecutionError` (preserving the cause); the enclosing subgraph's
`on_error` declaration decides whether that becomes a routed failure exit
or a workflow abort.

## Tracing

Tracing is on by default. The trace directory resolves as: constructor
`trace_dir` argument → `GAP_TRACE_DIR` environment variable → the workflow
directory (`gap run` passes `outputs/run_<timestamp>/`). The on-disk layout
is a stability guarantee consumed by `gap viz` and `gap trace-diff`; see
[Traces](../running/traces.md) for the full layout and tooling.

What is recorded, and when:

- **At `execute()` start** — the graph topology (node and edge
  registrations) and a snapshot copy of the executed `workflow.json` plus
  its `scripts/`.
- **As nodes run** — per-node resolved inputs, outputs, every `ctx.tool`
  sub-call (request/response), and observation-stream reads, all written
  incrementally under `node_data/<id>/`. Image-, mask-, depth- and
  pointcloud-shaped numpy values are extracted as PNG/NPY/NPZ assets and
  summarized in JSON (`<ndarray shape=…>`), so traces stay browsable
  without loading gigabytes.
- **At termination** — `dag_trace.json` (enriched node metadata: timing,
  status, edges, condition results, the full event log) is flushed when
  the workflow reaches an end node.

:::{warning}
`dag_trace.json` is only flushed when an end node is reached. A run that
crashes before reaching one leaves the incrementally written `node_data/`
on disk but no flushed `dag_trace.json`.
:::

`GAP_TRACE_STREAM_SAMPLE_N` (default 1) persists every Nth stream read;
`0` disables stream-read recording. See
[Environment Variables](environment-variables.md) for all knobs mentioned
on this page.
