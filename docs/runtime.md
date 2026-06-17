# The GaP runtime — v3 workflow schema & execution semantics

**Scope:** the canonical specification of the v3 `workflow.json` schema
(`gap/runtime/workflow.py`), the structural validator
(`gap/runtime/validate.py`), the super-step executor
(`gap/runtime/executor.py`), the tool dispatch layer (`gap/tools/`), and
execution-time checkpoint verification (`gap/runtime/verify/`).
**Audience:** graph authors (human or LLM), skill-bundle authors, and anyone
extending the runtime.

---

## 1. Abstract

A GaP workflow is a **dual control/data graph of subgraphs**. The top level
is a DAG-with-loops of `subgraph` and `end` nodes wired by edges and
conditional edges; each subgraph is a self-contained inner graph of
`tool` / `script` / `router` / `noop` nodes with the same edge vocabulary,
declared typed inputs/outputs, and a small set of **typed exit conditions**.
Control flow — loops ("clean all items on the table"), retries
("re-perceive on grasp slip"), and fallbacks — is expressed entirely as
edges between subgraphs routed on exit conditions. There are no retry
counters, no `branch_on` constructs, and no magic strings.

Data flows two ways:

- **Inside a subgraph** — tagged-object references
  (`{"$ref": "node.field.subfield"}`) pull values from upstream node outputs.
- **Across subgraphs** — by **output name**: a subgraph declares typed
  `inputs`, and at entry each input binds to the most recent upstream
  subgraph that declared an output of the same name.

Values on the wire are plain Python: TypedDicts from `gap.types` and numpy
arrays. There is no serialization layer in-process; the schema's type-name
strings ("OrientedBoundingBox", "PointCloud", …) resolve through the
`gap.schema` registry, and node I/O schemas come from Python type-hint
introspection.

The executor is a **frontier-based super-step scheduler**: each tick it runs
every ready node concurrently, folds the outputs back into the scope's
local-outputs dict, then advances the frontier along edges and conditional
edges. Subgraph nodes recurse into the same scheduler. Streaming nodes are
detached producers consumed by snapshot; Send fan-out spawns dynamic copies
of a target node.

## 2. Glossary

| Term | Meaning |
|---|---|
| **Workflow** | The top-level v3 graph: `nodes` + `edges` + `conditional_edges` + named `subgraphs`. |
| **Subgraph** | A self-contained inner graph owned by one skill; declares typed `inputs`/`outputs`, an `exit` declaration and optionally `on_error`. |
| **Node** | One unit in a graph. Types: `tool`, `script`, `router`, `subgraph`, `noop`, `end`. |
| **START / END** | Virtual node names. `START → x` edges seed the frontier; an edge `x → END` marks `x` as the scope's terminal node. They are never declared. |
| **Super-step** | One scheduler tick: all ready (non-streaming) nodes run concurrently, then the frontier advances. |
| **Exit condition / exit value** | The string a subgraph emits when it terminates; the top-level conditional edge on the subgraph node routes on it. |
| **`on_error`** | A subgraph's single failure exit: any exception inside the subgraph is caught and surfaced as this exit value. |
| **`$ref`** | Tagged-object reference `{"$ref": "head.field.sub"}` resolving against the current scope's outputs (or `in.<name>` bound inputs). |
| **Streaming node** | A `tool`/`script` node with `streaming: true`: spawned detached, publishes snapshots via `ctx.publish`, consumed via `$ref`. |
| **Send** | Dynamic fan-out: a `router` node returning a list of `{to, inputs}` dicts; one copy of the target runs per entry. |
| **Recovery** | Best-effort tool calls (`{tool, inputs}`) attached to an `end` node, run before the workflow terminates (e.g. open gripper, go home). |
| **Checkpoint** | An authored postcondition predicate over a privileged `World` snapshot, evaluated when its subgraph exits. |
| **Tool** | A typed callable dispatched by flat name through the `ToolRegistry` — connector tools (`robot.*`/`sim.*`), bundle tools (`sam3.segment_text`, `curobo.plan_to_pose`), or `@tool` plugins. |

## 3. Schema specification

### 3.1 Top-level workflow.json

```json
{
  "version": 3,
  "meta": {"name": "pick_and_place", "description": "..."},
  "nodes": {
    "target":    {"type": "subgraph", "ref": "target_sg"},
    "grasp":     {"type": "subgraph", "ref": "grasp_sg"},
    "transport": {"type": "subgraph", "ref": "transport_sg"},
    "done":      {"type": "end", "status": "success"},
    "abort":     {"type": "end", "status": "failure",
                  "recovery": [{"tool": "robot.open_gripper"},
                               {"tool": "robot.go_home"}]}
  },
  "edges": [["START", "target"]],
  "conditional_edges": {
    "target":    {"router_field": "exit",
                  "mapping": {"found": "grasp", "not_found": "abort"}},
    "grasp":     {"router_field": "exit",
                  "mapping": {"grasped": "transport", "failed": "abort"}},
    "transport": {"router_field": "exit",
                  "mapping": {"placed": "done", "blocked": "abort"}}
  },
  "subgraphs": {
    "target_sg":    { "...": "SubgraphDef" },
    "grasp_sg":     { "...": "SubgraphDef" },
    "transport_sg": { "...": "SubgraphDef" }
  }
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `version` | int | yes | Must be `3`. |
| `meta` | object | no | Free-form strings. `name`/`description` recommended; `observation_stream_hz` is consumed by the executor (§7.9). |
| `nodes` | object | yes, non-empty | Node name → NodeDef. Names `START`/`END` are reserved. |
| `edges` | list of `[src, dst]` | no | Static edges. Multiple outgoing edges from one node = parallel fan-out in the same super-step. |
| `conditional_edges` | object | no | Source node → `{router_field, mapping}` (§3.5). |
| `subgraphs` | object | no | Name → SubgraphDef. Top level only — subgraphs do not nest. |

Unknown keys anywhere are a hard load-time error (strict-key parsing).
Legacy v2 constructs are rejected with targeted migration messages: node
types `service`/`skill`/`policy` (rewrite as `type: "tool"` with a flat
dispatch name), recovery entries with `service`/`method` keys (rewrite as
`{"tool": ..., "inputs": ...}`), and the retired `exit.values` key
(split into `success_values` + `on_error`).

### 3.2 SubgraphDef

```json
{
  "skill": "perceiving-objects",
  "inputs":  {"target_obb": "OrientedBoundingBox"},
  "outputs": {"target_obb": {"$ref": "filter_obb.obb"}},
  "nodes": {
    "observe":    {"type": "tool", "tool": "robot.get_observation"},
    "perceive":   {"type": "script", "script": "scripts/target_sg/perceive.py",
                   "inputs": {"observation": {"$ref": "observe"},
                              "object_name": "alphabet soup"}},
    "filter_obb": {"type": "tool", "tool": "geometry.filter_and_compute_obb",
                   "inputs": {"points": {"$ref": "perceive.points"}}},
    "found":      {"type": "noop"}
  },
  "edges": [["START", "observe"], ["observe", "perceive"],
            ["perceive", "filter_obb"], ["filter_obb", "found"],
            ["found", "END"]],
  "conditional_edges": {},
  "exit": {"router_field": null, "success_values": ["found"]},
  "on_error": "not_found"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `skill` | string | yes | Owning skill bundle (legacy key `agent` is accepted as an alias). Scripts in this subgraph import under the bundle's synthetic package, which enables `load_prompt`. |
| `inputs` | object | no | `{name: type-name string}`. Type names resolve through the `gap.schema` registry. Referenced inside as `{"$ref": "in.<name>"}`. |
| `outputs` | object | no | `{name: {"$ref": ...}}` bindings from internal node fields to declared output names. |
| `nodes` / `edges` / `conditional_edges` | — | yes / no / no | Same shape as the top level. Valid inner node types: `tool`, `script`, `router`, `noop`. A node named `in` is rejected (reserved). |
| `exit` | object | yes | `{router_field, success_values}` — see §3.6. |
| `on_error` | string | no | The single failure exit value (§3.7). |
| `stage` | string | no | Canonical pick-and-place stage tag; ignored by the runtime. |

### 3.3 NodeDef by type

| Field | `tool` | `script` | `router` | `subgraph` | `noop` | `end` |
|---|---|---|---|---|---|---|
| `tool` (flat dispatch name) | **required** | — | — | — | — | — |
| `script` (path relative to workflow dir) | — | **required** | **required** (routing function) | — | — | — |
| `ref` (subgraph name) | — | — | — | **required** | — | — |
| `inputs` | yes | yes | yes | yes | yes | yes |
| `streaming` | yes | yes | — | — | — | — |
| `status` (`"success"`/`"failure"`) | — | — | — | — | — | **required** |
| `recovery` (list of ToolCalls) | — | — | — | — | — | optional |

- **`tool`** — dispatches the flat name through the `ToolRegistry`:
  connector tools (`robot.get_observation`, `sim.check_success`), tool-bundle
  functions (`sam3.segment_box`, `geometry.iou`), or a callable skill bundle
  registered under its own name (`pi05-libero.run`). Returns whatever
  the tool returns.
- **`script`** — imports the Python file and calls its typed
  `run(ctx: NodeContext, ...) -> Output` function, where `Output` is a
  TypedDict (or `None`). Resolved workflow inputs not in the `run()`
  signature are dropped with a warning; declared output keys missing from
  the returned dict are an execution error.
- **`router`** — runs its script like a script node, but the returned dict
  must carry a `route` key: a **string** (static dispatch — looked up in the
  node's `conditional_edges` mapping) or a **list** of `{"to", "inputs"}`
  dicts (Send fan-out, §7.7).
- **`subgraph`** — top level only; recurses into the referenced SubgraphDef
  (§7.5).
- **`noop`** — a passthrough marker with empty output; used as a named
  subgraph terminal (success exit) when `exit.router_field` is null.
- **`end`** — top level only; terminates the workflow with `status`, after
  running its `recovery` tool calls best-effort (§7.8).

### 3.4 Recovery ToolCall

```json
{"tool": "robot.open_gripper", "inputs": {}}
```

Recovery actions live outside the node-dispatch system: a short list of
best-effort tool calls run when the workflow lands on an end node. `tool` is
the same flat dispatch namespace as `type: tool` nodes; `inputs` are literal
values (no `$ref` resolution). Failures are logged and never raise.

### 3.5 Conditional edges

```json
"conditional_edges": {
  "grasp": {"router_field": "exit",
            "mapping": {"grasped": "transport", "failed": "abort"}}
}
```

| Source node type | `router_field` | Dispatch value |
|---|---|---|
| `router` | must be `null` | the routing function's returned `route` string |
| `subgraph` | `"exit"` (aliased to the reserved `_exit` output key) | the subgraph's exit value |
| any other | required string | that field of the source node's output |

The resolved value must be a key in `mapping` — anything else is a runtime
`PipelineError`. Mapping targets are node names (or `END` inside subgraphs).

### 3.6 Subgraph exit declaration

```json
"exit": {"router_field": null, "success_values": ["found", "table_clean"]}
```

The subgraph's terminal node is the one whose edge points to `END`. Two
modes:

- **`router_field: null`** (the common case): the terminal node's *name*
  becomes the exit value. Every entry of `success_values` must be a declared
  `noop` node — the success exits are literally named markers in the graph.
- **`router_field: "<field>"`** (data-dependent exit): the exit value is read
  from that field on the terminal node's output. Success values are returned
  strings and must **not** collide with node names.

At runtime the resolved exit value must be a member of
`success_values ∪ {on_error}`; anything else raises.

### 3.7 `on_error` — the failure exit

`on_error` names the exit value emitted when **any node in the subgraph
raises**. Without it, exceptions propagate and abort the workflow (the
exception surfaces in `ExecutionResult.error`). With it, the exception is
caught, the declared value becomes the subgraph's exit condition (bypassing
the terminal-node path), and the top-level conditional edge routes
accordingly — the idiom for "any failure → abort" wiring.

`on_error` is a pure symbol: it must not be a declared node (S9) and must
not appear as a conditional-edges mapping target inside the subgraph (S10).
Checkpoints are not evaluated on the `on_error` path (the postcondition is
moot).

### 3.8 Reference syntax (`$ref`)

```json
{"$ref": "observe"}                  // full output of node `observe`
{"$ref": "perceive.points"}          // field access
{"$ref": "filter_obb.obb.center.0"}  // nested walk + integer index
{"$ref": "in.target_obb"}            // bound subgraph input
{"$ref": "in.observation_stream"}    // the executor-injected stream (§7.9)
{"$ref": "tracker"}                  // streaming node → latest() snapshot
```

Resolution rules (`gap.runtime.workflow.resolve_ref`):

1. The head must be a node name in the current scope or the reserved
   pseudostate `in` (bound subgraph inputs). References **cannot escape the
   current subgraph** — cross-subgraph data crosses only through declared
   `inputs`/`outputs`.
2. If the head's value exposes a callable `.latest()` (a streaming
   `StreamSlot` or the observation stream), it is invoked transparently to
   snapshot the latest published value before walking sub-fields.
3. Each subsequent path part is, in order: an integer index into a
   list/tuple (negative indices allowed), a **dict key** (checked before
   attributes, so TypedDict keys like `"items"` are never shadowed by dict
   methods), then an attribute.
4. Inside literal lists, refs resolve element-wise:
   `["a", {"$ref": "x.y"}]` is a legal input value.

A tagged object is used instead of a string DSL so the validator can
distinguish refs from literals by structure, and literal lists are never
ambiguous with references.

## 4. Dataflow typing

- **Type-name strings** in subgraph `inputs` declarations ("Se3Pose",
  "PointCloud", "Mask", …) resolve through the `gap.schema` registry of
  `gap.types` TypedDicts. An unknown name fails validation, not
  mid-execution. `"Any"`/`"dict"` are open escape hatches;
  `"ObservationStream"` marks the injected stream.
- **Tool nodes** get their I/O schemas from the registry's `UnitSchema`
  (extracted from the tool function's type hints at registration).
- **Script nodes** get theirs from `run()`'s signature: every non-`ctx`
  parameter must carry a type annotation; the return annotation must be a
  TypedDict (or `None`). Supported hint shapes: scalars, TypedDicts,
  `list[...]`, `dict`, `T | None`, `Any`, `ObservationStream`,
  bare classes (`np.ndarray`).
- Cross-binding compatibility: same-name structured types must match by
  name; `int`/`float` interchange; `Any`/`dict` match anything; fields whose
  hints cannot be introspected degrade to lenient.

## 5. Validation rules

Validation has two layers. The **loader** (`load_workflow`) enforces syntax:
strict keys at every level, version check, per-type required fields,
reserved-name rejection (`START`/`END`/`in`), `streaming` only on
tool/script, and the v2-leakage migration errors (§3.1). The **structural
validator** (`validate_workflow`) returns a list of issues; the executor
hard-fails on any `error`-severity issue before running and logs warnings.

### 5.1 Workflow level

| # | Rule |
|---|---|
| W1 | `version == 3`. |
| W2 | At least one edge from `START`. |
| W3 | Every edge endpoint is `START`, `END`, or a declared node. |
| W4 | Every non-end node is reachable from `START` (through edges and conditional mappings). |
| W5 | Every node has an outgoing edge or a conditional-edges entry, unless it is `type: end` or streaming. |
| W6 | Conditional-edges sources are declared nodes; every mapping target is a declared node or `END`. |
| W7 | Every `subgraph` node's `ref` resolves to a declared subgraph. |
| W8 | Every reachable subgraph's declared input is supplied — by the executor-injected `observation_stream` or by a reachable upstream subgraph declaring an output of the same name. |
| — | At least one `end` node exists. |

### 5.2 Subgraph level

| # | Rule |
|---|---|
| S1 | At least one edge from `START`. |
| S2 | Every node reachable from `START` (streaming nodes that are direct `START` targets count). |
| S3 | Streaming nodes are pure sources: no outgoing edges, no conditional-edges entry. |
| S4 | Streaming-flag/skill-contract consistency: a `streaming: true` node must invoke a skill whose bundle contract declares `streaming: true`, and vice versa (enforced when a skill registry is provided). |
| S5 | `$ref` heads reference declared nodes, or `in.<name>` for declared inputs / `in.observation_stream`. Edge and conditional targets are declared. |
| S6 | Subgraph `outputs` bind to declared, non-end nodes. |
| S7 | `exit.success_values` is non-empty. |
| S8 | Conditional edges from a non-router source declare `router_field`; from a router source `router_field` is null. |
| S9 | `on_error` is not a declared node name. |
| S10 | `on_error` is not a conditional-edges mapping target. |
| S11 | `router_field: null` ⇒ every success value is a declared `noop` node; `router_field` set ⇒ success values must not collide with node names. |
| — | Non-streaming, non-end nodes need an outgoing edge or conditional entry. |
| — | Declared input type names resolve in the `gap.schema` registry. |

At graph-generation time the validator additionally reconciles each
subgraph's `success_values ∪ {on_error}` against the owning skill's declared
`exit_conditions` (from SKILL.md frontmatter); the executor skips this check.

Schema introspection failures (e.g. a `robot.*` tool not yet registered at
validate time) degrade to warning-level issues rather than blocking.

## 6. Tool layer

### 6.1 Registry and dispatch

`gap.tools.ToolRegistry` is the single flat dispatch surface. Three ways in:

- **`@tool(name=..., summary=..., tags=(...), scope="runtime")`** — declares
  a typed function (bundle `tools.py` files use this). Registrations are
  queued at import and drained by `discover_pending()`.
- **`register_callable(name, fn, summary=..., tags=...)`** — the connector's
  entry point; the **only** path allowed to claim the reserved `robot.*` /
  `sim.*` prefixes.
- **Directory discovery** — `default_tool_registry()` walks `gap/tools/`.

Name collisions raise (re-importing the *same* function is idempotent).
Schemas are extracted from type hints into a `UnitSchema` that drives both
validation and the LLM tool catalogs. Dispatch filters kwargs to the
function's signature and injects `ctx` iff the function declares it.

### 6.2 NodeContext — the skill-facing surface

Every script/tool node body receives a `NodeContext`:

- **`ctx.tool(name, **kwargs)`** — the sole dispatch call. Applies guard
  enforcement, invokes through the registry, and records the sub-call into
  the trace.
- **`ctx.publish(value)`** — streaming nodes only: publish a snapshot to the
  node's `StreamSlot` (§7.6).
- **`ctx.cancel_token.raise_if_set()`** — cooperative cancellation; long
  loops must call this once per iteration and let `TaskCancelled` propagate.
- `ctx.policy_executor` — shared websocket-client cache for learned-policy
  skills.

### 6.3 Guards (call-count safety limits)

Tools are classified by registry **tags**: `perception`, `planning`,
`sim_step`. At every `ctx.tool` dispatch the matching counter is incremented
and checked. Limits resolve in priority order: per-workflow
`gap.tools.guards.set_limits(...)` overrides → `GAP_MAX_PERCEPTION_CALLS` /
`GAP_MAX_PLANNING_CALLS` / `GAP_MAX_SIM_STEPS` environment variables →
unlimited. Counters reset at the start of every `execute()`.

`GuardLimitExceeded` subclasses **`BaseException`**, deliberately: a skill's
bare `except Exception` cannot swallow a tripped safety guard.

### 6.4 Errors

Skills raise `gap.errors.PipelineError` subclasses (`PerceptionFailed`,
`PlanningFailed`, `GraspFailed`, `ValidationFailed`, `ToolError`, …). The
node executors wrap any node-body exception into `NodeExecutionError`
(preserving the cause); the enclosing subgraph's `on_error` declaration
decides whether that becomes a routed failure exit or a workflow abort.

## 7. Executor semantics

### 7.1 Entry

`WorkflowExecutor.execute()` (or the `gap.execute(...)` facade):

1. resets guard counters and checkpoint results;
2. runs `validate_workflow` — error-severity issues raise
   `GraphValidationError` before anything executes;
3. initializes the trace (graph topology + a copy of the workflow dir);
4. starts the observation stream when the connector provides a poll fn;
5. runs the top-level scope; the terminal node must be `type: end`.
   Its `status` is recorded as `exit_status`, recovery runs, the trace
   flushes, and a `failure` status raises `PipelineError` (which
   `gap.execute` converts into `ExecutionResult.success=False`).

### 7.2 Scope scheduling (super-steps)

Each scope (top-level workflow or one subgraph visit) runs the same loop:

```
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

### 7.3 Conditional dispatch

After a node completes, its conditional edge (if any) resolves per §3.5.
For subgraph nodes the canonical `router_field: "exit"` reads the reserved
`_exit` key of the subgraph node's result. A resolved target of `END` marks
the source as the scope terminal.

### 7.4 Cross-subgraph data

The executor maintains `cross_subgraph_outputs: {sg_name: {output: value}}`
— **most recent run wins** (a loop's later visit overwrites the earlier
one). At subgraph entry each declared input is bound from the latest
producer, falling back to the facade-supplied initial inputs
(`gap.execute(..., inputs={...})`, also addressable at top level as
`in.<name>`). A missing producer raises (the validator's W8 catches this
statically for name-level wiring).

### 7.5 Subgraph node execution

1. Bind declared inputs (§7.4) into the reserved `in` pseudostate.
2. Run the inner scope (§7.2).
3. Compute the exit value (§3.6 / §3.7) and check it against
   `success_values ∪ {on_error}`.
4. Bind declared outputs by resolving their `$ref`s against the inner
   scope's outputs. Outputs unresolvable on this exit path are dropped with
   a debug log — downstream consumers simply see no value.
5. Enforce the subgraph's `validate=True` checkpoints (§8) — normal exits
   only.
6. Fire the `subgraph_exit_hook` (if installed) with a `SubgraphExitEvent`
   `{sg_name, visit_index, bound_outputs, exit_value, error_path,
   elapsed_s}`. Harnesses use this seam for world snapshots; production
   runs leave it `None` (zero overhead). Hook exceptions are logged, never
   propagated.
7. The subgraph node's result is `{**bound_outputs, "_exit": exit_value}`.

### 7.6 Streaming nodes

A `streaming: true` node is spawned into a detached single-thread future the
moment the frontier reaches it; it never blocks downstream readiness and is
removed from the frontier immediately.

- Its `StreamSlot` is placed in `local_outputs` **before** spawning, so
  same-super-step consumers resolving `$ref` to its name get the slot;
  `latest()` blocks (default 60 s) until the first `ctx.publish`, then
  returns the newest value.
- The skill convention: loop, `ctx.publish(snapshot)` each iteration, check
  `ctx.cancel_token.raise_if_set()`.
- On scope exit (success or error) the runtime fires each streaming node's
  cancel token, grace-waits `GAP_PARALLEL_CANCEL_GRACE_S` (default 2.0 s)
  for cooperative shutdown, then force-cancels stragglers.
- Streaming spawn is idempotent per scope, and validation guarantees
  streaming nodes are pure sources (S3) whose skill contract matches (S4).

### 7.7 Send (dynamic fan-out)

A `router` node's script returns `{"route": ...}`:

- **string** — static dispatch through the node's conditional-edges mapping.
- **list of `{"to": <node>, "inputs": {...}}`** — one copy of the target
  node is dispatched per entry, concurrently, with the entry's inputs merged
  over the target's declared inputs. Copies are traced as `<target>#<i>`,
  and the **results collect into a list stored under the router node's
  name**, so downstream `$ref`s to the router see the gathered outputs.

### 7.8 End nodes & recovery

Reaching an `end` node terminates the top-level scope. Its recovery tool
calls then run sequentially, best-effort — each failure is logged and the
remaining actions still run. The workflow's exit status is the end node's
`status`; `"failure"` makes `execute()` raise after recovery completes.

### 7.9 Observation stream

When the connector supplies a poll fn, the executor runs a background
thread polling `get_observation()` at a configured rate (workflow
`meta.observation_stream_hz` → `GAP_OBSERVATION_STREAM_HZ` → constructor
default 10 Hz). The stream is injected as the reserved bound input
`in.observation_stream`; node bodies receive a per-node handle whose
`.latest(timeout)` / `.latest_with_age(timeout)` reads are recorded into the
trace as addressable `(node_id, seq)` events for deterministic replay. The
stream stops in `execute()`'s `finally`.

## 8. Checkpoints (execution-time verification)

Checkpoints are per-subgraph postcondition predicates over **privileged
simulator state**, decoupled from the graph itself:

- A sidecar module `<workflow_dir>/checkpoints/<sg_name>.py` exports
  `CHECKPOINTS: list[gap.runtime.verify.Checkpoint]`. Authored by hand or
  via `gap.builder.Subgraph.add_checkpoint(name, predicate, rationale=...,
  validate=True)` + `dump_checkpoints_module()`.
- `Checkpoint.predicate` takes `(world)` or `(world, outputs)` — arity is
  introspected per call. `world` is a `gap.runtime.verify.World` snapshot
  (bodies, poses, AABBs, contacts, `is_grasped/is_in/is_on/is_above`,
  `eventually/always` over history); `outputs` is the subgraph's bound
  outputs at exit, so predicates can compare authored results (e.g. a
  perception OBB) against ground truth.
- `validate=True` checkpoints are hard postconditions; `validate=False`
  ones are probes — surfaced in reports, never enforced.

Enforcement is wired through `gap.execute(..., checkpoints=...)`:

| Mode | Behavior on a failed `validate=True` checkpoint |
|---|---|
| `"off"` | sidecars are never loaded |
| `"warn"` (default) | log and continue |
| `"raise"` | raise `gap.errors.VerificationFailed` naming the failed checkpoints |

Evaluation happens at each subgraph's **normal** exit (never on the
`on_error` path), against `connector.world_snapshot()`. A predicate that
raises is recorded as `passed=False` with the eval error captured. When the
connector exposes no ground truth (real robots), enforcement degrades to a
one-shot warning and the run proceeds. All results accumulate in
`ExecutionResult.checkpoint_results`.

## 9. Tracing

Tracing is on by default (`gap run` writes `outputs/run_<timestamp>/`;
programmatic default is `GAP_TRACE_DIR`, else the workflow dir). The
on-disk layout is a stability guarantee consumed by `gap viz` and
`gap trace-diff`:

```
dag_trace.json     # enriched node metadata: timing, status, edges, events
workflow.json      # copy of the executed workflow (plus scripts/)
node_data/<id>/    # per-node resolved inputs, outputs, ctx.tool sub-calls,
                   # stream reads, and extracted assets (PNG masks/images,
                   # NPZ depth/point clouds)
```

Asset extraction is structural: image-, mask-, depth- and pointcloud-shaped
numpy values are written as files and summarized in JSON
(`<ndarray shape=…>`), so traces stay browsable without loading gigabytes.

## 10. Authoring surfaces

Three equivalent ways to produce a v3 graph:

1. **`gap generate` / `gap.agent.generate(...)`** — the LLM pipeline
   (coordinator → per-subgraph agents → checkpoint agent → validate → fix
   loop) emits a workflow dir.
2. **`gap.builder`** — `Workflow` / `Subgraph` with
   `add_node` / `add_edge` / `add_conditional_edges` / `add_exit` /
   `set_outputs` / `add_checkpoint`; `save()` runs the same loader +
   validator.
3. **Hand-written JSON** — anything that passes §5.

The executor treats all three identically.
