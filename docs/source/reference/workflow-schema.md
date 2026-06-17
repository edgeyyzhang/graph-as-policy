# Workflow JSON Schema

This page is the reference for the v3 `workflow.json` format — the on-disk
representation of a GaP graph. The loader lives in
[gap/runtime/workflow.py](gh-engine:gap/runtime/workflow.py) and the
structural validator in
[gap/runtime/validate.py](gh-engine:gap/runtime/validate.py). For how a
loaded graph actually runs, see [Executor Semantics](executor.md).

A GaP workflow is a **dual control/data graph of subgraphs**. The top level
is a DAG-with-loops of `subgraph` and `end` nodes wired by edges and
conditional edges; each subgraph is a self-contained inner graph of
`tool` / `script` / `router` / `noop` nodes with the same edge vocabulary,
declared typed inputs/outputs, and a small set of typed exit conditions.
Control flow — loops, retries, fallbacks — is expressed entirely as edges
between subgraphs routed on exit conditions. There are no retry counters,
no `branch_on` constructs, and no magic strings.

Data flows two ways:

- **Inside a subgraph** — tagged-object references
  (`{"$ref": "node.field.subfield"}`) pull values from upstream node outputs.
- **Across subgraphs** — by **output name**: a subgraph declares typed
  `inputs`, and at entry each input binds to the most recent upstream
  subgraph that declared an output of the same name.

Values on the wire are plain Python: TypedDicts from `gap.types` and numpy
arrays. There is no in-process serialization layer; the schema's type-name
strings (`"OrientedBoundingBox"`, `"PointCloud"`, …) resolve through the
`gap.schema` registry, and node I/O schemas come from Python type-hint
introspection.

## Top-level workflow.json

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
| `meta` | object | no | Free-form strings. `name`/`description` recommended; `observation_stream_hz` is consumed by the executor (see [Observation stream](executor.md#observation-stream)). |
| `nodes` | object | yes, non-empty | Node name → NodeDef. Names `START`/`END` are reserved. |
| `edges` | list of `[src, dst]` | no | Static edges. Multiple outgoing edges from one node = parallel fan-out in the same super-step. |
| `conditional_edges` | object | no | Source node → `{router_field, mapping}` ([Conditional edges](#conditional-edges)). |
| `subgraphs` | object | no | Name → SubgraphDef. Top level only — subgraphs do not nest. |

Unknown keys anywhere are a hard load-time error (strict-key parsing).

:::{note}
Legacy v2 constructs are rejected with targeted migration messages:

- node types `service`/`skill`/`policy` — rewrite as `type: "tool"` with a
  flat dispatch name;
- recovery entries with `service`/`method` keys — rewrite as
  `{"tool": ..., "inputs": ...}`;
- the retired `exit.values` key — split into `success_values` + `on_error`.
:::

## Subgraph definition (SubgraphDef)

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
| `exit` | object | yes | `{router_field, success_values}` — see [Exit declaration](#subgraph-exit-declaration). |
| `on_error` | string | no | The single failure exit value — see [the on_error failure exit](#the-on_error-failure-exit). |
| `stage` | string | no | Canonical pick-and-place stage tag; ignored by the runtime. |

## Node definition by type (NodeDef)

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
  the tool returns. See [Connector Tools](connector-tools.md) for the
  `robot.*`/`sim.*` surface.
- **`script`** — imports the Python file and calls its typed
  `run(ctx: NodeContext, ...) -> Output` function, where `Output` is a
  TypedDict (or `None`). Resolved workflow inputs not in the `run()`
  signature are dropped with a warning; declared output keys missing from
  the returned dict are an execution error.
- **`router`** — runs its script like a script node, but the returned dict
  must carry a `route` key: a **string** (static dispatch — looked up in the
  node's `conditional_edges` mapping) or a **list** of `{"to", "inputs"}`
  dicts ([Send fan-out](executor.md#send-dynamic-fan-out)).
- **`subgraph`** — top level only; recurses into the referenced SubgraphDef
  ([Subgraph node execution](executor.md#subgraph-node-execution)).
- **`noop`** — a passthrough marker with empty output; used as a named
  subgraph terminal (success exit) when `exit.router_field` is null.
- **`end`** — top level only; terminates the workflow with `status`, after
  running its `recovery` tool calls best-effort
  ([End nodes and recovery](executor.md#end-nodes-and-recovery)).

The `streaming: true` flag is valid only on `tool` and `script` nodes; it
marks the node as a detached producer that publishes snapshots via
`ctx.publish` and is consumed via `$ref`
([Streaming nodes](executor.md#streaming-nodes)).

### Recovery ToolCall

```json
{"tool": "robot.open_gripper", "inputs": {}}
```

Recovery actions live outside the node-dispatch system: a short list of
best-effort tool calls run when the workflow lands on an end node. `tool` is
the same flat dispatch namespace as `type: tool` nodes; `inputs` are literal
values (no `$ref` resolution). Failures are logged and never raise.

## Conditional edges

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

## Subgraph exit declaration

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

## The on_error failure exit

`on_error` names the exit value emitted when **any node in the subgraph
raises**. Without it, exceptions propagate and abort the workflow (the
exception surfaces in `ExecutionResult.error`). With it, the exception is
caught, the declared value becomes the subgraph's exit condition (bypassing
the terminal-node path), and the top-level conditional edge routes
accordingly — the idiom for "any failure → abort" wiring.

`on_error` is a pure symbol: it must not be a declared node (S9) and must
not appear as a conditional-edges mapping target inside the subgraph (S10).
[Checkpoints](../running/checkpoints.md) are not evaluated on the `on_error`
path (the postcondition is moot).

## Reference syntax (`$ref`)

```json
{"$ref": "observe"}                  // full output of node `observe`
{"$ref": "perceive.points"}          // field access
{"$ref": "filter_obb.obb.center.0"}  // nested walk + integer index
{"$ref": "in.target_obb"}            // bound subgraph input
{"$ref": "in.observation_stream"}    // the executor-injected stream
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

## Dataflow typing

- **Type-name strings** in subgraph `inputs` declarations (`"Se3Pose"`,
  `"PointCloud"`, `"Mask"`, …) resolve through the `gap.schema` registry of
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
- Cross-binding compatibility: same-name structured types match by name;
  `int`/`float` interchange; `Any`/`dict` match anything; fields whose
  hints cannot be introspected degrade to lenient. Note that the structural
  validator currently enforces cross-subgraph wiring by *name* only (W8);
  this field-level compatibility relation is defined but not enforced at
  validation time.

## Validation rules

Validation has two layers:

1. The **loader** (`load_workflow`) enforces syntax: strict keys at every
   level, version check, per-type required fields, reserved-name rejection
   (`START`/`END`/`in`), `streaming` only on tool/script, and the v2-leakage
   migration errors above. Loader violations raise
   `WorkflowValidationError` immediately.
2. The **structural validator** (`validate_workflow`) returns a list of
   `ValidationIssue` objects; the executor hard-fails on any
   `error`-severity issue before running and logs warnings.

### Workflow level

| # | Severity | Rule |
|---|---|---|
| W1 | error | `version == 3`. |
| W2 | error | At least one edge from `START`. |
| W3 | error | Every edge endpoint is `START`, `END`, or a declared node. |
| W4 | error | Every non-end node is reachable from `START` (through edges and conditional mappings). |
| W5 | error | Every node has an outgoing edge or a conditional-edges entry, unless it is `type: end`. |
| W6 | error | Conditional-edges sources are declared nodes; every mapping target is a declared node or `END`. |
| W7 | error | Every `subgraph` node's `ref` resolves to a declared subgraph. |
| W8 | error | Every reachable subgraph's declared input is supplied — by the executor-injected `observation_stream` or by a reachable upstream subgraph declaring an output of the same name. |
| — | error | At least one `end` node exists. |

### Subgraph level

| # | Severity | Rule |
|---|---|---|
| S1 | error | At least one edge from `START`. |
| S2 | error | Every node reachable from `START` (streaming nodes that are direct `START` targets count). |
| S3 | error | Streaming nodes are pure sources: no outgoing edges, no conditional-edges entry. |
| S4 | error | Streaming-flag/skill-contract consistency: a `streaming: true` node must invoke a skill whose bundle contract declares `streaming: true`, and vice versa (enforced when a skill registry is provided). |
| S5 | error | `$ref` heads reference declared nodes, or `in.<name>` for declared inputs / `in.observation_stream`. Edge and conditional targets are declared. |
| S6 | error | Subgraph `outputs` bind to declared, non-end nodes. |
| S7 | error | `exit.success_values` is non-empty. |
| S8 | error | Conditional edges from a non-router source declare `router_field`; from a router source `router_field` is null. |
| S9 | error | `on_error` is not a declared node name. |
| S10 | error | `on_error` is not a conditional-edges mapping target. |
| S11 | error | `router_field: null` ⇒ every success value is a declared `noop` node; `router_field` set ⇒ success values must not collide with node names. |
| — | error | Non-streaming, non-end nodes need an outgoing edge or conditional entry. |
| — | error | Declared input type names resolve in the `gap.schema` registry. |

:::{note}
Schema **introspection** failures degrade to warning-level issues rather
than blocking. In particular, a `tool` node whose tool is not yet registered
at validate time (e.g. `robot.*` connector tools that only register when a
connector attaches) produces a warning listing the available tools — typos
in connector tool names surface at runtime, not validation.
:::

At graph-generation time the validator additionally reconciles each
subgraph's `success_values ∪ {on_error}` against the owning skill's declared
`exit_conditions` (from `SKILL.md` frontmatter); the executor skips this
check. See [Generating Graphs](../authoring/generation.md).

## Reserved names

| Name | Where | Meaning |
|---|---|---|
| `START` | edge endpoint | Virtual entry: `START → x` edges seed the frontier. Never declared as a node. |
| `END` | edge / mapping target | Virtual exit: an edge `x → END` marks `x` as the scope's terminal node. Never declared as a node. |
| `in` | `$ref` head | Pseudostate holding bound subgraph inputs (`in.<name>`). A subgraph node named `in` is rejected at load time. |
| `in.observation_stream` | `$ref` / input name | The executor-injected observation stream (when the connector polls). |
| `_exit` | output key | Reserved key on a subgraph node's result carrying its exit value; `router_field: "exit"` is an alias for it. |

## Authoring surfaces

Three equivalent ways to produce a v3 graph — the executor treats them
identically:

1. **`gap generate`** — the LLM pipeline emits a workflow dir
   ([Generating Graphs](../authoring/generation.md)).
2. **`gap.builder`** — `Workflow` / `Subgraph` with `add_node` /
   `add_edge` / `add_conditional_edges` / `add_exit` / `set_outputs` /
   `add_checkpoint`; `save()` runs the same loader + validator
   ([Builder API](../authoring/builder.md)).
3. **Hand-written JSON** — anything that passes the rules on this page.
