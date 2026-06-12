# Graph Patterns and Pitfalls

Field rules for writing graphs that work the first time — whether you author
them with [gap.builder](builder.md) or review what
[`gap generate`](generation.md) produced. Each section names the failure
mode it prevents; most of these failures are *silent* (a wrong default, an
unbound value, a skipped node), which is exactly why they are worth a page.
The normative contracts live in the
[workflow schema](../reference/workflow-schema.md) and
[executor](../reference/executor.md) references.

## `$ref` path syntax

A `Ref("head.part.part")` resolves left to right from a node's output. The
head must be a node name in the current scope (or `in` for bound subgraph
inputs); an integer segment always indexes a sequence, and every other
segment is tried as a **dict key first**, then an attribute:

```python
Ref("observe.cameras")                  # dict key
Ref("observe.arms.0.ee_pose")           # list index, then key
Ref("candidates.candidates.poses.0")    # node "candidates" → field → list[0]
Ref("in.target_obb")                    # bound subgraph input
```

Rules worth memorizing:

- **Dict keys win over attributes**, so a field named `items` is never
  shadowed by `dict.items`.
- **Integer segments index sequences** — negative indices work
  (`poses.-1`). There is no bracket syntax: `poses[0]` is not a path. If
  you need anything fancier than dotted walking and integer indices (slices,
  per-element transforms), write a small adapter script node.
- **Lists containing refs resolve element-wise**: an input value
  `[Ref("a.x"), Ref("b.x")]` becomes a list of the two resolved values.
- **A `$ref` never crosses a subgraph boundary.** Inside a subgraph you can
  only reference that subgraph's own nodes and `in.*`. To consume another
  subgraph's data, declare an input (`add_input`) and publish an output
  (`set_outputs`) — the binding happens by name (next section).

### Wrapped tool outputs

Tools return dicts, and many geometry tools wrap their single result in a
named field. Binding the bare node is the most common authoring error:

```python
sg.set_outputs(target_obb=Ref("filter_obb.obb"))   # correct
sg.set_outputs(target_obb=Ref("filter_obb"))       # wrong: binds {"obb": ...}
```

| Tool | Bind |
|---|---|
| `geometry.filter_and_compute_obb`, `geometry.compute_obb` | `.obb` |
| `geometry.filter_noise`, `geometry.mask_to_world_points` | `.points` |
| `geometry.compute_drop_position` | `.position` |
| `geometry.top_down_grasp_from_obb` | `.pose` |
| `geometry.top_down_grasp_candidates` | `.candidates` (a `GraspCandidates`, poses best-first) |

Point-cloud tools take `points=`, not the legacy `point_cloud=`. See the
[tool catalog](../skills/tool-catalog.md) for each tool's exact schema.

### Where a wrong path fails — and where it doesn't

- In a **node's `inputs`**, a path that doesn't resolve raises at runtime
  (and routes to the subgraph's `on_error` if one is declared).
- In a **subgraph's `outputs`**, a path that fails to resolve on a given
  exit is **silently unbound** (debug log only). Downstream consumers then
  see no value and scripts quietly fall back to parameter defaults.

The classic instance: the end-effector pose lives at
`observe.arms.0.ee_pose` — there is no `observe.ee_pose`. Bind
`ee_pose_at_grasp=Ref("observe.ee_pose")` as a subgraph output and it
silently unbinds; `compute_drop_pose.py` then runs with
`ee_pose_at_grasp=None` and falls back to an inferior height estimate. The
robot drops the object from too high, and nothing ever errored.

## Cross-subgraph dataflow binds by name

There is no explicit wiring between subgraphs. A subgraph input binds to the
**most recent upstream subgraph output with the same name** — validation
rule W8 checks that some producer exists; the executor takes the latest one.

```python
perceive_target.set_outputs(target_obb=Ref("filter_obb.obb"), ...)
grasp.add_input("target_obb", type_name="OrientedBoundingBox")   # binds by name
```

Two consequences:

- **Latest producer wins, silently.** If two perception subgraphs both
  publish `target_obb`, the one that ran most recently shadows the other
  with no warning. Prefix output names by role — `target_obb`,
  `container_obb`, `handle_obb` — as the
  [skill bundles](../skills/skill-catalog.md) do
  (`<name>_obb` / `<name>_mask` / `<name>_cloud`).
- **Renames are load-bearing on both ends.** The transport skill's drop
  computation takes the held object as `held_obb`, not `target_obb`. Pass
  `target_obb=` instead and the script *warns and discards* the extra
  kwarg, then computes the drop without the held object's height — wrong by
  the object's full extent, silently.

The reserved input name `observation_stream` is exempt from W8: the
executor injects the graph-scoped observation stream itself.

## Literal strings vs `Ref`

Perception prompts — `object_name`, `object_description`, `parent_prompt`,
`subpart_prompt`, `placement_description` — are **literal strings baked into
the graph**, not dataflow:

```python
sg.add_node("perceive", type="script", script="scripts/perceive_dino_vlm.py",
            inputs={"cameras": Ref("observe.cameras"),
                    "object_name": "blue and yellow alphabet soup can"})  # literal
```

Writing `Ref("in.object_name")` is a validation error unless you also
declared `object_name` as a subgraph input with an upstream producer —
which no coordinator does for prompt strings. Rule of thumb: data measured
at run time flows through `Ref`; task parameters known at authoring time are
literals.

## `on_error` is a semantic exit, not just a catch-all

A subgraph has exactly one failure symbol: any exception raised inside it
becomes the `on_error` exit value, which top-level conditional edges route
like any other exit. Never declare a `failed` node — failure exits exist
only as symbols (rules S9/S10), and without `on_error` an exception crashes
the entire run.

Because the symbol is yours to name, it can carry *meaning*, not just
"something broke":

```python
perceive.set_on_error("not_found")     # PerceptionFailed → "the table is clear"

wf.add_conditional_edges(
    "target", {"found": "grasp", "not_found": "done"},   # not_found = success!
    router_field="exit")
```

In a clean-all-items loop, perception failing to find another target *is*
the success condition — `not_found` (or `table_clean`) routes to the
success end node. The `perceiving-objects-oneshot` bundle is designed
around this: it returns a clean `found: False` instead of raising, so a
router or conditional can terminate the loop deliberately.

One enforcement note: `validate=True`
[checkpoints](../running/checkpoints.md) are skipped on the `on_error`
path — the postcondition of a subgraph that bailed out is moot.

## Loops are subgraph revisits, never inner cycles

The scheduler guards every scope with a completed-node set: an edge that
re-enters a node which already ran in the current scope is **silently
skipped** — a cycle attempt inside a subgraph doesn't error or retry, it
just stops. So:

- **Never wire a retry cycle between nodes inside a subgraph** (e.g.
  `verify → descend` back-edges). The re-entered node is skipped and the
  subgraph falls through. If a step needs retrying, put the retry inside a
  script (`for attempt in range(3): ...`), or make the whole subgraph the
  unit of repetition.
- **Express iteration at the workflow level, across subgraph visits.** Each
  visit to a subgraph opens a fresh scope, so its internal nodes run again.
  The canonical shape is the clean-all-items wiring of the
  [steered policy example](../examples/steered-policy.md): top-level
  conditional edges route `reset → target → approach → run → reset → ...`,
  and the loop terminates through a semantic exit
  (`target.not_found → done`, previous section).
- There is no `max_retries` field anywhere in the schema — repetition is
  graph wiring. The runaway guard is the super-step cap
  `GAP_ITERATION_CAP` (default `10000`): exceeding it raises
  `PipelineError: ... possible runaway loop`. See
  [environment variables](../reference/environment-variables.md).

## Streaming nodes and Send fan-out

Two escape hatches from lockstep super-step execution — both summarized
here, fully specified in the [executor reference](../reference/executor.md).

**Streaming nodes** (`streaming=True`, tool/script only) are spawned
detached and act as pure data sources: they must have no outgoing or
conditional edges (rule S3), and the skill contract must also declare
`streaming: true` (S4). Consumers simply `Ref` the node name and
transparently receive the latest published snapshot:

```python
sg.add_node("track", type="script", script="scripts/track.py",
            inputs={"target_prompt": "soup can"}, streaming=True)
sg.add_node("servo", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("track.box_pose")})   # latest() snapshot
```

The first read blocks up to 60 s waiting for the node's first publish, then
fails with `PipelineError`. At scope exit the executor fires the node's
cancel token and force-cancels after a grace period
(`GAP_PARALLEL_CANCEL_GRACE_S`, default 2.0 s) — streaming skills must
check `ctx.cancel_token` each iteration.

**Router nodes** (`type="router"`) run a script whose `run()` returns a
dict with a `route` key:

- a **string** — static dispatch through the node's conditional-edge
  mapping (the router's `router_field` must be `null`, rule S8), or
- a **list of `{"to": node, "inputs": {...}}` Send dicts** — dynamic
  fan-out: each entry spawns one copy of the target node with those inputs,
  all copies run in parallel, and the results collect as a list under the
  router's own name for downstream `Ref`s.

## Sim time: `settle_steps`, not `time.sleep`

`time.sleep()` never advances a simulator — the sim only steps when a tool
steps it. To let physics settle after a gripper command, pass
`settle_steps` (sim steps, not seconds):

```python
sg.add_node("open", type="tool", tool="robot.open_gripper",
            inputs={"settle_steps": 40})
sg.add_node("close", type="tool", tool="robot.close_gripper",
            inputs={"settle_steps": 60})
```

The shipped graphs use 40 on open and 60 on close. Note `settle_steps=0`
does not mean "skip settling" — it falls back to the connector's default.

## Units and conventions that bite

| Convention | gap | Elsewhere |
|---|---|---|
| Quaternion order | `{w, x, y, z}` — **scalar-first (wxyz)** everywhere in `gap.types` | scipy, MuJoCo, LIBERO use xyzw |
| OBB `extent` | **Half**-extents along local axes | many libraries use full extents |
| `GripperState.position` | meters; `0.0` = **closed** | — |
| `ArmState.gripper_fraction` | `0.0` closed → `1.0` = **open** | opposite feel to `position` |
| End-effector pose path | `observe.arms.0.ee_pose` | `observe.ee_pose` does not exist |

- Convert quaternions at the boundary with
  `gap.types.quat_xyzw_to_wxyz(q)` / `quat_wxyz_to_xyzw(q)` — inside a
  graph, every `Quaternion` dict is wxyz. Mixing the orders produces poses
  that are subtly rotated, not obviously broken.
- Because `extent` is half-extents, "0.15 m above the box top" is
  `center.z + extent.z + 0.15`, not `center.z + extent.z / 2 + 0.15`.
- A gripper check that reads `position == 0.0` as "open" (or
  `gripper_fraction == 1.0` as "closed") inverts the grasp logic; keep the
  two conventions straight when writing scripts and
  [checkpoints](../running/checkpoints.md).

## Quick reference

| Pitfall | Symptom | Fix |
|---|---|---|
| `Ref("filter_obb")` instead of `Ref("filter_obb.obb")` | downstream gets a wrapping dict | bind the named field |
| Wrong path in a subgraph *output* | silently unbound; script defaults kick in | verify paths against tool schemas; check trace `node_data` |
| Two subgraphs publish the same output name | latest producer silently shadows | prefix outputs by role (`target_*`, `container_*`) |
| `Ref("in.object_name")` for a prompt | validation error | prompts are literal strings |
| Declared `failed` node | S9 error if it matches `on_error`; else failure masquerades as a success value | failure exits via `set_on_error` only |
| No `on_error` on a subgraph | one exception crashes the whole run | always set a semantic failure symbol |
| Retry edge inside a subgraph | node silently skipped, subgraph stalls | loop at the workflow level across subgraph visits |
| `time.sleep` after gripper command | object not settled; grasp/release flaky | `settle_steps` on the tool call |
| xyzw quaternion fed to a graph | subtly wrong orientations | convert with `quat_xyzw_to_wxyz` |
| Treating `extent` as full size | grasps/drops off by half the object | `extent` is half-extents |
