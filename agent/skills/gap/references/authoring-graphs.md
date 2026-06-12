# Authoring graphs

A workflow (`workflow.json`, schema v3) is a top-level state machine
whose nodes are tools, scripts, routers, or **subgraphs** (one per skill
the task needs). Subgraphs own their inner state machines, declare typed
inputs/outputs, exit conditions, and postcondition checkpoints. In a
checkout, `docs/runtime.md` is the full spec; this is the working digest.

## Builder surface (`from gap.builder import Workflow, Subgraph, Ref`)

```python
sg = Subgraph(name="perceive_sg", skill="perceiving-objects-oneshot")
sg.add_input("object_name", type_name="str")     # cross-subgraph input
sg.add_node("observe", type="tool", tool="robot.get_observation")
sg.add_node("perceive", type="script", script="scripts/perceive_simple.py",
            inputs={"cameras": Ref("observe.cameras"),
                    "object_name": Ref("in.object_name")})
sg.add_exit("found")                  # success marker (noop node)
sg.set_on_error("not_found")          # error exit (NOT a declared node)
sg.add_edge("START", "observe"); sg.add_edge("observe", "perceive")
sg.add_edge("perceive", "found"); sg.add_edge("found", "END")
sg.set_outputs(target_obb=Ref("filter_obb.obb"))
sg.add_checkpoint("object_localized", lambda world: ..., rationale="...")
sg.dump_checkpoints_module(path)      # writes checkpoints/<sg>.py sidecar

wf = Workflow(name="task", description="...")
wf.add_subgraph(sg)
wf.add_node("perceive_node", type="subgraph", ref="perceive_sg")
wf.add_node("done", type="end", status="success")
wf.add_node("abort", type="end", status="failure")
wf.add_edge("START", "perceive_node")
wf.add_conditional_edges("perceive_node",
                         {"found": "done", "not_found": "abort"},
                         router_field="exit")
wf.save("my_graph/workflow.json")     # parse + structural validation
wf2 = Workflow.load("my_graph/workflow.json")
```

- Node types: `tool` | `script` | `router` | `subgraph` | `noop` | `end`.
- Dataflow: `Ref("node.field.subfield")`, `Ref("in.<input>")`; in JSON
  these are `{"$ref": "node.field"}`.
- Tool names exactly as `gap tools list` prints them; `robot.*`/`sim.*`
  are connector-provided at run time.
- Skill subgraphs should follow the skill's SKILL.md "recommended
  subgraph state flow" — the canonical scripts and wiring are the
  contract the skill was validated with.

## Validate → run loop

```bash
uv run gap run my_graph --validate-only      # rule-coded errors
MUJOCO_GL=egl uv run gap run my_graph --sim libero_object/0
```

Workflow rules: **W1** version == 3 · **W2** ≥1 edge from START ·
**W3** every edge endpoint is START/END/a declared node/a conditional
target · **W4** all non-END nodes reachable from START · **W5** every
non-END node has an outgoing edge (or is streaming / a conditional-edge
source) · **W6** conditional_edges source declared, targets valid ·
**W7** every `subgraph` node's `ref` resolves · **W8** declared inputs
are satisfiable (observation stream or upstream outputs).

Subgraph rules: **S1/S2** inner reachability from START · **S3**
streaming nodes have no outgoing edges · **S4** `streaming` flag matches
the skill contract · **S5** every `$ref` head is a declared producer ·
**S6** outputs bind to declared nodes (never END) · **S7**
`exit.success_values` non-empty · **S8** conditional edges declare
`router_field` (or `null` for router nodes) · **S9** `on_error` is not a
declared node · **S10** `on_error` is not a conditional-edge target ·
**S11** success values consistent with the router field.

## Execute programmatically

```python
import gap
conn = gap.connector.sim("libero", task="libero_object/0",
                         cameras=["agentview", "robot0_eye_in_hand"], seed=0)
result = gap.execute(
    "my_graph", conn,
    skills=None,              # default: resolved registries; or path(s)
    inputs={"object_name": "alphabet soup"},
    checkpoints="warn",       # off | warn | raise
)
# ExecutionResult: success, exit_status, outputs, trace_path,
#                  checkpoint_results, error, duration_s
```

`gap.execute` never raises on workflow failure — inspect
`result.success` / `result.error`. Checkpoints evaluate against
`connector.world_snapshot()` (sim); `raise` aborts on first violation.
Real connectors (`gap.connector.real("franka"|"ur_zed")`) are
SAFETY-GATED — see the skill's standing rules.

## Checkpoints

Declare postconditions per subgraph (`sg.add_checkpoint(name, predicate,
rationale=...)`); predicates receive a `World` snapshot (`world.body(name)`,
contacts, robot view). They are the difference between "the graph ran"
and "the task happened" — author them for every grasp/place/transport.
