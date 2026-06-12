# Hello, Graph

Your first graph in about two minutes — CPU only, no GPU, no API key,
no extras beyond `uv sync`.

This is the smallest real gap workflow: perceive a target object, then
grasp it — two subgraphs, each owned by an actual skill from
[open-robot-skills](gh-skills:skills). One command builds it with the
typed builder API, validates it with the same validator that gates
LLM-generated graphs, and renders it to a PNG. Source:
[examples/hello_graph](gh-engine:examples/hello_graph).

## Run it

```bash
uv run python examples/hello_graph/hello.py
```

```text
Built a real, validated robot-skill graph — no GPU, no API key.

  outputs/hello_graph/workflow.json   the graph (2 subgraphs, validated, 0 errors)
  outputs/hello_graph/graph.png   <-- open this
```

Open `outputs/hello_graph/graph.png`:

![Rendered hello_graph workflow](../_static/hello_graph.png)

Blue tool calls dispatch to connector/bundle tools, the green script node
is the perception bundle's canonical script, and green/red arrows are the
success/failure routes the executor follows. This render *is* the graph
that would run.

Two flags, both optional:

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | `outputs/hello_graph` | Output workflow directory |
| `--skills PATH` | auto-discovered | open-robot-skills checkout (`$GAP_SKILLS_PATH` or the sibling checkout) |

## The code

[hello.py](gh-engine:examples/hello_graph/hello.py) is ~140 lines. The
graph itself is two `Subgraph`s wired into a `Workflow`.

### Perceive

The perception subgraph belongs to the
[perceiving-objects](gh-skills:skills/perceiving-objects) skill: take an
observation, run the bundle's canonical DINO + VLM + SAM3 script, fit an
oriented bounding box:

```python
from gap.builder import Ref, Subgraph, Workflow

see = Subgraph(name="perceive_sg", skill="perceiving-objects")
see.add_node("observe", type="tool", tool="robot.get_observation")
see.add_node(
    "perceive", type="script", script="scripts/perceive_dino_vlm.py",
    inputs={"cameras": Ref("observe.cameras"), "object_name": target},
)
see.add_node(
    "fit_obb", type="tool", tool="geometry.filter_and_compute_obb",
    inputs={"points": Ref("perceive.cloud")},
)
see.add_exit("found")                      # noop success marker
see.set_on_error("not_found")              # any raise → "not_found" exit
for src, dst in [("START", "observe"), ("observe", "perceive"),
                 ("perceive", "fit_obb"), ("fit_obb", "found"),
                 ("found", "END")]:
    see.add_edge(src, dst)
see.set_outputs(target_obb=Ref("fit_obb.obb"))
```

`Ref("node.field")` is the dataflow primitive — it serializes to the JSON
`{"$ref": ...}` form. `add_exit` declares a named success exit;
`set_on_error` maps *any* exception raised inside the subgraph to one
failure exit, so the top level only ever sees `found` or `not_found`.

The `type: script` node references a per-workflow copy of the bundle's
canonical script — `hello.py` copies
`skills/perceiving-objects/scripts/perceive_dino_vlm.py` from the
open-robot-skills checkout into `<out>/scripts/`, exactly what
`gap generate` emits.

### Grasp

The grasp subgraph belongs to `grasping-direct-ik` — open, compute
top-down grasp candidates from the OBB, descend, close:

```python
grab = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
grab.add_input("target_obb", type_name="OrientedBoundingBox")
grab.add_node("open", type="tool", tool="robot.open_gripper")
grab.add_node(
    "candidates", type="tool", tool="geometry.top_down_grasp_candidates",
    inputs={"obb": Ref("in.target_obb")},
)
grab.add_node("descend", type="tool", tool="robot.go_to_pose",
              inputs={"pose": Ref("candidates.candidates.poses.0")})
grab.add_node("close", type="tool", tool="robot.close_gripper")
grab.add_exit("grasped")
grab.set_on_error("failed")
```

`Ref("in.target_obb")` reads the declared subgraph input. The input binds
to the perception subgraph's `target_obb` output *by name* — no explicit
cross-subgraph wiring. `Ref("candidates.candidates.poses.0")` shows a
list-index path into a node output.

### Top level

The workflow routes each subgraph's exit value:

```python
wf = Workflow(name="hello_graph",
              description=f"Perceive the {target}, then grasp it.")
wf.add_subgraph(see)
wf.add_subgraph(grab)
wf.add_node("perceive", type="subgraph", ref="perceive_sg")
wf.add_node("grasp", type="subgraph", ref="grasp_sg")
wf.add_node("done", type="end", status="success")
wf.add_node("abort", type="end", status="failure")
wf.add_edge("START", "perceive")
wf.add_conditional_edges(
    "perceive", {"found": "grasp", "not_found": "abort"},
    router_field="exit")
wf.add_conditional_edges(
    "grasp", {"grasped": "done", "failed": "abort"},
    router_field="exit")
```

### Validate twice, then render

`wf.save(out / "workflow.json")` runs parse + structural validation as it
writes. `hello.py` then re-validates with the real skill registry in the
loop — skill names, tool names, script schemas — exactly what
`gap run <dir> --validate-only` does:

```python
from gap.runtime.validate import validate_workflow
from gap.runtime.workflow import load_workflow
from gap.skills import find_skills_path, load_skills

skills_root = find_skills_path(args.skills, required=True)
issues = validate_workflow(load_workflow(out / "workflow.json"),
                           skill_registry=load_skills(skills_root))
```

Finally it renders the PNG:

```python
from gap.viz.render import render

render(out / "workflow.json", out / "graph.png")
```

:::{note}
`hello.py` sets `sys.dont_write_bytecode = True` before validating.
Validation imports the copied script for schema introspection, which would
otherwise drop a `__pycache__/` into the emitted `scripts/` directory.
:::

## What just happened

- **The typed builder produced the same artifact the LLM pipeline emits** —
  a workflow directory (`workflow.json` + `scripts/`) that `gap run` and
  `gap.execute` execute as-is.
- **The graph was validated twice**: structurally on `wf.save(...)`, then
  with the real skill registry in the loop.
- **The render is the graph that would run** — same nodes, same routes.

Confirm the CLI agrees:

```bash
uv run gap run outputs/hello_graph --validate-only
```

## Next steps

| Where | What you get |
|---|---|
| [build_a_graph](build-a-graph.md) | the full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, `--execute` |
| [generate_a_graph](generate-a-graph.md) | let the LLM write a graph like this from one instruction |
| [libero_quickstart](libero-quickstart.md) | run a graph like this end to end on the LIBERO sim (GPU) |
| [The quickstart](../getting-started/quickstart.md) | hello_graph → quickstart → generate, with the trace open |
| [Builder guide](../authoring/builder.md) | the full `gap.builder` API |
