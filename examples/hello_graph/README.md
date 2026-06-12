# hello_graph — your first graph in 2 minutes (CPU only)

> **What:** Build → validate → render a real robot-skill graph · **Needs:** `uv sync` (any OS, no GPU, no API key) · **Time:** ~2 min

The smallest real gap workflow: perceive a target object, then grasp it —
two subgraphs, each owned by an actual skill from
[open-robot-skills](https://github.com/graph-robots/open-robot-skills).
One command builds it, validates it with the same validator that gates
LLM-generated graphs, and renders it to a PNG:

```bash
uv run python examples/hello_graph/hello.py
```

```
Built a real, validated robot-skill graph — no GPU, no API key.

  outputs/hello_graph/workflow.json   the graph (2 subgraphs, validated, 0 errors)
  outputs/hello_graph/graph.png   <-- open this
  ...
```

Open `outputs/hello_graph/graph.png`:

![rendered hello_graph workflow](../../docs/assets/hello_graph.png)

## The essence

The core of the grasp subgraph, in builder calls (see [hello.py](hello.py)
for the complete ~140 lines):

```python
from gap.builder import Ref, Subgraph

grab = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
grab.add_input("target_obb", type_name="OrientedBoundingBox")
grab.add_node("open", type="tool", tool="robot.open_gripper")
grab.add_node("candidates", type="tool", tool="geometry.top_down_grasp_candidates",
              inputs={"obb": Ref("in.target_obb")})
grab.add_node("descend", type="tool", tool="robot.go_to_pose",
              inputs={"pose": Ref("candidates.candidates.poses.0")})
grab.add_node("close", type="tool", tool="robot.close_gripper")
grab.add_exit("grasped")
grab.set_on_error("failed")
```

hello.py then wires the edges (`grab.add_edge(...)`), pairs this with the
perception subgraph, and routes their exits at the top level of a
`Workflow` — `wf.save(...)` runs parse + structural validation on the
result.

## What just happened

- **The typed builder produced the same artifact the LLM pipeline emits** —
  a workflow directory (`workflow.json` + `scripts/`) that `gap run` and
  `gap.execute` execute as-is.
- **The graph was validated twice**: structurally on `wf.save(...)`, then
  with the real skill registry in the loop (skill names, tool names, script
  schemas) — exactly what `gap run <dir> --validate-only` does.
- **The render is the graph that would run**: blue tool calls dispatch to
  connector/bundle tools, the green script node is the perception bundle's
  canonical script, green/red arrows are the success/failure routes the
  executor follows.

## Next steps

| Where | What you get |
|---|---|
| `uv run gap run outputs/hello_graph --validate-only` | the CLI doing what this script just did |
| [build_a_graph](../build_a_graph/) | the full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, `--execute` |
| [generate_a_graph](../generate_a_graph/) | let the LLM write a graph like this from one instruction |
| [libero_quickstart](../libero_quickstart/) | run a graph like this end to end on the LIBERO sim (GPU) |
| [The 15-minute tour](../../docs/quickstart.md) | hello_graph → quickstart → generate, with the trace open |
