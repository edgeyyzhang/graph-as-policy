# Generating the tape-handover graph with an LLM

`build_graph.py` hand-authors the workflow; `gap generate` lets an LLM author
the same artifact from a natural-language instruction. The generated output is
a drop-in replacement — run it with `run.py <GRAPH_DIR>` or
`gap run <GRAPH_DIR> --sim libero_yam_tabletop/2`.

## Setup

Source the shared environment before running anything:

```bash
source /home/shamakg/gap/env.sh
```

This sets `GAP_LLM_PROVIDER=vertex`, `GAP_LLM_MODEL=claude-opus-4-8`, and
`MUJOCO_GL=egl` (needed for sim runs). All `gap generate` calls below inherit
these without extra flags.

## Generate

```bash
cd /home/shamakg/gap/graph-as-policy

.venv/bin/gap generate \
  "pick up the yellow tape with arm 0 and hand it to arm 1, then place it on the duct tape" \
  --skills /home/shamakg/gap/open-robot-skills \
  --skills /home/shamakg/gap/tsh-skills \
  --out examples/tape_handover_yam/generated
```

The agent writes `examples/tape_handover_yam/generated/task_00/` containing:
- `workflow.json` — the DAG + the `tsh-bimanual-handover` subgraph
- `scripts/` — copies of the skill's canonical scripts
- `agent_traces/` — per-agent LLM transcripts for debugging

## Run the generated graph

```bash
# via the example driver (adds the collision report + two free-camera renders):
.venv/bin/python examples/tape_handover_yam/run.py \
  examples/tape_handover_yam/generated/task_00

# or via the CLI (records outputs/run_*/run_video.mp4):
.venv/bin/gap run examples/tape_handover_yam/generated/task_00 \
  --sim libero_yam_tabletop/2 \
  --skills ../open-robot-skills --skills ../tsh-skills
```

## What the LLM has to work with

The registries expose **`tsh-bimanual-handover`** with five canonical scripts
(see its SKILL.md for the recommended 6-state flow and the data-flow refs):

| Script | Purpose |
|---|---|
| `perceive_duct` | RGB-D top-face centre of the place target (run FIRST, clean view) |
| `perceive_tape` | RGB-D tape grasp point + point cloud + ring radii |
| `pickup` | Ring-grasp at the perceived point, lift, measure `tape_in_giver` |
| `bimanual_exchange` | Giver presents (tape-as-end-effector) → receiver threads → grip swap → measure `receiver_offset` |
| `place` | Yaw-sweep a reachable pose, lay the tape flat on the duct |

The only values the workflow supplies are the object keys
(`"yellow_tape_1"`, `"duct_tape_1"`) and the arm ids (giver `0`, receiver `1`).
All numeric tuning (meeting point, offsets, clearances) lives in the skill's
`scripts/constants.py` — the SKILL.md hard rules forbid the agent from
inventing values for them.

## Comparing generated vs reference

```bash
# Validate structure only (no sim, no GPU)
.venv/bin/gap run examples/tape_handover_yam/generated/task_00 --validate-only

# Diff the workflow graphs
diff <(.venv/bin/gap viz --json examples/tape_handover_yam) \
     <(.venv/bin/gap viz --json examples/tape_handover_yam/generated/task_00)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `robot.get_observation` has no cameras | The graph was run without the LIBERO-YAM connector — use `--sim libero_yam_tabletop/2` (or `run.py`), not a bare/`libero` connector |
| `plan_tool_move: no plan …` | Target out of the arm's workspace — check the tape position against the reachability grid in CLAUDE.md |
| Workflow routes to `abort` | Read the failing node in `dag_trace.json` / `node_data/<node>/calls` |
| `GAP_LLM_PROVIDER not set` | `source /home/shamakg/gap/env.sh` was not sourced |
