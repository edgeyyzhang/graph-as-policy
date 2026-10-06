# rehearse_loop: improve a graph by rehearsing it

An authoring agent edits a policy graph, rehearses it in simulation, reads what
happened, and edits it again. The released GaP code runs graphs but does not feed
simulator results back to the author; this example adds that loop.

The only objective is task success, judged by the simulator. The feedback describes
what happened. It does not rank causes or suggest edits.

## The two sides

| | Workspace | Trusted directory |
|---|---|---|
| Who uses it | The authoring agent | The broker and you |
| Holds | The working graph, the skill library, results of the visible cases, two commands | The configuration, every round's full output, the held-out results, the ledger |
| Example location | `../gap_loops/soup_01` | `outputs/rehearse_loop/soup_01` |

The agent reaches the simulator only by writing a request file. The broker answers it.
Start exactly one broker per trusted directory: a second one would also answer every
request, so each rehearsal would run twice and the ledger would count it twice.
Each round's two rehearsals (visible and held-out, about 9 GB of GPU memory each) share
one GPU: the least-used one, or the one named by `CUDA_VISIBLE_DEVICES` when the broker is
started with it. Run one loop at a time; a rehearsal is CPU-heavy.

## Run a loop

All commands run from the graph-as-policy checkout. `gap.sh` runs the checkout's
own environment, `.venv/bin/python` (override with `GAP_PYTHON`), sets the Python
path, loads the OpenRouter key from `~/.config/gap/credentials.env` and points
LIBERO at the project's configuration. To build the environment:

```bash
UV_PYTHON=3.10 uv sync --extra pyroki --no-install-package nvidia-curobo
uv pip install --python .venv/bin/python "av>=14"
```

The authoring agent is Claude Code (`claude` on PATH, your subscription). The
perception tool bundles have their own venvs under `../open-robot-skills/tools/`.

```bash
# 1. Create the workspace and the trusted directory.
examples/rehearse_loop/gap.sh rehearse-loop init \
    ../gap_loops/soup_01 outputs/rehearse_loop/soup_01 \
    --graph examples/libero_quickstart/graph \
    --sim libero_object_all_variance/0 \
    --instruction "Pick the alphabet soup and place it in the basket" \
    --visible 1-8 --holdout 9-16 --budget 4

# 2. Start the broker. Leave it running.
examples/rehearse_loop/gap.sh rehearse-loop serve outputs/rehearse_loop/soup_01

# 3. In another terminal, start the agent.
examples/rehearse_loop/launch_agent.sh ../gap_loops/soup_01 --max-usd 25

# Steps 2 and 3 together, detached, with the broker stopped when the agent is done:
#   tmux new -d -s soup 'examples/rehearse_loop/run_loop.sh ../gap_loops/soup_01 outputs/rehearse_loop/soup_01 --max-usd 25'

# 4. Follow the agent, and read the ledger (it includes the held-out results).
examples/rehearse_loop/agent_log.py ../gap_loops/soup_01.agent.jsonl
examples/rehearse_loop/gap.sh rehearse-loop status outputs/rehearse_loop/soup_01
```

The final graph is `results/graph/` in the workspace, and the agent's report is
`results/notes/REPORT.md`.

## Report on a policy

```bash
CUDA_VISIBLE_DEVICES=2 examples/rehearse_loop/policy_report.sh \
    ../gap_loops/soup_01/results/graph libero_object_all_variance/0 outputs/policy_report/soup_01
```

This samples 4 of the 16 layouts at random (`--trajs N`, `--seed S`; `--trajs all`
for every layout), rehearses the graph on them with video, then writes
`outputs/policy_report/soup_01/report/report.html`: the task, the graph, a video of
one successful case with the active node highlighted, every evaluated case's result
with its own two-panel video, and the parameters of every node. The page embeds all
media and stays under 16 MB, so it can be published as a single page. `report.json`
beside it holds the numbers.

## Rehearse a graph without the loop

```bash
examples/rehearse_loop/run.sh GRAPH_DIR libero_object_all_variance/0 1-8 outputs/rehearse_loop/my_run --frames
```

Add `--previous OTHER_RUN` to compare with an earlier run of the same cases.

## What a rehearsal writes

```
<out>/
  feedback/main.md                 the main graph
  feedback/subgraphs/<name>.md     one file per subgraph, same sections
  trajectories/case_NNNN/*.md      state during each node, one row every 5 simulator steps
  cases/case_NNNN/
    case.json                      the full record of the case
    trajectory.jsonl               privileged state after every simulator step, every body
    frames/                        one camera image per node (with --frames)
    trace/                         the GaP trace: node inputs and outputs, tool calls, images
  feedback.md, feedback.json       the whole-run summary
```

Every feedback file has six sections: graph definition, task success, cases, counts
over cases, checkpoints, changes since the previous round. For each node visit it
gives the action (the node's inputs), the outputs and the change of state.

## What confines the agent

| Layer | What it does |
|---|---|
| Claude Code restricted mode | File tools reach the workspace only. User settings, memory, plugins and MCP servers are not loaded. |
| Permission rules in `launch_agent.sh` | Edits are allowed under `results/` and `.cache/` and denied under `eval/`, `runtime/` and `.requests/`. Shell commands other than `./eval/run` are refused, except read-only commands on files inside the workspace, which Claude Code allows by itself. |
| `./eval/run` | Runs each command under Landlock and seccomp: read access to the workspace, write access to `results/`, `.cache/` and `.requests/`, no network. |
| Script contract (`gap/rehearse/loop/protocol.py`) | A new or changed graph script may not import file, process or network modules, call `open`, or use `sim.*` tools. |
| Held-out cases | Run on the trusted side and recorded in the ledger only. |

Known limits:

- The script contract is a screen, not a security boundary. Graph scripts run inside
  the rehearsal process, next to the simulator.
- The feedback contains the simulator's ground truth. Nothing mechanical stops the
  agent from copying a true position into the graph as a constant. The prompt forbids
  it, and the held-out cases are what would expose it.
- The step sampler depends on the LIBERO backend. Other simulators would record the
  state at node boundaries only.
- Recording every step adds 10 to 23 percent to the run time of motion nodes.

## Files

| File | Role |
|---|---|
| `gap.sh` | Runs any `gap` command with the environment set |
| `run.sh` | Shortcut for one `gap rehearse` |
| `launch_agent.sh` | Starts Claude Code headless as the authoring agent |
| `run_loop.sh` | Broker and agent together for one loop; stops the broker when the agent finishes |
| `policy_report.sh` | Samples layouts, evaluates a graph on them with video and writes the policy card |
| `policy_card.py` | The policy card: one self-contained `report.html` with the task, the graph drawing, a two-panel policy video, per-case results with failure videos, and the parameters of every node |
| `agent_log.py` | Prints a readable summary of the agent's transcript |
| `reference_fix/` | A hand-written fix for the quickstart graph: a node that removes robot points from the container's point cloud. Kept out of every workspace. |

The code is in `gap/rehearse/` (rehearsal, feedback, trajectories) and
`gap/rehearse/loop/` (workspace, broker, contract). Tests are in `tests/rehearse/`.
