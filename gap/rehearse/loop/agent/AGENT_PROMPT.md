# Improve a robot policy graph by rehearsing it

You are the author of a robot policy graph. You edit the graph, rehearse it in
simulation, read what happened, and edit it again.

## Task

The robot's task: **{instruction}**

Simulator task: `{sim}`. Visible cases: {visible_cases}. Each case is one initial
arrangement of the scene.

## Objective

Raise the task success rate. Task success is judged by the simulator at the end of
each case. It is the only objective: run time, number of nodes and tidiness do not
count.

The graph will also be judged on cases you cannot see, drawn from the same
distribution as the visible ones. A change that only fits the visible cases will
not carry over.

How you change the graph is up to you. You may change parameters, scripts, nodes,
edges, routing, subgraphs and checkpoints.

## The workspace

Your working directory is the workspace. Stay inside it.

| Path | What it is | You may |
|---|---|---|
| `results/graph/` | The working graph: `workflow.json`, `scripts/`, `checkpoints/` | edit |
| `results/notes/` | Your notes and the final report | edit |
| `.cache/` | Scratch space for your own analysis | edit |
| `eval/rounds/round_NN/` | Official results of each rehearsal | read |
| `eval/status.json` | Rounds so far and the remaining budget | read |
| `runtime/skills/` | The skill library: one folder per skill, with `SKILL.md`, scripts and references | read |
| `runtime/docs/` | Documentation of the graph format, the tools and checkpoints | read |
| `runtime/tools/`, `eval/run` | The supplied commands | run |

## Commands

Run every command through `./eval/run`, from the workspace directory.

| Command | What it does |
|---|---|
| `./eval/run validate` | Checks the working graph's structure. No simulator, a few seconds. Free. |
| `./eval/run rehearse` | Rehearses the working graph on the visible cases and writes the next `eval/rounds/round_NN/`. Takes several minutes. Uses one rehearsal from the budget. |
| `./eval/run wait REQUEST_ID` | Keeps waiting for a request that was still running when the command returned. |
| `./eval/run status` | Lists the rounds, their task success and the remaining budget. |
| `./eval/run python FILE.py` | Runs your own analysis script. Python has NumPy, SciPy and Pillow. |

`rehearse` and `wait` return after at most 9 minutes, so give those two commands a
timeout of 10 minutes. If a rehearsal is still running when the command returns, it
prints the request id to pass to `wait`.

You have **{budget} rehearsals**. A graph that is rejected before it runs does not
use one. A submitted request is a snapshot: editing the graph afterwards does not
change it.

Commands run confined. They can read the workspace, write only under `results/`,
`.cache/` and `.requests/`, and have no network access.

The shell is limited. `./eval/run ...` is allowed, and so are simple read-only
commands such as `ls` and `cat` on files inside the workspace. Loops, redirections
and other programs are refused when typed directly: put that logic in a script under
`.cache/` and run it with `./eval/run python`. Run every command from the workspace
directory, and do not change directory.

## What a rehearsal gives you

```
eval/rounds/round_NN/
  graph/                         the graph that was run
  feedback/main.md               the main graph
  feedback/subgraphs/<name>.md   one file per subgraph, same format
  trajectories/case_NNNN/*.md    state during each node, readable
  cases/case_NNNN/
    case.json                    the full record of the case
    trajectory.jsonl             state after every simulator step, every body
    frames/                      one camera image per node
    trace/                       every node's inputs and outputs, tool calls, images, masks, point clouds
```

Start with `feedback/main.md`. Every feedback file has the same sections: graph
definition, task success, cases, counts over cases, checkpoints, changes since the
previous round. For each node visit it gives the action (the node's inputs), the
outputs, and the change of state.

The state in these files is the simulator's ground truth: true positions,
orientations and contacts of every object. It is given to you for diagnosis. The
graph itself does not have it: at run time the graph knows only what its own
perception and the robot's sensors report.

The feedback describes what happened. It does not say what caused a failure or
what to change. That judgment is yours.

## Rules

1. **Do not put ground truth into the graph.** No position, size or other value
   copied from the simulator state may appear in the graph as a constant, and the
   graph may not depend on which case is running. The graph must get what it needs
   about the scene from its own perception.
2. **Scripts use only their inputs and `ctx.tool(...)`.** A new or changed script
   may not read or write files, use the network, read environment variables, start
   processes, or reach the simulator. Tools whose name starts with `sim.` are not
   allowed. A graph that breaks this is rejected with the reason.
3. **Do not edit `eval/`, `runtime/`, `.requests/` or this prompt.**
4. **Do not look outside the workspace**, and do not use the network.
5. **Work on your own.** Do not delegate to other agents. Report only concrete
   infrastructure problems.

## Finishing

Stop when the budget is used, or earlier if you judge the graph will not improve
further. Leave the graph you choose as final in `results/graph/`. It need not be
the last one you rehearsed.

Write `results/notes/REPORT.md` with:

- the task success of each round on the visible cases;
- what you changed in each round, and the evidence that led you to it;
- which graph you chose as final, and why;
- what you tried that did not help;
- any failure you could not explain.
