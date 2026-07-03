# Examples

Examples ordered as a learning path — from the end-to-end quickstart to
the release gate and real robots. Install per
[Installation](../getting-started/installation.md); the **Needs** column
names the `uv sync --extra …` set, plus any credential or hardware.
**Time** is steady-state wall-clock: the first run of a session
additionally pays one-time costs — cold vision-model loads, and for
CuRobo examples a one-time CUDA-kernel JIT (~40 s) — so expect minutes
before the quoted per-trial rate kicks in.

All examples live under [examples/](gh-engine:examples) in the engine
repo, each with a README and runnable code.

## Start here

Zero to a verified rollout.

| Example | What it shows | Needs | Time |
|---|---|---|---|
| [libero_quickstart](libero-quickstart.md) | The end-to-end hero: DINO + SAM3 + VLM perception → OBB grasp → transport, verified against sim ground truth | `quickstart` + GPU + LLM key | ~25–55 s/trial |
| [grocery_packing](grocery-packing.md) | A static graph with a real **backward edge**: perceive → grasp → transport, looped until every item is in the basket, with unprivileged VLM termination | `grocery` (CUDA) + GPU + LLM key | ~min/item |

## Author & generate graphs

Two producers of the same artifact — `gap.builder` by hand, the LLM
pipeline from one instruction; both pass the same validator and run with
the same `gap run`.

| Example | What it shows | Needs | Time |
|---|---|---|---|
| [build_a_graph](build-a-graph.md) | The full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, optional `--execute` | `uv sync` | ~1 min |
| [generate_a_graph](generate-a-graph.md) | Instruction → validated workflow dir; CLI + Python, all providers | `uv sync` + LLM key | minutes |
| [agent_quickstart](agent-quickstart.md) | Step-by-step: Claude Code (one skill) generates a graph from a sentence → validate → sim run → video | `quickstart` + GPU + LLM key + Claude Code | ~10 min |

## Benchmarks & evaluation

The configs that gate releases.

| Example | What it shows | Needs | Time |
|---|---|---|---|
| [grocery_fulfillment](grocery-fulfillment.md) | The flagship acceptance family: graphs **LLM-generated per task**, nothing hand-written | `grocery` (CUDA) + LLM key | minutes (smoke) |
| [benchmark](benchmark.md) | Grid harness configs: 1-cell smoke → position-variance (posvar) grid → the release gate (`--gate` exits non-zero below the config's threshold) | `grocery` (CUDA) + LLM key | minutes → hours |

## Learned policies

Graphs and policies are complements: graphs steer, collect for, and verify
learned policies.

| Example | What it shows | Needs | Time |
|---|---|---|---|
| [steered_policy](steered-policy.md) | Hybrid graphs: perceive + hover above the target, then hand control to a VLA (vision-language-action) policy | `quickstart` + `policy` + LLM key | ~min/trial |
| [collect_and_train](collect-and-train.md) | Graph as scripted expert → HDF5/LeRobot dataset → train externally → policy back in a graph | `quickstart` + `policy` + LLM key | collection: s/episode |

## Real robots

**These move hardware. Read [Safety](../real-robots/safety.md) before
anything else.**

| Example | What it shows | Needs | Time |
|---|---|---|---|
| [cable_ur](cable-ur.md) | Perception-only UR + ZED connector — motion structurally impossible (read-only RTDE) | `real` + ZED SDK + UR arm | seconds |
| [real_franka_pick_place](real-franka-pick-place.md) | Franka + Robotiq pick-place loop via the robots_realtime bridge | `real` + Franka/Robotiq/ZED | s/cycle |

```{toctree}
:maxdepth: 1
:hidden:

libero-quickstart
grocery-packing
build-a-graph
generate-a-graph
agent-quickstart
grocery-fulfillment
benchmark
steered-policy
collect-and-train
cable-ur
real-franka-pick-place
```
