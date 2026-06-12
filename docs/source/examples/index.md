# Examples

Ten examples, ordered as a learning path — from a CPU-only hello-world to
the release acceptance gate and real robots. Install per
[Installation](../getting-started/installation.md); the **Needs** column
names the `uv sync --extra …` set, plus any credential or hardware.
**Measured** lists only numbers measured on this repo at the committed
`uv.lock` — a dash means no claim, not a failure.

All examples live under [examples/](gh-engine:examples) in the engine
repo, each with a README and runnable code.

## Start here

Zero to a verified rollout.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [hello_graph](hello-graph.md) | Build → validate → **render** your first graph; the same artifact the LLM emits | `uv sync` (any OS, no GPU, no API key) | ~2 min | — |
| [libero_quickstart](libero-quickstart.md) | The end-to-end hero: DINO + SAM3 + VLM perception → OBB grasp → transport, verified against sim ground truth | `quickstart` + GPU + LLM key | ~25–55 s/trial | **9/10 grasp · 7/10 task** (10 seeds) |

## Author & generate graphs

Two producers of the same artifact — `gap.builder` by hand, the LLM
pipeline from one instruction; both pass the same validator and run with
the same `gap run`.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [build_a_graph](build-a-graph.md) | The full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, optional `--execute` | `uv sync` (CPU to build) | ~1 min | — |
| [generate_a_graph](generate-a-graph.md) | Instruction → validated workflow dir; CLI + Python, all providers | `uv sync` + LLM key | minutes | — |
| [agent_quickstart](agent-quickstart.md) | Step-by-step: Claude Code (one skill) generates a graph from a sentence → validate → sim run → video | `quickstart` + GPU + LLM key + Claude Code | ~10 min | sentence → **task success in sim, 75 s, on video** |

## Benchmarks & evaluation

The numbers that gate releases.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [grocery_fulfillment](grocery-fulfillment.md) | The flagship acceptance family: graphs **LLM-generated per task**, nothing hand-written | `grocery` (CUDA) + LLM key | minutes (smoke) | **10/10 dev gate** (2026-06-11) |
| [benchmark](benchmark.md) | Grid harness configs: 1-cell smoke → position-variance (posvar) grid → the release gate (`--gate` exits non-zero below threshold) | `grocery` (CUDA) + LLM key | minutes → hours | release gate requires **≥90%** (10 tasks × 50 trials) |

## Learned policies

Graphs and policies are complements: graphs steer, collect for, and verify
learned policies.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [steered_policy](steered-policy.md) | Hybrid graphs: perceive + hover above the target, then hand control to a VLA (vision-language-action) policy | `quickstart` + `policy` + LLM key | ~min/trial | — |
| [collect_and_train](collect-and-train.md) | Graph as scripted expert → HDF5/LeRobot dataset → train externally → policy back in a graph | `quickstart` + `policy` + LLM key | collection: s/episode | — |

## Real robots

**These move hardware. Read [Safety](../real-robots/safety.md) before
anything else.**

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [cable_ur](cable-ur.md) | Perception-only UR + ZED connector — motion structurally impossible (read-only RTDE) | `real` + ZED SDK + UR arm | seconds | — |
| [real_franka_pick_place](real-franka-pick-place.md) | Franka + Robotiq pick-place loop via the robots_realtime bridge | `real` + Franka/Robotiq/ZED | s/cycle | — |

```{toctree}
:maxdepth: 1
:hidden:

hello-graph
libero-quickstart
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
