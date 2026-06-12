# gap examples

Ten examples, ordered as a learning path — from a CPU-only hello-world to
the release acceptance gate and real robots. Install per
[README › Get started](../README.md#get-started); the **Needs** column
names the `uv sync --extra …` set from the README's
[environment table](../README.md#pick-your-environment), plus any
credential or hardware. **Measured** lists only numbers measured on this
repo at the committed `uv.lock` — a dash means no claim, not a failure.

## Start here

Zero to a verified rollout.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [hello_graph](hello_graph/) | Build → validate → **render** your first graph; the same artifact the LLM emits | `uv sync` (any OS, no GPU, no API key) | ~2 min | — |
| [libero_quickstart](libero_quickstart/) | The end-to-end hero: DINO + SAM3 + VLM perception → OBB grasp → transport, verified against sim ground truth | `quickstart` + GPU + LLM key | ~25–55 s/trial | **9/10 grasp · 7/10 task** (10 seeds) |

## Author & generate graphs

Two producers of the same artifact — `gap.builder` by hand, the LLM
pipeline from one instruction; both pass the same validator and run with
the same `gap run`.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [build_a_graph](build_a_graph/) | The full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, optional `--execute` | `uv sync` (CPU to build) | ~1 min | — |
| [generate_a_graph](generate_a_graph/) | Instruction → validated workflow dir; CLI + Python, all providers | `uv sync` + LLM key | minutes | — |

## Benchmarks & evaluation

The numbers that gate releases.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [grocery_fulfillment](grocery_fulfillment/) | The flagship acceptance family: graphs **LLM-generated per task**, nothing hand-written | `grocery` (CUDA) + LLM key | minutes (smoke) | **10/10 dev gate** (2026-06-11) |
| [benchmark](benchmark/) | Grid harness configs: 1-cell smoke → position-variance (posvar) grid → the release gate (`--gate` exits non-zero below threshold) | `grocery` (CUDA) + LLM key | minutes → hours | release gate requires **≥90%** (10 tasks × 50 trials) |

## Learned policies

Graphs and policies are complements: graphs steer, collect for, and verify
learned policies.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [steered_policy](steered_policy/) | Hybrid graphs: perceive + hover above the target, then hand control to a VLA (vision-language-action) policy | `quickstart` + `policy` + LLM key | ~min/trial | — |
| [collect_and_train](collect_and_train/) | Graph as scripted expert → HDF5/LeRobot dataset → train externally → policy back in a graph | `quickstart` + `policy` + LLM key | collection: s/episode | — |

## Real robots

**These move hardware. Read [../docs/safety.md](../docs/safety.md) before
anything else.**

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [cable_ur](cable_ur/) | Perception-only UR + ZED connector — motion structurally impossible (read-only RTDE) | `real` + ZED SDK + UR arm | seconds | — |
| [real_franka_pick_place](real_franka_pick_place/) | Franka + Robotiq pick-place loop via the robots_realtime bridge | `real` + Franka/Robotiq/ZED | s/cycle | — |

## Adding an example

Match the header strip every example carries directly under its title:

```markdown
> **What:** <one clause> · **Needs:** <extras + credential/hardware> · **Time:** <estimate> · **Measured:** <repo-measured numbers>
```

Omit **Measured** when there is nothing measured; real-robot examples
append `· **Safety:** read [../../docs/safety.md](../../docs/safety.md) first`.
**Needs**, **Time**, and **Measured** values must match this gallery's row
for the example, verbatim; **What** may be condensed.
