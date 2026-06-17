# GaP examples

Examples ordered as a learning path — from the end-to-end quickstart to
the release gate and real robots. Install per
[README › Get started](../README.md#get-started); the **Needs** column
names the bundle install (`uv sync && uv run gap skills install --all`
gets the full set), plus any credential or hardware. **Measured** lists
only numbers measured on this repo at the committed `uv.lock` — a dash
means no claim, not a failure.

## Start here

Zero to a verified rollout.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [libero_quickstart](libero_quickstart/) | The end-to-end hero: DINO + SAM3 + VLM perception → OBB grasp → transport, verified against sim ground truth | `uv sync` + `gap skills install --all` + GPU + LLM key | ~25–55 s/trial | **9/10 grasp · 7/10 task** (10 seeds) |

## Author & generate graphs

Two producers of the same artifact — `gap.builder` by hand, the LLM
pipeline from one instruction; both pass the same validator and run with
the same `gap run`.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [build_a_graph](build_a_graph/) | The full authoring example: 4 subgraphs, a ground-truth checkpoint, recovery actions, optional `--execute` | `uv sync` | ~1 min | — |
| [generate_a_graph](generate_a_graph/) | Instruction → validated workflow dir; CLI + Python, all providers | `uv sync` + LLM key | minutes | — |
| [agent_quickstart](agent_quickstart/) | Step-by-step: Claude Code (one skill) generates a graph from a sentence → validate → sim run → video | `uv sync` + `gap skills install --all` + GPU + LLM key + Claude Code | ~10 min | sentence → **task success in sim, 75 s, on video** |

## Benchmarks & evaluation

The numbers that gate releases.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [grocery_fulfillment](grocery_fulfillment/) | The flagship acceptance family: graphs **LLM-generated per task**, nothing hand-written | `uv sync` + `gap skills install --all` (CUDA) + LLM key | minutes (smoke) | **10/10 dev gate** (2026-06-11) |
| [benchmark](benchmark/) | Grid harness configs: 1-cell smoke → position-variance (posvar) grid → the release gate (`--gate` exits non-zero below threshold) | `uv sync` + `gap skills install --all` (CUDA) + LLM key | minutes → hours | release gate requires **≥90%** (10 tasks × 50 trials) |

## Learned policies

Graphs and policies are complements: graphs steer, collect for, and verify
learned policies.

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [steered_policy](steered_policy/) | Hybrid graphs: perceive + hover above the target, then hand control to a VLA (vision-language-action) policy | `uv sync` + `gap skills install --all` + LLM key | ~min/trial | — |
| [collect_and_train](collect_and_train/) | Graph as scripted expert → HDF5/LeRobot dataset → train externally → policy back in a graph | `uv sync` + `gap skills install --all` + LLM key | collection: s/episode | — |

## Real robots

**These move hardware. Read [../docs/safety.md](../docs/safety.md) before
anything else.**

| Example | What it shows | Needs | Time | Measured |
|---|---|---|---|---|
| [cable_ur](cable_ur/) | Perception-only UR + ZED connector — motion structurally impossible (read-only RTDE) | `uv sync` + `gap skills install --all` + ZED SDK + UR arm | seconds | — |
| [real_franka_pick_place](real_franka_pick_place/) | Franka + Robotiq pick-place loop via the robots_realtime bridge | `uv sync` + `gap skills install --all` + Franka/Robotiq/ZED | s/cycle | — |

## Adding an example

Match the header strip every example carries directly under its title:

```markdown
> **What:** <one clause> · **Needs:** <install line + credential/hardware> · **Time:** <estimate> · **Measured:** <repo-measured numbers>
```

Omit **Measured** when there is nothing measured; real-robot examples
append `· **Safety:** read [../../docs/safety.md](../../docs/safety.md) first`.
**Needs**, **Time**, and **Measured** values must match this gallery's row
for the example, verbatim; **What** may be condensed.
