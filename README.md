<div align="center">

# gap — graph as policy

**The policy is the graph.**
A natural-language task is compiled by an LLM agent pipeline into a typed,
verified execution graph of robot skills — and the graph, not a monolithic
policy, is what runs on simulators and real robots.

<!-- TODO(release): real badge targets once the repos are public + CI is up:
     build status, PyPI version, docs, community chat. -->
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Skills: open-robot-skills](https://img.shields.io/badge/skills-gap--skills-orange.svg)](https://github.com/graph-robots/open-robot-skills)

</div>

```python
import gap

conn = gap.connector.sim("libero", task="libero_object/0")        # one process, no extra terminals
result = gap.execute("examples/libero_quickstart/graph", conn)    # skills auto-discovered
graph = gap.agent.generate_sync("pick up the soup can and put it in the basket")
gap.viz.serve("outputs")                                          # browse the trial trace
```

Skills live in the sibling **[open-robot-skills](https://github.com/graph-robots/open-robot-skills)** repo (Anthropic
Agent Skills format, contributable) and are discovered by path — clone the
two repos side by side and every command finds them, no flags needed.
Everything runs **in one process**: env, vision models, IK. No gRPC, no
protobufs, no self-hosted model servers.

## Get started

Requirements: **Linux + NVIDIA GPU (≥ ~10 GB VRAM) + EGL**, an LLM API key
(Anthropic / OpenAI-compatible / Vertex), and [uv](https://docs.astral.sh/uv/).
First run downloads ~3.5 GB of model weights (`HF_TOKEN` for gated repos).

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git

cd graph-as-policy
uv sync --extra quickstart            # one venv: engine + LIBERO sim + perception models
uv run gap skills check --download    # verify skill bundles + prefetch model weights
```

Run the quickstart graph — LIBERO sim + Grounding DINO + SAM3 + a hosted
VLM + in-process IK, executing a perceive → grasp → transport graph with
ground-truth checkpoint verification:

```bash
export ANTHROPIC_API_KEY=...          # or another provider, see "LLM providers"

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                        # browse the recorded trial at localhost:9432
```

Generate a graph from language instead of running the checked-in one:

```bash
uv run gap generate "pick up the alphabet soup can and place it in the basket"
```

(`uv run` needs no venv activation; `source .venv/bin/activate` once if you
prefer plain `gap …`. pip also works — see [Installation details](#installation-details).)

### Pick your environment

One `uv sync` per setup; everything is declared in `pyproject.toml` and
pinned by the committed `uv.lock` (the measured-release environment).

| Command | What you get |
|---|---|
| `uv sync` | engine only — tests, validation, graph authoring (CPU, any OS) |
| `uv sync --extra quickstart` | + LIBERO sim + SAM3 / Grounding DINO / geometry bundles |
| `CUDA_HOME=/usr/local/cuda uv sync --extra grocery` | + CuRobo motion planning (CUDA build) — the acceptance-benchmark set |
| `CUDA_HOME=/usr/local/cuda uv sync --extra all` | + policy serving, Gemini-ER, Vertex provider, Ray workers |

Per-bundle dependencies and setup are declared bundle-by-bundle in
[open-robot-skills/pyproject.toml](https://github.com/graph-robots/open-robot-skills/blob/main/pyproject.toml) (one extra per
bundle); the extras above are the curated sets spanning both repos.

## Why gap

- 🧭 **Language → typed graph.** A coordinator → subgraph-agents → checkpoint-agent
  pipeline compiles one instruction into a validated workflow of skills,
  with a script-fix loop on validation errors.
- ✅ **Verified, not hoped.** LLM-authored postcondition checkpoints are
  enforced against simulator ground truth at every subgraph exit
  (`--checkpoints warn|raise`).
- 🧩 **Skills are contributable.** Strategies and model tools live in
  [open-robot-skills](https://github.com/graph-robots/open-robot-skills) as Agent Skills bundles — one directory, one
  PR. The LLM composes them; you can too.
- ⚡ **One process, no servers.** Env + vision models + IK in-process; plain
  numpy + TypedDicts as the data contract. Ray is an opt-in extra, never a
  prerequisite.
- 🔍 **The trace is the product.** Every run records `workflow.json`,
  `dag_trace.json`, per-node I/O and assets; `gap viz` browses them,
  `gap trace-diff` compares them.
- 📊 **Benchmarked with a gate.** A grid harness (modes × families × seeds)
  with `--gate`: the grocery-fulfillment acceptance config must clear
  **≥90% success** for a release.

## Examples

| Example | What it shows | Needs |
|---|---|---|
| [build_a_graph](examples/build_a_graph/) | Author the pick-and-place graph in Python with `gap.builder` | none to build |
| [generate_a_graph](examples/generate_a_graph/) | Instruction → validated workflow dir, CLI + Python, all providers | LLM key |
| [libero_quickstart](examples/libero_quickstart/) | The end-to-end demo; **measured 9/10 grasp, 7/10 task success** over 10 seeded trials | `quickstart` |
| [grocery_fulfillment](examples/grocery_fulfillment/) | The flagship acceptance benchmark; **10/10** on the dev gate, ≥90% release gate | `grocery` |
| [steered_policy](examples/steered_policy/) | Hybrid graphs: perceive + hover, then hand control to a learned VLA policy | `quickstart` + `[policy]` |
| [collect_and_train](examples/collect_and_train/) | Graph as scripted expert → HDF5/LeRobot dataset → train → policy node | `quickstart` + `[policy]` |
| [benchmark](examples/benchmark/) | Benchmark configs: smoke, posvar grid, the acceptance gate | `grocery` |
| [cable_ur](examples/cable_ur/) | Real UR + ZED, perception-only connector (motion structurally impossible) | `[real]` + ZED SDK |
| [real_franka_pick_place](examples/real_franka_pick_place/) | Real Franka + Robotiq pick-place loop via robots_realtime — **read [docs/safety.md](docs/safety.md) first** | `[real]` + hardware |

## CLI

| Command | Purpose |
|---|---|
| `gap run <graph> [--sim SUITE/TASK \| --real {franka,ur_zed}] [--validate-only]` | Execute (or just validate) a graph; tracing on by default → `outputs/` |
| `gap generate "<instruction>" [--provider P] [--model M] [--out DIR]` | LLM pipeline: instruction → validated workflow dir |
| `gap benchmark <config.yaml> [--gate] [--resume]` | Benchmark grids; `--gate` exits non-zero below threshold |
| `gap viz [--root outputs] [--port 9432]` | Trial browser: graph swimlanes, per-node I/O, assets, videos |
| `gap skills list \| check [--download] \| table \| new <name>` | Bundle catalog, validation, weight prefetch, scaffolding |
| `gap policy serve <preset>` | One-command policy serving (`pi05-libero`, `molmoact-libero`) |
| `gap trace-diff <trial_a> <trial_b>` | Structural diff of two recorded traces |

Every command that takes a skills path accepts `--skills PATH`, but none
needs it: the open-robot-skills checkout is auto-discovered from `$GAP_SKILLS_PATH`
or the side-by-side layout.

## Authoring graphs in Python

LLM generation is one producer of graphs, not the only one. `gap.builder`
emits the identical artifact (`workflow.json` + `scripts/` + `checkpoints/`)
through the same validation:

```python
from gap.builder import Workflow, Subgraph, Ref

grasp = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
grasp.add_input("target_obb", type_name="OrientedBoundingBox")
grasp.add_node("candidates", type="tool", tool="geometry.top_down_grasp_candidates",
               inputs={"obb": Ref("in.target_obb")})
grasp.add_node("descend", type="tool", tool="robot.go_to_pose",
               inputs={"pose": Ref("candidates.candidates.poses.0")})
grasp.add_checkpoint("target_held",
                     lambda world: world.body("alphabet soup").is_grasped(),
                     rationale="gripper actually holds the can after close")
# ... edges, exits, more subgraphs ...

wf = Workflow(name="pick_into_basket")
wf.add_subgraph(grasp)
wf.save("my_graph/workflow.json")     # parse + structural validation
```

[examples/build_a_graph](examples/build_a_graph/) is the complete runnable
version; [docs/runtime.md](docs/runtime.md) specifies the JSON schema and
executor semantics.

## LLM providers

| Provider | Setup |
|---|---|
| `anthropic` (default) | `export ANTHROPIC_API_KEY=...` |
| `openai` (incl. OpenRouter / vLLM) | `export OPENAI_API_KEY=...`; custom endpoints via config YAML |
| `vertex` | `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT`; claude-* and gemini-* both route |

`--provider/--model` per call, or pin everything in a config YAML. The same
provider layer drives the `vlm` perception bundle (`GAP_VLM_PROVIDER`, …) —
see [examples/libero_quickstart](examples/libero_quickstart/README.md#vlm-provider).

## Architecture

```
┌────────────────────────── gap (engine) ──────────────────────────┐
│ agent/      instruction ─► coordinator ─► subgraph agents        │
│             ─► checkpoint agent ─► validate / fix ─► graph       │
│ runtime/    executor (super-steps, streaming, Send), validator,  │
│             tracing, policy loop, verify/ (World, checkpoints)   │
│ tools/      ToolRegistry: typed schemas, tags→guards, dispatch   │
│ connector/  sim()/real(), env registry, in-process pyroki IK,    │
│             rr_launcher, data collector                          │
│ envs/       libero (+perturbed), franka_real, ur_zed, msgpack    │
│ benchmark/  grid harness (modes × families × seeds), --gate      │
│ viz/        FastAPI+React trial browser, replay3d, PDF render    │
└──────────────┬───────────────────────────────▲───────────────────┘
               │ discovers (by path)           │ registers tools
        ┌──────▼───────────────────────────────┴──────┐
        │ open-robot-skills (Agent Skills format, contributable)
        │ tools/   sam3, grounding-dino, gemini-er,    │
        │          molmo, vlm, curobo, geometry        │
        │ skills/  perceiving-objects(-oneshot/        │
        │          -multiview/-parts), grasping-*,     │
        │          transporting-objects, tracking-     │
        │          objects, running-policies           │
        └──────────────────────────────────────────────┘
```

- **Tools vs skills.** Tools are *what the robot can compute* (model-named
  bundles exposing typed functions); skills are *what the robot can do*
  (strategies that own subgraphs). Connector tools (`robot.*`/`sim.*`) are
  the embodiment surface, shipped by the engine.

Docs: [runtime & schema](docs/runtime.md) · [skill authoring](docs/skills.md)
· [safety](docs/safety.md) · [design doc](docs/design.md)
· [contributing](CONTRIBUTING.md)

## Installation details

- **uv (recommended).** The commands above. `uv sync` reads the committed
  `uv.lock` — the exact environment the acceptance benchmark was measured
  on — and resolves the vendored sim submodules and the sibling open-robot-skills
  checkout automatically (`[tool.uv.sources]`).
- **Forgot `--recurse-submodules`?** `git submodule update --init` fixes
  `uv sync`'s "does not appear to be a Python project" error — the lockfile
  references the vendored submodules even for engine-only installs. The two
  repos cloned side by side are always required for the same reason.
- **pip.** Works, with the submodules and sibling named explicitly (from
  the parent directory of the two checkouts):

  ```bash
  pip install -e "graph-as-policy[libero]" \
    -e graph-as-policy/third_party/Variational-Automation-Benchmark \
    -e graph-as-policy/third_party/robosuite \
    -e "open-robot-skills[quickstart]"
  gap skills check --download
  ```

  Note: sam3's metadata over-pins `numpy==1.26`; uv applies the documented
  override, plain pip may need `pip install "numpy>=2" --force-reinstall`
  afterwards.
- **CuRobo** (`--extra grocery` / `--extra all` / `open-robot-skills[curobo]`)
  compiles CUDA extensions at install time: set `CUDA_HOME` to a toolkit
  matching torch's CUDA (pip users add `--no-build-isolation`).
- **Weights.** pip/uv own all code installs; `gap skills check` only
  verifies, and `--download` prefetches model weights — nothing installs
  behind your back.

## Roadmap

v1 draws the line at language → verified pick-and-place graphs on LIBERO +
real Franka/UR. Cut for scope, returning after v1: execution-feedback graph
repair, learned grasp planners (graspgen/m2t2), a remote model-serving tier,
bimanual support, Isaac-based rehearsal, non-pick-place domains
(articulated, contact-rich, long-horizon), self-hosted pointing VLMs.

## License & attribution

gap and open-robot-skills are MIT-licensed. They stand on third-party work —
LIBERO/LIBERO-PRO, the Variational-Automation-Benchmark fork, robosuite,
robots_realtime, pyroki, SAM3, Grounding DINO, and (optionally) NVIDIA
cuRobo — each under its own license. See **[NOTICE.md](NOTICE.md)** for the
full attribution table.
