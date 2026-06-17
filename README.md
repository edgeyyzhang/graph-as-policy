<div align="center">

# graph-as-policy

**The policy is the graph.**
GaP compiles a natural-language task into a typed, verified execution
graph of robot skills — and runs the graph, not a black-box policy, on
simulators and real robots.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Skills: open-robot-skills](https://img.shields.io/badge/skills-open--robot--skills-orange.svg)](https://github.com/graph-robots/open-robot-skills)
[![Paper: GaP @ CoRL '26](https://img.shields.io/badge/paper-GaP%20%40%20CoRL%202026-8a2be2.svg)](https://graph-robots.github.io/graph-as-policy-anonymous/)
[![Docs](https://img.shields.io/badge/docs-quickstart-0f766e.svg)](docs/quickstart.md)

<video src="docs/assets/grocery_packing_real.mp4"
       autoplay loop muted playsinline width="640"></video>

<sub>Franka arm packing varied grocery items — graph generated from
"pack the basket with the items on the table", refined in sim, run on
the real robot. (10× speed.)</sub>

</div>

```python
import gap

conn   = gap.connector.sim("libero", task="libero_object/0")
result = gap.execute("examples/libero_quickstart/graph", conn)
graph  = gap.agent.generate_sync("pick the soup can and put it in the basket")
print(graph)                  # the policy, as a graph
gap.viz.serve("outputs")      # browse the recorded trial
```

Skills live in the sibling
[**open-robot-skills**](https://github.com/graph-robots/open-robot-skills)
repo (Anthropic Agent Skills format, contributable) and are discovered
by path — clone the two repos side by side and every command finds them.

## Why GaP

- **Language → typed graph.** A coordinator → subgraph-agents →
  checkpoint-agent pipeline compiles one instruction into a validated
  workflow of skills, with a script-fix loop on validation errors.
- **Verified, not hoped.** LLM-authored postcondition checkpoints are
  enforced against simulator ground truth at every subgraph exit.
- **Skills are contributable.** Strategies and model tools live in
  [open-robot-skills](https://github.com/graph-robots/open-robot-skills)
  as Agent Skills bundles — one directory, one PR. The LLM composes them; you can too.
- **Per-bundle isolation, no monolith venv.** Each tool bundle (sam3, cuRobo,
  vlm, openpi, …) runs in its own venv via stdio-msgpack RPC. The engine
  stays ~150 MB; heavy ML stacks don't fight each other.

## Quickstart

**Requirements:** **1× NVIDIA RTX 4090 (≥24 GB VRAM, Linux + EGL)**, an
LLM API key (Anthropic / OpenAI-compatible / Vertex), and
[uv](https://docs.astral.sh/uv/). First run downloads ~3.5 GB of model
weights; `HF_TOKEN` (a free [HuggingFace token](https://huggingface.co/settings/tokens))
is needed only for the gated SAM3 weights.

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git    # sibling, auto-discovered
cd graph-as-policy
uv sync                                  # engine + LIBERO sim (now baseline)
uv run gap skills install --all          # per-bundle venvs (sam3, cuRobo, vlm, …)
uv run gap skills check --download       # weight prefetch + capability gate

export ANTHROPIC_API_KEY=...
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                           # browse the trial at localhost:9432
```

See the **[15-minute tour](docs/quickstart.md)** for the full walkthrough
(clone → run → generate → trace).

## Examples

| Example | What it shows |
|---|---|
| [libero_quickstart](examples/libero_quickstart/) | Hero: vision → OBB grasp → transport, ground-truth verified |
| [generate_a_graph](examples/generate_a_graph/) | Instruction → validated workflow dir; all LLM providers |
| [build_a_graph](examples/build_a_graph/) | Full Python authoring with `gap.builder`: checkpoints, recovery |
| [agent_quickstart](examples/agent_quickstart/) | Claude Code + one skill: sentence → graph → sim success |
| [grocery_fulfillment](examples/grocery_fulfillment/) | Flagship acceptance family; graphs LLM-generated per task |
| [benchmark](examples/benchmark/) | Grid harness: smoke → posvar → release gate |
| [steered_policy](examples/steered_policy/) | Perceive + hover, then hand to a learned VLA policy |
| [collect_and_train](examples/collect_and_train/) | Graph as scripted expert → dataset → train → policy node |
| [cable_ur](examples/cable_ur/) | Real UR + ZED perception (motion-disabled) |
| [real_franka_pick_place](examples/real_franka_pick_place/) | Real Franka pick-place via robots_realtime |

Full gallery with measured results, time estimates, and per-example
media in **[examples/README.md](examples/README.md)**. Real-robot examples
— read [docs/safety.md](docs/safety.md) first.

## Documentation

- [Quickstart (15-min tour)](docs/quickstart.md) — clone → run → generate → trace
- [Architecture](docs/design.md) — engine + skills split; tools vs. skills
- [Runtime & schema](docs/runtime.md) — workflow JSON, executor semantics, checkpoints
- [Skill authoring](docs/skills.md) — Agent Skills format, `gap.requires:` frontmatter
- [LLM providers](docs/source/authoring/llm-providers.md) — anthropic / openai / vertex
- [Skill registries](docs/source/skills/registries.md) — `--skills`, `$GAP_SKILLS_PATH`, `gap registry …`
- [CLI reference](docs/source/reference/cli.md) — every `gap` verb
- [Safety](docs/safety.md) — required reading before any real-robot example

## Use with Claude Code & AI agents

GaP ships an agent skill ([`agent/`](agent/)) that teaches AI coding
agents the workflows above:

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap
```

Then ask: *"what can this robot do right now?"*, *"run the quickstart graph
in sim"*, *"generate a graph that packs the groceries"*, *"why did this
trial fail?"*. Real-robot commands are gated on explicit human
confirmation. Other agents (Cursor, Codex, …) and the no-install path:
[agent/INSTALL.md](agent/INSTALL.md).

## License & attribution

graph-as-policy and open-robot-skills are MIT-licensed; they stand on third-party
work — LIBERO/LIBERO-PRO, Variational-Automation-Benchmark, robosuite,
robots_realtime, pyroki, SAM3, Grounding DINO, and (optionally) NVIDIA
cuRobo. Full attribution table: **[NOTICE.md](NOTICE.md)**.
