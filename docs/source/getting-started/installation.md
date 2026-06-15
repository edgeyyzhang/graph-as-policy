# Installation

gap is two repos in one virtual environment: **graph-as-policy** (the
engine) and **open-robot-skills** (the skill bundles, discovered by path).
This page takes you from zero to a verified install: clone both repos side
by side, pick an extras set, set your keys, and gate the result with
`gap skills check` and `gap check`.

## Requirements

What you need depends on how much of the stack you want to run:

| Setup | Requirements |
|---|---|
| Engine only — validate, author, and render graphs | Python ≥ 3.10, [uv](https://docs.astral.sh/uv/) or pip. CPU is fine; runs anywhere. |
| Simulation + perception (quickstart, benchmarks) | Linux, an NVIDIA GPU with **≥ ~10 GB VRAM**, EGL for headless rendering (`MUJOCO_GL=egl`), and an LLM API key (Anthropic / OpenAI-compatible / Vertex). |
| CuRobo motion planning (`grocery` / `all` extras) | All of the above plus a CUDA toolkit matching your torch build (`CUDA_HOME`). |
| Real robots | See [Connectors](../real-robots/connectors.md) and read [Safety](../real-robots/safety.md) first. |

The first simulation run downloads model weights (~3.5 GB total) from
HuggingFace; `HF_TOKEN` is needed only for the gated SAM3 weights — see
[Model weights](#model-weights) below.

:::{tip}
No GPU? You can still do real work — build, validate, and render
workflow graphs on a laptop. Start with
[gap in 2 minutes](quickstart-cpu.md).
:::

## Clone the two repos

Skills are discovered **by path**, not by pip entry points: clone the two
repos next to each other and every command finds the bundles with no flags.

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git
```

```text
your-workspace/
├── graph-as-policy/       # the engine
└── open-robot-skills/     # skill bundles — auto-discovered, no flags
```

:::{warning}
Both the side-by-side layout **and the directory names** are load-bearing.
Each repo's `pyproject.toml` resolves the other by relative path —
`[tool.uv.sources]` points at `../open-robot-skills` in the engine repo and
at `../graph-as-policy` in the skills repo. Rename or separate the
checkouts and `uv sync` fails in either repo.

`--recurse-submodules` matters too: the engine vendors its sim stack
(the LIBERO fork, robosuite, and friends) as git submodules, and the
lockfile references them even for engine-only installs.
:::

## Install with uv (recommended)

One `uv sync` per setup, run from inside `graph-as-policy/`. Everything is
declared in `pyproject.toml` and pinned by the committed `uv.lock` — the
exact environment the acceptance benchmark was measured on.

```bash
cd graph-as-policy
uv sync --extra quickstart
```

| Command | What you get |
|---|---|
| `uv sync` | Engine only: runtime, validator, `gap.builder`, viz, in-process IK (pyroki), plus the dev group (pytest, ruff, mypy). CPU, any OS. |
| `uv sync --extra quickstart` | + `[libero]` sim stack (`mujoco==3.6.0`, the vendored LIBERO fork and robosuite, `torch>=2.7`) + the open-robot-skills `quickstart` bundles: SAM3, Grounding DINO, geometry. |
| `CUDA_HOME=/usr/local/cuda uv sync --extra grocery` | Everything in `quickstart` + the CuRobo motion-planning bundle (CUDA build at install time). The acceptance-benchmark set. |
| `CUDA_HOME=/usr/local/cuda uv sync --extra all` | Everything in `grocery` + Ray workers (`ray[serve]`), the Vertex provider, the OpenPI policy client, and the Gemini-ER and learned-policy (`pi05-libero`, `molmoact-libero`) bundles. |

The `quickstart`, `grocery`, and `all` extras are meta-extras spanning both
repos: each combines the engine's `[libero]` sim stack with the matching
bundle set from `open-robot-skills/pyproject.toml` (one extra per bundle),
resolved from the side-by-side checkout. The vendored submodules are wired
the same way, so a single sync builds the whole environment.

:::{note}
**CuRobo builds CUDA extensions at install time** against your
environment's torch. The engine repo already declares it as a
no-build-isolation package for uv; you only need `CUDA_HOME` to point at a
CUDA toolkit matching torch's CUDA version when syncing
`--extra grocery` or `--extra all`.
:::

`uv run gap …` needs no venv activation; run
`source .venv/bin/activate` once if you prefer plain `gap …`.

## Install with pip

pip works, with the vendored submodules and the sibling repo named
explicitly. From the parent directory of the two checkouts:

```bash
pip install -e "graph-as-policy[libero]" \
  -e graph-as-policy/third_party/Variational-Automation-Benchmark \
  -e graph-as-policy/third_party/robosuite \
  -e "open-robot-skills[quickstart]"
gap skills check --download
```

Two pip-specific caveats:

- **numpy.** SAM3's published metadata over-pins `numpy==1.26`. uv applies
  a documented override (SAM3 runs fine on numpy 2.x); plain pip may
  downgrade numpy during resolution — fix it afterwards with
  `pip install "numpy>=2" --force-reinstall`.
- **CuRobo.** When installing `open-robot-skills[curobo]` (or the `grocery`
  / `all` sets), add `--no-build-isolation` and set `CUDA_HOME` to a
  toolkit matching torch's CUDA.

## Model weights

pip/uv own all *code* installs; model **weights** are a separate, lazy
concern. The quickstart perception stack pulls `facebook/sam3` and
`IDEA-Research/grounding-dino-base` from HuggingFace on the first call
into each model — about 3.5 GB total, cached locally after that.

- **SAM3 is gated**: set `HF_TOKEN` (a free
  [HuggingFace token](https://huggingface.co/settings/tokens) with access
  to the repo) before the first run, or the download fails with a 401.
- `gap skills check --download` runs each bundle's optional `prefetch()`
  hook after the checks. It is the supported mechanism for eager weight
  downloads — but no bundled tool currently defines a `prefetch()` hook,
  so today the command reports `declares no weights (no prefetch())` per
  bundle and weights still download lazily on first model call. Budget the
  download into your first run.

Nothing ever pip-installs or downloads behind your back outside these two
paths.

## LLM API keys

Generation (`gap generate`) and the hosted-VLM perception step both need an
LLM provider. Set one of:

| Provider | Setup |
|---|---|
| `anthropic` (default) | `export ANTHROPIC_API_KEY=...` |
| `openai` (incl. OpenRouter / vLLM) | `export OPENAI_API_KEY=...`; custom endpoints via config YAML |
| `vertex` | `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT`; needs the `vertex` extra |

See [LLM providers](../authoring/llm-providers.md) for per-call
`--provider`/`--model` overrides, config YAML, and the `GAP_VLM_*`
variables that route the perception VLM independently.

## Verify the install

Two commands gate a fresh install:

```bash
uv run gap skills check     # is every bundle well-formed and importable?
uv run gap check            # what can actually run on THIS machine?
```

`gap skills check` validates every discovered bundle — SKILL.md format
plus an import probe — and prints PASS/WARN/FAIL per bundle with an
install hint for anything missing (non-zero exit on FAIL). `gap check` is
the capability report: environment (GPU, LLM credentials), active
registries, per-bundle operational status, and the per-skill runnability
rollup with a fix hint per failure. It exits 0 whenever it can produce a
report — a laptop without a GPU is not an error; add `--strict` to turn
any not-ready bundle into exit 1 for CI.

Then smoke-test with the CPU-only example:

```bash
uv run python examples/hello_graph/hello.py
```

If that renders `outputs/hello_graph/graph.png`, the engine, the skills
checkout, and the validation path all work. Continue with
[gap in 2 minutes](quickstart-cpu.md) or jump straight to the
[15-minute tour](quickstart.md).

## Troubleshooting

**`uv sync` removed my extras.**
`uv sync` is *exact*: it makes the environment match exactly what you ask
for, so a bare `uv sync` after `uv sync --extra quickstart` uninstalls the
quickstart set again. Always repeat the `--extra` flag (or use
`uv sync --inexact …` for additive installs).

**`uv sync` fails with "does not appear to be a Python project".**
You cloned without `--recurse-submodules`. Run
`git submodule update --init` inside `graph-as-policy/` — the lockfile
references the vendored submodules even for engine-only installs. The
side-by-side `open-robot-skills` checkout is required for the same reason.

**pip downgraded numpy to 1.26.**
That is SAM3's over-strict metadata pin. Reinstall numpy 2.x
(`pip install "numpy>=2" --force-reinstall`); the measured release
environment runs SAM3 on numpy 2.x. uv users are unaffected — the
override is in both repos' `[tool.uv]` tables.

**Why is mujoco pinned to exactly 3.6.0?**
The pin is physics-load-bearing: the measured success rates were taken on
MuJoCo 3.6.0, and MuJoCo minor versions change contact dynamics enough to
move grasp/drop success rates materially. Don't loosen it if you want
results comparable to the published numbers.

**CuRobo fails to build.**
Check that `CUDA_HOME` points at a CUDA toolkit whose version matches your
torch build's CUDA, and (pip only) that you passed `--no-build-isolation`.

## Next steps

- [gap in 2 minutes](quickstart-cpu.md) — build and render a graph, no GPU.
- [The 15-minute tour](quickstart.md) — run a real rollout on LIBERO.
- [Concepts](concepts.md) — the vocabulary: workflows, subgraphs, skills, tools.
- [Skill registries](../skills/registries.md) — how bundles are discovered and layered.
