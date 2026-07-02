# Installation

GaP is two repos in one virtual environment: **graph-as-policy** (the
engine) and **open-robot-skills** (the skill bundles, discovered by path).
This page takes you from zero to a verified install: clone both repos side
by side, sync, install the skill bundles, set your keys, and gate the
result with `gap skills check` and `gap check`.

## Requirements

What you need to run the stack:

| Setup | Requirements |
|---|---|
| Run the quickstart, sim, perception, motion planning | Linux + an NVIDIA GPU + EGL, an LLM API key, [uv](https://docs.astral.sh/uv/). **≥24 GB VRAM (RTX 4090-class) fits the full stack**; the LIBERO quickstart alone runs in ~10 GB. |
| Real robots | See [Connectors](../real-robots/connectors.md) and read [Safety](../real-robots/safety.md) first. |

The first simulation run downloads model weights (~3.5 GB total) from
HuggingFace. `HF_TOKEN` is required in practice: the gated SAM3 weights
sit in the default perception path of the quickstart and grocery
examples — see [Model weights](#model-weights) below.

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

One `uv sync` plus one skills install, run from inside `graph-as-policy/`.
Everything is declared in `pyproject.toml` and pinned by the committed
`uv.lock` — the exact environment the acceptance benchmark was measured
on.

```bash
cd graph-as-policy
CUDA_HOME=/usr/local/cuda uv sync && uv run gap skills install --all
```

`uv sync` installs the engine and the LIBERO sim stack (`mujoco==3.6.0`,
the vendored LIBERO fork and robosuite, `torch>=2.7`, in-process IK via
pyroki) plus the dev group (pytest, ruff, mypy). `gap skills install --all`
then installs every bundle in the side-by-side `open-robot-skills`
checkout — SAM3, Grounding DINO, geometry, CuRobo, Gemini-ER, the
learned-policy bundles, and friends.

`--all` is the simplest path but heavier than most workflows need — the
learned-policy bundles (molmoact, lerobot, openpi/JAX) alone add many
gigabytes that the quickstart and grocery examples never touch. To install
only what one graph actually uses, point at its directory instead:

```bash
uv run gap skills install --workflow examples/libero_quickstart/graph
```

This resolves the bundles the graph references (both `type: tool` nodes
and `ctx.tool(...)` calls inside its scripts) and syncs just those venvs.

:::{note}
**CuRobo builds CUDA extensions at install time** against your
environment's torch. The engine repo already declares it as a
no-build-isolation package for uv; you only need `CUDA_HOME` to point at a
CUDA toolkit matching torch's CUDA version. A version **mismatch** shows
up as a CuRobo build failure (or a version-mismatch warning at import) —
compare `"$CUDA_HOME"/bin/nvcc --version` against torch's
`python -c "import torch; print(torch.version.cuda)"` before syncing.
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
  -e open-robot-skills
gap skills install --all
gap skills check --download
```

Two pip-specific caveats:

- **numpy.** SAM3's published metadata over-pins `numpy==1.26`. uv applies
  a documented override (SAM3 runs fine on numpy 2.x); plain pip may
  downgrade numpy during resolution — fix it afterwards with
  `pip install "numpy>=2" --force-reinstall`.
- **CuRobo.** The CuRobo bundle builds CUDA extensions at install time;
  add `--no-build-isolation` to the install command and set `CUDA_HOME` to
  a toolkit matching torch's CUDA.

## Model weights

pip/uv own all *code* installs; model **weights** are a separate, lazy
concern. The quickstart perception stack pulls `facebook/sam3` and
`IDEA-Research/grounding-dino-base` from HuggingFace on the first call
into each model — about 3.5 GB total, cached locally after that.

- **SAM3 is gated**: set `HF_TOKEN` (a free
  [HuggingFace token](https://huggingface.co/settings/tokens) with access
  to the repo) before the first run, or the download fails with a 401.
- `gap skills check --download` runs each bundle's optional `prefetch()`
  hook after the checks. The perception bundles (`sam3`, `grounding-dino`)
  define one, so the command fetches their weights eagerly — and surfaces
  a missing or under-privileged `HF_TOKEN` up front instead of mid-run.
  A bundle without a `prefetch()` reports
  `declares no weights (no prefetch())` and its weights download lazily on
  the first model call — budget that into your first run.

Nothing ever pip-installs or downloads behind your back outside these two
paths.

## LLM API keys

Generation (`gap generate`) and the hosted-VLM perception step both need an
LLM provider. Set one of:

| Provider | Setup |
|---|---|
| `openrouter` (default) | `export OPENROUTER_API_KEY=...`; point at any other OpenAI-compatible server (e.g. a local vLLM) by overriding the endpoint/base URL in the config YAML |
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
report; add `--strict` to turn any not-ready bundle into exit 1 for CI.

Then smoke-test the full stack:

```bash
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph --sim libero_object_all_variance/0 --validate-only
```

If that returns cleanly, the graph parses, every bundle is venv-ready, and
the sim launches. Continue with the [15-minute tour](quickstart.md).

## Troubleshooting

**`uv sync` fails with "does not appear to be a Python project".**
You cloned without `--recurse-submodules`. Run
`git submodule update --init` inside `graph-as-policy/` — the lockfile
references the vendored submodules even for engine-only installs. The
side-by-side `open-robot-skills` checkout is required for the same reason.

**`uv sync` fails with `Distribution not found at: file:///…/open-robot-skills`.**
The side-by-side `open-robot-skills` checkout is missing or renamed —
`[tool.uv.sources]` resolves it at `../open-robot-skills` relative to the
engine repo. Clone it next to `graph-as-policy/` (see
[Clone the two repos](#clone-the-two-repos)) and re-run `uv sync`.

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

- [The 15-minute tour](quickstart.md) — run a real rollout on LIBERO.
- [Concepts](concepts.md) — the vocabulary: workflows, subgraphs, skills, tools.
- [Skill registries](../skills/registries.md) — how bundles are discovered and layered.
