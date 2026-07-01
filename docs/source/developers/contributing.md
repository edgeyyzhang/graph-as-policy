# Contributing

Thanks for helping! This page covers contributions to the **engine repo**
([graph-as-policy](gh-engine:.)): how to set up a dev environment, how the
test suite is organized, what CI enforces, and how to build these docs.

:::{note}
**Skill and tool bundles** are contributed to the sibling
[open-robot-skills](gh-skills:.) repo instead — one directory, one PR. See
[Authoring bundles](../skills/authoring-bundles.md) for the bundle format
and its contribution checklist, and
[Testing bundles](../skills/testing-bundles.md) for the scaffolded tests.
:::

## Dev setup

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # side-by-side (uv resolves it by path)
cd graph-as-policy
uv sync                               # engine + dev tools (pytest/ruff/mypy) — enough for the default suite
# optional, for sim-marked work (needs Linux + EGL):
uv sync --extra libero
```

The default test suite runs on a **bare install** — no GPU, no sim stack,
no API keys:

```bash
uv run pytest tests -q
```

If you forgot `--recurse-submodules`, `git submodule update --init` fixes
`uv sync`'s "does not appear to be a Python project" error. More install
detail (extras, pip path, CUDA notes):
[Installation](../getting-started/installation.md).

## Test markers

`pyproject.toml` sets
`addopts = "-m 'not llm and not gpu and not sim and not real'"`, so the
heavy suites are deselected by default and opt-in. Every test also runs
under a 300-second timeout (`timeout = 300`, via pytest-timeout).

| Marker | Needs | Run with |
|---|---|---|
| *(none)* | nothing (CPU, core deps) | `pytest tests -q` — the PR gate, py3.10/3.11/3.12 |
| `sim` | `[libero]` extra (MuJoCo/EGL) | `pytest tests -q -m sim` |
| `gpu` | model weights + NVIDIA GPU | `pytest tests -q -m gpu` |
| `llm` | `OPENROUTER_API_KEY` (token budget!) | `pytest tests -q -m llm` |
| `real` | robot hardware | manual release checklist |

The split follows one design rule: **every LLM boundary has a recorded
artifact.** The deterministic suite exercises parsing, validation,
execution, tracing, and viz against canned LLM outputs and golden graphs;
`llm`-marked tests regenerate those artifacts live and check structural
*equivalence*, not byte-equality.

## What CI enforces

CI ([.github/workflows/ci.yml](gh-engine:.github/workflows/ci.yml)) runs
three jobs on every push to `main` and every pull request.

### Lint

```bash
ruff check gap tests
```

ruff is pinned to a minor line, `>=0.15,<0.16` — ruff adds and retires
rules between minors, and the repo is kept clean against exactly this line.
Keep your local ruff on it too:

```bash
uvx ruff@0.15 check gap tests
```

### Bare-install test matrix

The default suite must **collect and pass on a bare install**. CI runs
`pip install -e .` with no extras on Python 3.10, 3.11, and 3.12, then:

```bash
python -m pytest tests -q --collect-only   # the collect gate
python -m pytest tests -q                  # the default suite
```

The collect-only gate is the canary for the most common regression:
importing a heavy dependency at module scope. The practical rule — never
import mujoco/torch/transformers/gymnasium/pyzed at the top of any module a
test file imports. Use the lazy-import pattern: import inside the function
or factory, and raise an error with a pip hint when the dependency is
missing.

### Wheel contents

CI builds the wheel and asserts its contents with a small script:
`gap/viz/frontend/dist/` (including `index.html`) must ship — pip users get
`gap viz` without Node — and no `frontend/src/` or `node_modules` files may
leak into the wheel. `dist/` is checked into the repo for the same reason.

:::{warning}
If you touch `gap/viz/frontend/src/`, rebuild `dist/` and commit it **in
the same PR** — otherwise the wheel ships a stale frontend that no test
notices.
:::

## Pull request guidelines

- **Keep PRs focused** — one subsystem per PR.
- **New runtime behavior needs deterministic tests.** Use the public
  [gap.testing](gh-engine:gap/testing) fixtures — `FakeContext` (a
  `NodeContext` stand-in with scripted per-tool responses and a full call
  log), `make_test_observation` (geometrically consistent synthetic camera
  data), `assert_graph_valid` — not live models or API calls.
- **Respect the stability guarantees.** The on-disk trace layout
  (`dag_trace.json`, `node_data/<id>/` — see [Traces](../running/traces.md))
  and the skill-authoring import surface (`gap.NodeContext`, `gap.types`,
  `gap.errors`, `gap.skills`, `gap.testing`) are commitments to users.
  Changes to either need a strong justification and a migration note.
- **Don't bump submodule pins** (`third_party/`) as a side effect of an
  unrelated change.
- **Hardware-touching changes**: state what robot you ran on, and read
  [Safety](../real-robots/safety.md) first. Safety-relevant bugs get the
  `safety` label and priority review.

## Building the docs

The docs live in `docs/` (Sphinx + MyST markdown) with their own
requirements file:

```bash
pip install -r docs/requirements-docs.txt   # or: uv venv .venv-docs && uv pip install -r docs/requirements-docs.txt
./docs/build.sh                             # one-shot HTML build into docs/build/html
./docs/build.sh --watch 8000                # live-reload dev server (sphinx-autobuild)
./docs/build.sh --clean                     # remove build artifacts
```

The build treats **warnings as errors** (`sphinx-build -W --keep-going`) —
a broken cross-reference or unbalanced directive fails the build, so run it
locally before pushing docs changes.

## Where next

- [Architecture](architecture.md) — the design rationale behind the modules
  you are changing
- [Roadmap](roadmap.md) — what is deliberately out of scope right now
- [Authoring bundles](../skills/authoring-bundles.md) — contributing skills
  instead of engine code
