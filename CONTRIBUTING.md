# Contributing to gap

Thanks for helping! This page covers the engine repo. **Skill and tool
bundles** are contributed to the sibling
[open-robot-skills](https://github.com/graph-robots/open-robot-skills) repo instead — see
[docs/skills.md](docs/skills.md) for the bundle authoring guide and its
contribution checklist.

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

## Test markers

`pyproject.toml` sets `addopts = "-m 'not llm and not gpu and not sim and
not real'"` — the deselected suites are opt-in:

| Marker | Needs | Run with |
|---|---|---|
| *(none)* | nothing (CPU, core deps) | `pytest tests -q` — the PR gate, py3.10/3.11/3.12 |
| `sim` | `[libero]` extra (MuJoCo/EGL) | `pytest tests -q -m sim` |
| `gpu` | model weights + NVIDIA GPU | `pytest tests -q -m gpu` |
| `llm` | `ANTHROPIC_API_KEY` (token budget!) | `pytest tests -q -m llm` |
| `real` | robot hardware | manual release checklist |

## The rules CI enforces

1. **The default suite must collect and pass on a bare install.** CI
   installs `pip install -e .` with **no extras** on 3.10/3.11/3.12 and runs
   `pytest tests -q --collect-only` first. Practical consequence: never
   import mujoco/torch/transformers/gymnasium/pyzed at module scope in code
   a test file imports — use the lazy-import pattern (import inside the
   function/factory, with a pip-hint error message).
2. **Lint:** `ruff check gap tests` with ruff `>=0.15,<0.16` (pinned minor —
   keep your local ruff on the same line; `uvx ruff@0.15 check gap tests`).
3. **The wheel ships the viz frontend.** `gap/viz/frontend/dist/` is checked
   in (pip users get `gap viz` without node) and CI asserts it lands in the
   wheel. If you touch `gap/viz/frontend/src/`, rebuild `dist/` and commit
   it in the same PR.

## Pull requests

- Keep PRs focused; one subsystem per PR.
- New runtime behavior needs deterministic tests (use `gap.testing` —
  `FakeContext`, `make_test_observation` — not live models).
- The on-disk trace layout (`dag_trace.json`, `node_data/<id>/`) and the
  skill-authoring import surface (`gap.NodeContext`, `gap.types`,
  `gap.errors`, `gap.skills`, `gap.testing`) are **stability guarantees**;
  changes to either need a strong justification and a migration note.
- Don't bump submodule pins (`third_party/`) as a side effect of an
  unrelated change.
- Hardware-touching changes: state what robot you ran on, and read
  [docs/safety.md](docs/safety.md). Safety-relevant bugs get the `safety`
  label and priority review.

## Docs map

- [docs/runtime.md](docs/runtime.md) — v3 schema + executor semantics
- [docs/skills.md](docs/skills.md) — bundle authoring (open-robot-skills)
- [docs/safety.md](docs/safety.md) — real-robot safety
- [docs/design.md](docs/design.md) — the release design doc (architecture
  rationale, testing strategy, roadmap)
