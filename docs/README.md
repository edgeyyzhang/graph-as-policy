# GaP documentation site

This directory contains both the **legacy standalone docs** (`*.md` at this
level — still linked from the repo README) and the **documentation website**
(`source/` — Sphinx + [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/)
+ MyST markdown, modeled on [docs.skypilot.co](https://docs.skypilot.co)).

## Build

The pinned Sphinx (9.x) needs **Python ≥ 3.12**, while the engine venv pins
3.11 (`.python-version`) — build from a separate ≥3.12 environment, matching
Read the Docs:

```bash
# one-shot, no venv to manage (uv fetches 3.12 if needed):
uv run --no-project --python 3.12 --with-requirements docs/requirements-docs.txt bash docs/build.sh

# or with a persistent ≥3.12 environment of your own:
#   python3.12 -m venv .venv-docs && .venv-docs/bin/pip install -r docs/requirements-docs.txt
#   PATH=$PWD/.venv-docs/bin:$PATH ./docs/build.sh

./docs/build.sh                # one-shot build → docs/build/html (warnings are errors)
./docs/build.sh --watch 8000   # live-reload dev server
./docs/build.sh --clean        # remove build artifacts
```

Read the Docs builds from `.readthedocs.yml` at the repo root with
`fail_on_warning: true` — keep local builds warning-clean.

## Layout

```
docs/
├── source/
│   ├── conf.py               # theme, MyST, extensions
│   ├── index.rst             # redirect + top-navbar toctree (SkyPilot pattern)
│   ├── docs/index.md         # the real landing page + sidebar section toctrees
│   ├── getting-started/  examples/  running/  authoring/
│   ├── skills/  benchmarks/  real-robots/  agents/
│   ├── reference/  developers/
│   └── _static/              # custom.css, favicon, images
├── requirements-docs.txt
├── build.sh / Makefile
└── *.md                      # legacy docs (quickstart, runtime, skills, safety, design)
```

The left sidebar is defined by the captioned, hidden `{toctree}` blocks at the
bottom of `source/docs/index.md`; the top navbar by the hidden toctree in
`source/index.rst`. To add a page: create the `.md` file in its section
directory and add it to the matching toctree in `source/docs/index.md`
(example pages go in the toctree inside `source/examples/index.md`).

## Conventions

- Page titles **Title Case**; section headings **Sentence case** (SkyPilot style).
- MyST markdown with `colon_fence`: admonitions as `:::{note}` / `:::{warning}`,
  sphinx-design grids/cards with `::::` nesting.
- Code blocks: ` ```bash ` for command-only blocks (no `$`), ` ```console `
  with `$ ` prompts only when commands and output are mixed (sphinx-copybutton
  strips the prompts on copy).
- Link to repo sources with the custom URL schemes from `conf.py`:
  `[examples/build_a_graph](gh-engine:examples/build_a_graph)` and
  `[skills/sam3](gh-skills:tools/sam3)`.
- Cross-page links are relative `.md` paths; heading anchors are
  auto-generated for h1–h3 (`myst_heading_anchors = 3`).
- Facts over marketing: every command, flag, signature, and number in the site
  is verified against the source; measured results only as stated in the
  example READMEs.
- API/CLI references are hand-curated (no autodoc) so the docs build never
  imports `gap` or its GPU dependencies. When the CLI or public API changes,
  update `source/reference/cli.md` / `source/reference/api.md` — the agent
  skill's generated `agent/references/cli.md` is a good cross-check (CI pins
  it to the argparse parser).

## Deferred (post-v1 of the site)

- llms.txt + per-page markdown export (SkyPilot does this with custom local
  Sphinx extensions; adopt `extension/dynamic_llms_txt`-style generation later).
- Versioned docs / version switcher (RTD versions cover this once tags exist).
- Analytics, announcement banner content.
