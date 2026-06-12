# gap documentation site — design

Deliverable: a SkyPilot-style documentation website for **gap (graph as policy)** + **open-robot-skills**,
living in the engine repo at `graph-as-policy/docs/`, built with Sphinx.

## 1. Stack (mirrors docs.skypilot.co)

| Choice | Value | Why |
|---|---|---|
| Generator | Sphinx | Same as SkyPilot; RTD-ready |
| Theme | `pydata-sphinx-theme` | SkyPilot's active theme (their book-theme dep is vestigial) |
| Source format | MyST Markdown (`myst-parser`, `colon_fence` + `linkify`) | Repo docs are already markdown; writers produce md |
| Components | `sphinx-design` (grids/cards/tabs), `sphinx-copybutton` (`$ `-strip), `sphinx-togglebutton`, `sphinx-notfound-page`, `sphinx.ext.autosectionlabel` (prefixed), `intersphinx` | SkyPilot's exact extension set minus analytics |
| API reference | **Hand-curated pages** (no autodoc) | Avoids importing gap + GPU deps at build time; deterministic builds; accuracy enforced by adversarial review against source |
| CLI reference | Hand-written from `gap/cli/` source (argparse, so sphinx-click N/A) | gap uses argparse with lazy subcommand registration; `build_parser()` exists for future autogen |
| Build | `docs/Makefile` + `docs/build.sh` (fails on warnings), `requirements-docs.txt`, `.readthedocs.yml` | SkyPilot conventions |

Layout: `docs/source/` (conf.py + pages), existing `docs/*.md` files left untouched (README links keep working);
site pages adapt their content. Existing media (`docs/assets/`) copied to `source/_static/`.

**SkyPilot's two-level index trick is reproduced**: `source/index.md` is a redirect page whose hidden toctree
defines the top navbar (`Docs`, external GitHub links); the real landing page is `source/docs/index.md`
(URL: `/docs/index.html`, like SkyPilot) holding the captioned hidden toctrees that form the left sidebar.

## 2. Landing page (`docs/index.md`)

Hero: "**The policy is the graph.**" + one-paragraph value prop; the quickstart rollout GIF + rendered graph
side by side (existing assets); 6-line Python hero snippet; `sphinx-design` card grid linking to: Installation,
15-minute tour, Examples, Skill catalog, CLI reference, Use with Claude Code. "Why gap" bullets (typed graphs,
verified checkpoints, contributable skills, one process, trace-is-the-product, gated benchmarks) + measured
results table. Architecture diagram (ASCII from README, styled code block).

## 3. Navigation (left-sidebar sections → source dirs)

**Getting Started** (`getting-started/`)
1. `overview` — what is gap; thesis; two-repo model; tools vs skills; architecture
2. `installation` — hardware floor; uv/pip; extras matrix; submodules; CUDA/curobo; weights/HF_TOKEN; install verification (`gap skills check`); troubleshooting (uv-sync-is-exact, numpy/sam3, recurse-submodules)
3. `quickstart-cpu` — gap in 2 minutes, no GPU (hello_graph: build → validate → render)
4. `quickstart` — the 15-minute tour (run quickstart graph in LIBERO → read trace → gap viz → generate from language)
5. `concepts` — glossary: workflow/graph, node types, subgraph, tool vs skill, connector, checkpoint, trial/trace, registry

**Examples** (`examples/`)
6. `index` — gallery: learning-path tables w/ needs + measured results (cards)
7–16. one page per example: `hello-graph`, `libero-quickstart`, `build-a-graph`, `generate-a-graph`,
`agent-quickstart`, `grocery-fulfillment`, `benchmark`, `steered-policy`, `collect-and-train`,
`cable-ur`, `real-franka-pick-place` — each adapted from its README: what it shows, commands, results, link to source

**Running Graphs** (`running/`)
17. `execution` — `gap run` + `gap.execute()`: graph forms (dir/JSON/dict/builder), `--sim SUITE/TASK`, `--inputs`, `--validate-only`, tracing default-on, `--record-video`, ExecutionResult
18. `environments` — env registry; LIBERO classic suites + vab variance/packing; perturbed env; seeding semantics (`(seed-1)%n_inits`, seed 0 = unseeded); cameras; `MUJOCO_GL=egl`; `GAP_MUJOCO_EGL_DEVICES`
19. `checkpoints` — verification: off/warn/raise, World/Body/Robot predicate API, sim-only ground truth, hard checks vs probes, diagnostics, gotchas (default mismatch, silent degradation w/o world_snapshot)
20. `traces` — trial dir anatomy (workflow.json, dag_trace.json, node_data, assets), `gap viz` browser, `gap trace-diff`, agent_traces from generation

**Generating & Authoring** (`authoring/`)
21. `generation` — `gap generate` / `gap.agent.generate`: coordinator → subgraph agents → checkpoint agent → validate/fix loop; agent trace artifacts; missing-capability abort; LLM cache (GAP_LLM_CACHE_DIR / NO_CACHE)
22. `llm-providers` — anthropic/openai-compatible/vertex; default model; config YAML (composition, per-role models); VLM perception provider env vars (GAP_VLM_*)
23. `builder` — gap.builder guide: Workflow/Subgraph/Ref, edges & conditional edges, exits (`add_exit` vs `set_exit_router`), on_error, recovery, checkpoints + sidecars, save/validate
24. `patterns` — graph patterns & pitfalls: $ref paths, by-name cross-subgraph binding, on_error-as-semantic-exit, loops via subgraph revisits, streaming + Send, settle_steps not sleep, wxyz quaternions, OBB half-extents

**Skills & Tools** (`skills/`)
25. `registries` — resolution precedence (5 levels), full-override semantics of --skills/GAP_SKILLS_PATH, layering/shadowing, `gap registry` lifecycle, auto-discovery rules, `gap check` capability report
26. `skill-catalog` — all 10 skill bundles: purpose, owned subgraph, exits, inputs/outputs, requirements, selection guidance (perception tiers, 3 grasp strategies, transport, tracking, policies)
27. `tool-catalog` — all 7 tool bundles with per-tool signatures + the canonical perception→grasp recipe; env vars (GAP_MOLMO_BASE_URL, GAP_VLM_*)
28. `authoring-bundles` — SKILL.md format (spec-core + `gap:` block, all 15 keys, `gap.requires`), tools.py `@tool`, lazy loading contract, canonical scripts, prompts, one-extra-per-bundle, scaffolding (`gap skills new`)
29. `testing-bundles` — gap.testing (FakeContext, make_test_observation, assert_graph_valid, connector contract suite), `gap skills test`, markers (gpu/llm/sim/real)

**Benchmarks & Policies** (`benchmarks/`)
30. `benchmarking` — grid vs suites configs, 3 modes, policy A/B, `--gate` (≥0.90, errored-cell semantics), `--resume`, outputs (summary.json/tsv, videos), GPU spread, timeout guidance
31. `policies` — learned VLA policies: `gap policy serve` presets, running-policies skill (termination modes), steered-graph pattern, collect_and_train (DataCollector → LeRobot)

**Real Robots** (`real-robots/`)
32. `connectors` — franka (rr-session/msgpack bridge, autostart, heartbeat) + ur_zed (perception-only, ZED SDK, calibration); capabilities model; what's structurally disabled on real
33. `safety` — adapted safety.md: pre-session checklist, what gap enforces (guards, go_home no-op, gated agent commands), what it does NOT, reporting

**AI Agents** (`agents/`)
34. `claude-code` — plugin marketplace install, what the gap skill teaches, example prompts, safety gating, other agents + no-install path, agent_quickstart pointer

**Reference** (`reference/`)
35. `cli` — every command: run, generate, check, registry ×4, tools ×2, skills ×5, benchmark, viz, policy serve, trace-diff; all flags, defaults, exit-code conventions (0/1/2)
36. `api` — curated Python API: gap.execute, gap.connector.sim/real, gap.agent.generate(_sync), gap.builder.*, gap.benchmark.run, gap.viz.serve/render, gap.testing.*, gap.types core
37. `workflow-schema` — v3 JSON spec: top-level, SubgraphDef, NodeDef×6, conditional edges, exits, on_error, $ref syntax, validation rules W1–W8/S1–S11, dataflow typing
38. `executor` — semantics: super-steps, scope scheduling, conditional dispatch, cross-subgraph data, streaming, Send, end nodes/recovery, observation stream, guards
39. `connector-tools` — robot.* (13) + sim.* (6) tool tables with signatures, guard tags
40. `environment-variables` — every GAP_* + related env var, grouped, w/ defaults
41. `benchmark-config` — benchmark YAML reference (both shapes, all keys)
42. `faq` — troubleshooting/FAQ distilled from gotchas (install, EGL, weights, cache staleness, checkpoint false-fails, registry confusion)

**Developers** (`developers/`)
43. `architecture` — engine internals (adapted design.md): module map, data vocabulary, tool layer, connector design, executor, principles
44. `contributing` — engine PRs (CONTRIBUTING.md), test suite/markers, docs build how-to
45. `roadmap` — v1 scope line + post-v1 cuts

## 4. Conventions

- Page titles Title Case, sections Sentence case (SkyPilot style guide)
- Every code block copy-buttoned; shell blocks use `$ ` prompts only where mixed with output
- `:::{note}` / `:::{warning}` admonitions for gotchas; "Requirements" admonition atop GPU/key-needing pages
- Cross-refs via doc paths (myst) — no invented anchors; autosectionlabel prefixed
- Edit-on-GitHub enabled via html_context → graph-robots/graph-as-policy
- Light/dark pygments (tango/monokai); custom.css accent: deep teal on slate, distinct from SkyPilot blue
- llms.txt + markdown export: deferred (SkyPilot has custom extensions; out of v1 scope — noted in docs/README.md)

## 5. Build & verify

`docs/build.sh` → `sphinx-build -W --keep-going` (warnings fail); `--watch` → sphinx-autobuild.
Verification: clean `-W` build, linkcheck on internal refs, screenshot of landing + 3 key pages via headless browser.
