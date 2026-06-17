# CLI Reference

The `gap` command is the single entry point to the engine. Subcommands
register lazily, so `gap --help` stays fast even though some commands pull
in heavy dependencies (MuJoCo, JAX) when they actually run.

```text
gap {run,check,skills,tools,registry,generate,viz,trace-diff,benchmark,policy} ...
```

Running `gap` with no subcommand prints help and exits 1. The same applies
to the command groups `gap skills`, `gap tools`, `gap registry`, and
`gap policy` when called without a sub-subcommand.

:::{tip}
Ground truth is always `gap <command> --help` in your checkout. This page
is verified against the parser source; if the two ever disagree, trust
`--help`.
:::

## Exit codes

Every command follows the same convention:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | The operation ran but failed (run failure, validation errors, failed checks, no matches) |
| `2` | Usage or resolution error (bad arguments, unresolvable registries, broken config) |

Command-specific meanings (for example, `gap trace-diff` exit 2 = traces
did not align) are documented per command below.

## Common options: registries

Most commands accept the same two flags for choosing where skill bundles
come from:

| Flag | Type / default | Meaning |
|---|---|---|
| `--skills PATH` | string, repeatable; default: resolved registry set | Registry checkout root(s), precedence-ordered. A **full override**: configured and auto-discovered registries are suppressed. |
| `--registry NAME` | string; default: all active registries | Restrict the command to one active registry by name. |

Without `--skills`, registries resolve in precedence order:
`$GAP_SKILLS_PATH` → project `pyproject.toml` `[tool.gap].registries` →
user config `~/.config/gap/registries.toml` → an auto-discovered
`open-robot-skills` checkout next to the engine checkout.
`$GAP_SKILLS_PATH`, when set, is likewise a full override; the project,
user-config, and auto-discovered layers merge, and same-named bundles in
higher-precedence registries shadow lower ones. See
[Registries](../skills/registries.md) for the full resolution rules and
[Environment variables](environment-variables.md) for `GAP_SKILLS_PATH`.

## gap run

```text
gap run [--sim SUITE/TASK | --real {franka,ur_zed}] [options] GRAPH
```

Execute (or validate) a workflow graph. By default the graph runs
**tools-only** — no robot, no simulator — which is enough for graphs that
only call computation tools. Pass `--sim` to run against a LIBERO
simulation or `--real` to drive hardware. Tracing is **on by default**:
every run writes a trace directory under `outputs/run_<timestamp>` unless
you opt out. See [Execution](../running/execution.md) for the run
lifecycle and [Traces](../running/traces.md) for what lands in the trace
directory.

| Flag | Type / default | Meaning |
|---|---|---|
| `GRAPH` | path, required | Workflow directory or `workflow.json` (a directory gets `workflow.json` appended). |
| `--sim SUITE/TASK` | string; default: none (tools-only) | Run against a sim connector, e.g. `libero_object/0`. Only the LIBERO family is supported; the value is `suite/task_index`. |
| `--real {franka,ur_zed}` | choice; default: none | Run against real hardware. `franka` = robots_realtime msgpack bridge; `ur_zed` = perception-only UR + ZED. Mutually exclusive with `--sim`. |
| `--rr-config YAML` | path; default: `configs/franka/franka_robotiq_client.yaml` | franka only: rr-session config, relative to `third_party/robots_realtime`. |
| `--no-rr-autostart` | flag | franka only: do not spawn rr-session; run it yourself in a second terminal. |
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--validate-only` | flag | Validate the graph without executing it; prints each issue and an `OK`/`FAIL` summary. |
| `--no-trace` | flag | Disable trace output. |
| `--trace-dir DIR` | path; default: `outputs/run_<timestamp>` | Trace output directory. |
| `--no-video` | flag | Sim only: disable run-video recording. Video is **on by default** for sim runs (saved to `<trace-dir>/run_video.mp4` plus per-camera videos when the env buffers them); pass this to skip rendering. Silently a no-op for `--real` / tools-only runs, and skipped (with a `note:` line) when combined with `--no-trace`. |
| `--checkpoints {off,warn,raise}` | choice; default: `warn` | Checkpoint enforcement mode. See [Checkpoints](../running/checkpoints.md). |
| `--inputs K=V ...` | repeatable `k=v`; default: none | Initial workflow inputs. Values are parsed as JSON when possible, else kept as strings. An entry without `=` is an error. |
| `-v, --verbose` | flag | Enable debug logging. |

Exit codes: `0` run succeeded (or validation passed); `1` run failed,
validation found error-severity issues, the workflow failed to load,
`--sim` and `--real` were combined, or an `--inputs` entry was malformed.

```console
$ gap run outputs/generated_20260612_101500 --sim libero_object/0
video: outputs/run_20260612_102014/run_video.mp4 (412 frames)
SUCCESS (exit=success, 41.3s)
trace: outputs/run_20260612_102014
```

:::{note}
`--sim` needs the `[libero]` extra (MuJoCo is pinned to `==3.6.0` —
minor MuJoCo bumps shift contact dynamics enough to move grasp success
rates) and a GPU for the perception bundles. `--real ur_zed` needs the
`[real]` extra plus a manual ZED SDK (`pyzed`) install. Before any
`--real` run, read [Safety](../real-robots/safety.md).
:::

## gap generate

```text
gap generate [options] INSTRUCTION
```

Generate a workflow graph from a natural-language instruction using an
LLM pipeline over your active skill registries. On success it writes the
graph to the output directory, prints a colored text rendering of the
graph (color is suppressed when output is piped or `NO_COLOR` is set),
and tells you how to run it. See [Generation](../authoring/generation.md)
for how the pipeline works and
[LLM providers](../authoring/llm-providers.md) for provider setup.

| Flag | Type / default | Meaning |
|---|---|---|
| `INSTRUCTION` | string, required | The task, e.g. `"pick up the alphabet soup and put it in the basket"`. |
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). A `skills:` entry in `--config` also works. |
| `--provider {anthropic,openai,vertex}` | choice; default: `anthropic` | LLM provider override. |
| `--model MODEL` | string; default: the provider default | LLM model override. |
| `--out DIR` | path; default: `outputs/generated_<timestamp>` | Output directory. |
| `--config YAML` | path; default: none | Pipeline config YAML (llm / composition / skills knobs). |
| `-v, --verbose` | flag | Enable debug logging. |

Exit codes: `0` graph written; `1` generation failed (prints
`FAIL: <error>`); `2` registry resolution failed when no `--skills` or
config-provided registries were given.

```console
$ gap generate "pick up the alphabet soup and put it in the basket"
OK: wrote outputs/generated_20260612_103000 (2 subgraph(s), 3 generated file(s))
...
run it with: gap run outputs/generated_20260612_103000
```

:::{note}
Requires an LLM credential: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
Vertex application-default credentials. The `vertex` provider needs the
`[vertex]` extra.
:::

## gap check

```text
gap check [--skills PATH] [--registry NAME] [--format {pretty,json}] [--strict]
```

The capability report: which tool bundles are operational on this
machine, and which skills are therefore runnable. It probes the
environment (Python and gap versions, GPU, LLM credentials for all three
providers), lists active registries, reports each bundle as
`READY` / `NOT READY` / `SHADOWED` with dependency, requirement, and
weights probes plus fix hints, and rolls blocked skills up to the tools
that block them.

| Flag | Type / default | Meaning |
|---|---|---|
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict the report to one active registry. |
| `--format {pretty,json}` | choice; default: `pretty` | `json` emits a stable machine-readable schema. |
| `--strict` | flag | Exit 1 when any non-shadowed bundle is not ready (CI gating). |

Exit codes: `0` whenever a report is produced — the command is diagnostic
by design, so a laptop with no GPU still exits 0; `1` only with
`--strict` when any non-shadowed bundle is not ready **or zero registries
resolved**; `2` usage or resolution errors.

```bash
gap check --strict --format json
```

## gap registry

```text
gap registry {init,list,add,remove} ...
```

Manage skill registries — local bundle checkouts, layered by precedence.
See [Registries](../skills/registries.md) for the underlying model.

Registry names must match `^[a-z0-9]+(-[a-z0-9]+)*$` (lowercase letters,
digits, hyphens).

### gap registry init

```text
gap registry init [--name NAME] [--add] PATH
```

Scaffold a new, empty registry: a `pyproject.toml` (depending on
`graph-as-policy`, with pytest config that deselects `gpu`/`llm` markers
by default), `tools/`, `skills/`, a `tests/conftest.py` with
session-scoped registry fixtures, a `README.md`, and a `.gitignore`.

| Flag | Type / default | Meaning |
|---|---|---|
| `PATH` | path, required | Directory to scaffold the registry into. |
| `--name NAME` | string; default: the directory name | Registry/distribution name. |
| `--add` | flag | Also add it to the user config (highest precedence). |

Exit codes: `0` scaffolded; `1` the directory already contains a
`pyproject.toml`, `tools/`, or `skills/` (refuses to overwrite); `2`
invalid name.

```bash
gap registry init ~/skills/my-lab-skills --add
```

### gap registry list

```text
gap registry list [--format {pretty,json}]
```

Show the active registries in precedence order, with columns
PRIORITY / NAME / PATH / SOURCE / DIST / TOOLS / SKILLS / STATUS. This is
where typos show up: configured-but-broken entries are surfaced with a
status like `broken (not a registry checkout)` or `shadowed path`, and a
note is printed when a `--skills` or `$GAP_SKILLS_PATH` override is
suppressing your configured registries.

| Flag | Type / default | Meaning |
|---|---|---|
| `--format {pretty,json}` | choice; default: `pretty` | Output format. |

Exit code: always `0`.

### gap registry add

```text
gap registry add [--user | --project] NAME PATH
```

Add a registry to the user or project config. The new entry is
**prepended**, making it the highest-precedence entry in that config —
its bundles shadow same-named bundles in later entries and in
lower-precedence layers (a project-config entry still outranks
user-config ones). Remote URLs are rejected:
clone the registry locally first, then add the local path.

| Flag | Type / default | Meaning |
|---|---|---|
| `NAME` | string, required | Registry name (lowercase letters / digits / hyphens). |
| `PATH` | path, required | Registry checkout root (a local directory containing `tools/` or `skills/`). |
| `--user` | flag (default) | Write to `~/.config/gap/registries.toml`. |
| `--project` | flag | Write to the nearest `pyproject.toml` `[tool.gap].registries`. |

Exit codes: `0` added; `1` duplicate name, or `--project` with a
multi-line `registries = [...]` array (only single-line arrays are
edited automatically — fix by hand); `2` remote URL, a directory that is
not a registry checkout, an invalid name, or `--project` with no
`pyproject.toml` found.

```bash
gap registry add my-lab-skills ~/skills/my-lab-skills
```

### gap registry remove

```text
gap registry remove [--user | --project] NAME
```

Remove a registry from the user or project config.

| Flag | Type / default | Meaning |
|---|---|---|
| `NAME` | string, required | Registry name to remove. |
| `--user` | flag (default) | User config. |
| `--project` | flag | The nearest `pyproject.toml` `[tool.gap].registries`. |

Exit codes: `0` removed; `1` name not found; `2` `--project` with no
`pyproject.toml`.

## gap tools

```text
gap tools {list,show} ...
```

Inspect the merged tool catalog: connector `robot.*`/`sim.*` tools
(registered at run time), bundle `@tool` functions (live-introspected
with full schemas), core codegen meta-tools, and static rows parsed from
`SKILL.md` for bundles whose dependencies are not installed. See the
[Tool catalog](../skills/tool-catalog.md).

### gap tools list

```text
gap tools list [--skills PATH] [--registry NAME] [--format {pretty,markdown,json}]
               [--tag TAG] [--scope {runtime,codegen}] [--static]
```

List every tool in the active registries. Bundles whose deps are missing
still appear, as static rows with status
`static (deps not installed — <install hint>)`.

| Flag | Type / default | Meaning |
|---|---|---|
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict to one active registry. |
| `--format {pretty,markdown,json}` | choice; default: `pretty` | Output format. Pretty truncates long summaries. |
| `--tag TAG` | string; default: none | Only tools carrying this tag. |
| `--scope {runtime,codegen}` | choice; default: both | Only tools of this scope (`runtime` = dispatchable from graphs). |
| `--static` | flag | Never import bundles — catalog from `SKILL.md` declarations only. |

Exit codes: `0` at least one tool listed; `1` **zero tools matched** the
filters; `2` resolution errors.

```bash
gap tools list --tag perception --format markdown
```

### gap tools show

```text
gap tools show [--skills PATH] [--registry NAME] [--format {pretty,json}] NAME
```

Full input/output schema (type, required, default, description) and a
one-line runnability verdict for one tool — e.g. `ready`,
`not-ready — gpu: ...`, or `satisfied by the live connector at run time`.
When the bundle's deps are not installed, the summary degrades to the
`SKILL.md` declaration and the schemas are unavailable.

| Flag | Type / default | Meaning |
|---|---|---|
| `NAME` | string, required | Tool name, e.g. `geometry.compute_obb`. |
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict to one active registry. |
| `--format {pretty,json}` | choice; default: `pretty` | Output format. |

Exit codes: `0` found; `1` unknown tool (prints `did you mean:`
suggestions); `2` resolution errors.

## gap skills

```text
gap skills {list,check,table,new,test} ...
```

Inspect, verify, scaffold, and test skill bundles. All subcommands accept
`--skills` and `--registry` as in
[Common options](#common-options-registries). See
[Authoring bundles](../skills/authoring-bundles.md) and
[Testing bundles](../skills/testing-bundles.md) for the workflows behind
`new` and `test`.

### gap skills list

```text
gap skills list [--skills PATH] [--registry NAME]
```

List discovered bundles across registries, with columns
KIND / NAME / REGISTRY / DESCRIPTION / TOOLS. Shadowed bundles are marked
`(shadowed)`; bundles with a broken `SKILL.md` show `(SKILL.md rejected)`.

Exit codes: `0` bundles listed; `1` no bundles found; `2` resolution
errors.

### gap skills check

```text
gap skills check [--skills PATH] [--registry NAME] [--download]
```

Validate every bundle in two passes: a static format check (`SKILL.md`
frontmatter per kind, resources on disk, resolvable `allowed_tools`, type
names, pip-extra convention) and a per-bundle import probe. Each bundle
is registered individually so one broken bundle cannot mask others;
`ImportError`s are mapped to install hints. Prints
`[kind] name: PASS|WARN|FAIL` per bundle, then totals.

| Flag | Type / default | Meaning |
|---|---|---|
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict to one active registry. |
| `--download` | flag | After the checks, run each bundle's optional `prefetch()` hook to download model weights. A prefetch failure counts as a failure. |

Exit codes: `0` all bundles pass (warnings allowed); `1` any `FAIL`, any
prefetch failure, or no bundles found; `2` resolution errors.

```bash
gap skills check --download
```

:::{note}
`--download` is a hook mechanism: it invokes `prefetch()` on bundles that
define one. No bundle in [open-robot-skills](gh-skills:README.md)
currently defines a `prefetch()`, so model weights download lazily at
the first model call instead.
:::

### gap skills table

```text
gap skills table [--skills PATH] [--registry NAME]
                 [--format {pretty,markdown,json}] [--kind {tool,skill}]
```

Dump a catalog table of all discovered bundles using static `SKILL.md`
parsing only — no imports, so it works without any bundle deps installed.
Columns: Bundle / Kind / Registry / Description / Tools / Extra (the
`dist[bundle]` pip extra). Markdown output is paste-ready for registry
READMEs: bundle names link to `tools/<name>/` or `skills/<name>/`, and
multiple registries emit one `### <registry>` section each.

| Flag | Type / default | Meaning |
|---|---|---|
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict to one active registry. |
| `--format {pretty,markdown,json}` | choice; default: `pretty` | Output format. |
| `--kind {tool,skill}` | choice; default: both | Only bundles of this kind; markdown output then drops the Kind column. |

Exit codes: `0` bundles listed; `1` no bundles found; `2` resolution
errors.

### gap skills new

```text
gap skills new --kind {tool,skill} [--skills PATH] [--registry NAME] NAME
```

Scaffold a new bundle into the **primary** (highest-precedence) registry.
Tool kind: `SKILL.md` plus a `tools.py` with a `@tool`-decorated
`<name>.run`. Skill kind: `SKILL.md` (with `allowed_tools`,
`exit_conditions`, `produces_outputs`, `required_inputs`,
`canonical_scripts`) plus `scripts/example.py`, `prompts/`, and
`references/`. Both kinds get a generated unit test
(`tests/test_<name>.py`, hyphens mapped to underscores; skipped if it
already exists) and a `conftest.py`
if missing. Prints three next steps: declare the pip extra, run
`gap skills check`, run `gap skills test`.

| Flag | Type / default | Meaning |
|---|---|---|
| `NAME` | string, required | Bundle name (== directory name). |
| `--kind {tool,skill}` | choice, **required** | Bundle kind: `tools/<name>` or `skills/<name>`. |
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: primary registry | Restrict to one active registry. |

Exit codes: `0` scaffolded; `1` the bundle directory already exists; `2`
no registry resolved (printed with hints).

```bash
gap skills new measuring-depth --kind tool
```

### gap skills test

```text
gap skills test [--skills PATH] [--registry NAME] [BUNDLES ...] [-- PYTEST-ARGS]
```

Run bundle unit tests with `pytest`, invoked **from each owning
registry's root** so that registry's `[tool.pytest.ini_options]` governs.
The default registry scaffold deselects `gpu` and `llm` markers, so GPU
tests need an explicit `-- -m gpu`. With no bundle names, every
registry's full suite runs. Per-bundle the target is
`tests/test_<name>.py` (hyphens mapped to underscores), falling back to
`-k` on the underscored name.

Flags must come **before** bundle names; everything from the first
`-`-prefixed positional onward is passed through to pytest (use `--` as
the separator).

| Flag | Type / default | Meaning |
|---|---|---|
| `BUNDLES` | strings; default: everything | Bundle names to test. |
| `--skills PATH` | repeatable; default: resolved registries | See [Common options](#common-options-registries). |
| `--registry NAME` | string; default: all | Restrict to one active registry. |

Exit codes: `0` all invocations passed; `1` any pytest invocation failed
— pytest's "no tests collected" (rc 5) counts as a failure only when
bundles were named explicitly; `2` unknown bundle name or resolution
errors.

```bash
gap skills test sam3 -- -m gpu -x
```

:::{note}
Tests run with the **current** Python interpreter. A registry that keeps
its own virtual environment should run `uv run pytest` inside that
registry instead.
:::

## gap benchmark

```text
gap benchmark [--gate] [--resume] [--families F ...] [--modes M ...]
              [--output-dir DIR] [-v] CONFIG
```

Run a benchmark sweep or an acceptance gate. The config YAML is either a
`benchmark:` grid block (families × variations × modes) or a plain
pipeline config with `suites:` (suites mode). Prints a
`benchmark complete: S/N trials (success_rate=..., completion_rate=...)`
line and the path to the merged `summary.tsv`. See
[Benchmarking](../benchmarks/benchmarking.md) for the workflow and
[Benchmark config](benchmark-config.md) for the YAML schema.

| Flag | Type / default | Meaning |
|---|---|---|
| `CONFIG` | path, required | Benchmark YAML (grid or suites mode). |
| `--gate` | flag | Exit nonzero when the overall success rate is below the config's `gate_threshold` (default `0.90`), when any cell errored, or when no trial ran. Prints `gate PASS` or `gate FAIL`. |
| `--resume` | flag | Reuse the latest run dir, skip cells whose results exist, and rebuild the merged summary. |
| `--families FAMILY ...` | strings; default: all | Restrict the grid to these families (grid mode only). |
| `--modes MODE ...` | strings; default: all | Restrict the grid to these modes (grid mode only). |
| `--output-dir DIR` | path; default: from config | Override the config's output directory. |
| `-v, --verbose` | flag | Enable debug logging. |

Exit codes: `0` benchmark completed (and `gate PASS` when `--gate`); `1`
the run raised an exception, or `--gate` failed; `2` the config failed to
load, or `--families`/`--modes` were passed against a suites-mode config.

```bash
gap benchmark configs/acceptance.yaml --gate
```

:::{note}
Sim benchmarks need the `[libero]` extra and a GPU; install the `[ray]`
extra for parallel workers.
:::

## gap viz

```text
gap viz [--root ROOT] [--services PATH] [--port PORT] [--host HOST] [--no-browser] [-v]
```

Serve the interactive web-based trace visualizer (uvicorn/FastAPI, core
dependencies only). It scans the root directory recursively for trials,
prints the URL, root, trial count, and the first 10 trials, then opens
your browser. The UI renders PNG/JPEG trace assets; recorded
`run_video.mp4` files live in the trace directory and are viewed from
disk, not played in the browser. See [Traces](../running/traces.md).

| Flag | Type / default | Meaning |
|---|---|---|
| `--root ROOT` | path; default: `outputs` | Root output directory, scanned recursively for trials. |
| `--services PATH` | path; default: none | An open-robot-skills checkout; its bundles register tools used to enrich node tooltips with port schemas. (Note: this is `--services`, not `--skills`.) |
| `--port PORT` | int; default: `9432` | Port to serve on. |
| `--host HOST` | string; default: `127.0.0.1` | Host to bind to. |
| `--no-browser` | flag | Don't open the browser automatically (otherwise it opens after 1 second). |
| `-v, --verbose` | flag | Enable debug logging. |

Exit codes: `0` on shutdown; `1` when the root directory does not exist
(message to stderr).

```bash
gap viz --root outputs --services ../open-robot-skills
```

## gap policy

```text
gap policy {serve,list} ...
```

Manage learned-policy servers: spawn a server from a named preset, or
list the presets. The server process runs inside your openpi/MolmoAct
checkout (pointed to by `$GAP_OPENPI_DIR`, shell-expanded at spawn) — the
GPU lives there. On the gap side, learned-policy graph nodes need the
`[policy]` extra (a thin openpi websocket client, no JAX). See
[Policies](../benchmarks/policies.md).

### gap policy serve

```text
gap policy serve [--port PORT] [--startup-timeout SECS] PRESET
```

Spawn a policy server from a preset and block until Ctrl-C, then shut it
down.

| Flag | Type / default | Meaning |
|---|---|---|
| `PRESET` | string, required | Preset name (see `gap policy list`), e.g. `pi05-libero`. |
| `--port PORT` | int; default: an OS-allocated free port | Serve on this port. Pass it explicitly when you need a stable endpoint. |
| `--startup-timeout SECS` | float; default: `900.0` | How long to wait for the server port to open. The generous default exists because the first run downloads checkpoints. |

Exit codes: `0` after a clean Ctrl-C shutdown; `1` the server failed to
spawn; `2` unknown preset.

```bash
gap policy serve pi05-libero --port 8000
```

### gap policy list

```text
gap policy list
```

List the known policy presets. Currently two: `pi05-libero` (openpi
`pi05_libero` checkpoint, served by openpi's `serve_policy.py`) and
`molmoact-libero` (`allenai/MolmoAct-7B-D-LIBERO-0812`, a vLLM-style
server speaking the openpi websocket protocol). Exit code: always `0`.

## gap trace-diff

```text
gap trace-diff [--out PATH] [--quiet] TRACE_A TRACE_B
```

Align two run traces by node name and report per-node and aggregate
verdict agreement — built for rehearsal-vs-real comparisons in CI.
Positionals accept trace directories or `dag_trace.json` files directly.

| Flag | Type / default | Meaning |
|---|---|---|
| `TRACE_A` | path, required | First trace directory (or its `dag_trace.json`) — e.g. the rehearsal / reference run. |
| `TRACE_B` | path, required | Second trace directory (or its `dag_trace.json`) — e.g. the real run. |
| `--out PATH` | path; default: none | Write the structured diff as JSON to this path (parent dirs created). |
| `--quiet` | flag | Suppress the human-readable summary; only write `--out` if set. |

Exit codes: `0` at least one node matched **and** the verdict agreement
rate is exactly `1.0`; `1` the traces aligned but disagreed (agreement
< 1.0); `2` **no alignment** — zero nodes matched by name, a distinct
condition from disagreement.

```bash
gap trace-diff outputs/run_rehearsal outputs/run_real --out diff.json
```
