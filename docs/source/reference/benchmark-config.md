# Benchmark Config Reference

A benchmark YAML is a **standalone pipeline config** plus, optionally, a
`benchmark:` block. The loader parses the file twice: once as the shared
pipeline config (`task:` / `skills:` / `llm:` / `policies:` / `trials:` …)
and once for the `benchmark:` block, whose presence selects the shape:

- **Grid mode** — `benchmark:` present: sweep
  `modes × (family, variation) × policies` cells.
- **Suites mode** — no `benchmark:` block: the pipeline's own `suites:` list
  is the cell axis.

A YAML with neither a `benchmark:` block nor a `suites:` list raises at load.
For semantics (gating, resume, outputs, GPU spread) see the
[benchmarking guide](../benchmarks/benchmarking.md); for the checked-in
configs see [examples/benchmark](gh-engine:examples/benchmark).

:::{warning}
`base:` YAML inheritance is **rejected** with an error — inline every key.
Benchmark YAMLs must be standalone.
:::

## Shared pipeline keys

These keys apply to both shapes.

### `task`

| Key | Type | Default | Notes |
|---|---|---|---|
| `task` | str | `""` | The sentinel `"auto"` resolves per-task language prompts from LIBERO suite metadata. Grid cells force `task: "auto"` internally regardless. |

### `skills`

| Key | Type | Default | Notes |
|---|---|---|---|
| `skills` | path or list of paths | auto-discovered | Skill registry root(s) the codegen registries are built from. `${VAR}` env interpolation is applied; relative paths resolve against the YAML's directory; each path must exist (else `ValueError`). When omitted, registries are resolved from `$GAP_SKILLS_PATH`, project/user config, or an `open-robot-skills` checkout next to the gap repo — see [Registries](../skills/registries.md). |

### `llm`

The LLM used for graph codegen (`llm_generation` mode and suites mode). See
[LLM providers](../authoring/llm-providers.md).

| Key | Type | Default | Notes |
|---|---|---|---|
| `llm.provider` | str | `anthropic` | `anthropic` \| `openai` \| `vertex` |
| `llm.model` | str | `claude-opus-4-8` (anthropic only) | **Required** for `openai` / `vertex` — omitting it raises `ValueError`. |
| `llm.endpoint` | str | null | OpenAI-compatible base URL or full `/chat/completions` URL. |
| `llm.api_key` | str | null | Falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. |
| `llm.project_id` | str | null | GCP project (vertex). |
| `llm.region` | str | null (`"global"` at call time) | Vertex region. |
| `llm.temperature` | float | `0.7` | Silently omitted for models that reject sampling params. |
| `llm.max_tokens` | int | `20480` | |
| `llm.max_concurrent_requests` | int | `4` | Per-event-loop semaphore. |
| `llm.cache_dir` | path | null | Disk LLM response cache (also `GAP_LLM_CACHE_DIR`). |

### `gate_threshold`

| Key | Type | Default | Notes |
|---|---|---|---|
| `gate_threshold` | float | `0.90` | Acceptance bar for `--gate`: pooled success rate must reach it, with no errored cell and at least one trial. |

Placement rules: suites-mode configs put `gate_threshold` at the **top
level**; grid configs put it inside the `benchmark:` block. Both placements
are accepted everywhere — when both are present, the `benchmark:` block value
wins.

### `policies`

A learned policy is a **skill** (e.g. `pi05-libero`, `molmoact-libero`); the
launcher auto-boots its preset server when a workflow references it, so the
`policies:` block is **optional**. You only add an entry to **override** how a
skill is served, and the entry **key must equal the skill name**. Each entry
is **exactly one** of external, managed, or preset, and is referenced by the
`policy_id` axis (`benchmark.policies` / `mode_overrides.<mode>.policy_id`),
whose values are policy-skill names:

```yaml
policies:
  pi05-libero:                     # override: external server already running
    url: ws://127.0.0.1:9100
  molmoact-libero:                 # override: harness owns the subprocess
    start_cmd: "python serve.py --port {port}"
    env:
      CUDA_VISIBLE_DEVICES: "0"
```

| Key | Type | Notes |
|---|---|---|
| `policies.<id>.url` | str | External server, e.g. `ws://host:port` (`wss://` and bare `host:port` also accepted; the port is required). The harness never starts or stops it; grid mode preflight-probes it with a 5 s websocket handshake when the id is named in `benchmark.policies` or a `mode_overrides.<mode>.policy_id`. |
| `policies.<id>.start_cmd` | str | Managed server. The `{port}` placeholder is substituted with an OS-allocated free port; the command runs via the shell in its own process group and must open `127.0.0.1:{port}` within `policy_manager.startup_timeout_s`. Torn down (SIGTERM, then SIGKILL after `evict_grace_s`) when the run ends. |
| `policies.<id>.env` | mapping str→str | Extra environment for the managed subprocess — merged **over** `os.environ` (config values win key-by-key). Only meaningful with `start_cmd` / `preset`. |
| `policies.<id>.preset` | str | Named recipe (e.g. `pi05-libero`, `molmoact-libero`) expanded into a full managed entry; a sibling `env:` mapping overrides the preset's `env` key-by-key. See `gap policy list`. |

Specifying both `url` and `start_cmd` is a config error; so is specifying
neither.

### `policy_manager`

Tuning for the managed-policy lifecycle manager.

| Key | Type | Default | Notes |
|---|---|---|---|
| `policy_manager.startup_timeout_s` | float | `120.0` | Wall-clock budget for each managed policy's port to accept TCP connections. The posvar grid uses `900` — first run downloads checkpoints. |
| `policy_manager.evict_grace_s` | float | `10.0` | Seconds between SIGTERM and SIGKILL at teardown. |

### `trials`

| Key | Type | Default | Notes |
|---|---|---|---|
| `trials.trials_per_generation` | int | `30` | Trials per task (alias: `total`). |
| `trials.task_ids` | list of int | `[0]` | Tasks to run. |
| `trials.code_generations` | int | `1` | Independent codegen attempts per task. |
| `trials.regenerate_code_per_trial` | bool | `false` | Re-run codegen for every trial. |
| `trials.output_dir` | path | `./outputs` | Resolved to an absolute path at load time; a **relative** value resolves against the process working directory, not the YAML. Suites-mode runs write their timestamped run dir here. |
| `trials.num_workers` | int | `1` | Parallel sim worker processes. |
| `trials.record_video` | bool | `false` | Record per-trial MP4s. |
| `trials.enable_tracing` | bool | `false` | Per-node execution traces (the acceptance config disables them for throughput). |
| `trials.task_timeout_secs` | int | `0` | Per-trial wall-clock watchdog; `0` disables. Exceeding it hard-kills the trial, recorded as failed with `exit_code=124`. The acceptance config uses `900` — passing trials measured 391–537 s; a 600 s cap killed on-track trials. |

:::{note}
In **grid mode** the harness overrides most of `trials:` per cell:
`trials_per_generation` ← `benchmark.n_seeds`, `task_ids` ←
`benchmark.task_ids`/`n_tasks`, `num_workers` ← `benchmark.num_workers`,
`record_video` ← `benchmark.record_video`, `output_dir` ← the cell directory,
and `code_generations=1` / `regenerate_code_per_trial=false` are forced. In
**suites mode** the `trials:` block is load-bearing as written.
:::

The pipeline config also accepts `environment.cameras`, `safety_limits.*`,
`composition.*`, and `max_retries` — codegen knobs shared with
`gap generate`; see [Graph generation](../authoring/generation.md).

## Grid mode: the `benchmark:` block

```yaml
benchmark:
  families: [posvar]
  variations: [pos_var, permutation, basket_swap, all]
  modes: [llm_generation, llm_plus_policy, policy_only]
  n_tasks: 10
  n_seeds: 50
  num_workers: 8
  policies: []
  record_video: false
  output_dir: ../../benchmark_runs/posvar
  gate_threshold: 0.90
  mode_overrides:
    llm_plus_policy:
      workflow_dir: path/to/steered_policy_template
      num_workers: 4
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `benchmark.families` | list of str | `[posvar]` | Families to sweep — see [Families and variations](#families-and-variations). Unknown names raise at load. |
| `benchmark.variations` | list of str | null | In-family variation filter; null means every variation of each selected family. Unknown names raise at load. |
| `benchmark.modes` | list of str | all 3 | Subset of `llm_generation` / `llm_plus_policy` / `policy_only`. Unknown names raise at load. |
| `benchmark.n_tasks` | int | `10` | Task count when `task_ids` is unset (ids `0..n_tasks-1`). |
| `benchmark.task_ids` | list of int | null | Explicit task subset; overrides `n_tasks`. |
| `benchmark.n_seeds` | int | `50` | Trials per task. Seed `i` maps to LIBERO init state `(i-1) % len` — identical across modes. |
| `benchmark.num_workers` | int | `8` | Per-cell parallelism, passed to `launch()` as `trials.num_workers`. GPU spread is `GAP_MUJOCO_EGL_DEVICES=<csv>`. |
| `benchmark.policies` | list of str | `[]` | The A/B axis: policy-**skill** names (e.g. `pi05-libero`, `molmoact-libero`), each optionally overridden in the top-level `policies:` block. When non-empty, every policy-dependent mode runs once per skill and cell dirs gain a `/<policy_id>/` segment. |
| `benchmark.record_video` | bool | `trials.record_video` | Record + collate trial videos into `<run>/videos/`. |
| `benchmark.output_dir` | path | `./benchmark_runs` | Run-dir parent; relative paths resolve against the **YAML's directory**. Ignored in suites mode (which uses `trials.output_dir`). |
| `benchmark.gate_threshold` | float | `0.90` | Wins over a top-level `gate_threshold`. |
| `benchmark.smoke` | bool | `false` | Parsed and stored but **never read by the harness** — an inert marker. |
| `benchmark.mode_overrides` | mapping | `{}` | Per-mode knobs, below. |

### Families and variations

Each `(family, variation)` pair maps to a registered suite; the grid sweeps
the union over the selected families.

| Family | Variation | Suite |
|---|---|---|
| `libero` | `object` | `libero_object` |
| `libero_pro` | `object_swap` | `libero_object_swap` |
| `posvar` | `pos_var` | `libero_object_target_pos_var20x20` |
| `posvar` | `permutation` | `libero_object_target_permutation_variance` |
| `posvar` | `basket_swap` | `libero_object_target_basket_swap_variance` |
| `posvar` | `all` | `libero_object_all_variance` |
| `grocery_packing` | `object` | `libero_object_packing` |
| `grocery_packing` | `permutation` | `permutation_packing` |

The four `posvar` suites carry 10 tasks × 50 baked init-state rows each
(`all` is the union of the perturbation kinds). `libero_pro` deliberately
omits an `object` variation — it would duplicate `libero/object`.

### `mode_overrides.<mode>`

Per-mode overrides of the top-level grid sizing; an omitted key (or `null`)
inherits the top-level value.

| Key | Type | Default | Notes |
|---|---|---|---|
| `mode_overrides.<mode>.n_seeds` | int | inherit | |
| `mode_overrides.<mode>.num_workers` | int | inherit | Throttle policy modes when sharing one inference endpoint (posvar uses `4`). |
| `mode_overrides.<mode>.workflow_dir` | path | null | The workflow template directory. **Required** for `llm_plus_policy` and `policy_only` — missing it errors every cell of that mode (and therefore fails `--gate`). Relative paths are absolutized against the YAML's directory. |
| `mode_overrides.<mode>.policy_id` | str | null | Which policy **skill** fills the template's `{{policy_id}}` placeholder when `benchmark.policies` is empty; the value is a skill name (e.g. `pi05-libero` / `molmoact-libero`). Null falls back to `pi05-libero`. |
| `mode_overrides.<mode>.time_budget_s` | float | null | Parsed but **not consumed by any built-in mode** — inert. |
| `mode_overrides.<mode>.extra` | mapping | `{}` | Forwarded verbatim to the mode (no built-in mode reads more keys). |

### Workflow template placeholders

Template-driven modes copy `workflow_dir` per task and substitute `{{key}}`
tokens in `workflow.json`:

| Placeholder | Value |
|---|---|
| `{{target}}` | Target object phrase parsed from the LIBERO prompt |
| `{{target_full}}` | Same as `{{target}}` (the prompt has no richer descriptor) |
| `{{container}}` | Container phrase from the prompt; falls back to `"basket"` |
| `{{policy_id}}` | Resolved policy-**skill** name (e.g. `pi05-libero`), substituted to form the `<skill>.run` tool node; default `pi05-libero` |

Target/container come from matching the prompt against the pick-and-place
grammar `pick (up) (the) <target> and place it in (the) <container>`. An
unparseable prompt is a hard error only when the template actually contains a
`{{target` token — a template that only parameterizes `{{policy_id}}` is fine
with any prompt.

## Suites mode: the `suites:` list

Without a `benchmark:` block, each `suites:` entry is one benchmark cell; all
remaining cells launch concurrently, each bounded by its own `num_workers`.

```yaml
task: "auto"
gate_threshold: 0.90

suites:
  - suite_name: libero_object_all_variance
    task_ids: [0]
    task_prompts:
      0: "Pick up the alphabet soup can and put it in the basket."
    objects:
      target: "alphabet soup can with a blue and yellow label"
      expected_label: "Alphabet Soup"
      shape_hint: "metal cylinder about 7 cm wide, wrapped in a blue-and-yellow label"
    num_workers: 1

trials:
  trials_per_generation: 50
  num_workers: 25
  task_timeout_secs: 900
  record_video: true
  output_dir: ../../benchmark_runs/grocery_acceptance
```

Per-suite keys (a `null` override means "use the top-level `trials:` value"):

| Key | Type | Default | Notes |
|---|---|---|---|
| `suites[].suite_name` | str | `""` | Registered suite name. Duplicate names across entries get `_02`, `_03`, … cell-dir suffixes. |
| `suites[].task_prompts` | mapping int→str | `{}` | Per-task language prompts; tasks not listed fall back to `task` (and `"auto"` resolves from LIBERO metadata). |
| `suites[].task_ids` | list of int | null | Per-suite task subset. |
| `suites[].trials_per_generation` | int | null | Per-suite trial count override. |
| `suites[].timeout_secs` | int | null | Per-suite trial-watchdog override. |
| `suites[].num_workers` | int | null | Per-suite worker-pool bound; also sets the cell's GPU device-slot offset (cells are staggered by the cumulative `num_workers` of earlier suites). |
| `suites[].objects` | mapping str→str | `{}` | Arbitrary string pairs, consumed twice: appended to the codegen prompt as structured object hints, and substituted as `{{<key>}}` tokens into pre-built workflows. |

The `objects:` keys are free-form; the acceptance config's convention is
`target` (the perception phrase), `expected_label` (the success label), and
`shape_hint` (a geometry cue) so codegen writes unambiguous perception
prompts for the LIBERO grocery assets.

Suites-mode specifics:

- the run dir lives under `trials.output_dir` (there is no
  `benchmark.output_dir`);
- `gate_threshold` is read from the top level;
- `--families` / `--modes` CLI flags are rejected (exit 2);
- policy preflight is skipped — workers boot exactly what each workflow
  references;
- summary rows report mode `llm_generation` with family `suites` and the
  cell name as the variation.

## Validation, rejected and inert keys

| Input | Behavior |
|---|---|
| `base:` present | `ValueError` — inheritance is not supported. |
| Neither `benchmark:` nor `suites:` | `ValueError` — nothing to run. |
| Unknown family / variation / mode | `ValueError` at load. |
| `policies.<id>` with both or neither of `url` / `start_cmd` | Config error. |
| Nonexistent `skills:` path | `ValueError` at load. |
| `benchmark.smoke` | Parsed, never read — inert marker. |
| `mode_overrides.<mode>.time_budget_s` | Parsed, resolved, never consumed — inert. |

## Checked-in configs

| Config | Shape | What it is |
|---|---|---|
| [smoke.yaml](gh-engine:examples/benchmark/smoke.yaml) | grid | 1 cell: `grocery_packing/object` × `llm_generation` × 1 task × 1 seed |
| [posvar.yaml](gh-engine:examples/benchmark/posvar.yaml) | grid | 4 posvar variations × 3 modes × 10 tasks × 50 seeds, policy modes throttled to 4 workers |
| [grocery_acceptance_smoke.yaml](gh-engine:examples/benchmark/grocery_acceptance_smoke.yaml) | suites | Nightly smoke: tasks 0–1 × 10 trials = 20 trials, gated at 0.9 |
| [grocery_acceptance.yaml](gh-engine:examples/benchmark/grocery_acceptance.yaml) | suites | **The release gate**: 10 curated suites × 50 trials = 500 trials, gated at 0.90 |

See the [benchmarking guide](../benchmarks/benchmarking.md) for how to run
them and read the outputs.
