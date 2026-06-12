# Environment Variables

This page lists every environment variable the gap engine and the
open-robot-skills bundles read, grouped by subsystem. Variables are
optional unless marked required; "unset" means the documented in-code
default applies.

:::{tip}
Explicit function arguments and config-file values beat environment
variables throughout gap. Use env vars for per-shell or per-worker
configuration; use config files for anything you want recorded with a
run.
:::

## Engine runtime

Read by the executor and skill-registry resolution
([Execution](../running/execution.md),
[Executor semantics](executor.md)).

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_TRACE_DIR` | path; default: the workflow directory | Trace output directory when no `trace_dir` argument is given to `gap.execute()` / `WorkflowExecutor`. `gap run` passes `--trace-dir` explicitly (default `outputs/run_<timestamp>`), so this mostly matters for programmatic runs. See [Traces](../running/traces.md). |
| `GAP_ITERATION_CAP` | int; default `10000` | Super-step (node-visit) cap per execution. Exceeding it raises `PipelineError` ("possible runaway loop"). Overridable per-executor via `node_visit_cap`. |
| `GAP_PARALLEL_CANCEL_GRACE_S` | float; default `2.0` | Seconds a streaming node gets to honor its `CancelToken` at scope teardown before the executor force-cancels its future. |
| `GAP_OBSERVATION_STREAM_HZ` | float; default `10.0` | Poll rate of the graph-scoped observation stream. Precedence: `workflow.meta.observation_stream_hz` > this variable > the executor's `observation_hz` argument. |
| `GAP_TRACE_STREAM_SAMPLE_N` | int; default `1` | Persist every Nth observation-stream read into the trace's `stream_reads/` records. `0` disables stream-read recording entirely. See [Traces](../running/traces.md). |
| `GAP_SKILLS_PATH` | `os.pathsep`-separated paths; default: unset | Skill-registry roots, precedence-ordered. A **full override**: when set, project/user config and the auto-discovered sibling checkout are skipped. Every entry must be a valid skills checkout (a directory with a `tools/` or `skills/` bundle root) or resolution fails loudly. See [Registries](../skills/registries.md). |
| `GAP_PARALLEL_INPROC` | `1`; default: unset | With `num_workers <= 1`, runs benchmark trials **in-process** instead of spawning worker processes. Debug/test hook only — production always spawns, because in-process cells share one GPU, one GIL, and one set of guard counters. See [Benchmarking](../benchmarks/benchmarking.md). |

### Safety guards

Call-count caps for guard-tagged tools (`perception`, `planning`,
`sim_step` — see [Connector tools](connector-tools.md#guard-tags)).
Unset means **unlimited**. Per-workflow `gap.tools.guards.set_limits()`
overrides take priority over the environment; the benchmark harness sets
its `safety_limits:` config block this way per trial. Counters reset at
the start of each `execute()`; a violation raises `GuardLimitExceeded`
(a `BaseException` — not catchable with `except Exception`).

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_MAX_PERCEPTION_CALLS` | int; default: unlimited | Cap on `perception`-tagged tool calls per execution. |
| `GAP_MAX_PLANNING_CALLS` | int; default: unlimited | Cap on `planning`-tagged tool calls (e.g. `robot.solve_ik`, curobo planners). |
| `GAP_MAX_SIM_STEPS` | int; default: unlimited | Cap on `sim_step`-tagged tool calls (all motion tools, `sim.step`, `sim.apply_policy_action`). |

## Generation and LLM providers

Read by `gap generate` and the agent pipeline
([Generation](../authoring/generation.md),
[LLM providers](../authoring/llm-providers.md)).

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_LLM_CACHE_DIR` | path; default: unset (caching disabled) | Disk cache for LLM completions during generation. Keyed by provider + model + generation knobs + prompt; only correct for deterministic (temperature-0) calls. |
| `GAP_LLM_NO_CACHE` | `1`/`true`/`yes`/`on`; default: unset | Bypass the prompt cache even when a cache dir is configured. |
| `ANTHROPIC_API_KEY` | string | API key for the default `anthropic` provider, read by the Anthropic SDK. Probed by `gap check`. |
| `OPENAI_API_KEY` | string | API key for the `openai` provider. Custom OpenAI-compatible `endpoint:` configs may not need it. Probed by `gap check`. |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_REGION` | strings | Vertex AI project and region. The benchmark launcher exports them into spawned workers from the config's `llm.project_id` / `llm.region` so worker-side SDK clients can authenticate. |
| `GOOGLE_APPLICATION_CREDENTIALS` | path | Explicit Application Default Credentials file for the `vertex` provider; checked first by the `gap check` credential probe. |
| `CLOUDSDK_CONFIG` | path; default: `~/.config/gcloud` | Where `gap check` looks for the gcloud ADC file (`<dir>/application_default_credentials.json`) when `GOOGLE_APPLICATION_CREDENTIALS` is unset. |

The launcher also forwards a config-file `llm.api_key` into the matching
provider variable (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) for spawned
benchmark workers, since config-derived values would not otherwise
survive the process spawn.

## Simulation and environments

Read by the env layer and the parallel benchmark workers
([Environments](../running/environments.md),
[Connectors](../real-robots/connectors.md)).

| Variable | Type / default | Effect |
|---|---|---|
| `MUJOCO_GL` | string; workers default to `egl` | MuJoCo rendering backend. Spawned benchmark workers `setdefault` it to `egl` (headless GPU rendering); for interactive headless runs set `MUJOCO_GL=egl` yourself. See [Installation](../getting-started/installation.md). |
| `GAP_MUJOCO_EGL_DEVICES` | comma-separated GPU ids; default: unset | Round-robins benchmark workers across GPUs: worker *i* gets `devices[(offset + i) % len]`. E.g. `GAP_MUJOCO_EGL_DEVICES=1,2,3` keeps GPU 0 free for a co-tenant model server. |
| `GAP_LIBERO_JOINT_MOTION_MODE` | `teleport` \| `closed_loop`; default `teleport` | LIBERO joint-motion mode: `teleport` writes qpos directly with a short settle (fast, pure-graph workflows); `closed_loop` tracks targets through physics (mixed policy/graph workflows). Unknown values warn and fall back to `teleport`. Benchmark workers **overwrite** this per process: `closed_loop` when the run uses policies, else `teleport`. |
| `GAP_LIBERO_PERTURBED` | `1`/`true`/`yes`; default: unset | Opt into the perturbed LIBERO variant (scripted moving basket) for envs built without an explicit `perturbed=` kwarg. |
| `GAP_VAB_ROOT` | path; default: `<repo>/third_party/Variational-Automation-Benchmark` | Location of the vab LIBERO fork (variance/packing suites). |
| `GAP_LIBERO_PRO_ROOT` | path; default: `<repo>/third_party/LIBERO-PRO` | Location of the LIBERO-PRO fork (classic suites). |
| `GAP_UR_URDF` | path; default: unset (`ur5e_description`) | UR URDF used for forward kinematics in the UR+ZED connector. |
| `GAP_UR_ZED_CALIB` | path to a 4×4 camera→wrist `.npy`; default: unset | Hand-eye calibration for the UR+ZED connector. Missing calibration degrades **silently to identity** (warning only) — camera poses then equal the wrist pose. |

:::{note}
Two variables in this area are **set by gap itself**, not by you:

- Spawned benchmark workers export `CUDA_VISIBLE_DEVICES` and
  `MUJOCO_EGL_DEVICE_ID` to their assigned GPU id from
  `GAP_MUJOCO_EGL_DEVICES`, pinning the whole stack (torch, curobo, EGL)
  to one physical device. Don't set these manually for benchmark runs.
- `gap.envs.loader` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
  (a `setdefault`) to keep torch ≥ 2.6 loading the LIBERO init-state
  pickles.
:::

## Policies

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_OPENPI_DIR` | path | Your openpi (or MolmoAct-flavored openpi) checkout. The built-in policy presets (`pi05-libero`, `molmoact-libero`) expand `$GAP_OPENPI_DIR` in their server start commands. See [Policies](../benchmarks/policies.md). |

## Perception bundles (open-robot-skills)

Read by bundle code at import or call time; each bundle's `SKILL.md`
declares its requirements, surfaced by `gap check`. See the
[Tool catalog](../skills/tool-catalog.md).

### vlm bundle

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_VLM_PROVIDER` | `anthropic` \| `openai` \| `vertex`; default `anthropic` | Which backend `vlm.query` / `vlm.query_yes_no` use. |
| `GAP_VLM_MODEL` | string; default: `claude-opus-4-8` (anthropic only) | Model name. **Required** for the `openai` and `vertex` providers. |
| `GAP_VLM_BASE_URL` | URL | **Required** for the `openai` provider: an OpenAI-compatible chat-completions endpoint, e.g. `http://localhost:8000/v1`. |
| `GAP_VLM_API_KEY` | string; default: unset | Bearer token sent to the `openai`-provider endpoint (omitted when unset — local servers usually don't need it). |
| `GAP_VLM_PROJECT_ID` | string | Vertex project for the `vertex` provider. |
| `GAP_VLM_REGION` | string; default `global` | Vertex region for the `vertex` provider. |

### Other perception bundles

| Variable | Type / default | Effect |
|---|---|---|
| `GAP_SAM3_DEVICE` | torch device; default `cuda` | Device for the SAM3 image model and video tracker. |
| `GAP_DINO_DEVICE` | torch device; default `cuda` | Device for Grounding DINO. |
| `GAP_DINO_MODEL` | HF model id; default `IDEA-Research/grounding-dino-base` | Grounding DINO checkpoint. |
| `GAP_GEMINI_ER_MODEL` | string; default `gemini-robotics-er-1.5-preview` | Model for `gemini-er.detect`. |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | string | API key for the gemini-er bundle, via the google-genai SDK's default resolution (either works). |
| `GAP_MOLMO_BASE_URL` | URL; **required** for the molmo bundle | Molmo has no hosted API — point this at a self-hosted vLLM server's OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8122/v1`. |
| `GAP_MOLMO_MODEL` | string; default `allenai/Molmo2-8B` | Model name served at `GAP_MOLMO_BASE_URL`. |
| `GAP_PERCEPTION_CACHE` | `1` (default) or `0` | Disk cache for the `perceiving-objects` skill: a hit on identical camera frames + args short-circuits the whole DINO + VLM + SAM pipeline. Set `0` to disable. |
| `GAP_PERCEPTION_CACHE_DIR` | path; default: `<skills checkout>/.llm_cache/perceiving-objects` | Where that cache lives — per-checkout by default, easy to wipe. |

## Miscellaneous

| Variable | Type / default | Effect |
|---|---|---|
| `HF_TOKEN` | string | Hugging Face access token for gated model repos, used when bundle weights download from the Hub. Weights download lazily on first model call; `gap skills check --download` runs per-bundle `prefetch()` hooks where a bundle defines one. See [Installation](../getting-started/installation.md). |
| `CUDA_HOME` | path | Required when installing the curobo extra (`pip install -e "open-robot-skills[curobo]" --no-build-isolation`) — cuRobo JIT-compiles CUDA kernels against your toolkit. |
| `NO_COLOR` | any non-empty value | Disables colored graph rendering in `gap generate` output (color is also auto-disabled when stdout is not a TTY). |
| `CC` | compiler path | Used by Triton to JIT-compile the SAM3 video tracker's NMS kernels. The sam3 bundle repairs a missing or stale `CC` automatically, falling back to the first existing `gcc`/`clang`. |

## Related pages

- [CLI reference](cli.md) — flags that override these variables per
  command.
- [Registries](../skills/registries.md) — full `GAP_SKILLS_PATH`
  resolution order.
- [Benchmark config](benchmark-config.md) — config-file counterparts
  (`safety_limits:`, `llm:`, `policies:`).
- [FAQ](faq.md) — common misconfiguration symptoms.
