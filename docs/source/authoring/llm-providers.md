# LLM Providers

One provider layer (`gap.agent.LlmConfig` plus the `complete` /
`complete_with_tools` calls, [gap/agent/llm.py](gh-engine:gap/agent/llm.py))
drives every LLM call the codegen pipeline makes. Pick a provider per call
with `gap generate --provider/--model`, or pin everything — endpoint,
credentials, temperatures, per-role models — in a config YAML passed via
`--config` (or `gap.agent.generate(config=...)`).

There are **two separate provider surfaces**:

| Surface | Configured by | Governs |
|---|---|---|
| Codegen LLM | `--provider/--model`, `llm:` in the config YAML | Generation-time calls: coordinator, subgraph agents, checkpoint agent, validation-fix loop ([Generating graphs](generation.md)). |
| Runtime VLM | `GAP_VLM_*` environment variables | Execution-time perception: the `vlm.query` / `vlm.query_yes_no` tools used by perception skills ([Tool catalog](../skills/tool-catalog.md)). |

Setting one does not configure the other — a common surprise when a
generated graph runs perception against the default OpenRouter VLM even
though codegen was pointed at a local vLLM.

## Provider matrix

| Provider | Default model | Credentials | Tool-use loop |
|---|---|---|---|
| `openrouter` (default) | `gemini-3.1-flash-lite-preview` | `OPENROUTER_API_KEY` (or `api_key:` in YAML) | OpenAI tools API |
| `vertex` | `gemini-3.1-flash-lite-preview` | ADC (`gcloud auth application-default login`) | Gemini via `google-genai` SDK |

Both providers ship a default model, so a bare `gap generate "<task>"`
works with only a credential set; override per-call with `--model` (or
`llm.model`) for a specific one.

`openrouter` speaks OpenRouter's OpenAI-compatible chat-completions API;
its `endpoint:` can point at any other OpenAI-compatible server (a local
vLLM, a proxy). On OpenRouter model slugs are usually namespaced (e.g.
`google/gemini-3.1-flash-lite-preview`, `anthropic/claude-sonnet-4.5`); if
the bare default 404s for your account, set `--model` / `llm.model` /
`GAP_LLM_MODEL` to the namespaced slug.

The default (`gemini-3.1-flash-lite-preview`) was picked because on the
public `libero_quickstart` task it codegens a valid 4-subgraph graph in
~18 s (vs. ~50 s for `gemini-2.5-flash`) and picks the same hand-curated
topology (perceptions front-loaded). Override for other tasks where you
want a stronger or different model.

### openrouter (default; any OpenAI-compatible endpoint)

```bash
export OPENROUTER_API_KEY=...
gap generate "pick up the milk and put it in the basket"
```

By default this targets `https://openrouter.ai/api/v1`. Point it at a
custom OpenAI-compatible endpoint (a local vLLM server, a proxy) via
`endpoint:` in the config YAML:

```yaml
# openrouter.yaml — pin an explicit model + key
llm:
  provider: openrouter
  model: anthropic/claude-sonnet-4.5     # any model OpenRouter serves
  api_key: sk-or-...     # literal value — ${VAR} interpolation does not
                         # apply to api_key; or export OPENROUTER_API_KEY
```

```yaml
# vllm.yaml — local OpenAI-compatible server, no key needed
llm:
  provider: openrouter
  model: Qwen/Qwen2.5-Coder-32B-Instruct
  endpoint: http://localhost:8000/v1
```

```bash
gap generate "..." --config openrouter.yaml
```

Details worth knowing:

- `endpoint:` accepts a base URL (`http://host:port/v1`) **or** a full
  `.../chat/completions` URL — both work. Unset means OpenRouter.
- The key is read from `api_key:` (YAML) or `OPENROUTER_API_KEY`; a
  keyless endpoint (e.g. a local vLLM) works with neither set.
- Requests run over httpx with a 600 s timeout; 429 triggers exponential
  backoff, 5xx and transport errors retry (3 retries), any other 4xx
  raises immediately.
- vLLM with a reasoning parser can return `content: null` when the whole
  response was thinking tokens; GaP coalesces `reasoning_content` so
  downstream parsers don't crash.
- Tools are translated from the canonical descriptor shape to the OpenAI
  tools API automatically.

### vertex

```bash
pip install 'graph-as-policy[vertex]'     # google-genai
gcloud auth application-default login
gap generate "..." --config vertex.yaml
```

```yaml
# vertex.yaml
llm:
  provider: vertex
  model: gemini-3.1-pro-preview
  project_id: my-gcp-project
  region: global          # the default when omitted
```

Keep `region: global` for the preview Gemini models
(`gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`): they are
served from the global endpoint only — pinning a regional endpoint such
as `us-central1` returns a 404 ("publisher model not found").

Vertex serves **Gemini models only** (Claude-on-Vertex was removed):
requests go through `google-genai` with a native function-calling tool
loop. Authentication is Application Default Credentials. Set the project
with `llm.project_id`; when it is unset, the project falls back to the
`GOOGLE_CLOUD_PROJECT` environment variable. The benchmark launcher
exports `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_REGION` from the config
into spawned workers.

## Pipeline config YAML

`gap generate --config <yaml>` and `PipelineConfig.from_yaml` read one
file (parsed by [gap/agent/config.py](gh-engine:gap/agent/config.py)).
The generation-relevant keys, with defaults:

```yaml
llm:
  provider: openrouter         # openrouter | vertex
  model: null                  # null = provider default (gemini-3.1-flash-lite-preview)
  endpoint: null               # openrouter path: custom base URL (default openrouter.ai)
  api_key: null                # null = OPENROUTER_API_KEY (openrouter); vertex uses ADC
  project_id: null             # vertex
  region: null                 # vertex; null = "global"
  temperature: 0.7             # null = always omit the parameter
  max_tokens: 20480
  max_concurrent_requests: 4   # per-event-loop semaphore
  cache_dir: null              # null = $GAP_LLM_CACHE_DIR, unset = no cache

composition:                   # per-role overrides for the multi-agent pipeline
  coordinator_model: null      # null = llm.model
  subgraph_model: null         # subgraph/checkpoint/coder agents; null = llm.model
  subgraph_temperature: 0.3    # applied with subgraph_model (see note)
  max_validation_retries: 2    # LLM fix rounds after graph validation
  max_subgraph_retries: 2
  max_coordinator_retries: 2
  checkpoint_agent: true       # false = skip postcondition authoring

skills: ../open-robot-skills   # registry root(s): a path or a list
```

- **Per-role resolution:** the coordinator uses `coordinator_model`; the
  subgraph, checkpoint, and coder agents use `subgraph_model` and
  `subgraph_temperature`. Each falls back to `llm.model` /
  `llm.temperature` when unset. Note that in the current resolver
  `subgraph_temperature` only takes effect when `subgraph_model` is also
  set — pin both to control subagent sampling.
- `max_subgraph_retries` / `max_coordinator_retries` are parsed but not
  currently wired to the runner: every agent role gets 3 attempts. The
  live retry knob is `max_validation_retries` (the post-assembly fix
  loop — see [Generating graphs](generation.md)).
- `skills:` takes one path or a precedence-ordered list. `${VAR}`
  environment interpolation applies to these entries; relative paths
  resolve against the YAML file's directory; every entry must exist at
  load time or the config raises `ValueError`. When the key is omitted,
  the active registries are resolved as usual
  ([Registries](../skills/registries.md)).
- The remaining keys (`task`, `suites`, `trials`, `environment`,
  `safety_limits`, `policies`, `policy_manager`) drive benchmark-scale
  generation — see the [benchmark config reference](../reference/benchmark-config.md).

:::{warning} No `base:` inheritance
Earlier configs could inherit from a `platform.yaml` via `base:`. That is
gone — a config carrying `base:` raises `ValueError` with a migration
message. Inline the inherited keys instead.
:::

### Temperature and models that reject sampling parameters

The `openrouter` path forwards `temperature` on every request unless it is
`null`. Some models (for example certain reasoning models routed through
OpenRouter) reject sampling parameters and return a 400 — set
`temperature: null` to omit the parameter entirely.

## The runtime VLM provider (`GAP_VLM_*`)

Perception skills (e.g. `perceiving-objects`) call a hosted
vision-language model through the `vlm.query` / `vlm.query_yes_no` tools
in the [tools/vlm](gh-skills:tools/vlm) bundle. This surface is configured
**only** by environment variables, read at execution time. Each `GAP_VLM_*`
knob inherits from the matching `GAP_LLM_*` / google-SDK variable when
unset, so a shell already configured for the agent LLM doesn't need a
parallel set:

| Variable | Meaning |
|---|---|
| `GAP_VLM_PROVIDER` | `openrouter` (default) \| `vertex`. A per-call `provider=` tool argument overrides it; inherits `GAP_LLM_PROVIDER` when unset. |
| `GAP_VLM_MODEL` | Model id. Defaults to `gemini-3.1-flash-lite-preview` on both providers; inherits `GAP_LLM_MODEL` when unset. |
| `GAP_VLM_BASE_URL` | `openrouter` only, optional: point at another OpenAI-compatible endpoint (default `https://openrouter.ai/api/v1`), e.g. `http://localhost:8000/v1`. |
| `GAP_VLM_API_KEY` | `openrouter` only, optional bearer token (falls back to `OPENROUTER_API_KEY`). |
| `GAP_VLM_PROJECT_ID` | `vertex` only: GCP project (falls back to `GOOGLE_CLOUD_PROJECT`). |
| `GAP_VLM_REGION` | `vertex` only; defaults to `global`. |

The vertex path serves Gemini via `google-genai`, same as codegen, and
likewise needs the `[vertex]` extra. Unlike codegen, the VLM tools pin
`temperature: 0.0` and `max_tokens: 1024` on every provider — perception
callers make binary judgments that depend on deterministic decoding — and
expose no sampling knobs.

Example — Gemini on Vertex for runtime perception:

```bash
gcloud auth application-default login
export GAP_VLM_PROVIDER=vertex
export GAP_VLM_PROJECT_ID=my-gcp-project
export GAP_VLM_REGION=global
export GAP_VLM_MODEL=gemini-3.1-pro-preview

gap run my_graph/task_00 --sim libero_object_all_variance/0
```

Perception results are cached per checkout under
`<open-robot-skills>/.llm_cache/perceiving-objects` (override with
`GAP_PERCEPTION_CACHE_DIR`, disable with `GAP_PERCEPTION_CACHE=0`); the
cache key includes `GAP_VLM_PROVIDER` and `GAP_VLM_MODEL`, so swapping
models never returns stale picks. This cache is distinct from the codegen
cache (`GAP_LLM_CACHE_DIR` — see
[Generating graphs](generation.md)). The full list lives in the
[environment variable reference](../reference/environment-variables.md).

## Next steps

- [Generating graphs from language](generation.md) — the pipeline these
  providers power.
- [Benchmark config reference](../reference/benchmark-config.md) — the
  rest of the YAML schema.
- [Environment variables](../reference/environment-variables.md) — every
  `GAP_*` knob in one table.
