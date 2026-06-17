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
generated graph runs perception against the default Anthropic VLM even
though codegen was pointed at vLLM.

## Provider matrix

| Provider | Default model | Credentials | Tool-use loop |
|---|---|---|---|
| `anthropic` (default) | `claude-opus-4-8` | `ANTHROPIC_API_KEY` | native |
| `openai` (incl. OpenRouter / vLLM) | none — `model` **required** | `OPENAI_API_KEY` (or `api_key:` in YAML) | OpenAI tools API |
| `vertex` | none — `model` **required** | ADC (`gcloud auth application-default login`) | `claude-*` models only |

Only `anthropic` has a default model. On `openai` and `vertex` the
endpoint serves arbitrary models, so omitting `model` raises a
`ValueError` — always pass `--model` or set `llm.model`.

### anthropic

```bash
export ANTHROPIC_API_KEY=...
gap generate "pick up the milk and put it in the basket"
```

Uses the official SDK with streaming (`get_final_message()`), which keeps
long codegen outputs under request timeouts, and the native tool-use loop.
`api_key:` in the YAML overrides the environment variable.

### openai (any OpenAI-compatible endpoint)

```bash
export OPENAI_API_KEY=...
gap generate "..." --provider openai --model gpt-4o
```

This path is plain HTTP against any chat-completions API — the public
OpenAI endpoint by default, or a custom one (OpenRouter, a local vLLM
server, a proxy) via `endpoint:` in the config YAML:

```yaml
# openrouter.yaml
llm:
  provider: openai
  model: anthropic/claude-sonnet-4.5     # whatever the endpoint serves
  endpoint: https://openrouter.ai/api/v1
  api_key: sk-or-...     # literal value — ${VAR} interpolation does not
                         # apply to api_key; or export OPENAI_API_KEY
```

```yaml
# vllm.yaml — local server, no key needed
llm:
  provider: openai
  model: Qwen/Qwen2.5-Coder-32B-Instruct
  endpoint: http://localhost:8000/v1
```

```bash
gap generate "..." --config openrouter.yaml
```

Details worth knowing:

- `endpoint:` accepts a base URL (`http://host:port/v1`) **or** a full
  `.../chat/completions` URL — both work.
- Requests run over httpx with a 600 s timeout; 429 triggers exponential
  backoff, 5xx and transport errors retry (3 retries), any other 4xx
  raises immediately.
- vLLM with a reasoning parser can return `content: null` when the whole
  response was thinking tokens; GaP coalesces `reasoning_content` so
  downstream parsers don't crash.
- Tools are translated from the Anthropic descriptor shape to the OpenAI
  tools API automatically.

### vertex

```bash
pip install 'graph-as-policy[vertex]'     # AnthropicVertex + google-genai
gcloud auth application-default login
gap generate "..." --config vertex.yaml
```

```yaml
# vertex.yaml
llm:
  provider: vertex
  model: gemini-3.1-flash-lite-preview
  project_id: my-gcp-project
  region: global          # the default when omitted
```

Routing is by model name:

- `claude-*` models go through `AsyncAnthropicVertex` with the **full
  tool-use loop**.
- Anything else (Gemini) goes through `google-genai` as a **plain
  completion — tools are ignored with a warning**. The codegen meta-tools
  (skill references, examples, the coder subagent) are unavailable on
  this path; agents work from their system prompts alone.

Authentication is Application Default Credentials. Set the project with
`llm.project_id`; when it is unset, the Gemini path falls back to the
`GOOGLE_CLOUD_PROJECT` environment variable and the Claude path resolves
the project from the ADC credentials. The benchmark launcher exports
`GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_REGION` from the config into
spawned workers.

## Pipeline config YAML

`gap generate --config <yaml>` and `PipelineConfig.from_yaml` read one
file (parsed by [gap/agent/config.py](gh-engine:gap/agent/config.py)).
The generation-relevant keys, with defaults:

```yaml
llm:
  provider: anthropic          # anthropic | openai | vertex
  model: null                  # null = provider default (anthropic only)
  endpoint: null               # openai path: custom base URL
  api_key: null                # null = ANTHROPIC_API_KEY / OPENAI_API_KEY
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

### Temperature is silently dropped for some models

Models whose id contains `opus-4-7`, `opus-4-8`, or `fable` reject
sampling parameters (the API returns 400), so GaP omits `temperature` for
them on the Anthropic request path (the `anthropic` provider and
`claude-*` on `vertex`) — including the default `claude-opus-4-8`.
Setting `temperature:` therefore has no effect with the default model.
Set `temperature: null` to omit the parameter on every provider.

## The runtime VLM provider (`GAP_VLM_*`)

Perception skills (e.g. `perceiving-objects`) call a hosted
vision-language model through the `vlm.query` / `vlm.query_yes_no` tools
in the [tools/vlm](gh-skills:tools/vlm) bundle. This surface is configured
**only** by environment variables, read at execution time:

| Variable | Meaning |
|---|---|
| `GAP_VLM_PROVIDER` | `anthropic` (default) \| `openai` \| `vertex`. A per-call `provider=` tool argument overrides it. |
| `GAP_VLM_MODEL` | Model id. Default `claude-opus-4-8` on `anthropic`; **required** on `openai` and `vertex`. |
| `GAP_VLM_BASE_URL` | `openai` only, **required**: an OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1`. |
| `GAP_VLM_API_KEY` | `openai` only, optional bearer token. (`anthropic` uses `ANTHROPIC_API_KEY` via the SDK.) |
| `GAP_VLM_PROJECT_ID` | `vertex` only: GCP project. |
| `GAP_VLM_REGION` | `vertex` only; defaults to `global`. |

The vertex path routes `claude-*` to `AnthropicVertex` and everything
else to `google-genai`, same as codegen, and likewise needs the
`[vertex]` extra. Unlike codegen, the VLM tools pin `temperature: 0.0`
and `max_tokens: 1024` on every provider — perception callers make binary
judgments that depend on deterministic decoding — and expose no sampling
knobs.

Example — Gemini on Vertex for runtime perception:

```bash
gcloud auth application-default login
export GAP_VLM_PROVIDER=vertex
export GAP_VLM_PROJECT_ID=my-gcp-project
export GAP_VLM_REGION=global
export GAP_VLM_MODEL=gemini-3.1-flash-lite-preview

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
