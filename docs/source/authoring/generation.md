# Generating Graphs from Language

`gap generate` compiles a one-sentence instruction into a typed, verified
workflow directory — the same `workflow.json` + `scripts/` + `checkpoints/`
artifact that [gap.builder](builder.md) authors by hand and
[`gap run`](../running/execution.md) executes. Generation runs a multi-agent
LLM pipeline against your skill registries, validates the result with the
same validator the runtime uses, and writes every prompt and response to
disk for debugging.

:::{tip} End-to-end
The full loop **`gap generate "<task>"` → `gap run --sim`** works
end-to-end on the public quickstart task today: the codegen picks
`perceiving-objects` + `grasping-with-planner` + `transporting-objects`
from the registry, vendors their canonical scripts into a fresh graph
directory, and the runtime evaluates LLM-authored postcondition
checkpoints (`target_obb_is_plausible`, `target_held`,
`target_in_container`, …) against sim ground truth. With
`gemini-3.1-flash-lite-preview` on Vertex, codegen is on the order of
~20 s and the sim trial on the order of ~2 min.
:::

:::{note} Requirements
An LLM API key (`OPENROUTER_API_KEY` for the default provider — see
[LLM providers](llm-providers.md)) and at least one skill registry (an
open-robot-skills checkout). Generation is pure LLM calls plus static
validation — no simulator is needed.
:::

## CLI

```bash
export OPENROUTER_API_KEY=...

gap generate "pick up the alphabet soup can and place it in the basket" --out my_graph
```

| Flag | Meaning |
|---|---|
| `--skills PATH` | Skill registry root(s); repeatable, precedence-ordered. Default: the resolved registry set (see [Registries](../skills/registries.md)). |
| `--provider {openrouter,vertex}` | LLM provider override (default: `openrouter`). |
| `--model M` | LLM model override (default: the provider default — `gemini-3.1-flash-lite-preview` on `openrouter`, OpenRouter accepts Gemini slugs with or without the `google/` prefix; `vertex` requires an explicit gemini model). |
| `--out DIR` | Output directory (default: `outputs/generated_<timestamp>`). |
| `--config YAML` | Pipeline config YAML — `llm:` / `composition:` / `skills:` knobs (see [LLM providers](llm-providers.md)). |
| `-v, --verbose` | Debug logging. |

On success the command prints `OK: wrote <path> (N subgraph(s), M generated
file(s))`, renders the graph as box-drawing terminal text (color is
suppressed when stdout is not a TTY or `NO_COLOR` is set), and ends with
`run it with: gap run <path>`.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Graph written and loaded. |
| `1` | Generation failed (`FAIL: ...` — LLM retries exhausted, missing capabilities reported, or the assembled workflow did not load). |
| `2` | No skill registries could be resolved (`error: ...`). |

:::{warning} The output lands in a `task_00` subdirectory
`--out my_graph` does **not** make `my_graph` the workflow folder — the
workflow is written to `my_graph/task_00/` (the generation entry point is
shared with the multi-task benchmark launcher, which numbers tasks). Run it
with `gap run my_graph/task_00`, not `gap run my_graph`.
:::

## Python API

```python
import gap

graph = gap.agent.generate_sync(
    "pick up the alphabet soup can and place it in the basket",
    out_dir="my_graph",          # skills= omitted -> registries auto-discovered
)
print(graph)                     # the graph as box-drawing terminal text
print(graph.path)                # my_graph/task_00
```

Both entry points share one signature; `generate` is async,
`generate_sync` wraps it in `asyncio.run`:

```python
async def generate(
    instruction: str,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    model: str | None = None,
    provider: str | None = None,
    out_dir: str | Path | None = None,
    config: PipelineConfig | str | Path | None = None,
) -> GeneratedGraph
```

- `skills` — registry root(s); a path or a precedence-ordered sequence.
  Omitted means the active registries are resolved (`$GAP_SKILLS_PATH` >
  project `[tool.gap]` > user config > an open-robot-skills checkout next
  to the GaP checkout). Generation needs at least one; resolution failure
  raises `FileNotFoundError` listing what was tried.
- `model` / `provider` — per-call LLM overrides; they take precedence over
  the corresponding `config` fields.
- `out_dir` — defaults to `outputs/generated_<YYYYmmdd_HHMMSS>`; the
  workflow folder is always `<out_dir>/task_00`.
- `config` — a `PipelineConfig` instance or a YAML path for full control
  (per-role models, temperatures, retry budgets — see
  [LLM providers](llm-providers.md)).
- Raises `RuntimeError` when generation fails and `FileNotFoundError` when
  no skill registry is resolvable.

The result is a `GeneratedGraph` dataclass:

| Field | Contents |
|---|---|
| `path` | The written workflow folder (`<out_dir>/task_00`). |
| `workflow` | The parsed `workflow.json` dict. |
| `code` | LLM-generated sources keyed by workflow-relative path — inline scripts plus `checkpoints/<sg>.py` sidecars. Canonical skill-owned scripts are materialized on disk but not included here. |

`str(graph)` renders the same box-drawing text the CLI prints
(`gap.viz.text.to_text`).

## What gets written

```text
my_graph/task_00/           # = graph.path
├── workflow.json           # the v3 graph: top-level DAG + one entry per subgraph
├── scripts/                # scripts referenced by type:script nodes
├── checkpoints/            # postcondition sidecars, one module per subgraph
├── agent_traces/           # per-agent prompts + responses (below)
└── multi_agent_meta.json   # {"pipeline": "gap_multi_agent", "schema_version": 3}
```

See the [workflow schema reference](../reference/workflow-schema.md) for
the file format and [examples/generate-a-graph](../examples/generate-a-graph.md)
for a complete walkthrough with real output.

## Pipeline anatomy

```text
instruction
   │
   ▼
coordinator ──► subgraph agents (sequential, one per subgraph)
   │                  │
   ▼                  ▼
topology        structure + inline scripts
                      │
                      ▼
              checkpoint agent (one whole-workflow call)
                      │
                      ▼
              assemble workflow.json + materialize canonical scripts
                      │
                      ▼
              validate ⇄ LLM fix loop (bounded)
```

1. **Coordinator** — turns the task into a workflow topology: which skills
   to use, the subgraph input/output schemas, and the conditional edges
   between subgraphs. It emits a Python block binding
   `spec = WorkflowSpec(...)`, which is executed in a sandbox and checked
   at parse time (non-empty nodes/edges/subgraphs; every subgraph's
   `skill` must exist in the registry).
2. **Subgraph agents** — one call per subgraph, **sequential by design**:
   each agent sees the bound outputs of every upstream subgraph, so later
   prompts can reference earlier results. Each emits a builder block
   binding `sg = Subgraph(...)` plus optional inline scripts as fenced
   code blocks whose info string carries the path
   (`python:scripts/<sg>/<file>.py`), validated against
   the structural rules described in [Graph patterns](patterns.md)
   (declared nodes, exit values matching the skill's exit conditions,
   all outputs bound, ...).
3. **Checkpoint agent** — one whole-workflow call (enabled by
   `composition.checkpoint_agent`, default on) that authors postcondition
   checkpoints for every subgraph at once. Its output is AST-validated
   (only `add_checkpoint(...)` calls; predicate reads must match declared
   outputs) and semantically checked: every subgraph needs at least one
   `validate=True` checkpoint with a non-empty rationale, at most 6 per
   subgraph, and grasp/transport subgraphs that bind pose outputs must
   include a z-axis predicate. The sidecars land in `checkpoints/<sg>.py`.
4. **Assemble** — the per-subgraph dicts are stitched into a
   `{"version": 3, ...}` `workflow.json` and the folder is written.
5. **Validate + fix** — the workflow is loaded and checked with the
   runtime validator. Script errors trigger up to
   `composition.max_validation_retries` (default 2) LLM fix rounds, each a
   focused per-script "fix ONLY these errors" prompt.

### Retries and corrective feedback

Each agent role (coordinator, subgraph agent, checkpoint agent) gets **3
attempts**. When an attempt fails parse-time validation, the error text is
appended to the conversation as a corrective user turn ("Your previous
response had errors: ... Fix every error and re-emit the full block") and
the model tries again; exhausting the budget fails the whole generation.

Below the agent loop, each LLM call has its own transport retries: 3
retries with exponential backoff (2 s initial, plus jitter). On the
OpenAI-compatible path, 429 backs off, 5xx and transport errors retry, and
any other 4xx raises immediately. Agents may also call codegen meta-tools
(reading skill references and examples, requesting helper scripts from a
coder subagent); the tool-use loop is capped at 6 rounds per call.

If an agent calls the `report_missing_capability` meta-tool — the task
needs a skill or tool the registries do not provide — the build **aborts**
after the subgraph stage with a structured list of the gaps, rather than
emitting a graph that cannot work.

### Generated (invented) skills

When **no** skill in the resolved registries fits a step the task requires — but the step can
still be built from existing tools plus some custom Python — the coordinator can **invent a skill
inline** instead of aborting via `report_missing_capability`. It does this by passing
`generated=True` to `spec.declare_subgraph(...)` in the workflow spec (the same flag is also
available on `gap.builder.Subgraph` for hand-authored graphs). The subgraph_agent then implements
the invented skill from scratch by composing `type: tool` nodes and authoring `type: script`
nodes:

```python
spec.declare_subgraph(
    "insert_peg",
    skill="insert-peg-in-hole",      # invented name — NOT from the catalog
    generated=True,
    description="Insert the held peg into the hole on the fixture",
    inputs={"peg_pose": "Se3Pose", "hole_pose": "Se3Pose"},
    outputs={"inserted": "bool"},
    exit_success_values=["inserted"],
    on_error="failed",
)
```

There is **no canonical bundle on disk** and **no canonical script** for these skills — the LLM
emits both the skill declaration and its scripts as part of codegen, all in the workflow folder.
The runtime synthesises a transient `SkillInfo` (a minimal `SkillMeta` carrying the agent-declared
contract) so the rest of the pipeline can treat the invented skill uniformly: structural
validation still runs in full (the input/output contract, `exit_success_values` / `on_error`
consistency, the subgraph S1-S12 rules, and `allowed_tools` against the flat tool catalog), and
validation errors flow back to the authoring agent in the same self-repair feedback loop used for
registered-skill subgraphs.

Use this for a **one-off, task-specific micro-skill** that isn't worth promoting to the canonical
skill library — e.g. a `compute_target_drop_zone` step that's specific to one layout, or a
`insert-peg-in-hole` segment with bespoke fixture geometry. For anything you expect to reuse, ship
it as a proper bundle ([Authoring bundles](../skills/authoring-bundles.md)): an invented skill is
implemented from scratch every run, where a registry skill is tested and canonical. The coordinator
prompt enforces this preference: invent only when nothing in the Available Skills table fits.

The agent-side mirror of this rule lives in
[agent/skills/gap/references/authoring-graphs.md](gh-engine:agent/skills/gap/references/authoring-graphs.md)
(the authoring digest the in-IDE skill loads when you build graphs by hand).

### Canonical scripts always win

Skills can ship canonical (skill-owned) scripts, e.g. the perception
bundle's `perceive_dino_vlm`. When any script node references a path whose
stem matches a canonical script, the bundle's file is copied into the
workflow folder — **overwriting any LLM-emitted version of it**. LLM
reimplementations of canonical scripts have repeatedly introduced
regressions (wrong imports, simplified pose math, dropped parameters), so
the canonical source is authoritative; inline LLM scripts persist only
under non-canonical stems.

### Generation can "succeed" with residual validation errors

:::{warning}
If validation errors remain after the fix budget is exhausted, generation
still returns successfully — the residual errors are only logged as a
warning. Failures then surface at load or execution time. Always validate
before running:

```bash
gap run my_graph/task_00 --validate-only
```
:::

## Agent trace artifacts

Every prompt and every response is written under
`<workflow_dir>/agent_traces/`, one directory per agent:

```text
agent_traces/
├── _coordinator/
│   ├── system_prompt.md             # exact assembled system prompt
│   └── llm_response_attempt_0.md    # one file per attempt
├── <subgraph_name>/                 # one dir per subgraph
│   ├── system_prompt.md
│   ├── subgraph_spec.json           # the spec the coordinator handed down
│   └── llm_response_attempt_0.md
├── _checkpoint_agent/
│   ├── system_prompt.md
│   ├── user_prompt.md                # the workflow + builder sources it saw
│   └── llm_response_attempt_0.md
└── _validation_fix/
    └── attempt_0/                   # one dir per fix round
        ├── <script>_prompt.txt
        ├── <script>_response.txt
        └── <script>_fixed.py
```

When a generated graph misbehaves, read the responses in attempt order:
multiple `llm_response_attempt_N.md` files mean parse-time validation
rejected earlier attempts, and the corrective turns show exactly which
rule fired. These are generation-time traces; execution traces are
separate (see [Traces](../running/traces.md)).

## The LLM disk cache

Set `GAP_LLM_CACHE_DIR` (or `cache_dir:` under `llm:` in a config YAML) to
memoize plain completions on disk:

```bash
export GAP_LLM_CACHE_DIR=~/.cache/gap-llm
gap generate "pick up the milk and put it in the basket"
```

- The cache key is a SHA-256 hash over provider + model + temperature +
  `max_tokens` + system prompt + messages; entries are
  `<cache_dir>/<hash>.json`. Reruns of an identical prompt return the
  cached response without an API call.
- `GAP_LLM_NO_CACHE=1` (or `true`/`yes`/`on`) bypasses the cache without
  deleting it.
- **Tool-loop responses are never cached** — tool side effects make
  memoization unsound — so agents that call meta-tools always hit the API.

:::{warning} The cache collapses sampling
The cache only correctly memoizes deterministic calls. At temperature > 0
it collapses K stochastic samples into one stored response — every "fresh"
generation of the same prompt replays the first one. Set
`GAP_LLM_NO_CACHE=1` when you want diverse regenerations.
:::

This is the **codegen** cache. The perception pipeline has a separate
runtime cache (`GAP_PERCEPTION_CACHE_DIR`) keyed on the VLM provider
config — see [LLM providers](llm-providers.md) and the
[environment variable reference](../reference/environment-variables.md).

## Next steps

- [LLM providers](llm-providers.md) — provider setup and the pipeline
  config YAML (per-role models, temperatures, retry budgets).
- [examples/generate-a-graph](../examples/generate-a-graph.md) — the full
  walkthrough, from instruction to a validated run.
- [Authoring with the builder](builder.md) — produce the identical
  artifact in Python, no LLM involved.
- [Benchmarking](../benchmarks/benchmarking.md) — drive this pipeline over
  task × seed grids.
