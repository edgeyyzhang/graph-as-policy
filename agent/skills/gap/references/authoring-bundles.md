# Authoring skill & tool bundles

A bundle is one directory in a registry. The folder conveys the kind:

| | `tools/<name>/` (tool bundle) | `skills/<name>/` (skill bundle) |
|---|---|---|
| What it is | Model-backed typed callables | A manipulation strategy |
| Code | `tools.py` with `@tool` functions | `scripts/*.py` canonical scripts (+ optional class-based callable in `tools.py`) |
| Declares | `gap.tools: {name: summary}` | `gap.allowed_tools`, `gap.exit_conditions`, outputs/inputs, scripts |
| Examples | sam3, grounding-dino, geometry, curobo, vlm | perceiving-objects-oneshot, grasping-short-axis, transporting-objects |

Scaffold both kinds with `gap skills new <name> --kind tool|skill
[--registry NAME]` — it also writes `tests/test_<name>.py`.

> **What is `gap_core`?** As of the `gap-core` / `gap-runtime` workspace
> split, bundle authors depend only on **`graph-as-policy-core`** (~150
> MB) — no fastapi, JAX, or MuJoCo. The stable authoring surface
> (`@tool`, types, errors, `Skill`/`SkillMeta`/`Param`/`Serving`) lives
> under `gap_core.*`; runtime symbols (`gap.execute`, `gap.connector`,
> `gap.agent`) stay on the full `graph-as-policy` distribution.

## SKILL.md contract

Frontmatter (Agent Skills spec + gap extensions under one `gap:` key):

```yaml
---
name: my-skill                       # == directory name; ≤64 chars, kebab-case
description: One-line summary. ...   # ≤1024 chars; MUST contain "Use when …"
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, gpu]}
gap:
  requires: {gpu: true, env: [MY_KEY], env_any: [A, B], weights: true}
  # tool bundles:
  tools:
    - my-tool.detect: One-line summary of the @tool function.
  # skill bundles:
  allowed_tools: [robot.get_observation, grounding-dino.detect]
  exit_conditions: {found: "...", not_found: "..."}
  produces_outputs: {"<name>_obb": OrientedBoundingBox}
  required_inputs: {object_name: str}
  canonical_scripts:
    - perceive_simple: scripts/perceive_simple.py
  prompts: {vlm_pick: prompts/vlm_pick.md}
  references:
    - {title: Pipeline invariants, path: references/invariants.md}
  hard_rules: [invariants.md#emit-both-obb-and-mask]
  streaming: false
---
```

Hard rules enforced by the parser/validator (`gap skills check`):
`name` == dirname; description present, ≤1024, with a "Use when/for/…"
cue; gap extension keys only under `gap:` (top-level → error); declared
tool names namespaced `<bundle>.<fn>` (`robot.*`/`sim.*` reserved for
connectors); every referenced resource path exists; `allowed_tools`
resolve against connector tools + every declared bundle tool;
`produces_outputs`/`required_inputs` use gap.schema type names; one pip
extra per bundle; `gap.requires` keys limited to
`gpu`/`env`/`env_any`/`weights` (typos rejected). The body sections that
matter to planners: `## When to use`, `## When NOT to use`,
`## Recommended subgraph state flow` (skills), `## Gotchas` (tools).

## requires: — make `gap check` vouch for the bundle

The atomic capability question is "can this bundle run here?". Declare:
`gpu: true` (NVIDIA GPU needed), `env: [VARS...]` (all required),
`env_any: [A, B]` (at least one), `weights: true` (downloads on first
use). `requires: {}` = explicitly nothing. Bundles with weights may add
a **filesystem-only** hook next to `prefetch()` in `tools.py`:

```python
def weights_cached() -> bool | None:   # never downloads, never imports torch
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(MODEL_NAME, "config.json"), str)
```

## Code contracts

Skill script (`scripts/*.py`) — a pure function the executor calls:

```python
from typing import TypedDict
from gap_core.skills import load_prompt

class Output(TypedDict):
    found: bool
    score: float

def run(ctx, *, cameras, object_name: str, min_score: float = 0.0) -> Output:
    obs = ctx.tool("robot.get_observation")
    det = ctx.tool("grounding-dino.detect", image=..., query=object_name)
    prompt = load_prompt(__package__, "vlm_pick", object_name=object_name)
    ...
    return {"found": True, "score": 0.9}      # clean exits, not exceptions
```

`ctx` provides `ctx.tool(name, **kwargs)`, `ctx.publish(value)`
(streaming), `ctx.cancel_token.raise_if_set()`,
`ctx.observation_stream.latest(timeout)`. Long-running/stateful skills
subclass `gap_core.skills.Skill` (instance persists across visits within
one execution; set `gap.streaming: true` and `ctx.publish` per tick).

Tool function (`tools.py`):

```python
from typing import TypedDict
from gap_core.tools import tool

class DetectResult(TypedDict):
    detections: list[dict]

@tool(name="my-tool.detect", summary="One-liner.", tags=("perception",))
def detect(image, query: str, threshold: float = 0.3) -> DetectResult:
    model = _get_model()        # lazy singleton — heavy imports INSIDE
    ...
```

**Lazy-import discipline (test-enforced):** importing `tools.py` must
never import torch/transformers/cuda libs. Load models on first call via
a locked singleton. Typed signatures matter — gap introspects them into
the tool schema that `gap tools show` and the codegen prompts render.

## Authoring a policy bundle

A third kind, `kind=policy`, lives under `<registry>/policies/<name>/`
and wraps a learned-policy server (VLA, diffusion, IL) behind the
connector. Use it when the deliverable is *weights + a serving
entrypoint* rather than scripts or `@tool` callables. The SKILL.md adds
a `gap.serving:` block:

```yaml
gap:
  serving:
    command: "uv run python -m my_policy.serve --port {port}"
    protocol: websocket            # or stdio-msgpack
    env: [HF_TOKEN, GAP_DEVICE]
    requires_gpu: true
    weights_uri: "hf://my-org/my-policy-libero@v0.3"
```

`gap.execute` spawns the command, dials the protocol, and routes
observations/actions through the connector. See
`policies/pi05-libero` and `policies/molmoact-libero` in
open-robot-skills for working examples.

## Dependencies

One pip extra per bundle in the registry's `pyproject.toml`, named after
the bundle (`[]` when none), then `uv lock`. Never import another
bundle's Python; compose via `ctx.tool(...)` calls and `allowed_tools`.

## Testing

CPU-only by default — the registry's pytest config deselects `gpu`/`llm`
markers. Patterns from `gap.testing`:

```python
from gap.testing import FakeContext, make_test_observation

def test_happy_path(skills_registry):
    obs, gt = make_test_observation([("cube", (0, 0, 0.03), (0.06,) * 3)])
    ctx = FakeContext({
        "grounding-dino.detect": {"detections": [...]},   # canned responses
        "vlm.query": {"text": "A"},
        "robot.go_to_pose": [ok1, ok2],                    # sequenced
    })
    script = skills_registry.get("my-skill").canonical_scripts["example"].module
    out = script.run(ctx, cameras=obs["cameras"], object_name="cube")
    assert out["found"] and ctx.call_count("vlm.query") == 1
    assert ctx.calls_to("grounding-dino.detect")[0].kwargs["query"] == "object."
```

Real-model smokes: `@pytest.mark.gpu` / `@pytest.mark.llm`; run them
explicitly with `gap skills test my-skill -- -m gpu`. Graph-level checks:
`gap.testing.assert_graph_valid(graph, skill_registry=..., tool_registry=...)`.

## Pre-PR checklist

```bash
uv run gap skills check          # format + import probe: PASS, no WARN
uv run gap skills test my-skill  # the bundle's unit tests
uv run pytest tests -q           # whole-registry suite stays green
uv run gap check                 # bundle READY (or honest about requires)
uv lock                          # if deps changed
```

Plus: description has its "Use when …" cue; `requires:` declared (tool
bundles: mandatory); no heavy module-level imports; catalogs regenerate
cleanly (`gap skills table --format markdown`).
