# Testing Skill Bundles

Every bundle in a registry ships unit tests that run on a CPU-only
machine — no robot, no GPU, no LLM key. The engine makes this practical
by exporting its own test fakes as a public module, `gap.testing`, and by
giving the CLI a registry-aware test runner, `gap skills test`. This page
covers both, plus the marker conventions that keep heavy tests opt-in.

If you haven't written a bundle yet, start with
[Authoring Skill Bundles](authoring-bundles.md).

## The gap.testing surface

`gap.testing` exports exactly four names:

```python
from gap.testing import (
    FakeContext,            # scripted NodeContext stand-in
    ToolCallRecord,         # (tool, kwargs) dataclass for assertions
    make_test_observation,  # synthetic RGB-D scene + ground truth
    assert_graph_valid,     # loader + structural graph validation
)
```

These are the same fakes gap's own suite uses; anything else under
`gap.testing` is internal.

### FakeContext

```python
FakeContext(tool_responses: dict[str, Any] | None = None, *, node_id: str = "test_node")
```

A `NodeContext` stand-in with scripted per-tool responses. `tool_responses`
maps a tool name to one of **three forms**:

```python
ctx = FakeContext(tool_responses={
    "robot.get_observation": obs,          # plain value: returned on every call
    "vlm.query": ["A", "B"],               # list: popped front-to-back, one per call
    "grounding-dino.detect": fake_detect,  # callable: invoked with the call kwargs
})
```

- **Plain value** — returned on every call.
- **Callable** — invoked with the call's kwargs; its return value is
  returned. Raise inside it to exercise error paths.
- **List** — values popped front-to-back, one per call. When the list is
  exhausted, the next call raises `ToolError` with detail
  `"FakeContext: scripted responses exhausted"`.

Calling a tool with **no scripted response** also raises `ToolError`
(listing the scripted names), so tests fail loudly on unexpected calls.

:::{warning}
Lists are always interpreted as call sequences. To make a tool return a
list *value* on every call, wrap it in a callable:
`"my.tool": lambda **kw: [1, 2, 3]`.
:::

The context surface matches what scripts use at runtime:

| Attribute | Behavior |
|---|---|
| `ctx.tool(name, **kwargs)` | dispatches a scripted response; thread-safe; records every call |
| `ctx.publish(value)` | appends to `ctx.published` (for streaming skills) |
| `ctx.cancel_token` | a real `gap.runtime.context.CancelToken` |
| `ctx.policy_executor` | `None` |

Assertion helpers:

- `ctx.calls` — the full call log, a list of `ToolCallRecord`.
- `ctx.calls_to(tool)` — records for one tool name.
- `ctx.call_count(tool)` — number of calls to one tool name.
- `ctx.published` — everything passed to `ctx.publish`.

### ToolCallRecord

A two-field dataclass: `ToolCallRecord(tool: str, kwargs: dict)`. Useful
for asserting not just *that* a tool was called but *with what*:

```python
(call,) = ctx.calls_to("sam3.segment_box")
assert call.kwargs["text"] == "cube"
```

### make_test_observation

```python
make_test_observation(
    objects=None,               # list of (name, center_xyz, size_xyz) boxes
    *,
    camera_name="test_cam",
    image_hw=(120, 160),
    camera_eye=(0.0, -0.7, 0.9),
    camera_target=(0.0, 0.0, 0.1),
    table_z=0.0,
    fov_deg=60.0,
    arm_joints=7,
) -> tuple[Observation, dict]
```

Renders a synthetic tabletop scene — colored axis-aligned boxes seen by a
real pinhole camera model — into a gap `Observation`. The point is
**geometric consistency**: `depth`, `intrinsics`, and the camera `pose`
reproject exactly onto the box surfaces, so perception math
(mask → points → OBB) is tested against true numerics, not mocks.

- `objects` is a list of `(name, center_xyz, size_xyz)` boxes (full
  extents, meters). The default scene is one 6 cm cube at the origin,
  sitting on the table.
- Returns `(observation, ground_truth)`. `ground_truth` maps each object
  name to `{"center", "size", "color", "mask"}` — the mask is boolean
  pixel *ownership* after occlusion — plus a top-level
  `"camera_pose_mat"` (4×4 camera-to-world matrix).
- The observation holds one camera frame (float32 depth) and a stub
  7-DOF arm at gripper fraction 1.0. Sky pixels get depth `0.0`
  (invalid), matching sim conventions — not `inf`.

:::{note}
`make_test_observation` requires scipy (for the look-at camera rotation).
scipy is a base dependency of the gap engine, so it is already present
wherever gap is installed — your bundle's extra does not need it.
:::

### assert_graph_valid

```python
assert_graph_valid(graph, *, skill_registry=None, tool_registry=None)
```

Loads and structurally validates a workflow graph, raising
`AssertionError` with the full issue list when any **error**-severity
finding exists (warnings pass). `graph` may be a dict (written to a
temporary `workflow.json`), a file path, or a directory (in which case
`workflow.json` inside it is loaded). Pass registries to enable the
deeper schema checks. Use it to keep a bundle's `examples/` subgraphs
loadable:

```python
def test_example_subgraph_valid(skills_registry):
    bundle = skills_registry.get("my-skill")
    assert_graph_valid(bundle.bundle_dir / "examples" / "pick.json")
```

## Test layout and fixtures

Registry tests live centrally in `<registry>/tests/`, one file per bundle
or group, named `test_<bundle_with_underscores>.py`. Both
`gap registry init` and `gap skills new` scaffold a
`tests/conftest.py` with two session-scoped fixtures:

```python
SKILLS_ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def skills_registry():
    from gap.skills import load_skills
    return load_skills(SKILLS_ROOT)

@pytest.fixture(scope="session")
def tool_registry(skills_registry):
    from gap.tools import ToolRegistry
    reg = ToolRegistry()
    reg.discover_pending()
    return reg
```

`load_skills` imports each bundle's `tools.py` (firing its `@tool`
decorators into the pending queue), and `discover_pending()` drains them
into a `ToolRegistry` — so `tool_registry` contains every bundle tool
with its introspected schema. `gap skills new` also writes a starter
`tests/test_<name_with_underscores>.py` for the bundle (and never
overwrites an existing one).

## Writing the tests

The standard pattern: can the *model-backed* tools with `FakeContext`,
delegate the pure-CPU tools (like `geometry.*`) to their **real**
implementations, and check the result against `make_test_observation`'s
ground truth. From the open-robot-skills suite
([tests/test_skills_perception.py](gh-skills:tests/test_skills_perception.py)):

```python
import pytest
from gap.errors import PerceptionFailed
from gap.testing import FakeContext, make_test_observation


def _geometry_delegates(tool_registry):
    """Route geometry.* FakeContext calls to the real (CPU) geometry tools."""
    names = ("geometry.mask_to_world_points", "geometry.filter_and_compute_obb")
    return {
        name: (lambda _n: (lambda **kw: tool_registry.invoke(_n, **kw)))(name)
        for name in names
    }


def test_perceive_finds_cube(skills_registry, tool_registry):
    obs, gt = make_test_observation(
        [("cube", (0.0, 0.0, 0.03), (0.06, 0.06, 0.06))], image_hw=(240, 320)
    )
    script = skills_registry.get("my-skill").canonical_scripts["perceive"].module
    ctx = FakeContext({
        "robot.get_observation": obs,
        "grounding-dino.detect": {"detections": [
            {"box": [10, 10, 60, 60], "label": "cube", "score": 0.9}]},
        "sam3.segment_box": {"mask": gt["cube"]["mask"]},
        **_geometry_delegates(tool_registry),
    })
    out = script.run(ctx, object_name="cube")
    assert ctx.call_count("grounding-dino.detect") == 1


def test_perceive_not_found():
    def _no_detections(**kwargs):
        raise PerceptionFailed("nothing detected")   # exercise the error path
    ctx = FakeContext({"grounding-dino.detect": _no_detections})
    ...
```

Cover the happy path and at least one failure exit per skill. For tool
bundles, the scaffolded test asserts registration and a clean typed
schema:

```python
def test_my_model_tool_registered(tool_registry):
    assert "my-model.run" in tool_registry
    desc = tool_registry.get("my-model.run")
    assert desc.summary
    assert desc.schema.inputs  # the typed signature introspected cleanly
```

Anything that loads model weights gets `@pytest.mark.gpu`; anything that
hits a live LLM API gets `@pytest.mark.llm`.

## Running tests: gap skills test

```bash
gap skills test                  # every active registry's full tests/ suite
gap skills test my-skill         # one bundle's tests
gap skills test sam3 geometry    # several bundles, grouped per owning registry
gap skills test sam3 -- -m gpu -x   # pass pytest args after --
```

Mechanics worth knowing:

- **It runs your current interpreter**: `sys.executable -m pytest`, with
  the working directory set to each owning registry's root, so the
  registry's own `[tool.pytest.ini_options]` (marker deselects,
  testpaths) governs — exactly like running pytest there by hand. A
  registry with its own separate venv should be tested with
  `uv run pytest` in that registry instead.
- **Pytest passthrough**: everything from the first dash-prefixed
  positional token on is forwarded to pytest. Put `gap` flags
  (`--skills`, `--registry`) *before* the `--`; argparse folds the
  post-`--` tokens into the positional list, so flags placed after it
  would be swallowed as pytest args.
- **Per-bundle targeting**: the runner prefers
  `tests/test_<name_with_underscores>.py`; if that file doesn't exist it
  falls back to `pytest tests -k <name_with_underscores>` and prints a
  note.
- **Exit codes**: unknown bundle names exit 2 listing the known names;
  pytest exit 5 ("no tests collected") is an error for explicitly named
  bundles but only a note for full-suite runs; the command exits 1 if any
  group failed, else 0.
- Bundles are grouped per owning registry in precedence order; the first
  registry claiming a name owns it.

## Markers: keeping the default run CPU-only

The open-robot-skills pytest config deselects heavy tests by default:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not gpu and not llm'"
markers = [
  "gpu: needs model weights + an NVIDIA GPU",
  "llm: needs a live LLM API key",
]
```

So `gap skills test` (and a bare `pytest tests -q`) must stay green on a
CPU-only machine. The gap engine repo uses the same convention with two
extra markers, `sim` (MuJoCo/EGL) and `real` (robot hardware), and a
300-second per-test timeout.

Run the deselected smokes explicitly when you have the hardware:

```bash
gap skills test sam3 -- -m gpu        # GPU smoke tests for one bundle
gap skills test vlm -- -m llm         # live-API tests (needs a key)
```

:::{warning}
A plain `pytest tests` silently skips every `gpu`/`llm` test — passing
locally does not mean the GPU path works. Run `-- -m gpu` on a GPU
machine before claiming a model-backed bundle is done.
:::

## Pre-PR gate

From the registry root:

```bash
uv run gap skills check          # format validation + import probe
uv run gap skills test my-skill  # the bundle's own tests
uv run pytest tests -q           # the registry's full CPU suite
```

See the [contribution checklist](authoring-bundles.md#contribution-checklist)
for everything else reviewers look for.
