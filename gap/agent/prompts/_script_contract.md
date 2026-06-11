# Script contract — emitted Python files

Scripts (whether canonical scripts bundled in a skill or ad-hoc scripts
emitted by the `coder` subagent) follow a uniform contract.

## Shape

```python
from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose, Vec3


class Output(TypedDict):
    pose: Se3Pose


def run(ctx: NodeContext, obb: OrientedBoundingBox, z_offset: float) -> Output:
    ...
```

- `run(ctx: NodeContext, ...) -> Output` with type-annotated parameters.
- `Output` is a `TypedDict` declaring the output fields. Outputs from a
  state are referenced via `{"$ref": "<state>.<field>"}`.
- `NodeContext` exposes:
  - `ctx.tool(name, **kwargs)` — invoke any registered tool by its flat
    catalog name (connector `robot.*` / `sim.*` tools, bundle tools like
    `sam3.segment_text`, in-process `geometry.*` helpers). Returns the
    tool's result dict. This is the only dispatch surface.
  - Use `print()` for debug output.
- Scripts are Python: `True` / `False`, not JSON.

## The gap.types vocabulary (numpy-first TypedDicts)

`gap.types` defines plain `TypedDict`s carrying floats and numpy
arrays — access fields with **dict subscripts**, never attribute
access:

```python
from gap.types import (
    Mask,                 # np.ndarray uint8 [H, W] (0 = background, 255 = fg)
    Observation,          # {"cameras": [CameraFrame], "arms": [ArmState]}
    OrientedBoundingBox,  # {"center": Vec3, "extent": Vec3, "orientation": Quaternion}
    PointCloud,           # {"points": float32 [N, 3], "colors"?: float32 [N, 3]}
    Se3Pose,              # {"position": Vec3, "rotation": Quaternion}
    Vec3,                 # {"x": float, "y": float, "z": float}
)

z = pose["position"]["z"]          # ✅
z = pose.position.z                # ❌ AttributeError — these are dicts
```

| Type | Shape |
|---|---|
| `Vec3` | `{"x", "y", "z"}` (floats) |
| `Quaternion` | `{"w", "x", "y", "z"}` — **wxyz scalar-first**. Top-down gripper is `{"w": 0, "x": 1, "y": 0, "z": 0}`. |
| `Se3Pose` | `{"position": Vec3, "rotation": Quaternion}` — the key is **`rotation`** (NOT `orientation`) |
| `OrientedBoundingBox` | `{"center": Vec3, "extent": Vec3, "orientation": Quaternion}` — `extent` holds **half**-extents; the key is **`orientation`** (NOT `rotation`) |
| `CameraFrame` | `{"name": str, "rgb": uint8 [H,W,3], "depth": float32 [H,W] meters, "intrinsics": float64 [3,3] K, "pose": Se3Pose}` — image/depth are numpy arrays |
| `Mask` | bare `np.ndarray` uint8 `[H, W]` |
| `PointCloud` | `{"points": float32 [N,3]}` (+ optional `"colors"`) |
| `JointState` | `{"positions": float64 [dof]}` |
| `Trajectory` | `{"waypoints": list[JointState]}` |
| `Observation` | `{"cameras": list[CameraFrame], "arms": list[ArmState]}`; each arm has `joint_state`, `gripper_fraction`, `ee_pose` |

Note the asymmetry: `Se3Pose["rotation"]` vs
`OrientedBoundingBox["orientation"]` — it trips up generated code
constantly.

## Loading bundled prompt templates

Canonical scripts that belong to a registered bundle load VLM prompt
templates via:

```python
from gap.skills import load_prompt

vlm_prompt = load_prompt(
    __package__, "<prompt_name>",
    var1=value1, var2=value2,
)
```

The loader walks up to the bundle's SKILL.md and resolves
`prompts/<prompt_name>.md` against the bundle root. This works for
canonical scripts because the registry installs each script under a
synthetic package (`gap_skills.<kind>.<bundle>.scripts.<stem>`)
that points at the bundle directory.

It does NOT work for ad-hoc scripts emitted via `request_inline_script`
— those don't belong to a bundle, so they have no `prompts/` directory
and `__package__` is a synthetic `_gap_script_*` name. If an inline
script needs a prompt, embed the template literal in the script itself.
