# Cable UR (Perception-Only)

Locate a small adhesive marker on a cable with a UR arm and a wrist-mounted ZED
camera, fuse the segmentation with depth into a world-frame point cloud, fit an
oriented bounding box and a local plane, and report the marker's 3D position in
the robot base frame — then inspect the whole result in 3D with a bundled viser
viewer.

:::{note} Requirements
A UR arm reachable over the network, a wrist-mounted ZED camera, the ZED SDK
(manual install — `pyzed` is not on PyPI), a GPU with weights for the
`grounding-dino` and `sam3` tool bundles, and a VLM provider credential
(`ANTHROPIC_API_KEY` or a Vertex setup). The graph itself validates with no
hardware at all.
:::

This is the **standalone-connector** story: the `ur_zed` connector wires real
sensors into a GaP graph **without any motion stack**. Source:
[examples/cable_ur](gh-engine:examples/cable_ur).

## Why motion is structurally impossible

The `ur_zed` connector is perception-only by construction, not by convention:

- The env reads UR joint state over the **read-only RTDE receive interface**
  (`rtde_receive`) — there is no command channel wired at all.
- The connector is built with `motion_enabled=False`, so its tool registry
  contains **only** the five observation getters (`robot.get_observation`,
  `robot.get_camera_pose`, `robot.get_ee_pose`, `robot.get_gripper`,
  `robot.get_gripper_pose`). There is no `robot.go_to_pose`, no gripper tool,
  no trajectory tool — a graph cannot move the arm even by accident.

See [Real-Robot Connectors](../real-robots/connectors.md) for how this
capability model works, and read [Safety](../real-robots/safety.md) before any
hardware session regardless.

## What is in the example

```text
graph/workflow.json        v3 graph (perceive → filter → OBB → report)
graph/scripts/             DINO+VLM perception + RANSAC plane report
visualize.py               viser viewer for the recorded trace
task.yaml                  task metadata (prompt, suite, cameras)
```

## Hardware prerequisites

- **ZED SDK (manual install).** `pyzed` is not pip-installable here — install
  the ZED SDK and its Python API from
  <https://www.stereolabs.com/docs/app-development/python/install>. Everything
  else comes from `uv sync --extra real` (which provides `ur-rtde` /
  `rtde_receive`).
- A UR arm reachable over the network (default IP `172.22.22.2`; pass
  `robot_ip=` to the connector). Only the RTDE *receive* interface is used —
  the example never commands the arm.
- A wrist-mounted ZED with a 4x4 camera-to-wrist calibration matrix saved as
  `.npy`. Point `GAP_UR_ZED_CALIB` at it (or pass `calibration_path=`).
- A UR URDF for forward kinematics: set `GAP_UR_URDF=<path to ur5e.urdf>`, or
  leave it unset to use `robot_descriptions`' `ur5e_description`.

:::{warning}
Without a hand-eye calibration the env logs a warning and **falls back to an
identity transform** — the camera pose then equals the wrist pose, and every
reported 3D position is silently offset by the physical camera mount. Do not
trust positions from an uncalibrated run.
:::

## Run

```bash
# Validate the graph (no hardware needed):
uv run gap run examples/cable_ur/graph --validate-only

# Live perception run:
GAP_UR_ZED_CALIB=/path/to/camera_to_wrist_transform.npy \
uv run gap run examples/cable_ur/graph --real ur_zed
```

Programmatic, with more knobs:

```python
import gap

conn = gap.connector.real(
    "ur_zed",
    robot_ip="172.22.22.2",
    calibration_path="/path/to/camera_to_wrist_transform.npy",
)
result = gap.execute("examples/cable_ur/graph", conn)   # skills auto-discovered
conn.close()
```

The report node prints the marker position in base and camera frames and
writes `/tmp/white_tape_location.json`.

## How the graph works

The top-level graph routes one subgraph, `locate_white_tape` (skill
`perceiving-objects-multiview`), to a `done` or `abort` end node on its
`found` / `not_found` exit. Inside the subgraph:

1. `observe` — `robot.get_observation` captures the ZED RGB-D frame plus the
   UR joint state.
2. `perceive_vlm` — `scripts/locate_white_tape/perceive_dino_vlm.py` runs
   Grounding-DINO with `dino_prompt: "green sticker."`, asks a VLM to pick the
   right detection, falls back to `sam3.segment_text` over the `text_prompts`
   when DINO finds nothing, and lifts the mask into a world-frame point cloud
   using the ZED depth and the calibrated camera pose.
3. `filter_cloud` — `geometry.filter_noise` (DBSCAN, `eps: 0.003`,
   `min_samples: 10`) strips depth speckle.
4. `compute_obb` — `geometry.compute_obb` fits the oriented bounding box.
5. `report` — `scripts/report_location.py` fits a local plane with RANSAC,
   prints the base-frame and camera-frame positions, and writes the JSON
   result file.

Any perception failure raises and routes through the subgraph's
`on_error: "not_found"` exit — the graph aborts cleanly instead of reporting a
garbage pose.

## Visualize the result

```bash
uv run python examples/cable_ur/visualize.py     # latest outputs/run_*
uv run python examples/cable_ur/visualize.py \
    --trace outputs/run_20260610_120000 \
    --calib /path/to/camera_to_wrist_transform.npy
```

The viewer (viser, default `--port 8080`) reads the run's trace directory —
node outputs from `node_data/`, large point clouds from the report node's
`assets/*_cloud.npz` —
and renders the UR at the recorded joint configuration, the camera frustum
with the captured image, the perception cloud, the OBB wireframe, and the
fitted plane. See [Traces](../running/traces.md) for the trace layout.

## About task.yaml

The `task.yaml` next to the graph is **documentation metadata only**: the
`task:`, `suite:`, and `run:` keys record what the example is and how to run
it, but `gap run` never reads this file — you point it directly at the graph
directory. The open-robot-skills checkout is auto-discovered
(`$GAP_SKILLS_PATH` or a checkout next to the graph-as-policy repo); pass
`--skills` to override.

## Adapting to your marker

The prompts in `graph/workflow.json` (`object_name: "green circular sticker"`,
`dino_prompt: "green sticker."`, the `text_prompts` list, `min_points: 5`)
were tuned for a green circular sticker on a black cable. Edit them for your
own marker — they are plain input values on the `perceive_vlm` node, so no
script changes are needed for a different object description.

## Next steps

- [Real-Robot Connectors](../real-robots/connectors.md) — the `ur_zed` and
  `franka` connectors in detail, including the calibration env vars.
- [Real Franka Pick & Place](real-franka-pick-place.md) — the full-motion real
  robot example.
- [Safety](../real-robots/safety.md) — read before any hardware session.
