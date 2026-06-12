# cable_ur — locate a marker on a cable with a UR arm + ZED camera

> **What:** Perception-only UR + ZED connector — motion structurally impossible (read-only RTDE) · **Needs:** `real` + ZED SDK + UR arm · **Time:** seconds · **Safety:** read [../../docs/safety.md](../../docs/safety.md) first

Perception-only example: detect a small adhesive sticker on a cable,
fuse the segmentation with ZED depth into a world-frame point cloud, fit
an OBB and a local plane, and report the marker's 3D position and
orientation in the robot base frame. Then inspect the result in 3D with
`visualize.py`.

This is the **standalone-connector** story: the `ur_zed` connector wires
real sensors into a gap graph **without any motion stack**. The env
captures directly from the ZED (pyzed SDK) and reads UR joint state over
read-only RTDE; camera pose comes from URDF forward kinematics plus a
hand-eye calibration. The connector registers *only* observation/camera
tools — there is no `robot.go_to_pose`, no gripper, no trajectory tool
in its registry, so a graph cannot move the arm even by accident.

```
graph/workflow.json        v3 graph (perceive → filter → OBB → report)
graph/scripts/             DINO+VLM perception + RANSAC plane report
visualize.py               viser viewer for the recorded trace
task.yaml                  task metadata (prompt, suite, cameras)
```

## Hardware prerequisites

- **ZED SDK (manual install).** `pyzed` is not pip-installable here —
  install the ZED SDK and its python API from
  <https://www.stereolabs.com/docs/app-development/python/install>.
  Everything else comes from `uv sync --extra real`
  (which provides `ur-rtde` / `rtde_receive`).
- A UR arm reachable over the network (default IP `172.22.22.2`; pass
  `robot_ip=` to the connector). Only the RTDE *receive* interface is
  used — the example never commands the arm.
- A wrist-mounted ZED with a 4x4 camera→wrist calibration matrix saved
  as `.npy`. Point `GAP_UR_ZED_CALIB` at it (or pass
  `calibration_path=`); without it the camera pose falls back to the
  wrist pose and the reported 3D positions will be offset by the mount.
- UR URDF for FK: `GAP_UR_URDF=<path to ur5e.urdf>`, or leave unset to
  use `robot_descriptions`' `ur5e_description`.

## Run

```bash
# Validate the graph (no hardware needed):
uv run gap run examples/cable_ur/graph --validate-only

# Live perception run:
GAP_UR_ZED_CALIB=/path/to/camera_to_wrist_transform.npy \
uv run gap run examples/cable_ur/graph --real ur_zed
```

Programmatic (more knobs):

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

## Visualize

```bash
uv run python examples/cable_ur/visualize.py     # latest outputs/run_*
uv run python examples/cable_ur/visualize.py --trace outputs/run_20260610_120000 \
    --calib /path/to/camera_to_wrist_transform.npy
```

Renders the UR at the recorded joint config, the camera frustum (with
the captured image), the perception cloud, the OBB, and the fitted
plane in viser.

## Perception requirements

The perception script calls Grounding-DINO + SAM3 + a hosted VLM through
the open-robot-skills tool bundles (GPU weights + `ANTHROPIC_API_KEY` or a
vertex setup) — see the open-robot-skills README. The prompts in
`graph/workflow.json` (`object_name`, `dino_prompt`, `text_prompts`)
were tuned for a green circular sticker; edit them for your marker.
