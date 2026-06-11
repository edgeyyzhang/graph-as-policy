# real_franka_pick_place — Jello boxes into the bucket, on a real Franka

Pick up every Jello gelatin box on the table and drop it in the white
bucket, looping `perceive_target → grasp → transport` until the
perception subgraph reports the table clean. Runs on a Franka Panda with
a Robotiq gripper and a ZED camera, driven through the
[robots_realtime](../../third_party/robots_realtime) bridge.

```
graph/workflow.json        v3 graph (hand-migrated from the dev tree's v2 schema)
graph/scripts/             perception + approach/align/drop scripts
task.yaml                  task metadata (prompt, suite, cameras)
```

## Safety notice — read before running

This example moves a real robot arm.

- **Keep the E-stop within reach at all times.** Test it before the
  first run.
- **Clear the workspace**: nobody inside the arm's reach envelope; no
  fragile objects besides the task items; cables and the camera tripod
  outside the sweep volume.
- The connector pre-seeds a **hold-home** command before the realtime
  client connects, so the arm holds its pose at startup instead of
  jumping to a Viser gizmo target — still, start with the arm in a sane
  configuration near its home pose.
- `robot.go_home` is **disabled on real robots** (the connector logs a
  warning and does nothing) — the blind home move of the sim path could
  sweep through your scene. Move to free space manually before/after
  runs. The `abort` end node's recovery only opens the gripper.
- Approach heights and drop clearances in `graph/workflow.json` were
  tuned for a specific table height. Re-check them against your setup
  before the first descent.

## Hardware setup (rr-session)

The robots_realtime stack is vendored as a pinned submodule and runs in
its **own** process + environment — gap never imports it:

```bash
git submodule update --init third_party/robots_realtime
cd third_party/robots_realtime && uv sync   # one-time; needs the Franka/robotiq extras
```

The session config is
`third_party/robots_realtime/configs/franka/franka_robotiq_client.yaml`:
a `RobotNode` (Franka + Robotiq), a ZED `CameraNode`, and the
`FrankaOscClientCartesianAgent` client that connects to gap's msgpack
server on `127.0.0.1:9000`. Adjust the camera `device_id`, extrinsics
file, and robot config for your cell.

Wire protocol (both directions framed as 4-byte big-endian length +
msgpack with numpy support): the client sends observations
`{left: {joint_pos[8]}, <camera>: {images, depth_data, intrinsics,
pose_mat}}` and receives actions `{timestamp, left: {joint_pos[7],
gripper}}` republished at 50 Hz.

## Run

```bash
# Validate the graph first (no hardware needed):
uv run gap run examples/real_franka_pick_place/graph --validate-only

# Full run — spawns rr-session automatically, waits for the first camera
# frame, then executes the graph:
uv run gap run examples/real_franka_pick_place/graph --real franka
```

Two-terminal debug flow (watch the realtime client's own logs/TUI):

```bash
# terminal 1
uv run --directory third_party/robots_realtime \
    rr-session configs/franka/franka_robotiq_client.yaml

# terminal 2
uv run gap run examples/real_franka_pick_place/graph --real franka --no-rr-autostart
```

Programmatic:

```python
import gap

conn = gap.connector.real("franka")           # rr_autostart=True, port 9000
result = gap.execute("examples/real_franka_pick_place/graph", conn)
conn.close()
```

If startup hangs in `wait_ready`, the timeout error tells you which wire
stage is silent (no client connected / camera node not publishing / RGB
key missing) — also check the rr-session log it points at.

## Perception requirements

The perception subgraphs call Grounding-DINO + SAM3 + a hosted VLM
through the open-robot-skills tool bundles: you need a GPU with the model
weights for `grounding-dino` and `sam3`, and a configured VLM provider
(`ANTHROPIC_API_KEY` or a vertex setup) — see the open-robot-skills README.

## v2 → v3 migration notes

The dev-tree source graph was schema v2 (`states`/`transitions`,
per-state `on_success`/`on_failure`). The mapping applied here:

- each v2 subgraph state → a v3 node; the linear `on_success` chain →
  `edges`;
- all `on_failure` targets inside a subgraph collapsed into the single
  `on_error` exit symbol (v3 failures are exceptions, not edges);
- v2 terminal states → a `noop` node listed in `exit.success_values`;
- v2 `transitions` → the top-level `conditional_edges` mapping
  (including the `transport → perceive_target` loop);
- service/skill states → `type: tool` nodes (`gripper.Open` →
  `robot.open_gripper`, `RobotControl.GoToPose` → `robot.go_to_pose`,
  `top_down_grasp_poses_from_obb` → `geometry.top_down_grasp_candidates`,
  `filter_and_obb` → `geometry.filter_and_compute_obb`);
- the unreachable v2 `slip` end state was dropped.
