# Real Franka Pick & Place

Pick up every Jello gelatin box on the table and drop it in the white bucket,
looping `perceive_target → grasp → transport` until the perception subgraph
reports the table clean. Runs on a Franka Panda with a Robotiq gripper and a
ZED camera, driven through the vendored
[robots_realtime](gh-engine:third_party/robots_realtime) bridge. Source:
[examples/real_franka_pick_place](gh-engine:examples/real_franka_pick_place).

:::{note} Requirements
A Franka Panda with a Robotiq gripper, a ZED camera, the
`third_party/robots_realtime` submodule (own process and environment — gap
never imports it), a GPU with weights for the `grounding-dino` and `sam3` tool
bundles, and a VLM provider credential (`ANTHROPIC_API_KEY` or a Vertex
setup). The graph validates with no hardware at all.
:::

:::{warning}
This example moves a real robot arm. Read [Safety](../real-robots/safety.md)
in full before running it. In particular:

- Keep the **E-stop within reach** at all times, and test it before the first
  run.
- **Clear the workspace**: nobody inside the arm's reach envelope; no fragile
  objects besides the task items; cables and the camera tripod outside the
  sweep volume.
- Approach heights and drop clearances in `graph/workflow.json` were tuned for
  a specific table height. **Re-check them against your setup before the first
  descent.**
:::

## What is in the example

```text
graph/workflow.json        v3 graph (hand-migrated from the dev tree's v2 schema)
graph/scripts/             perception + approach/align/drop scripts
task.yaml                  task metadata (prompt, suite, cameras)
```

## Hardware setup (rr-session)

The robots_realtime stack is vendored as a pinned submodule and runs in its
**own** process and environment — gap talks to it over a TCP socket and never
imports it:

```bash
git submodule update --init third_party/robots_realtime
cd third_party/robots_realtime && uv sync   # one-time
```

The session config is
`third_party/robots_realtime/configs/franka/franka_robotiq_client.yaml`: a
`RobotNode` (Franka + Robotiq), a ZED `CameraNode`, and the
`FrankaOscClientCartesianAgent` client that connects back to gap's msgpack
server on `127.0.0.1:9000`. Adjust the camera `device_id`, extrinsics file,
and robot config for your cell.

Both directions of the wire are framed as a 4-byte big-endian length plus a
msgpack payload with numpy support: the client sends observations
`{left: {joint_pos[8]}, <camera>: {images, depth_data, intrinsics, pose_mat}}`
and receives actions `{timestamp, left: {joint_pos[7], gripper}}`, which gap
republishes at 50 Hz. The details live in
[Real-Robot Connectors](../real-robots/connectors.md#the-franka-connector).

## Run

```bash
# Validate the graph first (no hardware needed):
uv run gap run examples/real_franka_pick_place/graph --validate-only

# Full run — spawns rr-session automatically, waits for the first camera
# frame, then executes the graph:
uv run gap run examples/real_franka_pick_place/graph --real franka
```

Two-terminal debug flow, when you want to watch the realtime client's own
logs:

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

If startup hangs in `wait_ready`, the timeout error names the silent wire
stage (no client connected / camera node not publishing / RGB key missing) and
points at the rr-session log.

## How the graph works

The top level chains four subgraphs through conditional edges:

```text
START → perceive_container ─ found ──→ perceive_target ─ found ──→ grasp ─ grasped ──→ transport
                  │                          │                       │                     │
              not_found                  table_clean             collision              placed → perceive_target
                  ↓                          ↓                       ↓                  blocked
                abort                      done                    abort                   ↓
                                                                                         abort
```

- `perceive_container` and `perceive_target` use the
  `perceiving-objects-multiview` skill: `robot.get_observation` →
  a DINO+VLM perception script → `geometry.filter_and_compute_obb`.
- `grasp` (`grasping-direct-ik`, no motion planner): open gripper →
  `geometry.top_down_grasp_candidates` (`z_offset: -0.04`) → approach above
  the target (`approach_height: 0.35`) → compute an align pose → two
  `robot.go_to_pose` moves (**rotate at height, then straight descent** —
  never rotate during the descent) → `robot.close_gripper`
  (`settle_steps: 60`).
- `transport` re-observes the container, computes the drop pose
  (`approach_height: 0.2`, `drop_clearance: 0.15` — note that gap OBB
  `extent` values are **half-extents**), approaches, descends, releases, and
  retracts.

Two structural points are worth copying into your own real-robot graphs:

- **`on_error` encodes semantic outcomes.** v3 failures are exceptions routed
  to a single exit symbol per subgraph. Here `perceive_target_sg` sets
  `on_error: "table_clean"`: when the VLM answers that no Jello box remains
  (the curated `object_description` prompt instructs it to "answer with no
  letter"), the perception failure path *is* the loop's success exit.
- **The `abort` end node declares explicit recovery tools**
  (`robot.open_gripper`, then `robot.go_home`). On real robots
  `robot.go_home` is disabled — it logs a warning and does nothing — so the
  recovery effectively only opens the gripper. Nothing implicit moves the arm
  after an abort.

There is **no automatic success check**: `scripts/verify/check_placement.py`
exists (it raises if the target-container XY distance exceeds
`max_distance=0.06`) but is deliberately not wired into the graph. Real
connectors have no scripted success check, so you judge placement yourself or
wire the verify node in.

## About task.yaml

`gap run` never reads `task.yaml` — you point it directly at the graph
directory. The file's `task:` / `suite:` / `run:` keys are documentation
metadata for this example. Two of its details still matter:

- The camera name `robot0_robotview` under `environment.cameras` must match
  the `CameraNode` published by your rr-session config. The env auto-discovers
  the wire-level camera key; this is just gap's name for it.
- When a task config of this shape is consumed by the codegen/benchmark
  pipeline (`gap generate`, `gap benchmark`), its `policies:` block is read by
  the PolicyManager to boot or locate learned-policy servers — see
  [Collect & Train](collect-and-train.md). This example's graph uses no
  policies, so the file carries none.

## v2 → v3 migration notes

The dev-tree source graph used schema v2 (`states`/`transitions`, per-state
`on_success`/`on_failure`). The mapping applied here, useful if you are
porting your own v2 graphs:

- each v2 subgraph state → a v3 node; the linear `on_success` chain → `edges`;
- all `on_failure` targets inside a subgraph collapsed into the single
  `on_error` exit symbol (v3 failures are exceptions, not edges);
- v2 terminal states → a `noop` node listed in `exit.success_values`;
- v2 `transitions` → the top-level `conditional_edges` mapping (including the
  `transport → perceive_target` loop);
- service/skill states → `type: tool` nodes (`gripper.Open` →
  `robot.open_gripper`, `RobotControl.GoToPose` → `robot.go_to_pose`,
  `top_down_grasp_poses_from_obb` → `geometry.top_down_grasp_candidates`,
  `filter_and_obb` → `geometry.filter_and_compute_obb`).

See the [workflow schema reference](../reference/workflow-schema.md) for the
full v3 format.

## Next steps

- [Safety](../real-robots/safety.md) — the full pre-session checklist and what
  gap does and does not enforce.
- [Real-Robot Connectors](../real-robots/connectors.md) — startup ordering,
  heartbeat diagnostics, and the msgpack bridge in detail.
- [Cable UR](cable-ur.md) — the perception-only counterpart of this example.
