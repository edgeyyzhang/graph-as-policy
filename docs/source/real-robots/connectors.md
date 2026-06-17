# Real-Robot Connectors

:::{warning}
Real-robot runs can move hardware based on LLM-generated graphs and
perception-derived poses. Read [Safety](safety.md) before connecting a robot,
and keep the E-stop in hand for the entire session.
:::

`gap.connector.real()` builds a connector over real hardware the same way
`gap.connector.sim()` builds one over a simulator: it owns one env instance,
exposes a tool registry of `robot.*` tools, and plugs straight into
`gap.execute(graph, conn)`. Two robots are supported:

- **`franka`** — a Franka Panda driven through the vendored
  [robots_realtime](gh-engine:third_party/robots_realtime) bridge. Full
  motion: the standard `robot.*` motion, gripper, and trajectory tools.
- **`ur_zed`** — a UR arm + wrist ZED camera. **Perception-only**: the
  connector registers observation tools exclusively, so a graph cannot move
  the arm.

## The factory

```python
def real(
    robot: str = "franka",
    *,
    cameras: list[str] | None = None,
    rr_config: str | Path | None = None,
    rr_autostart: bool = True,
    rr_log_path: str | Path | None = None,
    port: int | None = None,
    wait_timeout_s: float = 60.0,
    **env_kwargs,
) -> RealConnector
```

| Argument | Applies to | Meaning |
|---|---|---|
| `robot` | — | `"franka"` or `"ur_zed"`; anything else raises `ValueError`. |
| `cameras` | both | Camera-name override (default: the env's `default_cameras` — `robot0_robotview` for franka, `zed_left` for ur_zed). |
| `rr_config` | franka | rr-session YAML, resolved relative to the `third_party/robots_realtime` checkout. Default `configs/franka/franka_robotiq_client.yaml`. |
| `rr_autostart` | franka | Spawn the rr-session client automatically (default). `False` restores the two-terminal debug flow; the connector still blocks in `wait_ready` until your client connects. |
| `rr_log_path` | franka | rr-session log tee destination (default `/tmp/gap_rr_session_<pid>.log`). |
| `port` | franka | msgpack server port (default 9000). |
| `wait_timeout_s` | both | Seconds `wait_ready` blocks for the first RGB observation before raising. |
| `**env_kwargs` | both | Extra env-factory kwargs: `host=` for franka (default `127.0.0.1`); `robot_ip=` (default `172.22.22.2`), `calibration_path=`, `resolution=` (default `"720p"`), `fps=` (default 15) for ur_zed. |

Passing `rr_config` or `port` with `robot="ur_zed"` raises `ValueError` —
there is no rr-session on that path. Both branches block in
`wait_ready(timeout_s=wait_timeout_s)` and close the connector before
re-raising if the first observation never arrives.

```python
import gap

conn = gap.connector.real("franka")          # spawns rr-session, waits for camera
result = gap.execute(graph, conn)            # open-robot-skills auto-discovered
conn.close()                                 # terminates the owned rr-session
```

Or from the CLI: `gap run <graph> --real franka` / `--real ur_zed` (plus
`--rr-config` and `--no-rr-autostart` for franka).

## The capabilities model

Harness code asks a connector what it can do instead of checking types:

```python
conn.capabilities
# Capabilities(reset=False, success_check=False, video=False, world_state=False)
```

On real connectors **all four flags are False**, even though the env classes
expose `reset`-shaped shims for interface parity: on hardware "reset" only
means "wait for an observation", there is no scripted success check, no
managed video capture, and no privileged ground-truth world state. Code that
branches on `capabilities` degrades gracefully on hardware; code that assumes
sim behavior does not.

## What differs from sim

- **`robot.go_home` is a no-op.** With `config.is_real` set (both real envs
  set it), `go_home` logs `"go_home skipped on real robot suite (safety)"`
  and returns without moving. A blind home move from an unknown pose is a
  collision generator; command explicit `robot.move_to_joints` if you
  genuinely need to re-home, with eyes on the arm.
- **No `sim.*` tools, ever.** `sim.reset`, `sim.step`, `sim.check_success`,
  `sim.apply_policy_action`, and the video tools are registered only by
  `SimConnector`. Graphs that call them fail validation or execution on a real
  connector.
- **Checkpoints are skipped.** `gap.execute(..., checkpoints=...)` enforces
  `validate=True` checkpoints against the connector's `world_snapshot()`
  ground truth. Real connectors expose none, so enforcement degrades to a
  **one-shot logged warning** and the run proceeds unchanged — your graph's
  postconditions are simply not checked on hardware. See
  [Checkpoints](../running/checkpoints.md).
- **No seeded resets or success rates.** Benchmark-style flows
  (`conn.reset(seed=...)`, `conn.check_success()`) are sim-only.

## The Franka connector

### Startup ordering (load-bearing)

The env constructor binds the msgpack server and **pre-seeds the action slot
with a hold-home command before the rr-session client is spawned**. The client
retries until the server is up, and its very first action request must see a
valid hold command — an empty reply makes it fall back to its own Viser IK
gizmo and jolt the arm before any workflow has commanded anything. The
factory encodes this ordering; if the arm moves at session start, stop and
check the rr-session config.

### rr-session lifecycle

With `rr_autostart=True` (default), the connector spawns
`uv run --directory third_party/robots_realtime rr-session <config> --no-tui`
in its own process group. A missing checkout raises `FileNotFoundError`
telling you to run `git submodule update --init`. On `conn.close()` the whole
process group is SIGTERMed and then **always SIGKILLed** — rr-session forks
node subprocesses, and killing only the direct child would leave the realtime
robot loop running headless.

`rr_autostart=False` (or `gap run --no-rr-autostart`) restores the
two-terminal flow: run rr-session yourself and watch its logs directly.

### The msgpack bridge

GaP runs a TCP server (default `127.0.0.1:9000`); the robots_realtime client
connects to it. Each exchange is one framed observation in, the latest action
out. Frames are a 4-byte big-endian length plus a msgpack payload with numpy
support:

```text
observation:  {left: {joint_pos: float32[8]},            # 7 joints + gripper
               <camera>: {images, depth_data, intrinsics, pose / pose_mat}}
action:       {timestamp: float, left: {joint_pos: [float]*7, gripper: float}}
```

Control is **absolute joint targets**: a background republisher re-stamps the
current target at **50 Hz** and is the sole writer of the action slot. Camera
frames are rate-limited by the client, so the env retains the last frame
between updates. The end-effector pose is computed by URDF forward kinematics
(`panda_description`) with the Robotiq TCP offset (−0.157 m along panda_hand
Z, π/4 Z rotation).

### Heartbeat diagnostics

A heartbeat thread logs one line per second and tells you exactly which side
of the link stopped:

| Field | Healthy | Diagnosis when off |
|---|---|---|
| `republish_hz` | ~50 | Drops toward 0 → the GaP-side republisher died. Anything below ~30 Hz is a problem. |
| `obs_age` | small (ms) | Keeps growing → the realtime side hung or the client disconnected. |
| `joint_max_diff` | nonzero while moving | ≈0 with fresh observations while commanding → the robot is physically stuck. |
| `target_age` | small while commanding | Grows large → GaP stopped issuing new targets (`move_to_joints` not being called). |
| `cmd_vs_obs` | small | Persistent gap between commanded and observed joints → the arm is not tracking. |

`env.obs_stale` flips to `True` (and the heartbeat logs at warning level) when
no fresh wire observation has arrived for 2 s.

### wait_ready

`conn.wait_ready(timeout_s=60.0)` blocks until the first RGB observation
arrives. On timeout, the error names the silent wire stage — no client
connected, camera node not publishing, or RGB key missing — and includes the
rr-session exit code and log path. Do not work around a missing RGB stream by
skipping `wait_ready`.

## The UR + ZED connector

`ur_zed` captures RGB-D directly from the ZED via the **pyzed SDK** (manual
install from the ZED SDK distribution — it is not on PyPI; the import error
tells you where to get it) and reads UR joint state over the **read-only RTDE
receive interface** (`rtde_receive`, from `uv sync --extra real` /
`pip install 'graph-as-policy[real]'`).

Because there is no command channel, the connector is built with
`motion_enabled=False` and registers **only** the five observation getters:
`robot.get_observation`, `robot.get_camera_pose`, `robot.get_ee_pose`,
`robot.get_gripper` (always reports 1.0 — there is no gripper), and
`robot.get_gripper_pose`. Accidental motion is structurally impossible.

Defaults: `robot_ip="172.22.22.2"`, 720p at 15 fps, NEURAL depth mode, image
and depth rotated 180° for the upside-down wrist mount, depth NaN/inf mapped
to 0 ("no depth").

### Hand-eye calibration

The camera pose is computed as URDF forward kinematics to `wrist_3_link`
composed with the inverse hand-eye calibration — not from the UR controller's
TCP pose. Provide the calibration as a 4x4 camera-to-wrist `.npy` via the
`calibration_path=` kwarg or the `GAP_UR_ZED_CALIB` env var.

:::{warning}
If no calibration is found, the env logs a warning and **silently falls back
to an identity transform**: camera poses then equal the wrist pose, and every
world-frame point your graph computes is offset by the physical camera mount.
The run does not fail — check the log.
:::

The URDF used for FK resolves from `GAP_UR_URDF` (a path to a `.urdf` file),
falling back to `robot_descriptions`' `ur5e_description`.

## Environment variables

| Variable | Effect |
|---|---|
| `GAP_UR_ZED_CALIB` | Path to the 4x4 camera-to-wrist calibration `.npy` (ur_zed). Missing → identity fallback with a warning. |
| `GAP_UR_URDF` | UR URDF path for forward kinematics (ur_zed). Unset → `ur5e_description`. |
| `GAP_MAX_PERCEPTION_CALLS` / `GAP_MAX_PLANNING_CALLS` / `GAP_MAX_SIM_STEPS` | Call-count guard limits on tagged tools — see [Safety](safety.md#call-count-guards). |

The full list lives in
[Environment variables](../reference/environment-variables.md).

## Next steps

- [Safety](safety.md) — pre-session checklist, what GaP enforces, what it
  does not.
- [Real Franka Pick & Place](../examples/real-franka-pick-place.md) — the
  full-motion example on this connector.
- [Cable UR](../examples/cable-ur.md) — the perception-only example.
- [Connector tools](../reference/connector-tools.md) — the complete `robot.*`
  tool reference.
