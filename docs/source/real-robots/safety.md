# Safety

:::{warning}
GaP can drive real hardware (`gap.connector.real(...)`, `gap run --real`).
Graphs may be LLM-generated, and even hand-authored graphs execute
perception-derived motions. **Treat every run as capable of unexpected
motion.** Read this page before connecting a robot.
:::

## Before every session

- **Hardware E-stop in hand.** Keep the physical emergency stop within reach
  for the entire session, and know your robot's software stop (e.g. the
  Franka user-stop). Software-level guards in GaP are *not* a substitute —
  they run in the same process they are trying to stop.
- **Clear the workspace.** Remove people, pets, cables, and anything fragile
  from the arm's full reachable envelope — not just the tabletop. Perception
  errors put grasps in surprising places; an OBB fit to a mis-segmented
  background object is a commanded target like any other.
- **Verify the scene matches the task.** Object descriptions in the graph were
  written for a specific scene; stale graphs on rearranged tables move the arm
  through whatever is there now.
- **Dry-run first.** `gap run <graph> --validate-only` checks the graph
  without executing; running the same graph in sim first catches most logic
  errors for free.
- **Start with conservative guard limits** (see
  [Call-count guards](#call-count-guards)) so a runaway perceive/plan loop
  halts the workflow instead of hammering the robot.

## What GaP itself enforces

These mitigations ship in the connector and runtime — understand them, don't
rely on them:

- **`robot.go_home` is disabled on real robots.** With `config.is_real` set
  (both real envs set it), `go_home` logs a warning and returns without
  moving. A blind home move from an unknown pose, possibly while holding an
  object, is a collision generator — recovery sequences on real robots should
  open the gripper and stop, not re-home. Command explicit joint moves if you
  genuinely need to re-home, with eyes on the arm.
- **Hold-home pre-seeding (Franka).** `FrankaRealEnv` binds the msgpack server
  and pre-seeds the action slot with a hold-current-pose command *before* the
  realtime client connects. Without the seed, the client's first action
  request returns empty and it falls back to its own IK gizmo, jolting the arm
  before any workflow has commanded anything. The arm should not move at
  session start; if it does, stop and check the rr-session config.
- **Heartbeat / staleness detection.** The 50 Hz republisher and heartbeat
  threads detect a stale realtime link; observations stop flowing rather than
  silently going stale. See
  [heartbeat diagnostics](connectors.md#heartbeat-diagnostics) for the field
  semantics.
- **Perception-only connectors register no motion tools.** The `ur_zed`
  connector wires cameras and read-only joint state only — no
  `robot.go_to_pose`, no gripper, no trajectories — so accidental motion is
  structurally impossible on that path.
- **No `sim.*` tools on real connectors.** There is no reset, no scripted
  success check, and no privileged world state on hardware; checkpoint
  enforcement degrades to a logged skip.
- **Call-count guards.** Perception-, planning-, and sim-step-tagged tool
  calls are counted and limited (next section). `GuardLimitExceeded`
  subclasses `BaseException`, so a bare `except Exception:` in skill code
  cannot accidentally silence a safety limit.
- **Recovery actions are best-effort and explicit.** Failure end nodes run
  only the tool calls the graph declares (typically `robot.open_gripper`);
  nothing implicit moves the arm after an abort.

## Call-count guards

Tools are classified by their registry tags (`perception`, `planning`,
`sim_step`), and every tagged call is counted. Limits resolve in priority
order: per-workflow `gap.tools.guards.set_limits(...)` overrides, then the
environment variables, else unlimited.

| Environment variable | Caps |
|---|---|
| `GAP_MAX_PERCEPTION_CALLS` | Calls to perception-tagged tools (detectors, segmenters, VLM queries). |
| `GAP_MAX_PLANNING_CALLS` | Calls to planning-tagged tools (IK, motion planning). |
| `GAP_MAX_SIM_STEPS` | Calls to step-tagged control tools (`robot.go_to_pose`, `robot.move_to_joints`, gripper and trajectory tools). |

```bash
GAP_MAX_PERCEPTION_CALLS=30 GAP_MAX_PLANNING_CALLS=10 GAP_MAX_SIM_STEPS=2000 \
uv run gap run my_graph --real franka
```

Or programmatically, per workflow:

```python
from gap.tools import guards

guards.set_limits(perception=30, planning=10, sim_step=2000)
```

When a limit trips, the run halts with `GuardLimitExceeded`. The codegen and
benchmark pipelines (`gap generate`, `gap benchmark`) apply limits from the
task config's `safety_limits:` block on every trial (defaults:
`max_perception_calls: 50`, `max_planning_calls: 20`,
`max_sim_steps: 5000`); standalone `gap run` / `gap.execute` rely on the
environment variables — **unset means unlimited**, so set them explicitly for
hardware sessions.

## What GaP does NOT do

- No force/torque limiting, no speed scaling, no collision detection on real
  hardware — those belong to your robot's controller configuration (e.g.
  Franka's collision thresholds) and to the motion-planning bundle you choose.
  Direct-IK motions (`robot.go_to_pose`) solve IK for the target and drive
  the joints straight to the solution with **no collision awareness**.
- No workspace fencing. If your controller supports virtual walls, configure
  them there.
- No human detection. Keep the cell clear.

## Franka + robots_realtime specifics

- The realtime loops run in the vendored `robots_realtime` subprocess
  (`rr-session`), spawned in a process group and killed on connector close. If
  GaP crashes hard, verify the rr-session process is gone before re-launching
  — two clients fighting over one arm is undefined.
- `rr_autostart=False` (or `gap run --no-rr-autostart`) restores the
  two-terminal debug flow so you can watch the realtime log directly.
- First-observation timeout surfaces camera diagnostics; do not work around a
  missing RGB stream by skipping `wait_ready`.

## Reporting

Found a safety-relevant bug (a guard that can be bypassed, a motion that
ignores `is_real`, …)? Open an issue with the `safety` label — these are
prioritized over everything else.

## Next steps

- [Real-Robot Connectors](connectors.md) — the `franka` and `ur_zed`
  connectors in detail.
- [Real Franka Pick & Place](../examples/real-franka-pick-place.md) — a
  full-motion example with its safety notice.
- [Cable UR](../examples/cable-ur.md) — the perception-only path.
