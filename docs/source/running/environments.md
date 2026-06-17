# Simulators and Environments

GaP's built-in simulation family is LIBERO tabletop manipulation: classic
benchmark suites, the vab variance/packing suites, and a perturbed moving-target
variant, all running in one process. This page covers the environment registry,
the suites, seed semantics, cameras and video, and the motion-control details
that matter when you debug a graph in sim.

:::{note} Requirements
The sim stack needs the `[libero]` extra (Linux + NVIDIA GPU + EGL) and
`MUJOCO_GL=egl` exported for headless rendering — see
[Installation](../getting-started/installation.md). `mujoco` is pinned to
exactly `3.6.0`: the acceptance numbers were measured on it, and MuJoCo minor
versions change contact dynamics enough to move grasp/drop success rates.
:::

## The environment registry

Env names resolve through [gap/envs/registry.py](gh-engine:gap/envs/registry.py).
Each entry maps a name to a *lazy* factory (a `"module.path:attr"` string), so
importing the registry never pulls mujoco — the sim stack loads only when a
factory is actually called.

```python
from gap.envs.registry import registered_envs

registered_envs()   # {name: canonical suite key}
```

Resolution: an exact name match wins; otherwise the longest registered
`prefix=True` entry matches, with the full name passed through as the suite key.
The `"libero"` prefix entry routes **any** `libero*` name to the LIBERO factory,
which then decides per suite between the vab task-dir format and the LIBERO-PRO
benchmark registry. Unknown names raise `KeyError` listing every registered name.

| Name | Kind |
|---|---|
| `libero` (prefix) | Routes any `libero*` name to the LIBERO factory |
| `libero_object`, `libero_spatial`, `libero_goal`, `libero_90`, `libero_10`, `libero_object_swap`, `libero_object_basket_swap` | Classic LIBERO-PRO suites |
| `libero_object_all_variance`, `libero_object_target_pos_var20x20`, `libero_object_target_permutation_variance`, `libero_object_target_basket_swap_variance`, `libero_object_packing`, `permutation_packing` | vab suites |
| `libero_grocery_packing_object` → `libero_object_packing`, `libero_grocery_packing_permutation` → `permutation_packing` | Aliases |
| `franka_real`, `ur_zed` | Real hardware — see [Connectors](../real-robots/connectors.md) |

Register your own env with the same factory contract:

```python
from gap.envs.registry import EnvConfig, register_env

register_env("my_sim", "my_pkg.envs:make_env")

# my_pkg/envs.py
def make_env(suite_name, task_id, camera_names, enable_render, **extra):
    ...
    return env, EnvConfig(arm_dof=7, action_mode="velocity_joints", ...)
```

`EnvConfig` carries the static robot metadata the connector needs: DOF, action
mode, control frequency, home joints, TCP offset, default cameras, and an
`is_real` flag that gates motion safety behavior.

:::{note}
`gap run --sim SUITE/TASK` always resolves through the `"libero"` entry — the
CLI reaches every LIBERO-family suite (classic and vab), but custom env families
are reachable only from Python via `gap.connector.sim("my_sim", ...)`.
:::

## LIBERO suites

Two vendored forks back the suites, and both ship a top-level Python package
named `libero`. The loader swaps `sys.modules` entries when it switches forks,
so suites from both load in the same process (live envs keep references to
their own modules and keep working):

- **Classic suites** (`libero_object`, `libero_spatial`, `libero_goal`,
  `libero_90`, `libero_10`, `libero_object_swap`, `libero_object_basket_swap`)
  load through the LIBERO-PRO benchmark registry — BDDL task files plus
  pruned-init states — vendored at `third_party/LIBERO-PRO` (override with
  `GAP_LIBERO_PRO_ROOT`).
- **vab suites** (the `*_variance` and packing suites) are self-contained YAML
  task directories under
  `third_party/Variational-Automation-Benchmark/tasks/<suite>/` (override with
  `GAP_VAB_ROOT`). Each YAML bakes the arena, objects, 50 init-state
  variations, and a success predicate.

Pick a task as `SUITE/TASK_ID`:

```bash
MUJOCO_GL=egl gap run my_graph --sim libero_object_all_variance/0
```

In Python, `task` also accepts a `(suite, id)` tuple, or a bare int when the
env name itself is a suite:

```python
import gap

conn = gap.connector.sim("libero", task="libero_object/0")
conn = gap.connector.sim("libero_object", task=3)
```

:::{warning} vab task numbering follows classic LIBERO order
The `libero_object*variance` suites are numbered by the classic
`libero_object` task map, **not** by alphabetical YAML order: task 0 is the
alphabet soup, task 1 is the cream cheese — not the alphabetically-second bbq
sauce (which is task 3). Sorting the YAML filenames would silently renumber 8
of the 10 tasks, so every config that addresses tasks by classic id (benchmark
YAMLs, generated-graph recipes) stays in sync. The packing suites use
lexicographic order, which is their intended numbering.
:::

## Seeds and initial states

Seeds index the suite's baked initial-state variations; the mapping is
`init_index = (seed - 1) % n_inits` on both the classic (pruned-init files) and
vab (YAML-baked inits) paths. Seeds 1..N therefore walk the init variations in
order and wrap around.

Three places take a seed:

- `gap.connector.sim(..., seed=N)` stashes the seed; the connector's **first**
  no-arg `reset()` consumes it (one reset, not two).
- The `sim.reset` graph tool takes `seed: int = 0`, where **0 means unseeded** —
  pass 1 or higher for a reproducible init.
- `SimConnector.reset(seed=...)` in Python, with `None` meaning unseeded.

`gap run --sim` has no seed flag; runs are unseeded unless the graph calls
`sim.reset` with a seed. Benchmark trials are seeded per cell by the harness —
see [Benchmarking](../benchmarks/benchmarking.md).

## Cameras, observations, and video

Default cameras are `agentview` and `robot0_eye_in_hand`
(`EnvConfig.default_cameras`); override with `gap.connector.sim(cameras=[...])`.
Frames render offscreen at 800×512. Per camera, each observation carries:

- `rgb` — uint8 image (extracted upright; the env handles MuJoCo's flip),
- `depth` — metric depth, with EGL's occasional out-of-range pixels clamped,
- `segmentation` — element-level instance segmentation,
- `pose` / `pose_mat` — the camera pose in the robot base frame,
- `intrinsics` — a 3×3 K matrix computed from the MuJoCo `fovy` at render size.

Observations also include joint state, the gripper pose in the base frame, a
policy-training-exact proprio vector, and ground-truth object poses
(`cube_poses`) used by [checkpoints](checkpoints.md).

Video capture buffers one frame every 4 sim steps and encodes at 20 fps
(libx264) — real-time playback for the 20 Hz sim. Enable it with
`gap run --record-video` (see [Executing Graphs](execution.md#traces-and-video))
or `gap.connector.sim(record_video=True)`; graphs can also drive it with the
`sim.enable_video` / `sim.save_video` tools. Saving writes the main mp4 plus a
`<stem>_<camera>.mp4` per camera. Videos land in the trace directory and are
watched from disk — the `gap viz` UI renders image assets, not mp4s.

MuJoCo needs a GL backend to render those frames headlessly; on NVIDIA machines
that is EGL, hence the `MUJOCO_GL=egl` prefix on every sim command in the docs.
Parallel benchmark workers set it automatically if unset.

## How graph motion is executed

The robosuite env is constructed with the **OSC_POSE** controller, because that
is the action space LIBERO policies emit: the `sim.apply_policy_action` tool
forwards 7-dim `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` actions verbatim (positive
gripper = close — the `pi05_libero` convention). A second **JOINT_POSITION**
controller is built at each reset and hot-swapped in whenever a graph node moves
the arm through joint-space targets (`robot.move_to_joints`,
`robot.execute_trajectory`, the IK-based `robot.go_to_pose`). See the
[connector tools reference](../reference/connector-tools.md).

Joint moves run in one of two modes, selected per process via
`GAP_LIBERO_JOINT_MOTION_MODE`:

- `teleport` (default) — write the target joint qpos directly, then settle up to
  10 sim steps (≈0.5 s of physics) so contact forces resolve. Fast; built for
  pure graph workflows stepping through planner waypoints.
- `closed_loop` — track the target under physics until the joint error drops
  below tolerance. Slower, but it leaves OSC's interpolator clean — required
  when a policy node interleaves OSC actions with joint moves in the same
  workflow.

An unknown value logs a warning and falls back to `teleport`.

:::{note} Payload-aware teleport fallback
Teleporting is only safe with an empty hand: once the gripper has closed on an
object (gripper fraction < 0.5), a qpos jump displaces the finger contacts and
the next physics step ejects the grasp. The env therefore silently switches to
closed-loop tracking while carrying a payload, and `robot.execute_trajectory`
always forces closed-loop for its duration.
:::

## Perturbed moving-basket env

For dynamic-target experiments, `FrankaLiberoPerturbedEnv` teleports the
`basket_1` body along a scripted trajectory on every sim tick (before rendering,
so cameras see the motion). Opt in per process with `GAP_LIBERO_PERTURBED=1`
(also accepts `true`/`yes`) or `make_env(..., perturbed=True)`:

```bash
GAP_LIBERO_PERTURBED=1 MUJOCO_GL=egl gap run my_graph --sim libero_object/0
```

The default motion is circular — radius 0.05 m, period 8 s — with a linear
alternative (`velocity` + `duration_s`). The trajectory is a module-level
constant (`_BASKET_MOTION` in
[gap/envs/libero_perturbed_env.py](gh-engine:gap/envs/libero_perturbed_env.py));
edit the constant to change the motion — there is deliberately no config
plumbing. The basket's rest pose is re-resolved after each reset, and its
velocity is zeroed on every teleport.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MUJOCO_GL` | — | Set to `egl` for headless GPU rendering |
| `GAP_LIBERO_JOINT_MOTION_MODE` | `teleport` | `teleport` or `closed_loop` joint motion |
| `GAP_LIBERO_PERTURBED` | off | `1`/`true`/`yes` selects the moving-basket env |
| `GAP_VAB_ROOT` | `third_party/Variational-Automation-Benchmark` | vab fork location |
| `GAP_LIBERO_PRO_ROOT` | `third_party/LIBERO-PRO` | LIBERO-PRO fork location |

The full surface lives in the
[environment-variables reference](../reference/environment-variables.md).

## Next steps

- [Executing Graphs](execution.md) — `gap run --sim` flags, tracing, video
- [Benchmarking](../benchmarks/benchmarking.md) — suites × seeds at scale
- [Connectors](../real-robots/connectors.md) — the `franka_real` and `ur_zed` envs
- [LIBERO quickstart](../examples/libero-quickstart.md) — a full graph on `libero_object_all_variance/0`
