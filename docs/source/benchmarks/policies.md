# Learned Policies

gap does not replace learned policies — it gives them structure. A vision-language-action (VLA) model
is one **node** in a graph, not the whole program: the graph perceives, pre-positions the arm, hands
control to the policy for the dexterous segment, decides when the policy is done, and verifies the
outcome. This page covers the full surface: a learned policy as a first-class **skill**, the two
shipped policy skills ([pi05-libero](gh-skills:skills/pi05-libero) and
[molmoact-libero](gh-skills:skills/molmoact-libero)), how a policy node auto-boots its server,
when to register a `policies:` override, the steered-policy pattern, what happens inside the loop,
and collecting data to train your own policy.

:::{note} Requirements
Serving a VLA needs a GPU and an [openpi](https://github.com/Physical-Intelligence/openpi) (or
MolmoAct) checkout pointed to by `GAP_OPENPI_DIR`. The gap-side websocket client is thin (no JAX):
install it with the `policy` extra in the engine repo (`pip install "graph-as-policy[policy]"`) or
the matching policy-skill extra in the skills repo (`uv sync --extra pi05-libero` and/or
`--extra molmoact-libero`, both of which pull `openpi-client`; `--extra all` includes both).
:::

## A learned policy is a skill

A VLA is not a generic "policy runner" parameterized by an opaque id. Each model **checkpoint** is
its own skill bundle, and the coordinator picks between them the same way it picks any other skill:
by reading the capability-rich `SKILL.md` description. The two shipped policy skills are:

| Skill | Checkpoint | What it is for |
|---|---|---|
| [pi05-libero](gh-skills:skills/pi05-libero) | openpi π0.5 LIBERO (`pi05_libero`) | The LIBERO Franka pick-and-place distribution |
| [molmoact-libero](gh-skills:skills/molmoact-libero) | MolmoAct LIBERO (`allenai/MolmoAct-7B-D-LIBERO-0812`) | Same task family — the MolmoAct alternative to `pi05-libero` |

Both wrap the same closed-loop body and the same load-bearing LIBERO observation encoding; they
differ only in the checkpoint they serve. Their `SKILL.md` descriptions state the task family
(LIBERO Franka pick-and-place), the action space (robosuite `OSC_POSE` deltas), and — explicitly —
what they are **not** for: deformables / cloth folding, articulated objects, non-Franka embodiments,
or anything outside the LIBERO pick-place distribution. That "not for" list is what lets the
coordinator decline to delegate an out-of-envelope task to a learned policy and instead report a
missing capability.

This replaces the old "emit a generic `running-policies` node with a `policy_id`" handoff. There is
no `policy_id` input anymore — the skill *is* the model.

## Where a VLA fits in a graph

A policy stage is a regular tool node that calls a policy skill's `.run` tool — e.g.
`pi05-libero.run` (or `molmoact-libero.run`). The skill owns its model, so the node carries **no
`policy_id`**. Everything around it stays a typed graph:

- **Before** the policy runs, perception and Cartesian moves put the end-effector somewhere close
  to the policy's training distribution (see [the steered-policy pattern](#the-steered-policy-pattern)).
- **During** the run, the loop reads frames from the graph-scoped observation stream — the same
  10 Hz stream the rest of the graph sees (see [Execution](../running/execution.md)).
- **After** the policy terminates, the graph decides what happens next: loop back to re-perception,
  hand off to a geometric place skill, or end the workflow.

This is also how the benchmark harness ablates structure: the `llm_plus_policy` mode materializes
the steered-policy template per task, while `policy_only` runs the bare VLA with no
perception/approach scaffolding as its baseline — isolating exactly what the graph buys over the
raw policy on perturbed layouts. See [Benchmarking](benchmarking.md).

## Serving: the skill owns its preset

Each policy skill **owns a serving preset whose name equals the skill name** (`pi05-libero`,
`molmoact-libero`). You do not normally wire this up: when a workflow references a policy skill, the
launcher scans for it and **auto-boots the matching preset server** from
`gap.runtime.policy_presets.PRESETS` — no `policies:` config block is needed for the common case
(see [policy_boot.py](gh-engine:gap/runtime/policy_boot.py)). The presets:

| Skill / preset | Checkpoint | Serve recipe |
|---|---|---|
| `pi05-libero` | `s3://openpi-assets/checkpoints/pi05_libero` | openpi's `scripts/serve_policy.py` |
| `molmoact-libero` | `hf://allenai/MolmoAct-7B-D-LIBERO-0812` | a vLLM-style serve script speaking the openpi websocket protocol |

Both presets' start commands begin with `cd $GAP_OPENPI_DIR` — set it to your openpi (or MolmoAct)
checkout; the shell expands it at spawn time. The server runs inside that checkout with its own
GPU dependencies; gap only connects as a websocket client.

To run a server by hand (so several workflows or workers can share one endpoint), the one-command
path spawns the skill's preset directly:

```bash
gap policy serve pi05-libero --port 9100      # or: gap policy serve molmoact-libero
```

`gap policy list` prints the available presets (the two shipped skills). Two flags matter:

- `--port N` — **the default port is OS-allocated (random)**. Pass `--port` whenever anything else
  needs a stable endpoint, e.g. a `policies:` override's `url:` entry.
- `--startup-timeout SECS` — default **900**. The first run downloads the checkpoint through the
  serve script itself, which legitimately takes minutes; the command waits for the TCP port to open.

The command blocks until Ctrl-C, then tears the server down.

## Overriding the serving recipe in config

The common case needs no config at all — referencing the skill auto-boots its preset. You only add a
`policies:` block when you want to **override** how a skill is served: point it at an external server
you already run, or hand it a custom `start_cmd`. The override key **must equal the skill name**;
that entry then wins over the auto-resolved preset. The block is consumed by the engine's
`PolicyManager` ([gap/runtime/policy_manager.py](gh-engine:gap/runtime/policy_manager.py)). Two
override styles, plus the explicit preset form:

```yaml
policies:
  pi05-libero:
    url: ws://127.0.0.1:9100        # external: you run the server; gap only records the URL
  molmoact-libero:
    start_cmd: "python serve.py --port {port}"   # managed: gap spawns and tears down
    env:
      CUDA_VISIBLE_DEVICES: "1"

policy_manager:
  startup_timeout_s: 900            # PolicyManager default is 120 s; raise it for first-run downloads
```

(A `preset: pi05-libero` entry is also accepted, but is only needed to attach `env:` overrides — a
bare skill reference already resolves to its shipped preset.)

Rules (violations raise `PolicyConfigError`):

- A **managed** entry's `start_cmd` must contain a `{port}` placeholder; the manager substitutes an
  OS-allocated free port, spawns the command through the shell in its own process group, waits for
  TCP readiness, and sends SIGTERM (then SIGKILL) at shutdown. Optional `env:` entries overlay the
  spawn environment.
- A **preset** entry may carry its own `env:` mapping, which overrides the preset's key by key.
- Specifying both `url` and `start_cmd` (or `preset` plus either) is an error, as is an entry with
  neither and an unknown preset name.
- The launcher boots **all** policy servers the graph requires up front (auto-resolved presets plus
  any `policies:` overrides); if any fails to come up, every subprocess it started is torn down and
  the error propagates. There is no eviction or runtime registration.

The benchmark harness preflights `url:` override entries but never owns those servers — start
`gap policy serve` yourself before launching policy-mode benchmarks; auto-resolved presets and
managed/`start_cmd` entries are spawned and torn down by the benchmark workers themselves
(see [Benchmark config](../reference/benchmark-config.md)).

## The policy `.run` tool

A policy skill's `.run` tool (`pi05-libero.run`, `molmoact-libero.run`) is the entry point for a VLA
stage. It is a class-based stateful skill: the websocket connection is cached per preset on the
executor and reused across invocations within one workflow, so a clean-all-items loop does not
reconnect on every iteration. The node names the skill and carries **no `policy_id`**:

```json
{
  "type": "tool",
  "tool": "pi05-libero.run",
  "inputs": {
    "observation_stream": {"$ref": "in.observation_stream"},
    "prompt": "pick up the object and place it in the basket",
    "gripper_cycle_termination": true,
    "max_windows": 70
  }
}
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `observation_stream` | required | The graph-scoped observation stream (`{"$ref": "in.observation_stream"}`). The loop reads `.latest()` once per window instead of calling `robot.get_observation`. |
| `prompt` | required | Task instruction passed verbatim to the policy. |
| `termination_prompt` | `""` | Optional VLM yes/no question; empty skips all VLM calls. |
| `gripper_cycle_termination` | `false` | End the stage after one commanded open→close→open gripper cycle. |
| `max_windows` | `20` | Hard ceiling on closed-loop iterations. |
| `replan_every` | `5` | Action rows consumed per window before replanning (the π-series LIBERO cadence). |
| `term_period` | `2` | VLM termination check cadence, in windows. |
| `arm_id` | `0` | Index into the observation's arm states. |
| `vlm_camera` | `0` | Camera index for VLM termination screenshots. |
| `settle_steps` | `10` | Zero-EE-delta / gripper-open dummy actions sent before the first inference so freshly spawned objects settle. |

Output: `{status, num_windows, num_steps}` where `status` is `completed_by_vlm`, `gripper_cycle`,
or `max_windows`, and `num_steps` counts the action rows applied across all windows.

### Exit conditions

Each policy skill declares its exit conditions in `SKILL.md`. The three success statuses above are
the subgraph's success exits (`gripper_cycle`, `completed_by_vlm`, `max_windows`); the single
failure exit is `failed`, taken when the loop raises (server / inference / execution error). It is
the subgraph's `on_error` exit.

**An exit value never means the task succeeded.** `gripper_cycle` says the gripper opened, closed,
and reopened — not that the right object ended up in the right place. Whether the *task* actually
succeeded is verified by a **postcondition checkpoint** that checks the world (e.g. the object is in
the container), never by an exit status. Each policy skill ships its own canonical capability
checkpoint for exactly this.

### Termination

Three additive terminators; the loop exits with the corresponding success status on whichever fires
first:

- **`max_windows`** — the hard backstop; always on.
- **VLM termination** — when `termination_prompt` is non-empty, every `term_period` windows the
  loop calls `vlm.query_yes_no` on the `vlm_camera` frame and exits with `completed_by_vlm` on yes.
- **Gripper-cycle termination** — when enabled, the loop watches the commanded gripper column the
  VLA emits each window (LIBERO convention: −1 open, +1 closed, with a ±0.5 hysteresis dead-band)
  and exits with `gripper_cycle` once a full open→close→open cycle completes. The close must hold
  for at least 3 windows, so a momentary failed grasp does not fire. This is the per-item
  terminator for multi-item loops: it needs no VLM calls, and it hands control back to the graph
  exactly when one item has been picked and released.

The loop also checks the executor's cancel token every window, so a policy stage inside a parallel
branch can be cooperatively cancelled.

## The steered-policy pattern

A VLA performs best near its training distribution. The steered-policy pattern uses geometric
perception to *put it there* before handing over control — robust object localization on perturbed
layouts, then closed-loop dexterity from a familiar starting pose:

1. **Perceive** — an OBB-producing perception subgraph localizes the next target. A clean
   `not_found` exit (the VLM answers "none") ends a clean-all-items loop with success.
2. **Hover** — a Cartesian approach translates the end-effector above the target OBB
   (`approach_z = max(obb_top + 0.12, 0.30)` in the example), **preserving the current
   end-effector rotation**: forcing a top-down quaternion pushes the π0.5 proprio state
   out-of-distribution. The move uses a tight tolerance (`0.003`, `max_steps=400`) because
   `robot.go_to_pose`'s default convergence (0.01 rad, 120 steps) stops early and leaves the
   rotation a few degrees off, drifting the hand-off proprio.
3. **Hand off** — the policy skill's `.run` tool (e.g. `pi05-libero.run`) takes over for the
   pick-and-place, terminating on the gripper cycle, then the graph loops back to re-perception.

The example graphs ship with a `{{policy_id}}.run` tool placeholder so one graph works against any
policy skill: the `{{policy_id}}` token is substituted with a policy-**skill** name (e.g.
`pi05-libero` or `molmoact-libero`) to form the concrete `.run` tool. The benchmark harness
materializes it per cell; for standalone `gap run` you must substitute it first (template the
workflow, or sed the placeholder). The full walkthrough, including a VLA-grasp + geometric-place
variant, is [Steered policy](../examples/steered-policy.md); the graphs live at
[examples/steered_policy](gh-engine:examples/steered_policy).

## Inside the loop

The closed-loop body (`run_policy_loop` in [gap/runtime/policy.py](gh-engine:gap/runtime/policy.py))
speaks the openpi websocket protocol: each window it encodes the latest observation into the openpi
obs dict, calls `client.infer`, and forwards the first `replan_every` rows of the returned action
chunk to the env via `sim.apply_policy_action` — untranslated, in the checkpoint's native action
space. Connectors without a VLA passthrough do not register `sim.apply_policy_action`, so the call
fails loudly rather than driving the robot with the wrong action space. For joint-space checkpoints
on trajectory-capable robots, the `chunk_to_trajectory` helper packs an `(H, D)` chunk into a
`gap.types.Trajectory` for `robot.execute_trajectory` instead. The LIBERO observation encoding is
**load-bearing**: images are W-flipped (composing with the connector's H-flip into the 180°
rotation the checkpoints trained on), center-cropped to a square, and `resize_with_pad`-ed to
224×224 uint8, and the 8-dim state is `[eef_pos(3), axis-angle(3), gripper_qpos(2)]` — not joint
angles. Skip any of this and the policy "looks lost" even with the action space wired correctly.

## Collecting data and training your own policy

The loop closes in the other direction too: run a gap graph as a scripted expert, record
demonstrations, train a policy externally, and bring it back into gap — either as its own
policy-skill bundle (the path the two shipped skills take; see
[Authoring bundles](../skills/authoring-bundles.md)) or, for a one-off, as a managed `policies:`
entry.

`gap.connector.collector.DataCollector` hooks the connector's step callback and records one
synchronized row per control step into a flat, append-only HDF5 file:

```text
/observations/<camera>_rgb   uint8   [N, H, W, 3]
/observations/state          float32 [N, dof+1]   (arm joints + gripper)
/actions                     float32 [N, A]
/rewards                     float32 [N]
/dones                       bool    [N]
/episode_ends                int64   [E]
/episode_success             bool    [E]
```

The layout maps one-to-one onto a LeRobot dataset (`observation.images.<camera>`,
`observation.state`, `action`), so you can convert and train with LeRobot recipes or fine-tune an
openpi checkpoint. Filter to `success=True` episodes during conversion.

```python
import gap
from gap.connector.collector import DataCollector

with gap.connector.sim("libero", task="libero_object_all_variance/0") as conn:
    collector = DataCollector(conn, "demos.hdf5")
    try:
        for episode in range(50):
            conn.reset(seed=episode + 1)
            collector.start_episode()
            gap.execute("examples/libero_quickstart/graph", conn, checkpoints="warn")
            success, reward = conn.check_success()
            collector.end_episode(success=success)
    finally:
        collector.close()
```

:::{warning}
`close()` is what writes the episode index tables (`/episode_ends`, `/episode_success`) and closes
the file — always call it (or use the collector as a context manager), or the dataset is left
without episode boundaries. An episode still open at `close()` time is ended and marked as a
failure.
:::

Once trained, serve your checkpoint behind any websocket server that speaks the openpi protocol. To
make it a first-class skill the coordinator can pick, package it as a policy-skill bundle (subclass
`gap.runtime.policy_skill.PolicyLoopSkill`, set `preset` to the bundle name, and add a preset
recipe). For a quick one-off you can instead reference it from a graph by a placeholder name and
supply a managed `policies:` entry keyed by that same name:

```yaml
policies:
  my_policy:
    start_cmd: "python my_serve.py --checkpoint /path/to/ckpt --port {port}"
```

The end-to-end walkthrough is [Collect and train](../examples/collect-and-train.md).

## See also

- [Steered policy example](../examples/steered-policy.md) — the hybrid graphs in detail
- [Collect and train example](../examples/collect-and-train.md) — demonstrations → HDF5 → training
- [Benchmarking](benchmarking.md) — running `llm_plus_policy` / `policy_only` modes at scale
- [CLI reference](../reference/cli.md) — `gap policy serve` / `gap policy list`
- [Environment variables](../reference/environment-variables.md) — `GAP_OPENPI_DIR` and friends
