# Checkpoints and Verification

A graph that *runs to its success end node* has only proven that control flow
worked. Checkpoints prove the **physics** worked: they are per-subgraph
postcondition predicates evaluated against privileged simulator ground truth
at each subgraph exit, independent of anything the graph's own perception
reported. The graph says "I closed the gripper"; the checkpoint asks the
simulator "is the can actually touching a finger link?"

Checkpoints are authored three ways — by hand, via
[`gap.builder`](../authoring/builder.md)'s `Subgraph.add_checkpoint(...)`, or
by the [generation pipeline](../authoring/generation.md)'s checkpoint agent —
and all three produce the same artifact: a sidecar Python module next to the
workflow.

## Sidecar modules

Checkpoints live outside `workflow.json`, in
`<workflow_dir>/checkpoints/<subgraph_name>.py`. Each sidecar must export a
module-level `CHECKPOINTS: list[Checkpoint]`. The executor loads the file
fresh on every evaluation (so you can edit predicates between runs without
restarting anything), and a *missing* sidecar is silently skipped — subgraphs
without checkpoints cost nothing.

A real example, the quickstart's grasp postcondition
([examples/libero_quickstart/graph/checkpoints/grasp_sg.py](gh-engine:examples/libero_quickstart/graph/checkpoints/grasp_sg.py)):

```python
from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    """The soup can is in contact with a robot link after `close`."""
    return world.body("alphabet soup").is_grasped()


def _target_held_diagnostics(world) -> dict:
    body = world.body("alphabet soup")
    held = world.held_body()
    return {
        "resolved_body": body.name,
        "contacts": sorted(body.contacts),
        "held_body": held.name if held is not None else None,
        "gripper_open_fraction": (
            world.robot().gripper_open_fraction
            if world.robot_view is not None else None
        ),
        "body_z": body.z,
    }


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "After close, the grasp target must be held: ground-truth "
            "contacts between the can and a robot finger link prove the "
            "gripper actually closed ON the object (not an empty grip)."
        ),
        validate=True,
        diagnostics_fn=_target_held_diagnostics,
    ),
]
```

`Checkpoint` is a frozen dataclass with these fields:

| Field | Default | Meaning |
|---|---|---|
| `name` | — | Identifier reported in results and failure messages |
| `subgraph` | — | Owning subgraph name |
| `predicate` | — | `(world) -> bool` or `(world, outputs) -> bool` |
| `rationale` | `""` | Why this postcondition matters (surfaced in feedback) |
| `validate` | `True` | `True` = hard postcondition, `False` = probe |
| `weight` | `1.0` | Relative importance for scoring/triage |
| `diagnostics_fn` | `None` | Optional `dict`-returning context collector, same arity rules |

### Predicate signatures

The evaluator inspects each callable's arity per call:

- **`predicate(world)`** — the classic privileged-only check.
- **`predicate(world, outputs)`** — `outputs` is the subgraph's bound
  outputs (the values declared via `Subgraph.set_outputs(...)`, resolved at
  exit). This lets a predicate compare what the *graph* computed (e.g. a
  perception OBB) against what the *simulator* knows is true.

Callables declared with bare `*args` are treated as one-argument.

A predicate that raises never crashes the run — it is recorded as
`passed=False` with the exception captured in `eval_error`. A
`diagnostics_fn` that raises is treated as a missing diagnostic
(`diagnostics=None`); the predicate's verdict stays authoritative.

### Hard checkpoints vs probes

`validate=True` checkpoints are **hard postconditions**: they are the ones
enforcement acts on. `validate=False` checkpoints are **probes** — they are
useful as extra signal during graph generation and triage, but the executor
never enforces them.

## Enforcement modes

Enforcement runs at each subgraph's **normal** exit — including declared
failure exits, but *not* the `on_error` path (if the subgraph's body raised,
you already know something went wrong). Every result is appended to
`ExecutionResult.checkpoint_results` (a list of `CheckpointResult` with
`name`, `subgraph`, `passed`, `eval_error`, `diagnostics`, `eval_time_s`).

| Mode | Behavior on a failed `validate=True` checkpoint |
|---|---|
| `off` | Sidecars are never loaded; nothing is evaluated |
| `warn` | Log a warning and continue |
| `raise` | Raise `gap.errors.VerificationFailed` naming the failed checkpoints |

You select the mode from the CLI or the Python facade (the raw
`WorkflowExecutor` takes the same `checkpoints=` argument):

```bash
gap run path/to/graph --sim libero_object_all_variance/0 --checkpoints raise
```

```python
result = gap.execute(graph, connector=conn, checkpoints="raise")
for cp in result.checkpoint_results:
    print(cp.subgraph, cp.name, cp.passed, cp.diagnostics)
```

:::{warning}
The default differs between API layers. `gap run` and the `gap.execute()`
facade default to `checkpoints="warn"`; the raw `WorkflowExecutor`
constructor defaults to `checkpoints="off"`. If you drive the executor
directly and expect checkpoint evaluation, pass `checkpoints=` explicitly.
An invalid value raises `ValueError` at construction.
:::

In `warn` mode the run continues past failures, so a "successful" run can
still carry failed checkpoints — always inspect
`ExecutionResult.checkpoint_results` (or the run log) rather than assuming
`success=True` implies every postcondition held.

## Ground truth is sim-only

Evaluation needs a privileged `World` snapshot, which the executor obtains
from the connector's `world_snapshot` method (wired as `world_snapshot_fn`).
Simulator connectors provide it; real-robot
[connectors](../real-robots/connectors.md) generally cannot — there is no
oracle for "is the can really in the basket" on hardware.

When `checkpoints != "off"` but no `world_snapshot_fn` is available,
enforcement **silently degrades**: the executor logs *one* warning for the
whole run and skips all checkpoint evaluation. The run proceeds unchanged,
and `checkpoint_results` stays empty. This is deliberate — the same graph
runs in sim (verified) and on a real robot (unverified) without edits — but
it means checkpoints are a rehearsal tool, not a hardware safety mechanism.
See [Safety](../real-robots/safety.md) for what gates real-robot runs.

## The predicate vocabulary

Predicates are written against three snapshot types from
`gap.runtime.verify`: `World`, `Body`, and `Robot`
([gap/runtime/verify/world.py](gh-engine:gap/runtime/verify/world.py)). The
harness constructs them; your code only reads them.

### Looking up bodies

`world.body(name)` resolves names through four stages: exact match →
snake-case slug (`"alphabet soup"` → `"alphabet_soup"`) → token-subset match
(a verbose phrase like `"small blue and white cream cheese"` resolves to
`"cream_cheese"`) → unique substring-token match (`"frying pan"` →
`"chefmate_8_frypan"`). The last stage only resolves when exactly one body
wins; ambiguity falls through. An unresolvable name raises
`BodyNotFoundError` listing the scene's canonical spec ids and body names —
useful feedback when an LLM-authored predicate guesses wrong.

Related helpers: `world.has_body(name)`, `world.body_names()`,
`world.region(name)` / `world.region_names()` for physics-free drop-target
regions.

### Body predicates

All geometry is world-frame; tolerances are meters and radians.

| Predicate | Defaults | Notes |
|---|---|---|
| `is_grasped()` | link prefixes `("robot", "panda_", "Robotiq", "finger_")` | True if any robot link is in the body's contact set |
| `is_grasped_by(prefix)` | — | Explicit robot prefix |
| `is_on(other)` | `tol_m=0.05` | **Lenient**: AABB overlap, see warning below |
| `is_on_strict(other)` | `tol_m=0.03, tol_xy_m=0.0, max_penetration_m=0.005` | Asymmetric z check + centroid containment |
| `is_in(container)` | `require_contact=True, tol_xy_m=0.02, tol_z_m=0.02` | Cavity AABB containment; contact check auto-skipped for regions |
| `contains(other)` | same tolerances | Inverse of `is_in`, no contact requirement |
| `is_above(other)` | `min_clearance_m=0.0, require_xy_overlap=True` | Clearance over the other body's top face |
| `is_settled()` | `speed_thresh=0.08` | Linear velocity norm below threshold |
| `is_axis_aligned()` | `local_axis="z", world_axis="z", tol_rad=0.20` | Orientation check |
| `distance_to(other)` / `xy_distance_to(other)` | — | Center-to-center meters |
| `bottom_footprint_xy()` / `xy_coverage_over(other)` | `z_slack_m=0.005` | Mesh-projected true footprint (needs trimesh/scipy/shapely; returns `None`/`0.0` without them) |

Plus scalar sugar: `body.x/.y/.z`, `body.top_z/.bottom_z`,
`body.left_x/.right_x`, `body.near_y/.far_y`.

:::{warning}
`is_on` uses AABB *overlap*, which is documented-lenient: an elongated
object like a frypan passes with only its handle hanging over the burner
while the body rests on the table 2–3 cm below. Use `is_on_strict` whenever
the predicate you mean is "the object body is actually resting on the
support" — it requires the bottom face within `-5 mm … +3 cm` of the
support's top *and* the centroid inside the support's XY extent.
:::

### World-level helpers

These avoid hard-coding body names, which keeps predicates robust when
perception's naming differs from ground truth:

- `world.held_body()` — the body currently grasped (closest to the
  end-effector when several qualify), or `None`.
- `world.bodies_displaced(min_xy_m=0.05)` — non-static bodies that moved
  since the rollout's initial snapshot, sorted by displacement.
- `world.body_over(region)` / `world.body_inside(region)` — which body sits
  over / inside a region's cavity.
- `world.moved_body_inside(region, min_xy_m=0.05)` — composition of the two
  above: "something that moved during the rollout ended up in the basket".

### Robot state

`world.robot()` returns a `Robot` view with `joint_pos`, `ee_position`,
`ee_quaternion_wxyz`, and `gripper_open_fraction` (0.0 closed → 1.0 open),
plus `gripper_is_closed(threshold=0.1)` and `gripper_is_open(threshold=0.9)`.

### Temporal operators

A `World` may carry a history of earlier snapshots (`world.history()`,
chronological). Three operators quantify over `history() + [self]`:

```python
# "the can was held at some point, and at exit it sits in the basket"
def placed(world) -> bool:
    return (
        world.eventually(lambda w: w.body("alphabet soup").is_grasped())
        and world.at_end(lambda w: w.body("alphabet soup").is_in(w.body("basket")))
    )
```

`eventually(fn)` (any snapshot), `always(fn)` (every snapshot), and
`at_end(fn)` (current snapshot only — provided for symmetry).

### Dry-running predicates with StubWorld

`StubWorld(body_names=[...], container_names=[...], region_names=[...])`
builds a minimal `World` with no simulator: every body is a 10 cm cube at
the origin, containers and regions get an 8 cm interior cavity, and a stub
7-joint robot is attached. Validators use it to catch predicates that
reference bodies that won't exist — a `BodyNotFoundError` at authoring time
instead of at rollout time. It is also handy in your own unit tests; see
[Testing bundles](../skills/testing-bundles.md) for the wider testing
toolkit.

## Authoring with the builder

`gap.builder` keeps checkpoints attached to the subgraph they verify:

```python
sg.add_checkpoint(
    "target_held",
    predicate=lambda w: w.held_body() is not None,
    rationale="grasp must physically hold something at exit",
    validate=True,
)
```

`Subgraph.add_checkpoint(name, predicate, *, diagnostics=None, rationale="",
validate=True, weight=1.0)` records the declaration;
`dump_checkpoints_module(path)` writes the sidecar. Note the builder kwarg is
`diagnostics`, while the dataclass field is `diagnostics_fn`. See
[the builder guide](../authoring/builder.md) for the full surface.

Generated graphs get their sidecars from a dedicated checkpoint agent that
runs after the per-subgraph agents — see
[Generating graphs](../authoring/generation.md).

## Diagnostics

`diagnostics_fn` runs after the predicate (pass or fail) and its dict lands
in `CheckpointResult.diagnostics` and the run log. Use it to capture the
*why* while the privileged snapshot is still in hand: contact sets, body
heights, `world.held_body()`, gripper fraction. Two `World` fields exist
specifically for this kind of post-mortem: `world.raw_contact_diagnostics`
(per-sensor contact force norms) and `world.sim_state` (a kitchen-sink dump
of joint states, body poses, action targets, and grasp metrics).

## Reading checkpoint failures

Not every `passed=False` is a task failure. Generated graphs often include
*perception-audit* checkpoints (e.g. `target_obb_matches_truth`) that compare
a perception OBB from the graph's outputs against the privileged body pose —
and the OBB may be expressed in the robot's base frame while the `World`
snapshot is world-frame. A constant-offset mismatch there is a bug in the
generated predicate, not in the rollout.

When triaging, trust the **physical** checkpoints first — `target_held`
(ground-truth contacts) and placement checks like `is_in` /
`moved_body_inside` are frame-free and directly measure task progress. In
the default `warn` mode the run continues either way, so a passed
`target_held` alongside a failed perception audit usually means the task
succeeded. See the [agent quickstart](../examples/agent-quickstart.md) for a
worked example of this pattern.

## Next steps

- [Executing graphs](execution.md) — where checkpoints sit in the execution lifecycle.
- [Traces](traces.md) — checkpoint results land in the run log alongside the recorded trial.
- [LIBERO quickstart](../examples/libero-quickstart.md) — a measured run with the `target_held` checkpoint in the loop.
