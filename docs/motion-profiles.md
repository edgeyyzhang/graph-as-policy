> The task-0 example graph referenced below has been deleted. Its experiment commands are historical; the reusable profile API remains implemented. Official graph generation is pending.

# Additive motion profiles

A motion profile adjusts a node's exposed scalar inputs and its arm-controller
gains for one blocking invocation. Zero offsets retain nominal behavior.

```python
from gap.runtime.motion_profile import ControllerGainOffsets, MotionProfile
import gap

profiles = {
    "object0_transport": MotionProfile(
        controller=ControllerGainOffsets(kp_offset=10.0, kd_offset=2.0),
        input_offsets={"safe_height": 0.02},
    ),
}
result = gap.execute(graph, connector, motion_profiles=profiles, max_node_workers=1)
```

Use the graph's actual node names, or `subgraph.node` for subgraph members.
Unknown names are rejected. The profile input mapping is copied and immutable.
The executor resolves graph references and declared defaults before applying
input offsets. Nested scalar paths such as `pose.position.z` are supported;
this does not add support for nested `$ref` dictionaries in workflow JSON.
Unknown input paths and nonfinite adjustments fail rather than being ignored.

Controller offsets are added to each controller's own native gain vectors.
OSC and joint PD receive the same offsets, with their existing six/seven-entry
shapes. `kp` and `kd` are independent: changing one never recomputes the other.
Both controllers must use fixed impedance mode and resulting gains must remain
positive. Existing controller gains are restored on return or exception,
including internal servo-to-IK fallback. Profiles do not alter null-space
posture gains or gripper control.

LIBERO requires closed-loop joint motion for gain profiles. Configure it before
constructing the connector, for example `GAP_LIBERO_JOINT_MOTION_MODE=closed_loop`.
Profile-enabled execution requires one node worker and rejects streaming nodes
and routers/Send. Nonblocking arm commands and simulator resets inside an active
gain scope are rejected. Blocking joint-trajectory streaming is supported.
Backends without a gain-scope capability reject requested controller profiles.

## Automatic inputs and primitive hooks

The generic schema cannot identify an automatic-value sentinel or a physical
clearance constraint. A script may provide these optional, non-motion hooks:

```python
def resolve_motion_profile_inputs(ctx, inputs, paths):
    # Resolve automatic nominal values only for the requested paths.
    return inputs

def validate_motion_profile_inputs(ctx, inputs, paths):
    # Raise if effective values violate the primitive's contract.
    pass
```

The official `transporting-objects/scripts/waypoint_move.py` uses a shared helper
to resolve a nonpositive `safe_height` to workspace `transport_z` before applying
an offset. It rejects a corrected height that would re-enter automatic mode.
Its retry ladder stays fixed. Other automatic parameters need their own explicit
resolver; do not assume a numeric sentinel is a physical nominal value.

A corrected input does not mutate upstream graph outputs. When a later node
needs the same corrected target, explicitly wire the effective target through
outputs. Transport now provides `commanded_drop_x` / `commanded_drop_y` for this
purpose. These are commanded coordinates, not achieved end-effector positions.

## Learning integration and traces

`WorkflowExecutor` accepts either a static `motion_profiles` mapping or a
`motion_profile_resolver(full_id, node, resolved_inputs)` callback returning a
`MotionProfile` (or None) per visit. Supply `controller_gain_scope` from the
connector when constructing the executor directly.

GOPI's optimization schema version 2 treats every parameter as an additive
offset with default zero. `destination` is `input` or `controller`; controller
names are `kp_offset` and `kd_offset`. Version 1 keeps its existing absolute
input semantics. The actor already initializes its deterministic mean to zero
while retaining stochastic exploration.

Every profiled visit writes `motion_profile.json` containing nominal inputs,
selected offsets, and effective inputs under `node_data/<node>/`, with history
under `iters/`. `resolved_inputs.json` contains the effective arguments.

The separate `gopi/graphs/libero10_task0_profiles` experiment retains the original
pilot graph alongside it. It enables two gain offsets at each of 10 motion nodes,
and XYZ/height-related transport inputs (`drop_x`, `drop_y`, `safe_height`) at the
two transport nodes: 26 coordinates across the graph, two per ordinary motion
node and five per transport node. Other nodes have no tunable coordinates.
The lowering wrapper receives commanded XY through top-level `$ref` inputs;
its nominal drop Z and orientation are retained. Observation dimension is 337,
including corrected XY in graph memory. Bounds are initial experiment settings,
not a demonstrated stable range or an improvement guarantee.

From the inner `gopi/` directory:

```bash
../graph-as-policy/.venv/bin/python -m scripts.sac_graph.train \
  --graph graphs/libero10_task0_profiles \
  --gap-root ../graph-as-policy --skills-root ../open-robot-skills \
  --output outputs/sac_graph/profiles_run \
  --episodes 1 --eval-episodes 1 --train-inits 1 \
  --batch-size 1 --updates-per-episode 1 --hidden 8
```

The output directory must be new. Old checkpoints/replay are incompatible with
the new action and observation schema. Source manifests include untracked local
Python extensions as well as dependency diffs.
