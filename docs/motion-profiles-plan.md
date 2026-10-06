# Node motion profiles: implementation plan

Status: core implementation completed. See [motion-profiles.md](motion-profiles.md)
for the implemented API, restrictions, and runnable experiment. This document
retains the design rationale; graph-wide discovery reports and orientation
corrections remain future extensions.

## Agreed behavior

Each motion-node invocation receives one profile containing two independent
gain offsets (`kp_offset`, `kd_offset`) and enabled geometric offsets.
All optimized scalar parameters use additive adjustments:
effective = nominal + offset, with identity 0. Physical domains are enforced
through offset bounds and validation of the resulting values. Identity
initialization is recommended, not required for learning.

Both OSC and joint-position control use the same offset pair, added to their
own unmodified baseline vectors. Adjusting `kp` does not recompute `kd`. The node
owns the learned adjustments; the environment applies controller fields.
Existing nominal values still come from controller configuration, resolved
workflow inputs, upstream geometry, or primitive defaults.

A missing profile or an identity profile preserves native behavior. Previous
gains are restored when the node returns or raises. Primitive function
signatures stay unchanged for existing exposed inputs; internal constants
are excluded from this implementation. The scope is exposed motion-choice
inputs plus the two explicitly added controller-gain offsets. Do not expose
additional internal constants, retry logic, or planner settings for learning.

## 1. Profile types and public execution interface

Add `gap/runtime/motion_profile.py` with immutable profile types:

```python
ControllerGainOffsets(kp_offset: float = 0.0, kd_offset: float = 0.0)
MotionProfile(
    controller: ControllerGainOffsets | None,
    input_offsets: Mapping[str, float],
)
```

Validate finite offsets and resulting positive gain entries for both
controllers. Enforce declared physical lower/upper bounds on effective
quantities, including clearance distances. Copy the input
mapping so callers cannot mutate an active profile. Input names are exact
primitive arguments or validated nested scalar paths (for example
`safe_height` or `pose.position.z`); a generic `clearance` field must not
silently map to unrelated geometry arguments. Resolve an omitted argument's
declared default before applying an adjustment. Require an explicit semantic
binding for sentinel values such as a height of -1 meaning automatic height;
do not numerically adjust a sentinel and treat it as a physical quantity.
Resolve automatic values before applying the adjustment; where resolution
occurs inside the primitive, an explicit binding there is required. For
`waypoint_move.safe_height`, factor the existing automatic-height resolution
into a small shared helper used by the primitive and profile preparation:
nonpositive height selects `robot.describe_workspace()["transport_z"]`.
Apply the offset to that resolved height and check its domain before dispatch.
Register this resolver only for the corresponding primitive; do not guess
automatic semantics from parameter names. A profiled unresolved sentinel with
no registered resolver is an explicit unsupported binding, not a silent edit.

Extend `gap/runtime/execute.py::execute` with an optional `motion_profiles`
mapping keyed by fully qualified node ID. Example of the proposed API:

```python
profiles = {
    "object0_transport": MotionProfile(
        controller=ControllerGainOffsets(kp_offset=0.0, kd_offset=0.0),
        input_offsets={"safe_height": 0.0},
    ),
}
gap.execute(graph, connector, motion_profiles=profiles, max_node_workers=1)
```

This identity profile preserves the nominal transport height and each
controller's own gains. For a subgraph node, use the executor's existing
`subgraph.node` identity. Reject unknown profile targets before execution.

Keep the workflow JSON schema unchanged in this first implementation. The
mapping can be saved/loaded by the caller; GaP will not implicitly load GOPI's
RL-specific `optimization.json`.

## 2. Scope profiles around the actual node body

Extend `gap/runtime/executor.py::WorkflowExecutor` with optional dependencies:

- A profile resolver called once per node invocation, receiving node identity,
  definition, and resolved inputs. Static execution uses a mapping-backed
  resolver; GOPI supplies a callback that selects the RL action.
- A controller-gain scope factory supplied by the connector. The generic
  executor does not import or reach into a LIBERO environment.

In `_dispatch_node`, after resolving `$ref` inputs:

1. Resolve the invocation's profile.
2. Copy nominal primitive inputs and apply each declared additive offset exactly once.
   Validate bindings against the actual
   script signature or tool schema, including nested paths and defaults.
   This validation is necessary because
   `execute_script_node` currently warns and drops unknown script arguments.
3. Record the effective profile and resolved inputs in the trace.
4. Enter the gain scope, run `execute_tool_node` or `execute_script_node`, and
   exit the scope before returning or routing an exception.

All calls made by the script remain inside this scope, including a Cartesian
servo attempt followed by joint-control recovery and trajectory execution.
Do not add gain setter/resetter nodes to the graph or duplicate gain handling
in the five primitive implementations.

Initial scope: serial, blocking tool/script nodes. Profile-enabled execution
requires `max_node_workers=1`, rejects streaming nodes and Send fan-out, and
rejects nonblocking arm commands while a gain scope is active. These checks
prevent a profile from being restored before its motion ends or overwritten
by another node. Nested subgraphs with ordinary blocking nodes are supported.
Ordinary joint-trajectory streaming is a blocking primitive and is supported;
it is distinct from executor nodes declared `streaming: true`.

## 3. Apply independent gain offsets in the existing LIBERO environment

Add a context-managed gain scope to `gap/envs/libero_env.py::FrankaLiberoEnv`.
Expose it through a thin capability method on `gap/connector/core.py::Connector`;
unsupported backends fail explicitly when controller gains are requested.
Wire that method into the executor in `gap/runtime/execute.py`.

On entry, before mutating either controller:

- Validate both controllers use fixed impedance mode, so their `set_goal`
  methods will not recalculate the gains.
- Validate closed-loop joint motion is configured; reject teleport mode.
- Copy the current `kp` and `kd` vectors from both controller objects.

Capture immutable native baseline vectors in `_rebuild_controllers` whenever
controllers are constructed. Add offsets to these baselines, not gains left by a
previous or enclosing profile. Entry snapshots are only for restoration.

Then assign new vectors to both objects:

```python
ctrl.kp = baseline_kp + offsets.kp_offset
ctrl.kd = baseline_kd + offsets.kd_offset
```

Use `try/finally` to restore both snapshots, including partial-entry failures.
Use a scope stack to restore the preceding values correctly for nested scopes.
Controller switches through `_use_controller` preserve the applied offsets.

Environment resets are allowed between nodes and rebuild controller objects
normally. Reject resets inside an active profile scope in version one, before
resetting the simulator, to avoid restoring snapshots into obsolete objects.
Episode boundaries must have no outstanding gain scope.

No edits are required to robosuite's `osc.py`, `joint_pos.py`,
`single_arm.py`, or `LiberoHandle.step`. Their existing fixed-impedance code
already consumes `self.kp` and `self.kd`. Gripper commands and OSC null-space
posture gains remain outside this two-scalar parameterization.

## 4. Connect the GOPI parameter registry and learner

The native GaP profile API is usable without RL. The existing GOPI integration
will use it through these additional changes outside the GaP repository:

- `gopi/relax/env/graph_spec.py`: extend parameter declarations with a
  destination (`input` by default for backward compatibility, or `controller`).
  Use additive-offset semantics for each binding in the new schema. Validate
  duplicates by destination/target, and restrict controller fields to
  `kp_offset` and `kd_offset`. Decode into one MotionProfile, with no controller
  fields leaking into primitive kwargs. Keep legacy absolute input parameters
  on their existing code path; version the new schema and fingerprint.
- Graph `optimization.json`: register destination, target, units,
  offset bounds, and identity default (0) for
  every new optimization variable. Preserve the old pilot configuration
  separately when introducing the new experiment.
- `gopi/scripts/sac_graph/runtime.py`: use the native executor profile resolver
  and connector scope. Retain observation, transition, and reward handling;
  remove duplicate input-override execution logic. Select and record exactly
  one action per visit, including repeat visits.

The existing task-0 graph has 10 motion nodes, hence 20 gain coordinates across
the graph. Geometry adds only the explicitly enabled dimensions. Two transport
`safe_height` fields would produce 22 total coordinates for that experiment.
This does not imply a 22-dimensional action at every node: each node selects
only its own gains and enabled geometric fields. Non-motion nodes have no
new gain parameters. Use a new experiment/checkpoint schema for changed action
dimensions; do not reuse incompatible replay or checkpoints.

Adjustment bounds will be explicit experiment settings, chosen after simulator
smoke tests. Gain-offset bounds must keep both controllers' resulting gains
positive. Geometry must respect each field's physical domain. Ranges need not be equal across fields.

Recommend initializing the actor's deterministic output to normalized zero,
which the decoder maps to zero offsets. A known useful adjustment
or a saved policy is also a valid starting point. Initialize the actor's mean
output head appropriately, not every neural-network weight to zero. Retain
nonzero exploration variance; stochastic SAC samples need not be identity
even when the deterministic output is. Run a separate deterministic
identity-profile evaluation to verify the unchanged nominal baseline.

Geometric ownership must be explicit: in the current task-0 graph,
`safe_height` belongs to the transport node's inputs, whereas `drop_clearance`
is an input of the upstream `object*_drop` geometry node. The latter is not
automatically a motion-node parameter. For motion-only optimization, use an
explicit correction to an already exposed motion target. Keep geometry-node
parameters and hardcoded constants fixed in this implementation. If several
nodes need a common adjusted target, pass it through
graph outputs/references; do not independently recreate inconsistent targets.
Orientations require a specified-frame rotation composition for future angle
offsets, not componentwise quaternion addition or multiplication.

## Parameter discovery and selection

Existing discovery surfaces:

- `gap-core/src/gap_core/tools/schema.py::extract_schema` extracts declared
  input names, types, required flags and defaults from `run(ctx, ...)` modules.
- Registered connector/tool inputs are available from
  `connector.tool_registry.get(tool_name).schema.inputs`.
- `gap/runtime/validate.py::build_tool_node_schema` and
  `build_script_node_schema` already handle both node types for validation.

These describe interfaces, not a complete catalog of tunable parameters.
They do not discover internal constants, controller gain fields, physical
domains, automatic-value semantics, or whether a parameter is actually used
in a given execution branch. Add an explicit optimization binding registry
on top of this discovery, with source/provenance, domain, units,
bounds, default, consumer, activation conditions, and motion-node ownership.
Expose one graph-specific report joining discovered inputs and these bindings,
including fields excluded from learning and their exclusion reason. This report
is proposed functionality, not an existing GaP command.

Initial selection:

| Quantity | Treatment |
| --- | --- |
| Controller `kp`, `kd` | Learn additive offsets at every motion node, with positive effective gains. |
| Carry/hover target world Z, placement XYZ | Bounded additive corrections, when bound to the final consumed target. World coordinates are not inherently positive. |
| Grasp/lowering depth | Learn a bounded correction to the exposed target position along the approach axis; do not modify internal descent constants. |
| Approach/lift/placement clearance distances | Include only when already an exposed, consumed motion-node input; upstream geometry settings and hardcoded clearances stay fixed. |
| Task-relevant orientation correction | Optional bounded rotation-vector/yaw correction with explicit frame and consistent downstream propagation; never raw quaternion components. |
| OBBs, observations, candidate lists | Input data, not learned parameter vectors. |
| TCP offsets, robot geometry, joint/actuator limits, physical contact margins | Keep fixed as calibration/model/constraint quantities. |
| Tolerances, timeouts, iteration/step budgets, subsampling, success checks | Keep fixed in this motion-profile experiment, for consistent execution and evaluation. |
| Arm IDs, booleans, planner/controller selection, recovery routing | Keep fixed; discrete behavior choices are outside this continuous-parameter experiment. |
| Gripper settle steps and gripper-specific controls | Excluded by the current arm-motion-only scope. |

Additional positive-distance examples in the local skills include
`drop_clearance`, `approach_height`, resolved direct-IK `clearance`, and positive
approach-offset magnitudes. Planner tolerances and margins also have positive
domains, but that alone does not justify learning them. Some physical margins
may allow zero; record nonnegative versus strictly positive domains explicitly.
The current graph does not consume the `approach_pose` output from
`compute_drop_pose`, so optimizing its `approach_height` would be ineffective.

Do not enable every candidate correction by default. Check that the affected
quantity reaches an executed motion, is not overwritten or inactive in the
selected branch, and does not duplicate another parameter's effect. A shared
target correction must have one owner and propagate to its downstream users.

Concrete transport integration:

- Allowlist `drop_x`, `drop_y`, `safe_height`; initially leave orientation
  unchanged until its frame/composition binding and downstream propagation
  are explicitly defined. Add the two controller offsets separately.
- Decode five scalars for this selection: three input offsets and two gain
  offsets. Use the native schemas for validation, not to automatically enable
  every numeric input.
- In the executor, resolve refs/defaults and registered automatic values,
  deep-copy nominal inputs, then add the selected offsets once. Record both
  nominal/effective inputs and the selected offsets. Pass the resulting kwargs
  to the existing script/tool call inside the gain scope.
- If XY placement corrections are enabled, extend transport outputs to expose
  `commanded_drop_x` and `commanded_drop_y`, and explicitly wire the lowering
  node's target XY to those outputs. Retain its nominal placement Z and
  orientation. Validate/update the declared output schema and workflow refs.
  These outputs describe the commanded target, not the measured achieved pose.
  Until that wiring exists, do not enable transport XY corrections in that graph.
- Input edits affect a per-invocation copy only. Restore only mutable controller
  gains on exit; do not overwrite upstream nominal geometry or graph definitions.

## 5. Tests and acceptance criteria

Add focused tests alongside `tests/runtime` and `tests/envs`:

- No-profile execution preserves existing gains, calls, and outputs.
- Identity profiles reproduce no-profile execution, including native gain
  differences between controllers and unchanged geometry.
- Gain offsets broadcast to OSC's six and Panda's seven entries;
  changing one scalar leaves the other independent.
- One profiled node switches OSC to joint PD internally; both modes observe
  their own baselines plus the selected offsets, and both recover their distinct
  previous vectors afterward.
- Exceptions, nested scopes, and repeated visits restore exact previous gains.
- Unknown overrides fail instead of being dropped; input geometry changes are
  local to the invocation and do not mutate workflow definitions.
- Additive bindings use resolved nominal values or declared defaults;
  repeated visits do not compound adjustments. Nested paths, unsupported
  sentinels, and deterministic actor identity initialization are covered.
- Reset, streaming-node, parallel-execution, nonblocking-command, and teleport
  restrictions fail explicitly before an unsupported profiled motion starts.
- A script and a direct tool node work, including inside a subgraph; static
  profiles through `gap.execute` and dynamic profiles through GOPI agree.

Run the relevant existing workflow/executor, connector, and trajectory tests.
Then run simulator smoke tests for a direct OSC move, a joint-controlled move,
and forced servo fallback, recording applied gains and restoration. Compare
no-profile execution to the existing baseline on matched seeds. Finally run
a short RL integration smoke test with the new action schema. A reward or
sample-efficiency improvement is an experimental result, not an implementation
acceptance condition.

## Implementation order

1. Profile types and LIBERO gain scope, with lifecycle tests.
2. Native executor and public `gap.execute` integration, with input/trace tests.
3. GOPI schema/runtime integration and a separate motion-profile experiment.
4. Simulator validation and documentation of the measured gain ranges.
