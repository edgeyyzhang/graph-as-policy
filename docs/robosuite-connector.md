# Host-constructed Robosuite environments

`gap.connector.robosuite(...)` connects a benchmark-owned Robosuite or
MimicGen environment to the ordinary GAP runtime. It is not a second graph
engine and it is not task-specific.

```text
gap.execute(workflow, connector)
  -> RobosuiteConnector (standard robot.* and sim.* tools)
     -> RobosuiteEnvAdapter (reset, OSC step, video, native success)
     -> MujocoSceneAdapter (entities, OBBs, features, joints, contacts)
```

This is structurally parallel to the registered LIBERO path:

```text
gap.execute(workflow, connector)
  -> SimConnector
     -> FrankaLiberoEnv
     -> LiberoWorldAdapter (a compatibility subclass of MujocoSceneAdapter)
```

The host benchmark remains responsible for constructing its task and for
passing its native success function:

```python
from gap.connector import robosuite

raw_env = benchmark_adapter.make_env(...)
connector = robosuite(
    raw_env,
    task_completed=benchmark_adapter.success,
    task_flags=benchmark_adapter.task_flags,
    condition_definitions=benchmark_adapter.task["definitions"],
    seed=seed,
    max_steps=horizon,
)
connector.reset()
result = gap.execute(workflow, connector)
```

Robosuite motion tools use normalized OSC_POSE, not joint targets. A blocking
`robot.go_to_pose` or `robot.go_to_pose_cartesian` call defaults to 200 control
ticks and reports a stall after 25 consecutive non-improving ticks. Contact
moves can therefore stall by construction; the error reports the commanded and
achieved TCP pose, position/orientation residual, tick budget and nearest joint
limits. Position and orientation share the same OSC solve, so large simultaneous
rotations can starve translation. All public robot poses and privileged scene
geometry use connector-world (robot-base); raw demonstration EEF channels must
first be converted to the public TCP convention.

Kitchen's cooking history remains owned by its benchmark adapter rather than
reimplemented in GAP. The connector exposes those adapter-owned latches through
`sim.get_task_flags`, and `sim.evaluate_condition` evaluates the same
named milestone definitions used by Code-as-Policy. Coffee, Mug, Hammer,
Assembly, Threading, Stack, and Square use the same connector with their own
flags and definitions. Simulator vocabulary stays below
`PrivilegedSceneAdapter`, and graph skills continue to consume portable
`SceneEntity`, `SceneFeature`, and `ArticulationState` values.
