---
name: libero-yam
description: Connector tools for the LIBERO-YAM bimanual simulation. Provides
  ground-truth object poses, per-arm planning frame info, and PD-controlled
  trajectory execution. These tools are registered at runtime by
  LiberoYamSimConnector; this bundle declares their typed signatures so skill
  validators and the subgraph agent can see them statically. Use when authoring
  skills that run on the LIBERO-YAM connector.
compatibility: requires gap>=0.1
metadata: {category: sim, tags: [libero-yam, sim, mujoco, curobo]}
gap:
  tools:
    - libero-yam.object_pose: World-frame [x, y, z] position of a scene object by MJCF body name.
    - libero-yam.arm_base_pose: Arm base world pose + TCP offset + current joints — the frame info for planning this arm via the curobo bundle.
    - libero-yam.execute_trajectory: Stream-execute a cuRobo joint Trajectory on the given arm, PD-controlled (contact-respecting).
  requires:
    env: [MUJOCO_GL]
---

# libero-yam connector tools

Static declarations for the tools registered by `LiberoYamSimConnector` in
`gap.connector.libero_yam`. Scripts call these via
`ctx.tool("libero-yam.<name>", ...)`.

## `libero-yam.object_pose`

```python
object_pose(object_name: str) -> dict
# Returns: {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]}
```

Reads the world-frame centroid of `object_name` (an MJCF body name, e.g.
`"yellow_tape_1"`, `"duct_tape_1"`) directly from `model`/`data` — no
perception, no noise. Only valid when the LIBERO-YAM connector is active.

## `libero-yam.arm_base_pose`

```python
arm_base_pose(arm_id: int = 0) -> dict
# Returns: {"position": [x, y, z], "rotation": [w, x, y, z],
#           "tcp_offset": [x, y, z], "joints": [j1..j6]}
```

Everything a skill needs to plan this arm with `curobo.plan_to_pose`
(`robot_file="yam.yml"`): the base world pose for the world->base transform
(pure translation — YAM bases are upright), the link_6->TCP offset to pass as
`tcp_offset`, and the current joints for `start_joint_position`.

## `libero-yam.execute_trajectory`

```python
execute_trajectory(trajectory: Trajectory, arm_id: int = 0, max_steps: int = 200) -> dict
```

Streams the planned joint waypoints onto the sim one step each, then converges
blocking on the final waypoint. PD-controlled — use this (NOT the core
`robot.execute_trajectory`, which qpos-teleports) so contact-rich legs like the
grasp descent respect the physics.
