"""Type documentation for LIBERO-YAM connector tools.

These tools are registered at runtime by LiberoYamSimConnector via
reg.register_callable — they need live MuJoCo env access and cannot be
provided as static @tool stubs. This file documents the signatures only.

NOTE: gap skills check will report these as unknown in allowed_tools — this is
a known framework limitation for connector-custom tools (same as tsh-pi05 /
sim.tsh_phase). The tools work correctly at runtime.
"""

from __future__ import annotations

_CONNECTOR_HINT = (
    " requires the LIBERO-YAM connector (LiberoYamSimConnector) — use "
    "gap.connector.libero_yam() to build it."
)


def object_pose(object_name: str) -> dict:
    """Return ``{"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]}``
    for ``object_name`` (MJCF body name) from live MuJoCo model data.

    Registered as ``libero-yam.object_pose`` by LiberoYamSimConnector.
    """
    raise RuntimeError("libero-yam.object_pose" + _CONNECTOR_HINT)


def arm_base_pose(arm_id: int = 0) -> dict:
    """Frame info for planning ``arm_id`` with the canonical curobo bundle.

    Returns ``{"position": [x, y, z], "rotation": [w, x, y, z],
    "tcp_offset": [x, y, z], "joints": [j1..j6]}`` — the arm's base world pose,
    the fixed link_6->TCP offset, and its current joints: everything a skill
    needs to convert a world TCP target into the arm-base frame cuRobo plans
    in and to seed ``start_joint_position``. YAM bases are mounted upright,
    so world->base is a pure translation by ``position``.

    Registered as ``libero-yam.arm_base_pose`` by LiberoYamSimConnector.
    """
    raise RuntimeError("libero-yam.arm_base_pose" + _CONNECTOR_HINT)


def execute_trajectory(trajectory: dict, arm_id: int = 0,
                       max_steps: int = 200) -> dict:
    """Stream-execute a cuRobo joint ``Trajectory`` (from ``curobo.plan_to_pose``
    et al.) on ``arm_id``: dense waypoints one sim step each, then a blocking
    converge on the final waypoint. PD-controlled (contact-respecting) —
    distinct from the core ``robot.execute_trajectory``, which qpos-teleports
    and would phase the fingers through the tape on a grasp descent.

    Registered as ``libero-yam.execute_trajectory`` by LiberoYamSimConnector.
    """
    raise RuntimeError("libero-yam.execute_trajectory" + _CONNECTOR_HINT)
