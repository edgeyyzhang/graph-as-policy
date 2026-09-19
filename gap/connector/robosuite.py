"""GAP connector for host-constructed robosuite / MimicGen environments.

This is parallel to :func:`gap.connector.sim`: it presents the identical
robot.*, sim.*, privileged-provider, checkpoint, and executor surfaces.  The
only difference is ownership of scene construction.  A benchmark constructs
its registered task and hands the resulting environment to :func:`robosuite`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.types import Se3Pose

from gap.connector.core import OSC_MAX_TICKS
from gap.connector.sim import SimConnector
from gap.connector.world_adapter import MujocoSceneAdapter
from gap.envs.registry import EnvConfig, GripperSpec, WorkspaceSpec
from gap.envs.robosuite_env import RobosuiteEnvAdapter


def _live_config(env: Any, camera_names: list[str] | None) -> EnvConfig:
    """Derive embodiment metadata from the live robot instead of task names."""
    obs = getattr(env, "raw_observation", {}) or {}
    raw = env.raw_env
    robots = list(getattr(raw, "robots", []) or [])
    fallback_joints = getattr(robots[0], "init_qpos", np.zeros(7)) if robots else np.zeros(7)
    joints = np.asarray(obs.get("robot0_joint_pos", fallback_joints), dtype=np.float64)
    control_freq = float(getattr(raw, "control_freq", 20.0) or 20.0)
    arm_dof = int(joints.size or 7)
    joint_names: list[str] = []
    joint_limits: list[tuple[float, float]] = []
    model = getattr(getattr(raw, "sim", None), "model", None)
    for index in range(1, arm_dof + 1):
        name = f"robot0_joint{index}"
        try:
            jid = int(model.joint_name2id(name))
            limits = tuple(float(value) for value in np.asarray(model.jnt_range)[jid])
        except Exception:
            limits = (-np.inf, np.inf)
        joint_names.append(name)
        joint_limits.append(limits)
    return EnvConfig(
        arm_dof=arm_dof,
        num_arms=len(robots or [None]),
        action_mode="osc_pose",
        control_freq=control_freq,
        home_joints=tuple(float(v) for v in joints) if joints.size else None,
        joint_names=tuple(joint_names),
        joint_limits=tuple(joint_limits),
        default_cameras=tuple(camera_names or ("agentview",)),
        gripper=GripperSpec(
            name=type(getattr(robots[0], "gripper", None)).__name__ if robots else "parallel_jaw",
            span_m=0.08,
            close_axis=(0.0, 1.0, 0.0),
            approach_axis=(0.0, 0.0, 1.0),
            finger_reach_m=None,
            width_at_closed_m=0.0,
        ),
        # The privileged provider measures the live support when one exists.
        workspace=WorkspaceSpec(surface_z=0.0, transport_z=None),
    )


class RobosuiteConnector(SimConnector):
    """A :class:`SimConnector` whose motion backend is normalized OSC_POSE.

    Skill code and workflow execution are unchanged.  The connector overrides
    only how the standard robot.go_to_pose operations are realized, just as a
    real-robot connector supplies a different transport below the same tools.
    """

    def supports_joint_position_control(self) -> bool:
        return False

    def joint_position_control_reason(self) -> str:
        return (
            "unsupported by the Robosuite OSC_POSE connector: its action space "
            "contains Cartesian deltas, not joint targets; direct qpos writes are forbidden"
        )

    def _osc_target(
        self,
        pose: Se3Pose,
        *,
        arm_id: int = 0,
        max_ticks: int = OSC_MAX_TICKS,
        tool_name: str = "robot.go_to_pose",
    ) -> None:
        if not pose:
            raise ToolError("robot.go_to_pose", "pose required")
        with self._quiet_motion():
            self._servo_to_pose(
                pose,
                arm_id=arm_id,
                max_ticks=max(1, int(max_ticks)),
                tool_name=tool_name,
            )

    def go_to_pose(
        self,
        pose: Se3Pose,
        tolerance: float = 0.0,
        max_steps: int = 0,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
        z_approach: float = 0.0,
        nonblocking: bool = False,
    ) -> None:
        del tolerance, tcp_offset
        if nonblocking:
            raise ToolError(
                "robot.go_to_pose",
                "nonblocking pose commands are not supported by the OSC connector",
            )
        if z_approach > 0 and pose.get("position"):
            p = pose["position"]
            approach: Se3Pose = {
                "position": {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p["z"]) + float(z_approach),
                },
                "rotation": dict(pose["rotation"]),
            }
            self._osc_target(
                approach, arm_id=arm_id, max_ticks=max_steps or OSC_MAX_TICKS
            )
        self._osc_target(pose, arm_id=arm_id, max_ticks=max_steps or OSC_MAX_TICKS)

    def go_to_pose_cartesian(self, pose: Se3Pose, arm_id: int = 0) -> None:
        self._osc_target(
            pose, arm_id=arm_id, tool_name="robot.go_to_pose_cartesian"
        )


def robosuite(
    env: Any,
    *,
    task_completed: Callable[[Any], bool] | None = None,
    task_flags: Callable[[Any], Any] | None = None,
    condition_definitions: dict[str, Any] | None = None,
    cameras: list[str] | None = None,
    seed: int | None = None,
    record_video: bool = False,
    max_steps: int | None = None,
    video_stride: int = 1,
) -> RobosuiteConnector:
    """Bind a host-created robosuite environment to the normal GAP runtime.

    `task_completed` is the benchmark's native predicate.  It is deliberately
    injected instead of reimplemented here, so stateful tasks such as Kitchen
    retain their exact benchmark latch semantics.
    """
    wrapped = (
        env
        if isinstance(env, RobosuiteEnvAdapter)
        else RobosuiteEnvAdapter(
            env,
            task_completed=task_completed,
            camera_names=cameras,
            max_steps=max_steps,
            video_stride=video_stride,
        )
    )
    config = _live_config(wrapped, cameras)
    provider = MujocoSceneAdapter(wrapped, arm_dof=config.arm_dof)
    return RobosuiteConnector(
        wrapped,
        config,
        camera_names=cameras,
        record_video=record_video,
        seed=seed,
        scene_adapter=provider,
        task_flags_provider=(
            (lambda: task_flags(env)) if callable(task_flags) else None
        ),
        condition_definitions=condition_definitions,
    )


__all__ = ["RobosuiteConnector", "robosuite"]
