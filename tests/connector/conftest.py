"""Shared fakes for the connector test suite.

``FakeEnv`` implements the pinned env surface (reset / step /
get_observation → sim-native obs dict, ``_step_once`` / ``_set_gripper`` /
``compute_reward`` / ``task_completed``, video methods) with first-order
joint dynamics so the connector's convergence loops actually iterate.

``FakeEnvConfig`` mirrors ``gap.envs.registry.EnvConfig``'s field surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass
class FakeEnvConfig:
    arm_dof: int = 7
    num_arms: int = 1
    action_mode: str = "velocity_joints"
    control_freq: float = 20.0
    home_joints: tuple = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    tcp_offset: tuple | None = None
    tcp_rotation_z: float | None = None
    arm_bases: tuple | None = None
    robot_urdf_path: str | None = "panda_description"
    default_cameras: tuple = ("cam0",)
    is_real: bool = False


class FakeEnv:
    """Pinned env surface with simple first-order joint dynamics.

    Deliberately does NOT define ``move_to_joints_blocking`` so the
    connector's own manual convergence loop (velocity-mode action
    synthesis) is exercised.
    """

    def __init__(self, dof: int = 7, camera_names=("cam0",)):
        self._dof = dof
        self.camera_names = list(camera_names)
        self.max_steps = 100000
        self._sim_step_count = 0
        self._joints = np.zeros(dof, dtype=np.float64)
        self._gripper_fraction = 1.0
        self._gripper_pos = 1.0  # settles toward fraction
        self._current_done = False
        self._reward = 0.0
        self._success = False
        self.step_actions: list[np.ndarray] = []   # via handle.step
        self.reset_seeds: list[int | None] = []
        self.closed = False
        self._video_enabled = False
        self._frames: list[np.ndarray] = []
        self.handle = SimpleNamespace(step=self._handle_step)

    # --- env surface ----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self._joints[:] = 0.1
        self._gripper_fraction = 1.0
        self._gripper_pos = 1.0
        self._sim_step_count = 0
        return self.get_observation(), {"task_prompt": "fake task"}

    def _handle_step(self, action):
        """Velocity-joints semantics with 50% tracking per step."""
        action = np.asarray(action, dtype=np.float64)
        self.step_actions.append(action.copy())
        delta = action[: self._dof] / 20.0  # control_freq
        self._joints = self._joints + 0.5 * delta
        self._advance_gripper()
        self._sim_step_count += 1
        return self.get_observation(), 0.0, False, {}

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        self._gripper_fraction = float(fraction)

    def _step_once(self) -> None:
        self._advance_gripper()
        self._sim_step_count += 1
        if self._video_enabled:
            self._frames.append(np.zeros((4, 4, 3), dtype=np.uint8))

    def _advance_gripper(self) -> None:
        err = self._gripper_fraction - self._gripper_pos
        self._gripper_pos += float(np.clip(err, -0.05, 0.05))

    def compute_reward(self) -> float:
        return self._reward

    def task_completed(self) -> bool:
        return self._success

    def get_observation(self) -> dict:
        obs: dict = {}
        for cam in self.camera_names:
            obs[cam] = {
                "images": {
                    "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                    "depth": np.full((8, 8), 0.5, dtype=np.float32),
                },
                "intrinsics": np.eye(3, dtype=np.float64),
                # [x, y, z, qw, qx, qy, qz] — wxyz per the env-boundary
                # convention (converted from sim-native xyzw inside the env).
                "pose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 1.0, 0.0]),
            }
        obs["robot_joint_pos_0"] = np.concatenate(
            [self._joints, [self._gripper_pos]]
        )
        obs["robot_cartesian_pos_0"] = np.array(
            [0.4, 0.05, 0.3, 0.0, 1.0, 0.0, 0.0, self._gripper_pos]
        )
        return obs

    # --- video ------------------------------------------------------------

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True):
        self._video_enabled = enabled
        if clear:
            self._frames.clear()

    def get_video_frames(self, *, clear: bool = False):
        frames = list(self._frames)
        if clear:
            self._frames.clear()
        return frames

    def close(self) -> None:
        self.closed = True


class FakeIK:
    """Scripted IK backend: returns preset joints, records calls."""

    trajectory_needs_joint_reverse = False

    def __init__(self, joints=None):
        self.joints = list(joints) if joints is not None else [0.3] * 7
        self.calls: list[dict] = []
        self.plan_calls: list[dict] = []

    def solve_ik(self, pose, *, arm_id=0, seed_joints=None, tcp_offset=None):
        self.calls.append({
            "pose": pose, "arm_id": arm_id,
            "seed_joints": seed_joints, "tcp_offset": tcp_offset,
        })
        return list(self.joints)

    def plan_linear(self, start, end, *, arm_id=0, tcp_offset=None, **kw):
        self.plan_calls.append({"start": start, "end": end, "arm_id": arm_id})
        return {
            "waypoints": [
                {"positions": np.full(7, j)} for j in (0.1, 0.2)
            ]
        }


@pytest.fixture
def fake_env():
    return FakeEnv()


@pytest.fixture
def fake_config():
    return FakeEnvConfig()


@pytest.fixture
def connector(fake_env, fake_config):
    from gap.connector import SimConnector

    return SimConnector(fake_env, fake_config, ik=FakeIK())
