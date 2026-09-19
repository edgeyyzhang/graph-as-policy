"""Simulator-neutral tests for the host-constructed robosuite env adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gap.connector.robosuite import RobosuiteConnector
from gap.envs.registry import EnvConfig
from gap.envs.robosuite_env import RobosuiteEnvAdapter


class _Model:
    def body_name2id(self, name: str) -> int:
        if name != "robot0_base":
            raise KeyError(name)
        return 0


class _RawEnv:
    horizon = 50
    control_freq = 20

    def __init__(self) -> None:
        self.pos = np.array([-0.2, 0.0, 1.0])
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                body_xpos=np.array([[-0.6, 0.0, 0.9]]),
                body_xquat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            ),
        )

    def _obs(self) -> dict:
        return {
            "robot0_eef_pos": self.pos.copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_joint_pos": np.zeros(7),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }

    def reset(self) -> dict:
        self.pos = np.array([-0.2, 0.0, 1.0])
        return self._obs()

    def step(self, action: np.ndarray) -> tuple:
        self.pos += np.asarray(action[:3]) * 0.05
        return self._obs(), 0.0, False, {}

    def _check_success(self) -> bool:
        return False

    def close(self) -> None:
        pass


def test_adapter_uses_gap_robot_base_frame_and_normalized_osc() -> None:
    env = RobosuiteEnvAdapter(_RawEnv(), max_steps=10)
    obs, info = env.reset(seed=3)
    assert info == {}
    # World eef x=-0.2 and base x=-0.6 -> public x=0.4.
    assert np.isclose(obs["robot_cartesian_pos_0"][0], 0.4)
    assert np.isclose(obs["robot_joint_pos_0"][-1], 1.0)
    env.apply_policy_action(np.array([0.2, 0, 0, 0, 0, 0, -1.0]))
    assert env._sim_step_count == 1
    assert np.isclose(env.get_observation()["robot_cartesian_pos_0"][0], 0.41)


def _osc_connector(raw: _RawEnv) -> RobosuiteConnector:
    wrapped = RobosuiteEnvAdapter(raw, max_steps=100)
    wrapped.reset()
    config = EnvConfig(
        arm_dof=7,
        action_mode="osc_pose",
        joint_names=tuple(f"joint{i + 1}" for i in range(7)),
        joint_limits=tuple((-1.0, 1.0) for _ in range(7)),
        home_joints=tuple(0.0 for _ in range(7)),
    )
    return RobosuiteConnector(wrapped, config)


def test_osc_connector_hides_unsupported_joint_motion_tools() -> None:
    connector = _osc_connector(_RawEnv())
    registry = connector.tool_registry
    assert "robot.get_joint_state" in registry
    assert "robot.move_to_joints" not in registry
    assert "robot.execute_trajectory" not in registry
    assert "robot.go_home" not in registry
    capability = connector.describe_arm()["joint_position_control"]
    assert capability["available"] is False
    assert "OSC_POSE" in capability["reason"]
    assert "direct qpos writes are forbidden" in capability["reason"]


def test_osc_stall_error_names_nearest_joint_limits() -> None:
    class StuckRawEnv(_RawEnv):
        def _obs(self) -> dict:
            obs = super()._obs()
            obs["robot0_joint_pos"] = np.array([0.0, 0.99, -1.0, 0.0, 0.0, 0.0, 0.0])
            return obs

        def step(self, action: np.ndarray) -> tuple:
            return self._obs(), 0.0, False, {}

    connector = _osc_connector(StuckRawEnv())
    pose = connector.get_ee_pose()
    pose["position"]["x"] += 0.2
    import pytest
    from gap_core.errors import ToolError

    with pytest.raises(ToolError) as excinfo:
        connector.go_to_pose_cartesian(pose)
    message = str(excinfo.value)
    assert "servo stalled" in message
    assert "joint3=-1.0000, lower=-1.0000, margin=0.000" in message
    assert "joint2=0.9900, upper=1.0000, margin=0.005" in message
