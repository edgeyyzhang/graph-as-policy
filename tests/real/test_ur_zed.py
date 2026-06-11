"""URZedEnv: lazy-import errors, perception-only registration, obs assembly.

No ZED / UR hardware: SDK imports are asserted to fail with actionable
messages, and the observation path runs against mocked env internals.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gap.connector import Capabilities, RealConnector

_HAS_PYZED = importlib.util.find_spec("pyzed") is not None
_HAS_RTDE = importlib.util.find_spec("rtde_receive") is not None


# ---------------------------------------------------------------------------
# Lazy-import errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_PYZED, reason="pyzed installed — error path not reachable")
class TestPyzedMissing:
    def test_constructor_error_mentions_zed_sdk(self):
        from gap.envs.ur_zed_env import URZedEnv

        with pytest.raises(ImportError) as ei:
            URZedEnv()
        msg = str(ei.value)
        assert "ZED SDK" in msg
        assert "stereolabs.com" in msg

    def test_real_factory_propagates_clear_error(self):
        from gap.connector.real import real

        with pytest.raises(ImportError, match="ZED SDK"):
            real("ur_zed")


@pytest.mark.skipif(_HAS_RTDE, reason="rtde installed — error path not reachable")
class TestRtdeMissing:
    def test_error_points_at_real_extra(self, monkeypatch):
        import gap.envs.ur_zed_env as mod

        # Get past the ZED stage with a stub camera.
        monkeypatch.setattr(mod, "ZedDepthCamera", lambda **kw: object())
        with pytest.raises(ImportError) as ei:
            mod.URZedEnv()
        assert "graph-as-policy[real]" in str(ei.value)


# ---------------------------------------------------------------------------
# Perception-only tool registration (mocked env internals)
# ---------------------------------------------------------------------------


OBSERVATION_TOOLS = {
    "robot.get_observation", "robot.get_camera_pose", "robot.get_ee_pose",
    "robot.get_gripper", "robot.get_gripper_pose",
}


class _StubPerceptionEnv:
    camera_names = ["zed_left"]

    def get_observation(self):
        return {
            "robot_joint_pos_0": np.zeros(7, dtype=np.float32),
            "robot_cartesian_pos_0": np.array(
                [0.1, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
            ),
            "zed_left": {
                "images": {
                    "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
                    "depth": np.zeros((4, 4), dtype=np.float32),
                },
                "intrinsics": np.eye(3, dtype=np.float32),
                "pose": np.array([0, 0, 0.5, 1, 0, 0, 0], dtype=np.float32),
            },
        }

    def close(self):
        pass


def _ur_config():
    from gap.envs.registry import EnvConfig

    return EnvConfig(
        arm_dof=6, num_arms=1, action_mode="absolute_joints",
        control_freq=15.0, default_cameras=("zed_left",), is_real=True,
    )


class TestPerceptionOnlyRegistration:
    def _conn(self):
        return RealConnector(_StubPerceptionEnv(), _ur_config(), motion_enabled=False)

    def test_only_observation_tools(self):
        reg = self._conn().tool_registry
        names = set(reg._tools.keys())
        assert names == OBSERVATION_TOOLS

    def test_no_motion_no_sim(self):
        reg = self._conn().tool_registry
        for forbidden in (
            "robot.go_to_pose", "robot.go_to_pose_cartesian",
            "robot.move_to_joints", "robot.execute_trajectory",
            "robot.go_home", "robot.open_gripper", "robot.close_gripper",
            "robot.solve_ik", "sim.reset", "sim.step", "sim.check_success",
        ):
            assert forbidden not in reg

    def test_capabilities_all_false(self):
        assert self._conn().capabilities == Capabilities(
            reset=False, success_check=False, video=False, world_state=False,
        )

    def test_observation_assembles(self):
        obs = self._conn().get_observation()
        cam = obs["cameras"][0]
        assert cam["name"] == "zed_left"
        assert cam["rgb"].shape == (4, 4, 3)
        arm = obs["arms"][0]
        assert arm["joint_state"]["positions"].shape == (6,)
        assert arm["ee_pose"]["position"]["z"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Observation capture with mocked internals (no pyzed / rtde / URDF download)
# ---------------------------------------------------------------------------


class _StubZed:
    def get_images_and_depth(self):
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]  # marker pixel to verify the 180° rotation
        depth = np.zeros((4, 6), dtype=np.float32)
        depth[0, 0] = 1.5
        return rgb, rgb.copy(), depth

    def get_intrinsics_matrix(self):
        return np.eye(3) * 50.0

    def close(self):
        self.closed = True


class _StubRtde:
    def getActualQ(self):
        return [0.1, -1.5, 1.2, -1.3, 1.5, -0.2]


class _StubUrdf:
    link_map = {"wrist_3_link": object()}  # no ee_link → wrist fallback

    def update_cfg(self, cfg):
        assert len(cfg) == 6
        self.cfg = cfg

    def get_transform(self, link, base=None):
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.1, 0.5]
        return T


def _mocked_ur_env():
    from gap.envs.ur_zed_env import URZedEnv

    env = object.__new__(URZedEnv)  # skip hardware __init__
    env.camera_names = ["zed_left"]
    env.zed = _StubZed()
    env.rtde_recv = _StubRtde()
    env._urdf = _StubUrdf()
    env._T_cam_to_wrist = np.eye(4)
    env._obs = {}
    env._sim_step_count = 0
    env.max_steps = 999999
    return env


class TestCaptureWithMockedInternals:
    def test_obs_shapes_and_rotation(self):
        env = _mocked_ur_env()
        obs = env.get_observation()

        jp = obs["robot_joint_pos_0"]
        assert jp.shape == (7,)  # 6 joints + synthetic open gripper
        assert jp[6] == pytest.approx(1.0)

        cart = obs["robot_cartesian_pos_0"]
        assert cart.shape == (8,)
        np.testing.assert_allclose(cart[:3], [0.3, 0.1, 0.5], atol=1e-6)
        np.testing.assert_allclose(cart[3:7], [1, 0, 0, 0], atol=1e-6)  # wxyz

        cam = obs["zed_left"]
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        # Upside-down mount: both rgb and depth rotated together by 180°.
        assert rgb.shape == (4, 6, 3) and depth.shape == (4, 6)
        np.testing.assert_array_equal(rgb[-1, -1], [255, 0, 0])
        assert depth[-1, -1] == pytest.approx(1.5)
        assert cam["pose"].shape == (7,)
        np.testing.assert_allclose(cam["intrinsics"], np.eye(3) * 50.0)

    def test_reset_and_step_count(self):
        env = _mocked_ur_env()
        obs, info = env.reset()
        assert "zed_left" in obs
        env.step(None)
        assert env._sim_step_count == 1
        assert env.compute_reward() == 0.0
        assert env.task_completed() is False


class TestRegistryRow:
    def test_ur_zed_registered(self):
        from gap.envs.registry import registered_envs

        assert registered_envs()["ur_zed"] == "ur_zed"

    def test_franka_real_registered(self):
        from gap.envs.registry import registered_envs

        assert registered_envs()["franka_real"] == "franka_real"
