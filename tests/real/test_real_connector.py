"""RealConnector + real() factory against the fake rr client.

real("franka", rr_autostart=False) exercises the genuine wire path:
FrankaRealEnv binds the msgpack server, the fake client (standing in for
rr-session) feeds frames, wait_ready unblocks on the first RGB.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gap.connector import Capabilities, RealConnector
from gap.connector.real import real

from .conftest import ClientFeeder, free_port

# The full motion surface registered for franka (matches core.Connector).
MOTION_TOOLS = {
    "robot.go_to_pose", "robot.go_to_pose_cartesian", "robot.move_to_joints",
    "robot.execute_trajectory", "robot.go_home", "robot.open_gripper",
    "robot.close_gripper", "robot.solve_ik",
}
OBSERVATION_TOOLS = {
    "robot.get_observation", "robot.get_camera_pose", "robot.get_ee_pose",
    "robot.get_gripper", "robot.get_gripper_pose",
}


@pytest.fixture
def franka_conn():
    port = free_port()
    feeder = ClientFeeder(port, cam_key="camera_top")
    conn = real(
        "franka",
        rr_autostart=False,
        port=port,
        cameras=["robot0_robotview"],
        wait_timeout_s=20.0,
    )
    yield conn
    conn.close()
    feeder.stop()


class TestFrankaFactory:
    def test_wait_ready_and_tool_surface(self, franka_conn):
        assert isinstance(franka_conn, RealConnector)
        reg = franka_conn.tool_registry
        for name in OBSERVATION_TOOLS | MOTION_TOOLS:
            assert name in reg, f"missing {name}"
        # NEVER any sim.* tools on a real connector.
        sim_tools = [n for n in reg._tools if n.startswith("sim.")]
        assert sim_tools == []

    def test_capabilities_all_false(self, franka_conn):
        assert franka_conn.capabilities == Capabilities(
            reset=False, success_check=False, video=False, world_state=False,
        )

    def test_observation_assembles(self, franka_conn):
        obs = franka_conn.get_observation()
        assert len(obs["cameras"]) == 1
        cam = obs["cameras"][0]
        assert cam["name"] == "robot0_robotview"
        assert cam["rgb"].shape == (8, 8, 3)
        assert obs["arms"][0]["joint_state"]["positions"].shape == (7,)

    def test_config_marks_real(self, franka_conn):
        assert franka_conn.is_real is True
        assert franka_conn.config.control_freq == 50.0
        assert franka_conn.config.arm_dof == 7
        # Robotiq TCP from the source server's franka_real branch.
        np.testing.assert_allclose(
            franka_conn.config.tcp_offset, (0.0, 0.0, -0.157)
        )
        assert franka_conn.config.tcp_rotation_z == pytest.approx(np.pi / 4)

    def test_go_home_guard_no_wire_command(self, franka_conn, caplog):
        """GoHome on a real robot is skipped (ported server.py guard)."""
        before = franka_conn.env._target_update_count
        with caplog.at_level(logging.WARNING):
            franka_conn.go_home()
        assert franka_conn.env._target_update_count == before
        assert any("go_home skipped" in r.message for r in caplog.records)


class TestWaitReadyTimeout:
    def test_timeout_surfaces_diagnostics(self):
        port = free_port()
        with pytest.raises(TimeoutError) as ei:
            real(
                "franka",
                rr_autostart=False,
                port=port,
                cameras=["cam0"],
                wait_timeout_s=1.0,
            )
        # The env's _diagnose_missing_rgb text rides on the error.
        assert "latest_observation is None" in str(ei.value)


class TestFactoryErrors:
    def test_unknown_robot(self):
        with pytest.raises(ValueError, match="unknown real robot"):
            real("spot")

    def test_ur_zed_rejects_rr_kwargs(self):
        with pytest.raises(ValueError, match="rr_config/port only apply"):
            real("ur_zed", rr_config="x.yaml")


# ---------------------------------------------------------------------------
# go_home guard unit (no sockets — fake env + config)
# ---------------------------------------------------------------------------


class _MotionRecorder:
    """Minimal env recording motion commands."""

    def __init__(self):
        self.move_calls: list = []

    def move_to_joints_blocking(self, joints, **kw):
        self.move_calls.append(np.asarray(joints))

    def get_observation(self):
        return {"robot_joint_pos_0": np.zeros(8)}

    def close(self):
        pass


def _config(is_real: bool):
    return SimpleNamespace(
        arm_dof=7, num_arms=1, action_mode="absolute_joints",
        control_freq=50.0, home_joints=None, tcp_offset=None,
        tcp_rotation_z=None, arm_bases=None, robot_urdf_path=None,
        default_cameras=("cam0",), is_real=is_real,
    )


class TestGoHomeGuardUnit:
    def test_is_real_blocks_motion(self, caplog):
        env = _MotionRecorder()
        conn = RealConnector(env, _config(is_real=True))
        with caplog.at_level(logging.WARNING):
            conn.go_home()
        assert env.move_calls == []
        assert any("go_home skipped" in r.message for r in caplog.records)

    def test_sim_config_would_move(self):
        """Contrast: with is_real=False the same code path issues motion."""
        env = _MotionRecorder()
        conn = RealConnector(env, _config(is_real=False))
        conn.go_home()
        assert len(env.move_calls) == 1
        np.testing.assert_allclose(
            env.move_calls[0], [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        )
