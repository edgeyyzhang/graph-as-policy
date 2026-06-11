"""Msgpack protocol round-trip: fake rr client ↔ FrankaRealEnv.

The fake client is a plain socket speaking the wire format read from the
robots_realtime submodule (4-byte BE length + msgpack-numpy; replies
decoded raw=True → byte keys), against a real FrankaRealEnv bound to an
ephemeral loopback port. No hardware, no robots_realtime import.
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from .conftest import FakeRRClient, free_port, full_frame, joints_frame


def _wait(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestRoundTrip:
    def test_preseeded_hold_home_action(self, franka_env):
        """The very first reply must be the hold-home pre-seed, never {}."""
        client = FakeRRClient(franka_env._test_port)
        try:
            reply = client.send_request(joints_frame())
            assert reply != {}
            assert isinstance(reply[b"timestamp"], float)
            left = reply[b"left"]
            joint_pos = np.asarray(left[b"joint_pos"], dtype=np.float64)
            assert joint_pos.shape == (7,)
            np.testing.assert_allclose(joint_pos, franka_env._current_joints, atol=1e-4)
            assert left[b"gripper"] == pytest.approx(1.0)
        finally:
            client.close()

    def test_first_observation_assembled(self, franka_env):
        client = FakeRRClient(franka_env._test_port)
        try:
            joints = np.array([0.1, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7], dtype=np.float32)
            client.send_request(full_frame(joints7=joints, gripper=0.25))

            assert _wait(
                lambda: franka_env.get_observation().get("cam0") is not None
            ), "camera observation never assembled"
            obs = franka_env.get_observation()

            # Joints: 8 floats (7 arm + gripper), passed through verbatim.
            jp = obs["robot_joint_pos_0"]
            assert jp.shape == (8,)
            np.testing.assert_allclose(jp[:7], joints, atol=1e-6)
            assert jp[7] == pytest.approx(0.25)

            # EE pose via FK + Robotiq TCP: [xyz, wxyz quat, gripper].
            cart = obs["robot_cartesian_pos_0"]
            assert cart.shape == (8,)
            assert cart[7] == pytest.approx(0.25)
            quat = cart[3:7]
            assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-3)

            # Camera payload converted: rgb/depth/intrinsics/pose (wxyz).
            cam = obs["cam0"]
            assert cam["images"]["rgb"].shape == (8, 8, 3)
            assert cam["images"]["depth"].shape == (8, 8)
            np.testing.assert_allclose(
                cam["intrinsics"], np.eye(3, dtype=np.float32) * 100.0
            )
            pose = cam["pose"]
            assert pose.shape == (7,)
            np.testing.assert_allclose(pose[:3], [0.1, 0.2, 0.3], atol=1e-6)
            np.testing.assert_allclose(pose[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        finally:
            client.close()

    def test_camera_kept_when_rate_limited(self, franka_env):
        """Joints-only frames must not clobber the last good camera frame."""
        client = FakeRRClient(franka_env._test_port)
        try:
            client.send_request(full_frame())
            assert _wait(lambda: "cam0" in franka_env.get_observation())
            # The realtime client strips cameras on most ticks.
            client.send_request(joints_frame(joints7=np.full(7, 0.3, dtype=np.float32)))
            assert _wait(
                lambda: franka_env.get_observation()["robot_joint_pos_0"][0]
                == pytest.approx(0.3)
            )
            obs = franka_env.get_observation()
            assert obs["cam0"]["images"]["rgb"] is not None
        finally:
            client.close()

    def test_republish_rate_plausible(self, franka_env):
        """Action frames are re-stamped at ~50 Hz by the background republisher."""
        client = FakeRRClient(franka_env._test_port)
        try:
            stamps = []
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                reply = client.send_request(joints_frame())
                stamps.append(reply[b"timestamp"])
                time.sleep(0.005)
            unique = sorted(set(stamps))
            # 50 Hz over 0.6 s ≈ 30 fresh stamps; demand a plausible band.
            assert 10 <= len(unique) <= 60, f"{len(unique)} unique stamps"
            assert stamps == sorted(stamps), "timestamps must be non-decreasing"
        finally:
            client.close()

    def test_streaming_target_reaches_wire(self, franka_env):
        """move_to_joints_blocking(max_steps=0) supersedes the wire command."""
        client = FakeRRClient(franka_env._test_port)
        try:
            target = np.array([0.5, -0.4, 0.3, -2.1, 0.1, 1.6, 0.0])
            franka_env.move_to_joints_blocking(target, max_steps=0)

            def _sees_target():
                reply = client.send_request(joints_frame())
                got = np.asarray(reply[b"left"][b"joint_pos"], dtype=np.float64)
                return np.allclose(got, target, atol=1e-4)

            assert _wait(_sees_target, timeout=2.0), "republisher never emitted target"
        finally:
            client.close()


class TestHeartbeatStaleness:
    def test_staleness_triggers_on_client_silence(self, franka_env):
        client = FakeRRClient(franka_env._test_port)
        try:
            client.send_request(full_frame(timestamp=time.time()))
            # Fresh frame: heartbeat (period 0.05 s) should report not-stale.
            assert _wait(lambda: bool(franka_env.last_heartbeat), timeout=2.0)
            # Now go silent. stale_after_s=0.3 → obs_stale flips.
            assert _wait(lambda: franka_env.obs_stale, timeout=3.0), (
                f"staleness never triggered: {franka_env.last_heartbeat}"
            )
            assert franka_env.last_heartbeat["obs_age"] > 0.3
            # A fresh frame clears it again.
            client.send_request(full_frame(timestamp=time.time()))
            assert _wait(lambda: not franka_env.obs_stale, timeout=3.0)
        finally:
            client.close()


class TestDiagnostics:
    def test_diagnose_no_client(self, franka_env):
        msg = franka_env._diagnose_missing_rgb("cam0")
        assert "latest_observation is None" in msg

    def test_diagnose_no_camera_subdict(self, franka_env):
        client = FakeRRClient(franka_env._test_port)
        try:
            client.send_request(joints_frame())
            assert _wait(lambda: franka_env.server.latest_observation is not None)
            msg = franka_env._diagnose_missing_rgb("cam0")
            assert "no camera-shaped subdict" in msg
            assert "left" in msg  # wire keys listed for debugging
        finally:
            client.close()

    def test_diagnose_missing_rgb_key(self, franka_env):
        client = FakeRRClient(franka_env._test_port)
        try:
            frame = joints_frame()
            frame["camera_top"] = {
                "images": {},
                "depth_data": np.zeros((4, 4), dtype=np.float32),
            }
            client.send_request(frame)
            assert _wait(lambda: franka_env.server.latest_observation is not None)
            msg = franka_env._diagnose_missing_rgb("cam0")
            assert "no rgb / left_rgb / right_rgb" in msg
        finally:
            client.close()


class TestTeardown:
    def test_close_stops_threads_and_server(self):
        from gap.envs.franka_real_env import FrankaRealEnv

        port = free_port()
        env = FrankaRealEnv(
            camera_names=["cam0"], port=port,
            heartbeat_period=0.05, stale_after_s=0.3,
        )
        client = FakeRRClient(port)
        client.send_request(full_frame())
        client.close()

        env.close()
        env._republish_thread.join(timeout=2.0)
        env._heartbeat_thread.join(timeout=2.0)
        assert not env._republish_thread.is_alive()
        assert not env._heartbeat_thread.is_alive()

        # The listening socket closes: fresh connections are refused.
        def _refused():
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return False
            except OSError:
                return True

        assert _wait(_refused, timeout=5.0), "msgpack server still accepting"

    def test_close_idempotent(self, franka_env):
        franka_env.close()
        franka_env.close()
