"""Hardware-in-the-loop checks — run manually with real robots attached.

Excluded from the default suite (``addopts`` deselects ``-m real``). Run:

    pytest tests/real/test_hardware.py -m real --no-header -s

Pre-flight checklist (see examples/real_franka_pick_place/README.md):
  - E-stop in reach and tested; workspace clear of people and clutter
  - robots_realtime submodule synced + ``uv sync``'d
  - Franka: FCI enabled, arm near home, Robotiq mounted, ZED connected
  - UR: reachable at the configured IP; ZED SDK installed; calibration
    .npy exported (GAP_UR_ZED_CALIB)
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.real


def test_franka_session_first_observation():
    """Spawns rr-session, waits for the first camera frame, holds home.

    The arm should NOT move during this test (hold-home pre-seed).
    """
    import gap.connector

    conn = gap.connector.real("franka", wait_timeout_s=120.0)
    try:
        obs = conn.get_observation()
        assert obs["cameras"], "no camera frame from the rr-session ZED node"
        rgb = obs["cameras"][0]["rgb"]
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        joints = obs["arms"][0]["joint_state"]["positions"]
        assert joints.shape == (7,)
        assert np.all(np.isfinite(joints))
    finally:
        conn.close()


def test_franka_gripper_cycle():
    """Open → close → open; verifies the 50 Hz action path end to end."""
    import gap.connector

    conn = gap.connector.real("franka", wait_timeout_s=120.0)
    try:
        conn.open_gripper()
        closed = conn.close_gripper()
        assert closed["position"] < 0.5
        opened = conn.open_gripper()
        assert opened["position"] > 0.5
    finally:
        conn.close()


def test_ur_zed_capture():
    """Perception-only: one RGB-D capture + FK camera pose from the UR."""
    import gap.connector

    conn = gap.connector.real("ur_zed")
    try:
        obs = conn.get_observation()
        cam = obs["cameras"][0]
        assert cam["rgb"].ndim == 3
        assert cam["depth"].ndim == 2
        assert cam["intrinsics"].shape == (3, 3)
        # Perception-only registry: no motion tools.
        assert "robot.go_to_pose" not in conn.tool_registry
    finally:
        conn.close()
