"""Unit tests for ``Connector._servo_to_pose`` — the policy-like OSC Cartesian
servo (Phase 2). Drives the real method against a stub whose
``apply_policy_action`` integrates the EE pose toward the commanded OSC delta,
so it needs no mujoco/EGL. Covers convergence and the stall -> ToolError path
that triggers the caller's cuRobo fallback.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

Connector = pytest.importorskip("gap.connector.core").Connector
# _servo_to_pose imports robosuite's orientation_error at call time.
pytest.importorskip("robosuite")

_DOWN = {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}


def _stub(start_pos, *, move: bool = True, track: float = 1.0):
    state = {"pos": np.asarray(start_pos, dtype=float), "n": 0}

    def get_ee_pose(arm_id: int = 0):
        p = state["pos"]
        return {
            "position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "rotation": dict(_DOWN),
        }

    def apply(action):
        a = np.asarray(action, dtype=float)
        if move:
            # OSC: action[:3] in [-1,1] -> up to output_max (0.05 m) toward goal.
            state["pos"] = state["pos"] + track * a[:3] * 0.05
        state["n"] += 1

    env = SimpleNamespace(apply_policy_action=apply, _gripper_fraction=1.0)
    return SimpleNamespace(env=env, get_ee_pose=get_ee_pose), state


def test_servo_converges_to_target():
    stub, st = _stub([0.40, 0.0, 0.30])
    target = np.array([0.45, 0.04, 0.25])
    pose = {"position": {"x": 0.45, "y": 0.04, "z": 0.25}, "rotation": dict(_DOWN)}
    Connector._servo_to_pose(stub, pose, pos_tol=0.005, rot_tol=0.05, max_ticks=200)
    assert float(np.linalg.norm(st["pos"] - target)) < 0.005
    assert 0 < st["n"] < 60  # few OSC ticks, not a dense trajectory


def test_servo_robust_to_pd_undertracking():
    stub, st = _stub([0.40, 0.0, 0.30], track=0.6)
    target = np.array([0.46, 0.0, 0.28])
    pose = {"position": {"x": 0.46, "y": 0.0, "z": 0.28}, "rotation": dict(_DOWN)}
    Connector._servo_to_pose(stub, pose, pos_tol=0.005, max_ticks=200)
    assert float(np.linalg.norm(st["pos"] - target)) < 0.006


def test_servo_stall_raises_tool_error():
    """A stuck EE (apply does nothing) must raise so the caller falls back to
    the collision-aware cuRobo planner."""
    from gap_core.errors import ToolError

    stub, _ = _stub([0.40, 0.0, 0.30], move=False)
    pose = {"position": {"x": 0.45, "y": 0.04, "z": 0.25}, "rotation": None}
    with pytest.raises(ToolError) as excinfo:
        Connector._servo_to_pose(stub, pose, stall_ticks=10, max_ticks=200)
    message = str(excinfo.value)
    assert "after 11/200 ticks (patience=10)" in message
    assert "target=(0.4500,0.0400,0.2500)" in message
    assert "achieved=(0.4000,0.0000,0.3000)" in message
    assert stub._last_motion_diagnostic["status"] == "stalled"
    assert stub._last_motion_diagnostic["position_error_m"] > 0.0
