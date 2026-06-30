"""Unit tests for ``FrankaLiberoEnv.stream_joint_trajectory`` — the policy-like
joint-space path-following servo (Phase 1 of policy-like motion execution).

Drives the real method against a faithful *stub* of robosuite's JOINT_POSITION
integration (normalized action in [-1, 1] -> up to ``output_max`` rad/step), so
it needs no mujoco/EGL — unlike the ``-m sim`` smoke tests. Validates the
pursuit/settle algorithm and its latency instrumentation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Importing the class pulls viser + the libero loader; skip cleanly where the
# sim extra isn't installed. The tests themselves drive a stub, not a real sim.
FrankaLiberoEnv = pytest.importorskip("gap.envs.libero_env").FrankaLiberoEnv

_OUTPUT_MAX = 0.05


def _stub(start, *, dof: int = 7, track: float = 1.0, frac: float = 1.0):
    """Bind the real method to a stub whose ``handle.step`` integrates qpos the
    way the JOINT_POSITION controller does. ``track`` < 1 mimics PD
    under-tracking per env step; ``frac`` is the per-tick step clamp.
    """
    qpos = np.zeros(dof + 3, dtype=np.float64)
    addrs = list(range(dof))
    qpos[addrs] = np.asarray(start, dtype=np.float64)

    def step(action):
        a = np.asarray(action, dtype=np.float64)
        qpos[addrs] = qpos[addrs] + track * a[:dof] * _OUTPUT_MAX
        return (None, 0.0, False, {})

    data = SimpleNamespace(
        qpos=qpos, xquat=np.array([[1.0, 0.0, 0.0, 0.0]]), xpos=np.zeros((1, 3))
    )
    handle = SimpleNamespace(
        step=step, env=SimpleNamespace(sim=SimpleNamespace(data=data))
    )
    env = SimpleNamespace(
        handle=handle,
        _panda_joint_qpos_addrs=addrs,
        _joint_output_max=_OUTPUT_MAX,
        _stream_max_step_frac=frac,
        _gripper_fraction=1.0,
        gripper_link_idx=0,
        _record_frames=False,
        _subsample_rate=4,
        _sim_step_count=0,
        _command_count=0,
        _sim_physics_wall_s=0.0,
        _use_controller=lambda mode: None,
        _current_obs=None,
        _current_reward=None,
        _current_done=None,
        _current_info=None,
    )
    return env, qpos, addrs


def _run(env, wps, **kw):
    FrankaLiberoEnv.stream_joint_trajectory(env, [list(w) for w in wps], **kw)


def test_single_waypoint_converges():
    env, q, a = _stub(np.zeros(7))
    _run(env, [np.full(7, 1.0)], settle_tolerance=0.01, settle_max_steps=60)
    assert np.linalg.norm(q[a] - 1.0) < 0.01
    assert env._command_count > 0
    # Streamed (commanded) ticks + a bounded settle tail.
    settle_ticks = env._sim_step_count - env._command_count
    assert 1 <= settle_ticks <= 60


def test_polyline_followed_to_last_waypoint():
    env, q, a = _stub(np.zeros(7))
    wps = [np.full(7, v) for v in (0.2, 0.4, 0.6, 0.8, 1.0)]
    _run(env, wps, settle_tolerance=0.005, settle_max_steps=80)
    assert np.linalg.norm(q[a] - 1.0) < 0.01


def test_no_settling_tax_on_far_move():
    """A 1.0 rad move costs ~= dist/output_max ticks (~20), NOT a 120-step
    per-waypoint convergence — this is the whole point of streaming."""
    env, q, a = _stub(np.zeros(7))
    target = np.zeros(7)
    target[0] = 1.0
    _run(env, [target], settle_tolerance=0.01, settle_max_steps=60)
    assert 18 <= env._sim_step_count <= 32


def test_already_at_target_is_cheap():
    env, q, a = _stub(np.full(7, 0.5))
    _run(env, [np.full(7, 0.5)], settle_tolerance=0.01, settle_max_steps=60)
    assert env._command_count == 0  # no commanded motion needed
    assert env._sim_step_count >= 1  # one settle check


def test_robust_to_pd_undertracking():
    env, q, a = _stub(np.zeros(7), track=0.6)
    _run(env, [np.full(7, 0.8)], settle_tolerance=0.01, settle_max_steps=120)
    assert np.linalg.norm(q[a] - 0.8) < 0.01


def test_step_clamp_fraction_still_converges():
    # frac=0.5 halves the per-tick joint step; the move still reaches target.
    env, q, a = _stub(np.zeros(7), frac=0.5)
    _run(env, [np.full(7, 0.5)], settle_tolerance=0.01, settle_max_steps=80)
    assert np.linalg.norm(q[a] - 0.5) < 0.01


def test_empty_trajectory_is_noop():
    env, q, a = _stub(np.full(7, 0.3))
    _run(env, [], settle_tolerance=0.01, settle_max_steps=60)
    assert env._command_count == 0 and env._sim_step_count == 0
