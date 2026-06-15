"""make_env factory knob tests — env classes stubbed, no sim required.

The perturbed / joint-motion-mode opt-ins are explicit factory kwargs with
``GAP_LIBERO_PERTURBED`` / ``GAP_LIBERO_JOINT_MOTION_MODE`` env-var
fallbacks (the per-worker configuration channel). These tests pin that
dispatch without constructing a real LIBERO env.
"""

from __future__ import annotations

import pytest

import gap.envs.libero_env as libero_env_mod
import gap.envs.libero_perturbed_env as perturbed_mod
from gap.envs.registry import EnvConfig


class _FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.camera_names = kwargs.get("camera_names") or [
            "agentview",
            "robot0_eye_in_hand",
        ]
        self._control_freq = kwargs.get("control_freq", 20)


class _FakePerturbedEnv(_FakeEnv):
    pass


@pytest.fixture()
def stubbed_envs(monkeypatch):
    monkeypatch.setattr(libero_env_mod, "FrankaLiberoEnv", _FakeEnv)
    monkeypatch.setattr(perturbed_mod, "FrankaLiberoPerturbedEnv", _FakePerturbedEnv)
    monkeypatch.delenv("GAP_LIBERO_PERTURBED", raising=False)
    monkeypatch.delenv("GAP_LIBERO_JOINT_MOTION_MODE", raising=False)


def test_factory_defaults(stubbed_envs):
    env, config = libero_env_mod.make_env("libero_object", 0, None, False)
    assert isinstance(env, _FakeEnv) and not isinstance(env, _FakePerturbedEnv)
    assert env.kwargs["suite_name"] == "libero_object"
    assert env.kwargs["task_id"] == 0
    assert env.kwargs["joint_motion_mode"] == "closed_loop"
    assert isinstance(config, EnvConfig)
    assert config.action_mode == "velocity_joints"
    assert config.arm_dof == 7
    assert config.home_joints == (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)


def test_factory_env_var_fallbacks(stubbed_envs, monkeypatch):
    monkeypatch.setenv("GAP_LIBERO_PERTURBED", "1")
    monkeypatch.setenv("GAP_LIBERO_JOINT_MOTION_MODE", "closed_loop")
    env, _ = libero_env_mod.make_env("libero_object", 3, ["agentview"], True)
    assert isinstance(env, _FakePerturbedEnv)
    assert env.kwargs["joint_motion_mode"] == "closed_loop"
    assert env.kwargs["camera_names"] == ["agentview"]
    assert env.kwargs["enable_render"] is True


def test_factory_explicit_kwargs_override_env_vars(stubbed_envs, monkeypatch):
    monkeypatch.setenv("GAP_LIBERO_PERTURBED", "1")
    monkeypatch.setenv("GAP_LIBERO_JOINT_MOTION_MODE", "closed_loop")
    env, _ = libero_env_mod.make_env(
        "libero_object", 0, None, False,
        perturbed=False, joint_motion_mode="teleport",
    )
    assert isinstance(env, _FakeEnv) and not isinstance(env, _FakePerturbedEnv)
    assert env.kwargs["joint_motion_mode"] == "teleport"


def test_factory_invalid_motion_mode_falls_back(stubbed_envs, monkeypatch):
    monkeypatch.setenv("GAP_LIBERO_JOINT_MOTION_MODE", "warp_drive")
    env, _ = libero_env_mod.make_env("libero_object", 0, None, False)
    assert env.kwargs["joint_motion_mode"] == "closed_loop"


def test_factory_extra_kwargs_pass_through(stubbed_envs):
    env, config = libero_env_mod.make_env(
        "libero_object", 0, None, False, max_steps=500, seed=7, control_freq=30,
    )
    assert env.kwargs["max_steps"] == 500
    assert env.kwargs["seed"] == 7
    assert config.control_freq == 30.0
    assert config.default_cameras == ("agentview", "robot0_eye_in_hand")
