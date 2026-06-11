"""LIBERO sim smoke tests — need the [libero] extra + EGL (``-m sim``).

Run with::

    MUJOCO_GL=egl .venv/bin/python -m pytest gap/tests/envs -q -m sim

Covers both LIBERO sources behind ``gap.envs.loader``:

- vab (Variational-Automation-Benchmark YAML tasks): construction via the
  registry factory, baked-init seeding, raw-obs assembly (cameras +
  proprio + ground-truth object poses), success check, video round-trip.
- classic LIBERO-PRO benchmark registry: the same smoke on
  ``libero_object`` task 0, constructed *after* the vab env to exercise
  the in-process fork switch.

Module-scoped fixtures share the (expensive) envs across tests; test
order within this file is load-bearing for the fork-switch coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from gap.envs.registry import EnvConfig, resolve

pytestmark = pytest.mark.sim

_VAB_SUITE = "libero_object_all_variance"
_PRO_SUITE = "libero_object"


def _make(suite: str):
    factory, key = resolve(suite)
    assert key == suite
    return factory(key, 0, camera_names=None, enable_render=False)


@pytest.fixture(scope="module")
def vab_env():
    env, config = _make(_VAB_SUITE)
    yield env, config
    env.handle.env.close()


@pytest.fixture(scope="module")
def pro_env():
    env, config = _make(_PRO_SUITE)
    yield env, config
    env.handle.env.close()


def _assert_smoke_obs(obs: dict) -> None:
    """Shared obs-shape assertions for both LIBERO sources."""
    for cam in ("agentview", "robot0_eye_in_hand"):
        assert cam in obs
        rgb = obs[cam]["images"]["rgb"]
        assert rgb.shape == (512, 800, 3) and rgb.dtype == np.uint8
        depth = obs[cam]["images"]["depth"]
        assert depth.shape[:2] == (512, 800)
        assert np.isfinite(depth).all()
        assert obs[cam]["intrinsics"].shape == (3, 3)
        assert obs[cam]["pose"].shape == (7,)
        assert obs[cam]["pose_mat"].shape == (4, 4)
    assert obs["robot_joint_pos_0"].shape == (8,)
    assert obs["robot_cartesian_pos_0"].shape == (8,)
    assert obs["robot_proprio_pi05_libero_0"].shape == (8,)
    # Ground-truth object poses: {name: [x, y, z, qw, qx, qy, qz]}
    assert obs["cube_poses"], "expected ground-truth object poses"
    for pose in obs["cube_poses"].values():
        assert pose.shape == (7,)
        assert np.isfinite(pose).all()


# ---------------------------------------------------------------------------
# vab suite
# ---------------------------------------------------------------------------


def test_vab_factory_and_envconfig(vab_env):
    env, config = vab_env
    assert isinstance(config, EnvConfig)
    assert config.action_mode == "velocity_joints"
    assert config.arm_dof == 7
    assert config.num_arms == 1
    assert config.control_freq == 20.0
    assert config.home_joints == (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    assert config.tcp_offset == (0.0, 0.0, -0.1)
    assert config.robot_urdf_path == "panda_description"
    assert config.default_cameras == ("agentview", "robot0_eye_in_hand")
    assert not config.is_real
    assert env.handle.task_language


def test_vab_reset_obs_and_success(vab_env):
    env, _ = vab_env
    obs, info = env.reset(seed=1)
    assert "task_prompt" in info and info["task_prompt"]
    _assert_smoke_obs(obs)
    # Task 0 of the suite is the alphabet-soup pick-and-place.
    assert "alphabet_soup" in obs["cube_poses"]
    assert "basket" in obs["cube_poses"]
    # Success check is callable and falsy at episode start. (The classic
    # path returns np.bool_ — the connector, like the source server, wraps
    # it in bool().)
    assert not env.task_completed()
    assert float(env.compute_reward()) == 0.0


def test_vab_seed_selects_baked_init(vab_env):
    """Seeds map onto the YAML's 50 baked inits: (seed - 1) % n_inits."""
    env, _ = vab_env
    assert len(env.handle.init_states) == 50
    obs1, _ = env.reset(seed=1)
    basket_1 = obs1["cube_poses"]["basket"][:2].copy()
    obs2, _ = env.reset(seed=2)
    basket_2 = obs2["cube_poses"]["basket"][:2].copy()
    # init 0 and init 1 place the basket >10 cm apart in XY.
    assert np.linalg.norm(basket_1 - basket_2) > 0.05
    # Same seed → same baked init.
    obs1b, _ = env.reset(seed=1)
    assert np.linalg.norm(obs1b["cube_poses"]["basket"][:2] - basket_1) < 0.02


def test_vab_step_zero_action(vab_env):
    env, _ = vab_env
    env.reset(seed=1)
    # OSC_POSE is active after reset: 7-dim [Δpose(6), gripper].
    obs, reward, terminated, truncated, info = env.step(np.zeros(7))
    _assert_smoke_obs(obs)
    assert isinstance(float(reward), float)
    assert terminated is False
    assert truncated is False


def test_vab_video_roundtrip(vab_env, tmp_path):
    env, _ = vab_env
    env.enable_video_capture(True, clear=True)
    for _ in range(2 * env._subsample_rate):
        env.step(np.zeros(7))
    env.enable_video_capture(False, clear=False)
    frames = env.get_video_frames()
    assert len(frames) >= 2
    assert frames[0].shape == (512, 800, 3)
    out = tmp_path / "vab_smoke.mp4"
    written = env.save_video(str(out), fps=20)
    assert written == len(frames)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# classic LIBERO-PRO suite (constructed after vab → exercises fork switch)
# ---------------------------------------------------------------------------


def test_pro_factory_and_envconfig(pro_env):
    env, config = pro_env
    assert config.action_mode == "velocity_joints"
    assert config.arm_dof == 7
    assert config.robot_urdf_path == "panda_description"
    assert env.handle.task_language


def test_pro_reset_obs_step_and_success(pro_env):
    env, _ = pro_env
    obs, info = env.reset(seed=1)
    assert "task_prompt" in info
    _assert_smoke_obs(obs)
    # Classic path additionally renders element-level segmentation.
    assert "segmentation" in obs["agentview"]["images"]
    assert not env.task_completed()
    obs, reward, terminated, truncated, _ = env.step(np.zeros(7))
    _assert_smoke_obs(obs)
    assert terminated is False and truncated is False


def test_vab_env_survives_fork_switch(vab_env, pro_env):
    """A live vab env keeps stepping after the classic fork was activated."""
    vab, _ = vab_env
    pro, _ = pro_env
    obs, *_ = vab.step(np.zeros(7))
    assert "alphabet_soup" in obs["cube_poses"]
    assert not vab.task_completed()
    obs, *_ = pro.step(np.zeros(7))
    assert obs["robot_joint_pos_0"].shape == (8,)
