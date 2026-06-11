"""DataCollector HDF5 schema + episode synchronization."""

from __future__ import annotations

import h5py
import numpy as np

from gap.connector import DataCollector, SimConnector

from .conftest import FakeEnv, FakeEnvConfig, FakeIK


def _make(tmp_path):
    env = FakeEnv()
    conn = SimConnector(env, FakeEnvConfig(), ik=FakeIK())
    collector = DataCollector(conn, tmp_path / "demo.h5")
    return env, conn, collector


def test_hdf5_schema(tmp_path):
    env, conn, collector = _make(tmp_path)
    collector.start_episode()
    conn.open_gripper(settle_steps=5)              # 5 hold steps
    conn.move_to_joints([0.2] * 7, max_steps=50)   # N control steps
    collector.end_episode(success=True)
    collector.close()

    with h5py.File(tmp_path / "demo.h5") as f:
        n = f.attrs["num_steps"]
        assert n == 5 + len(env.step_actions)
        rgb = f["/observations/cam0_rgb"]
        assert rgb.shape == (n, 8, 8, 3) and rgb.dtype == np.uint8
        state = f["/observations/state"]
        assert state.shape == (n, 8) and state.dtype == np.float32
        actions = f["/actions"]
        assert actions.shape == (n, 8) and actions.dtype == np.float32
        assert f["/rewards"].shape == (n,)
        assert f["/dones"].shape == (n,)
        assert list(f["/episode_ends"][...]) == [n]
        assert list(f["/episode_success"][...]) == [True]
        assert list(f.attrs["cameras"]) == ["cam0"]


def test_actions_synchronized_with_state(tmp_path):
    env, conn, collector = _make(tmp_path)
    collector.start_episode()
    conn.move_to_joints([0.3] * 7, max_steps=50)
    collector.end_episode(success=False)
    collector.close()

    with h5py.File(tmp_path / "demo.h5") as f:
        actions = f["/actions"][...]
        states = f["/observations/state"][...]
    # Row 0: o_0 is the pre-step obs (joints at 0), a_0 the velocity command.
    np.testing.assert_allclose(states[0, :7], 0.0, atol=1e-6)
    np.testing.assert_allclose(actions[0, :7], 0.3 * 20.0, atol=1e-5)
    # The recorded actions mirror what the env actually stepped.
    np.testing.assert_allclose(actions[:, :7], np.stack(env.step_actions)[:, :7])


def test_episode_boundaries(tmp_path):
    env, conn, collector = _make(tmp_path)
    # Steps outside an episode are not recorded.
    conn.open_gripper(settle_steps=3)

    collector.start_episode()
    conn.open_gripper(settle_steps=2)
    collector.end_episode(success=True)

    collector.start_episode()
    conn.close_gripper(settle_steps=4)
    collector.end_episode(success=False)
    collector.close()

    with h5py.File(tmp_path / "demo.h5") as f:
        assert f.attrs["num_steps"] == 6
        assert list(f["/episode_ends"][...]) == [2, 6]
        assert list(f["/episode_success"][...]) == [True, False]


def test_close_detaches_hook_and_finalizes_open_episode(tmp_path):
    env, conn, collector = _make(tmp_path)
    collector.start_episode()
    conn.open_gripper(settle_steps=2)
    collector.close()  # open episode → recorded as failure

    with h5py.File(tmp_path / "demo.h5") as f:
        assert list(f["/episode_success"][...]) == [False]
    assert collector._on_step not in conn._step_callbacks
    # Further steps must not raise (hook detached, file closed).
    conn.open_gripper(settle_steps=1)


def test_context_manager(tmp_path):
    env = FakeEnv()
    conn = SimConnector(env, FakeEnvConfig(), ik=FakeIK())
    with DataCollector(conn, tmp_path / "cm.h5") as collector:
        collector.start_episode()
        conn.open_gripper(settle_steps=2)
        collector.end_episode(success=True)
    with h5py.File(tmp_path / "cm.h5") as f:
        assert f.attrs["num_steps"] == 2
