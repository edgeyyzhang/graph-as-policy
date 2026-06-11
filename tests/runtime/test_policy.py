"""Tests for gap.runtime.policy — gripper-cycle detector, obs encoding, loop.

No real policy server: the openpi websocket client is replaced by a
``FakePolicyClient`` returning scripted action chunks, and the NodeContext
by a ``RecordingCtx`` that records every ``ctx.tool`` dispatch. The loop's
action execution is the per-row ``sim.apply_policy_action`` passthrough
(the source's behavior); trajectory-capable consumers pack windows via
``chunk_to_trajectory`` and dispatch ``robot.execute_trajectory``.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from gap.runtime.policy import (
    _LIBERO_DUMMY_ACTION,
    PolicyExecutor,
    _GripperCycleDetector,
    _parse_ws_url,
    _primary_rgb,
    chunk_to_trajectory,
    encode_obs,
    run_policy_loop,
)

_OPENPI_MISSING = importlib.util.find_spec("openpi_client") is None


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class RecordingCtx:
    """NodeContext stand-in: records every ctx.tool dispatch."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, dict]] = []
        self._responses = dict(responses or {})
        self.cancel_token = None

    def tool(self, name, **kwargs):
        self.calls.append((name, kwargs))
        responder = self._responses.get(name)
        if callable(responder):
            return responder(**kwargs)
        return responder

    def names(self):
        return [name for name, _ in self.calls]

    def calls_for(self, name):
        return [kw for n, kw in self.calls if n == name]


class FakePolicyClient:
    """openpi WebsocketClientPolicy stand-in returning scripted chunks.

    Chunks are consumed in order; the last one repeats forever.
    """

    def __init__(self, chunks):
        self._chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
        self.infer_requests: list[dict] = []

    def infer(self, obs):
        self.infer_requests.append(obs)
        chunk = self._chunks.pop(0) if len(self._chunks) > 1 else self._chunks[0]
        return {"actions": chunk}


def make_camera(name="agentview", h=8, w=8, seed=0):
    rgb = (np.arange(h * w * 3).reshape(h, w, 3) + seed) % 251
    return {"name": name, "rgb": rgb.astype(np.uint8)}


def make_observation(*, n_cameras=1, proprio=None, h=8, w=8):
    cameras = [
        make_camera(name=f"cam{i}", h=h, w=w, seed=i * 7) for i in range(n_cameras)
    ]
    arm = {
        "joint_state": {"positions": np.zeros(7)},
        "gripper_fraction": 1.0,
        "ee_pose": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    }
    if proprio is not None:
        arm["proprio_state"] = np.asarray(proprio, dtype=np.float64)
    return {"cameras": cameras, "arms": [arm]}


_PROPRIO = [0.4, -0.1, 0.9, 0.0, 0.0, 1.5, 0.038, -0.038]


# ---------------------------------------------------------------------------
# Gripper-cycle detector
# ---------------------------------------------------------------------------


def test_detector_full_cycle_fires_once():
    det = _GripperCycleDetector()
    # open -> close (held >= 3 windows) -> open completes the cycle.
    assert [det.update(c) for c in [-1.0, 1.0, 1.0, 1.0, -1.0]] == [
        False, False, False, False, True,
    ]


def test_detector_monotone_signal_never_fires():
    det = _GripperCycleDetector()
    assert not any(det.update(-1.0) for _ in range(10))  # stays open
    det = _GripperCycleDetector()
    assert not any(det.update(1.0) for _ in range(10))  # closes, never reopens


def test_detector_brief_close_is_failed_grasp_and_resets():
    det = _GripperCycleDetector()
    # Two-window close is below the min-hold threshold: no fire, reset.
    assert [det.update(c) for c in [1.0, 1.0, -1.0]] == [False, False, False]
    # The real grasp afterwards completes normally.
    assert [det.update(c) for c in [1.0, 1.0, 1.0, -1.0]] == [
        False, False, False, True,
    ]


def test_detector_deadband_neither_counts_nor_resets():
    det = _GripperCycleDetector()
    # Mid-range commands inside the hysteresis dead-band keep HOLDING
    # without contributing to the hold count.
    assert [det.update(c) for c in [1.0, 0.0, 0.2, 1.0, 1.0, -1.0]] == [
        False, False, False, False, False, True,
    ]


def test_detector_none_is_ignored():
    det = _GripperCycleDetector()
    assert det.update(None) is False


# ---------------------------------------------------------------------------
# Observation encoding (gap.types.Observation -> openpi obs dict)
# ---------------------------------------------------------------------------


def test_encode_obs_layout_keys_shapes_dtypes():
    obs = make_observation(n_cameras=2, proprio=_PROPRIO, h=64, w=96)
    out = encode_obs(obs, prompt="put the bowl in the sink")

    assert list(out.keys()) == [
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    ]
    assert out["observation/image"].shape == (224, 224, 3)
    assert out["observation/image"].dtype == np.uint8
    assert out["observation/wrist_image"].shape == (224, 224, 3)
    assert out["observation/wrist_image"].dtype == np.uint8
    assert out["observation/state"].shape == (8,)
    assert out["observation/state"].dtype == np.float32
    assert out["prompt"] == "put the bowl in the sink"


def test_encode_obs_proprio_state_passthrough():
    obs = make_observation(proprio=_PROPRIO)
    out = encode_obs(obs, prompt="x")
    np.testing.assert_array_equal(
        out["observation/state"], np.asarray(_PROPRIO, dtype=np.float32),
    )


def test_encode_obs_image_is_wflip_of_square_input():
    # Square input + image_size == H makes the fallback resize an identity,
    # isolating the W-flip ([:, ::-1]) the policy expects on top of gap's
    # already-H-flipped render.
    obs = make_observation(proprio=_PROPRIO, h=8, w=8)
    out = encode_obs(obs, prompt="x", image_size=8)
    rgb = obs["cameras"][0]["rgb"]
    np.testing.assert_array_equal(out["observation/image"], rgb[:, ::-1])


def test_encode_obs_center_crops_nonsquare_to_square():
    obs = make_observation(proprio=_PROPRIO, h=8, w=12)
    out = encode_obs(obs, prompt="x", image_size=8)
    rotated = obs["cameras"][0]["rgb"][:, ::-1]
    np.testing.assert_array_equal(out["observation/image"], rotated[:, 2:10])


def test_encode_obs_single_camera_reuses_primary_for_wrist():
    obs = make_observation(n_cameras=1, proprio=_PROPRIO, h=8, w=8)
    out = encode_obs(obs, prompt="x", image_size=8)
    np.testing.assert_array_equal(
        out["observation/wrist_image"], out["observation/image"],
    )


def test_encode_obs_ee_pose_fallback_synthesizes_state():
    # No proprio_state: state falls back to ee_pose + synthetic gripper qpos.
    half = math.sqrt(0.5)
    obs = {
        "cameras": [],
        "arms": [{
            "joint_state": {"positions": np.zeros(7)},
            "gripper_fraction": 0.5,
            "ee_pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                # 90 deg about z, gap wxyz order.
                "rotation": {"w": half, "x": 0.0, "y": 0.0, "z": half},
            },
        }],
    }
    out = encode_obs(obs, prompt="x")
    assert list(out.keys()) == ["observation/state", "prompt"]
    np.testing.assert_allclose(
        out["observation/state"],
        np.asarray(
            [1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2, 0.02, -0.02],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


def test_encode_obs_raw_gripper_qpos_preferred_over_synthetic():
    obs = make_observation()
    obs["arms"][0]["gripper_qpos"] = np.asarray([0.031, -0.012])
    out = encode_obs(obs, prompt="x", image_size=8)
    np.testing.assert_allclose(
        out["observation/state"][6:], np.asarray([0.031, -0.012], np.float32),
    )


def test_encode_obs_empty_observation_has_prompt_only():
    assert encode_obs({"cameras": [], "arms": []}, prompt="p") == {"prompt": "p"}


def test_primary_rgb_index_fallback():
    obs = make_observation(n_cameras=2)
    assert _primary_rgb(obs, 1) is obs["cameras"][1]["rgb"]
    # Out-of-range index falls back to camera 0; no cameras -> None.
    assert _primary_rgb(obs, 5) is obs["cameras"][0]["rgb"]
    assert _primary_rgb({"cameras": []}, 0) is None


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------


def test_loop_executes_action_rows_per_window_until_max_windows():
    obs = make_observation(proprio=_PROPRIO)
    chunk = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
    client = FakePolicyClient([chunk])
    ctx = RecordingCtx()

    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="stack the cups",
        max_windows=3, replan_every=2, settle_steps=2,
        obs_provider=lambda: obs,
    )

    assert result == {"status": "max_windows", "num_windows": 3, "num_steps": 6}
    applied = ctx.calls_for("sim.apply_policy_action")
    # 2 settle actions + 2 rows per window x 3 windows.
    assert len(applied) == 2 + 3 * 2
    assert applied[0]["action"] == _LIBERO_DUMMY_ACTION
    assert applied[1]["action"] == _LIBERO_DUMMY_ACTION
    # Only the first replan_every rows of each chunk execute, in order.
    assert applied[2]["action"] == chunk[0].tolist()
    assert applied[3]["action"] == chunk[1].tolist()
    assert all(kw["arm_id"] == 0 for kw in applied)
    # One inference per window, each fed the encoded prompt.
    assert len(client.infer_requests) == 3
    assert all(req["prompt"] == "stack the cups" for req in client.infer_requests)


def test_loop_default_obs_provider_uses_robot_get_observation():
    ctx = RecordingCtx(responses={
        "robot.get_observation": lambda **kw: make_observation(proprio=_PROPRIO),
    })
    client = FakePolicyClient([np.zeros((4, 7))])
    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        max_windows=2, replan_every=4, settle_steps=0,
    )
    assert result["status"] == "max_windows"
    assert ctx.names().count("robot.get_observation") == 2


def test_loop_injected_observation_stream_wins_over_tool():
    class StubStream:
        def __init__(self):
            self.reads = 0

        def latest(self):
            self.reads += 1
            return make_observation(proprio=_PROPRIO)

    stream = StubStream()
    ctx = RecordingCtx()
    client = FakePolicyClient([np.zeros((2, 7))])
    run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        max_windows=3, replan_every=2, settle_steps=0,
        obs_provider=stream.latest,
    )
    assert stream.reads == 3
    assert "robot.get_observation" not in ctx.names()


def test_loop_vlm_termination_fires():
    obs = make_observation(proprio=_PROPRIO)
    answers = iter([False, True])
    ctx = RecordingCtx(responses={
        "vlm.query_yes_no": lambda **kw: {"answer": next(answers)},
    })
    client = FakePolicyClient([np.zeros((3, 7))])

    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        termination_prompt="is the bowl in the sink?",
        max_windows=10, replan_every=3, term_period=1, settle_steps=0,
        obs_provider=lambda: obs,
    )

    assert result == {"status": "completed_by_vlm", "num_windows": 2, "num_steps": 6}
    vlm_calls = ctx.calls_for("vlm.query_yes_no")
    assert len(vlm_calls) == 2
    assert vlm_calls[0]["prompt"] == "is the bowl in the sink?"
    assert vlm_calls[0]["image"] is obs["cameras"][0]["rgb"]


def test_loop_vlm_checked_every_term_period_and_max_windows_caps():
    obs = make_observation(proprio=_PROPRIO)
    ctx = RecordingCtx(responses={
        "vlm.query_yes_no": lambda **kw: {"answer": False},
    })
    client = FakePolicyClient([np.zeros((2, 7))])

    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        termination_prompt="done?",
        max_windows=6, replan_every=2, term_period=2, settle_steps=0,
        obs_provider=lambda: obs,
    )

    assert result == {"status": "max_windows", "num_windows": 6, "num_steps": 12}
    assert len(ctx.calls_for("vlm.query_yes_no")) == 3  # windows 2, 4, 6


def test_loop_gripper_cycle_termination():
    obs = make_observation(proprio=_PROPRIO)
    # One (1, 7)-row chunk per window; the trailing gripper column traces
    # open -> close (3 windows) -> open.
    grip = [-1.0, 1.0, 1.0, 1.0, -1.0]
    chunks = [np.asarray([[0.0] * 6 + [g]]) for g in grip]
    ctx = RecordingCtx()
    client = FakePolicyClient(chunks)

    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        max_windows=20, replan_every=1, settle_steps=0,
        gripper_cycle_termination=True,
        obs_provider=lambda: obs,
    )

    assert result == {"status": "gripper_cycle", "num_windows": 5, "num_steps": 5}


def test_loop_disabled_gripper_cycle_runs_to_max_windows():
    obs = make_observation(proprio=_PROPRIO)
    chunks = [np.asarray([[0.0] * 6 + [g]]) for g in [-1.0, 1.0, 1.0, 1.0, -1.0]]
    ctx = RecordingCtx()
    client = FakePolicyClient(chunks)
    result = run_policy_loop(
        ctx, client=client, policy_id="pi", prompt="x",
        max_windows=8, replan_every=1, settle_steps=0,
        obs_provider=lambda: obs,
    )
    assert result["status"] == "max_windows"
    assert result["num_windows"] == 8


# ---------------------------------------------------------------------------
# PolicyExecutor
# ---------------------------------------------------------------------------


class StubManager:
    def __init__(self, urls):
        self._urls = dict(urls)

    def url_for(self, policy_id):
        return self._urls[policy_id]


def test_policy_executor_run_uses_cached_client_and_tool_seam():
    executor = PolicyExecutor(StubManager({"pi": "ws://127.0.0.1:9001"}))
    fake = FakePolicyClient([np.zeros((5, 7))])
    executor._clients["pi"] = fake  # bypass the websocket connect

    ctx = RecordingCtx(responses={
        "robot.get_observation": lambda **kw: make_observation(proprio=_PROPRIO),
    })
    result = executor.run(ctx, policy_id="pi", prompt="x", max_windows=1)

    assert result["status"] == "max_windows"
    assert executor.client_for("pi") is fake
    # Observation fetched through the connector tool; rows forwarded
    # through the sim passthrough (10 settle steps + 5 chunk rows).
    assert ctx.names().count("robot.get_observation") == 1
    assert len(ctx.calls_for("sim.apply_policy_action")) == 10 + 5

    executor.close()
    assert executor._clients == {}


@pytest.mark.skipif(not _OPENPI_MISSING, reason="openpi-client is installed")
def test_policy_executor_missing_openpi_client_pip_hint():
    executor = PolicyExecutor(StubManager({"pi": "ws://127.0.0.1:9001"}))
    with pytest.raises(RuntimeError, match=r"graph-as-policy\[policy\]"):
        executor.client_for("pi")


# ---------------------------------------------------------------------------
# chunk -> Trajectory packing (robot.execute_trajectory payload)
# ---------------------------------------------------------------------------


def test_chunk_to_trajectory_packs_rows_as_waypoints():
    chunk = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    traj = chunk_to_trajectory(chunk)
    assert list(traj.keys()) == ["waypoints"]
    assert len(traj["waypoints"]) == 2
    for i, wp in enumerate(traj["waypoints"]):
        assert wp["positions"].dtype == np.float64
        np.testing.assert_array_equal(wp["positions"], chunk[i])


def test_chunk_to_trajectory_rejects_non_2d():
    with pytest.raises(ValueError, match="expected 2-D"):
        chunk_to_trajectory(np.zeros(7))


def test_parse_ws_url():
    assert _parse_ws_url("ws://127.0.0.1:8000") == ("127.0.0.1", 8000)
    assert _parse_ws_url("localhost:9090") == ("localhost", 9090)
    with pytest.raises(ValueError, match="missing port"):
        _parse_ws_url("ws://localhost")
