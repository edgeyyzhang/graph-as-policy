"""Per-node loop for learned VLA policy states.

A policy node names a registered policy (set up by
:class:`gap.runtime.policy_manager.PolicyManager`). This module drives
the inference → execution → termination loop for a single such state.

Semantics of one policy node execution:

1. Ask the manager for the websocket URL of the named policy.
2. For ``max_windows`` iterations:
   a. Call ``robot.get_observation`` (or read the injected
      observation stream) for fresh obs.
   b. Encode the obs + prompt into the dict the OpenPI client expects.
   c. ``client.infer(obs_dict)`` returns ``{"actions": (H, D) array}``.
   d. Forward the first ``replan_every`` rows to the env via
      ``sim.apply_policy_action`` (trajectory-capable robots instead
      pack rows into a :class:`gap.types.Trajectory` and call
      ``robot.execute_trajectory`` — see :func:`chunk_to_trajectory`).
   e. Every ``term_period`` windows, ask
      ``vlm.query_yes_no`` whether the task is done. Break if yes.
3. Return ``{"status", "num_windows", "num_steps"}``.

Action semantics (absolute joints, joints + gripper trailing, etc.) are
the user's responsibility: whatever checkpoint they host must emit rows
that the embodied robot can execute. We pass the rows through without
any embodiment-specific translation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .context import NodeContext

if TYPE_CHECKING:
    from gap.types import JointState, Trajectory

    from .policy_manager import PolicyManager

logger = logging.getLogger(__name__)


# Valid status strings surfaced in the node output.
_STATUS_COMPLETED = "completed_by_vlm"
_STATUS_MAX = "max_windows"
# Stage ended because the policy's commanded gripper completed one
# open->close->open grasp/release cycle (one item picked & placed) —
# the per-item terminator for long-horizon clean-all-items loops.
_STATUS_GRIPPER_CYCLE = "gripper_cycle"

# One-shot logging guard so encode_obs prints its first-window payload
# (eef_pos, axisangle, gripper_qpos) once per process for verification,
# without flooding the trial log with one entry per inference window.
_ENCODE_OBS_LOGGED = False
# Per-camera one-shot guard for image-preprocess logging (raw shape/mean
# vs. policy-input shape/mean).  Keyed by camera tag so agentview and
# wrist each log once.
_ENCODE_OBS_IMG_LOGGED: dict[str, bool] = {}

# Default values if the workflow doesn't supply a given input.
_DEFAULT_MAX_WINDOWS = 20
# 5 matches openpi/examples/libero/main.py — the LIBERO π-series
# checkpoints (pi05_libero) were tuned around a 5-step replan cadence.
_DEFAULT_REPLAN_EVERY = 5
_DEFAULT_TERM_PERIOD = 2
_DEFAULT_ARM_ID = 0
_DEFAULT_VLM_CAMERA = 0
# LIBERO drops objects on reset; openpi waits 10 sim steps with a
# zero-action / open-gripper command so they settle before the first
# inference.  Without this, frame 0 has objects mid-air and the policy's
# first action is conditioned on a transient scene.
_DEFAULT_SETTLE_STEPS = 10
# LIBERO/robosuite OSC_POSE convention: 6 zero EE deltas + gripper=-1
# (open) — see ``LIBERO_DUMMY_ACTION`` in openpi's reference.
_LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

# Gripper-cycle stage-termination tuning. The detector watches the
# gripper column the VLA emits each window (LIBERO OSC_POSE convention:
# -1 = open, +1 = closed) — a binary, object-width-independent signal,
# unlike the observed gripper_fraction which a wide grocery item holds
# mid-range. The two thresholds form a hysteresis dead-band.
_GRIP_CMD_CLOSED_ABOVE = 0.5   # commanded gripper > this -> CLOSED intent
_GRIP_CMD_OPEN_BELOW = -0.5    # commanded gripper < this -> OPEN intent
_GRIP_MIN_HOLD_WINDOWS = 3     # CLOSED must persist >= N windows before a
                               # re-open completes the grasp/release cycle


class PolicyExecutor:
    """Runs the per-node loop for all policy states in a workflow.

    One instance is shared across a :class:`WorkflowExecutor` run. It
    caches one websocket client per ``policy_id`` to avoid reconnecting
    across repeated policy nodes in the same workflow.
    """

    def __init__(self, manager: PolicyManager):
        self._manager = manager
        self._clients: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main entry point (called per policy node)
    # ------------------------------------------------------------------

    def run(
        self,
        ctx: NodeContext,
        *,
        policy_id: str,
        prompt: str,
        termination_prompt: str = "",
        max_windows: int = _DEFAULT_MAX_WINDOWS,
        replan_every: int = _DEFAULT_REPLAN_EVERY,
        term_period: int = _DEFAULT_TERM_PERIOD,
        arm_id: int = _DEFAULT_ARM_ID,
        vlm_camera: int = _DEFAULT_VLM_CAMERA,
        gripper_cycle_termination: bool = False,
    ) -> dict[str, Any]:
        """Legacy entry point for policy states.

        Acquires observations through ``ctx.tool("robot.get_observation")``
        each window. The ``run_policy`` skill (which reads from a graph-
        scoped ``observation_stream``) is the preferred entry point for new
        workflows; both share the same closed-loop body via
        :func:`run_policy_loop`.
        """
        client = self._client(policy_id)
        return run_policy_loop(
            ctx,
            client=client,
            policy_id=policy_id,
            prompt=prompt,
            termination_prompt=termination_prompt,
            max_windows=max_windows,
            replan_every=replan_every,
            term_period=term_period,
            arm_id=arm_id,
            vlm_camera=vlm_camera,
            gripper_cycle_termination=gripper_cycle_termination,
            obs_provider=lambda: ctx.tool("robot.get_observation"),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Drop cached websocket clients. Subprocess lifecycle is the
        :class:`PolicyManager`'s responsibility.
        """
        self._clients.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _client(self, policy_id: str) -> Any:
        return self.client_for(policy_id)

    def client_for(self, policy_id: str) -> Any:
        """Return (lazily creating) the WebsocketClientPolicy for ``policy_id``.

        Public so the ``run_policy`` skill can borrow the same cached
        client lifecycle (one connection per policy id, reused across all
        windows / multiple skill invocations within one workflow).
        """
        client = self._clients.get(policy_id)
        if client is not None:
            return client

        url = self._manager.url_for(policy_id)
        host, port = _parse_ws_url(url)

        try:
            from openpi_client.websocket_client_policy import (
                WebsocketClientPolicy,
            )
        except ImportError as exc:  # pragma: no cover - optional dep path
            raise RuntimeError(
                "policy nodes require the 'openpi-client' package; install it "
                "via `pip install \"graph-as-policy[policy]\"`"
            ) from exc

        logger.info(
            "[policy:%s] connecting WebsocketClientPolicy to %s:%d",
            policy_id, host, port,
        )
        client = WebsocketClientPolicy(host=host, port=port)
        self._clients[policy_id] = client
        return client


# ---------------------------------------------------------------------------
# Closed-loop body (shared between PolicyExecutor.run and run_policy skill)
# ---------------------------------------------------------------------------


class _GripperCycleDetector:
    """Detect one commanded open->close->open grasp/release cycle.

    Fed the VLA's commanded gripper value (LIBERO ``action[-1]``: -1 open,
    +1 closed) once per window via :meth:`update`, which returns True on
    the window the cycle completes — the caller then ends the policy
    stage. A momentary close that never reaches ``_GRIP_MIN_HOLD_WINDOWS``
    (a failed grasp) does not fire: the machine resets and waits for the
    real grasp. The detector is loop-local, so each ``run_policy_loop``
    invocation (each loop iteration of a clean-all-items workflow) gets a
    fresh one with no carried-over state.
    """

    _WAIT_CLOSE = 0
    _HOLDING = 1

    def __init__(self) -> None:
        self._state = self._WAIT_CLOSE
        self._hold = 0

    def update(self, cmd: float | None) -> bool:
        """Advance the state machine with one window's commanded gripper.

        Returns True exactly once — on the window a full open->close->open
        cycle completes.
        """
        if cmd is None:
            return False
        cmd = float(cmd)
        if self._state == self._WAIT_CLOSE:
            if cmd > _GRIP_CMD_CLOSED_ABOVE:
                self._state = self._HOLDING
                self._hold = 1
            return False
        # self._state == self._HOLDING
        if cmd > _GRIP_CMD_CLOSED_ABOVE:
            self._hold += 1
            return False
        if cmd < _GRIP_CMD_OPEN_BELOW:
            if self._hold >= _GRIP_MIN_HOLD_WINDOWS:
                return True
            # Failed grasp: closed too briefly to be a real pick. Reset
            # and wait for the next close.
            self._state = self._WAIT_CLOSE
            self._hold = 0
            return False
        # Hysteresis dead-band: stay HOLDING, neither count nor reset.
        return False


def run_policy_loop(
    ctx: NodeContext,
    *,
    client: Any,
    policy_id: str,
    prompt: str,
    termination_prompt: str = "",
    max_windows: int = _DEFAULT_MAX_WINDOWS,
    replan_every: int = _DEFAULT_REPLAN_EVERY,
    term_period: int = _DEFAULT_TERM_PERIOD,
    arm_id: int = _DEFAULT_ARM_ID,
    vlm_camera: int = _DEFAULT_VLM_CAMERA,
    settle_steps: int = _DEFAULT_SETTLE_STEPS,
    gripper_cycle_termination: bool = False,
    obs_provider: Any | None = None,
) -> dict[str, Any]:
    """Run the closed-loop replan/execute body for one policy invocation.

    ``obs_provider`` is a zero-arg callable that returns a fresh
    :class:`gap.types.Observation` each window. The legacy
    ``PolicyExecutor.run`` binds it to ``ctx.tool("robot.get_observation")``;
    the ``run_policy`` skill binds it to ``observation_stream.latest()``. The
    callable is invoked once per window-tick.

    Returns ``{"status", "num_windows", "num_steps"}`` matching the legacy
    contract.
    """
    if obs_provider is None:
        def obs_provider() -> Any:
            return ctx.tool("robot.get_observation")

    max_windows = int(max_windows)
    replan_every = max(1, int(replan_every))
    term_period = max(1, int(term_period))
    arm_id = int(arm_id)
    vlm_camera = int(vlm_camera)
    settle_steps = max(0, int(settle_steps))

    logger.info(
        "[policy:%s] starting loop max_windows=%d replan_every=%d "
        "term_period=%d settle_steps=%d arm_id=%d gripper_cycle=%s "
        "prompt=%r",
        policy_id, max_windows, replan_every, term_period, settle_steps,
        arm_id, gripper_cycle_termination, prompt,
    )

    # Settle: send N zero-EE-delta + gripper-open actions so freshly
    # spawned objects can fall onto the table before the policy sees its
    # first frame.  Mirrors openpi/examples/libero/main.py:num_steps_wait.
    for _ in range(settle_steps):
        ctx.tool(
            "sim.apply_policy_action",
            action=list(_LIBERO_DUMMY_ACTION),
            arm_id=arm_id,
        )

    num_steps = 0
    grip_detector = (
        _GripperCycleDetector() if gripper_cycle_termination else None
    )
    for i in range(max_windows):
        # Cooperative cancellation point: if this loop is running inside a
        # parallel branch and a sibling has signaled, exit immediately.
        cancel_token = getattr(ctx, "cancel_token", None)
        if cancel_token is not None:
            cancel_token.raise_if_set()
        obs = obs_provider()

        openpi_obs = encode_obs(obs, prompt=prompt, arm_id=arm_id)
        result = client.infer(openpi_obs)
        chunk = np.asarray(result["actions"])
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        chunk = chunk[:replan_every]

        # Forward each row to the env via the low-level passthrough tool.
        # The action layout is the policy checkpoint's native space (e.g.
        # ``[Δx,Δy,Δz,Δrx,Δry,Δrz, gripper]`` for pi05_libero on LIBERO's
        # OSC_POSE controller); the bridge does no translation, so the
        # checkpoint-emitted action shape must match what the simulator's
        # controller is configured to accept.  Connectors without a direct
        # VLA passthrough don't register ``sim.apply_policy_action`` — the
        # call fails loudly rather than driving the robot with the wrong
        # action space (same env-capability semantics as the source).
        for row in chunk:
            ctx.tool(
                "sim.apply_policy_action",
                action=[float(x) for x in row],
                arm_id=arm_id,
            )
        num_steps += chunk.shape[0]

        # Gripper-cycle stage termination: end as soon as the policy's
        # commanded gripper has completed one open->close->open cycle —
        # the item has been picked and released. Checked before the VLM
        # block so the cheap deterministic signal wins when both modes
        # are enabled. ``chunk[-1, -1]`` is the gripper command at the
        # end of this window; the chunk is already in hand, so no extra
        # observation call.
        if grip_detector is not None and chunk.shape[0]:
            if grip_detector.update(float(chunk[-1, -1])):
                logger.info(
                    "[policy:%s] gripper-cycle termination fired at "
                    "window %d (num_steps=%d)",
                    policy_id, i + 1, num_steps,
                )
                return {
                    "status": _STATUS_GRIPPER_CYCLE,
                    "num_windows": i + 1,
                    "num_steps": num_steps,
                }

        if termination_prompt and (i + 1) % term_period == 0:
            rgb = _primary_rgb(obs, vlm_camera)
            if rgb is None:
                logger.warning(
                    "[policy:%s] no RGB available for VLM termination "
                    "check at window %d", policy_id, i + 1,
                )
            else:
                done = ctx.tool(
                    "vlm.query_yes_no",
                    prompt=termination_prompt,
                    image=rgb,
                )
                if _vlm_yes(done):
                    logger.info(
                        "[policy:%s] VLM termination fired at window %d",
                        policy_id, i + 1,
                    )
                    return {
                        "status": _STATUS_COMPLETED,
                        "num_windows": i + 1,
                        "num_steps": num_steps,
                    }

    logger.info(
        "[policy:%s] reached max_windows=%d without VLM termination",
        policy_id, max_windows,
    )
    return {
        "status": _STATUS_MAX,
        "num_windows": max_windows,
        "num_steps": num_steps,
    }


def _vlm_yes(response: Any) -> bool:
    """Coerce a ``vlm.query_yes_no`` response into a bool.

    The vlm tool bundle returns ``{"answer": bool}``; a bare bool (or any
    object exposing an ``answer`` attribute) is also accepted.
    """
    if isinstance(response, dict):
        return bool(response.get("answer", False))
    return bool(getattr(response, "answer", response))


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


def encode_obs(
    obs: Any,
    *,
    prompt: str,
    arm_id: int = 0,
    image_size: int = 224,
) -> dict[str, Any]:
    """Translate a :class:`gap.types.Observation` into an OpenPI obs dict.

    Matches the obs preprocessing in
    ``third_party/openpi/examples/libero/main.py`` — the LIBERO π-series
    checkpoints (pi05_libero etc.) were trained against this exact shape:

    - ``observation/image``        agentview RGB, 180°-rotated, padded+resized
                                   to ``image_size`` (default 224), uint8.
    - ``observation/wrist_image``  wrist RGB, same preprocessing.
    - ``observation/state``        8-dim float32: ``[eef_pos(3),
                                   axisangle(eef_quat)(3), gripper_qpos(2)]``
                                   — NOT joint angles.  ``gripper_qpos`` is
                                   the simulator's raw 2-dim per-finger qpos
                                   when present, falling back to a synthetic
                                   ``[+frac, -frac] * 0.04`` mirror that
                                   matches robosuite's panda gripper.
    - ``prompt``                    language instruction.

    The 180° rotation (``[::-1, ::-1]``) is the LIBERO convention — the
    agentview camera is mounted upside-down, and openpi's training data
    was rotated to match.  Skipping this is what makes the policy "look
    lost" even with the action space wired up correctly.
    """
    cameras = list(obs.get("cameras") or [])
    arm_states = list(obs.get("arms") or [])

    result: dict[str, Any] = {}
    if cameras:
        primary_rgb = _rgb_as_array(cameras[0].get("rgb"))
        if primary_rgb is not None:
            result["observation/image"] = _libero_preprocess(primary_rgb, image_size)
            if not _ENCODE_OBS_IMG_LOGGED.get("agentview"):
                logger.info(
                    "[encode_obs] agentview raw=%s mean=%.1f → policy=%s mean=%.1f",
                    primary_rgb.shape, float(primary_rgb.mean()),
                    result["observation/image"].shape,
                    float(result["observation/image"].mean()),
                )
                _ENCODE_OBS_IMG_LOGGED["agentview"] = True
        wrist_rgb = _rgb_as_array(
            cameras[1].get("rgb") if len(cameras) > 1 else cameras[0].get("rgb")
        )
        if wrist_rgb is not None:
            result["observation/wrist_image"] = _libero_preprocess(wrist_rgb, image_size)

    if arm_states:
        idx = arm_id if 0 <= arm_id < len(arm_states) else 0
        arm = arm_states[idx]

        # Preferred path: connectors that host a VLA policy (LIBERO,
        # franka_real with pi05 wiring, …) populate ``proprio_state`` with
        # the raw 8-dim ``[eef_pos, axisangle, gripper_qpos]`` vector that
        # the checkpoint was trained on.  Any frame conversion / TCP
        # offset / ee-site-vs-body-quat mismatch that would distribution-
        # shift the model is the connector's responsibility, not the
        # policy's.
        proprio_raw = arm.get("proprio_state")
        proprio = (
            [] if proprio_raw is None
            else [float(x) for x in np.asarray(proprio_raw).ravel()]
        )
        used_proprio = False
        if len(proprio) >= 8:
            eef_pos = np.asarray(proprio[0:3], dtype=np.float64)
            axisangle = np.asarray(proprio[3:6], dtype=np.float64)
            gripper_qpos = [float(x) for x in proprio[6:8]]
            synthesized = False
            used_proprio = True
        else:
            ee_pose = arm.get("ee_pose")
            if ee_pose is not None and ee_pose.get("position") is not None:
                pos = ee_pose["position"]
                eef_pos = np.array(
                    [pos["x"], pos["y"], pos["z"]], dtype=np.float64,
                )
                # gap Quaternion is WXYZ; ``_quat2axisangle`` expects XYZW.
                rot = ee_pose["rotation"]
                quat_xyzw = np.array(
                    [rot["x"], rot["y"], rot["z"], rot["w"]], dtype=np.float64,
                )
                axisangle = _quat2axisangle(quat_xyzw)
            else:
                eef_pos = np.zeros(3, dtype=np.float64)
                axisangle = np.zeros(3, dtype=np.float64)

            qpos_raw = arm.get("gripper_qpos")
            gripper_qpos = (
                [] if qpos_raw is None
                else [float(x) for x in np.asarray(qpos_raw).ravel()]
            )
            synthesized = False
            if len(gripper_qpos) < 2:
                # Connectors that don't surface raw finger qpos: synthesize the
                # symmetric ±0.04·fraction layout robosuite's panda gripper
                # produces.  ``gripper_fraction`` is 1.0 = open in gap, which
                # maps to qpos ~ +0.04 / -0.04.
                frac = float(arm.get("gripper_fraction", 1.0))
                gripper_qpos = [0.04 * frac, -0.04 * frac]
                synthesized = True
            gripper_qpos = gripper_qpos[:2]

        if not _ENCODE_OBS_LOGGED:
            logger.info(
                "[encode_obs] state shape=8 eef_pos=%s axisangle=%s "
                "gripper_qpos=%s (proprio_state=%s, synthesized=%s)",
                np.round(eef_pos, 4).tolist(),
                np.round(axisangle, 4).tolist(),
                [round(float(x), 4) for x in gripper_qpos],
                used_proprio,
                synthesized,
            )
            globals()["_ENCODE_OBS_LOGGED"] = True

        state = np.concatenate([
            eef_pos.astype(np.float32),
            axisangle.astype(np.float32),
            np.asarray(gripper_qpos, dtype=np.float32),
        ])
        result["observation/state"] = state

    result["prompt"] = prompt
    return result


def _libero_preprocess(rgb: np.ndarray, image_size: int) -> np.ndarray:
    """Match ``openpi/examples/libero/main.py`` preprocessing.

    Training/reference convention is ``raw_libero_render[::-1, ::-1]`` —
    a 180° rotation applied to robosuite's raw (upside-down) MuJoCo
    output.  gap's LIBERO connector ``get_observation`` already
    pre-flips the H axis (``[::-1]``) to surface a right-side-up image
    for VLM / perception consumers, so the rotation that *this* function
    needs to apply on top is just a W-flip (``[:, ::-1]``).  Net effect:
    ``raw[::-1][:, ::-1] == raw[::-1, ::-1]`` — exactly what training
    saw.  Forgetting this composition is what makes the policy "look
    lost" with the action space wired up correctly.

    Pipeline: W-flip (``[:, ::-1]``) → **center-crop to a square** →
    ``resize_with_pad`` to ``image_size`` × ``image_size`` → uint8.

    The center-crop is the bridge between gap and openpi's reference: the
    π-series LIBERO checkpoints were trained on 256×256 *square* renders,
    then resized-with-pad to 224 (no letterboxing because the source was
    already square).  gap renders agentview / wrist at the env's native
    aspect ratio (e.g. 800×512), so feeding that straight into
    ``resize_with_pad`` would introduce ~40px of black bars top/bottom —
    a distribution shift the model has never seen.  Squaring first keeps
    the env render untouched and feeds the policy what it expects.
    """
    rotated = np.ascontiguousarray(rgb[:, ::-1])
    h, w = rotated.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    squared = rotated[top:top + side, left:left + side]
    try:
        from openpi_client import image_tools  # type: ignore[import-not-found]
        out = image_tools.resize_with_pad(squared, image_size, image_size)
        return image_tools.convert_to_uint8(out)
    except ImportError:
        # Fallback: simple resize via numpy.  Less faithful but keeps the
        # obs shape correct so the policy doesn't reject it.
        return _resize_with_pad_np(squared, image_size).astype(np.uint8)


def _resize_with_pad_np(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    # Nearest-neighbor — only used as a fallback when openpi-client is missing.
    ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
    xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
    resized = arr[np.ix_(ys, xs)]
    pad_h, pad_w = size - new_h, size - new_w
    top, left = pad_h // 2, pad_w // 2
    out = np.zeros((size, size, arr.shape[2]), dtype=arr.dtype)
    out[top:top + new_h, left:left + new_w] = resized
    return out


def _quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Robosuite-style axis-angle from XYZW quaternion.

    Mirrors ``openpi/examples/libero/main.py:_quat2axisangle`` which is in
    turn copied from robosuite. Returns the axis multiplied by the rotation
    angle (radians); zero rotation maps to a zero vector.
    """
    import math
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (q[:3] * 2.0 * math.acos(w)) / den


def _rgb_as_array(rgb: Any) -> np.ndarray | None:
    """Validate a :class:`gap.types.CameraFrame` ``rgb`` array as (H, W, 3).

    gap observations carry numpy arrays directly (no byte decode).
    Returns ``None`` for a missing or malformed frame so callers skip it.
    """
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] != 3:
        return None
    return arr


def _primary_rgb(obs: Any, cam_index: int) -> Any:
    """Return the RGB array for the selected VLM camera.

    Returns ``None`` if the observation has no cameras or the index is
    out of range — callers are expected to handle that by skipping the
    VLM termination check for the current window.
    """
    cameras = list(obs.get("cameras") or [])
    if not cameras:
        return None
    idx = cam_index if 0 <= cam_index < len(cameras) else 0
    return cameras[idx].get("rgb")


# ---------------------------------------------------------------------------
# Action chunk → Trajectory
# ---------------------------------------------------------------------------


def chunk_to_trajectory(chunk: np.ndarray) -> Trajectory:
    """Pack a ``(H, D)`` action-chunk array into a :class:`gap.types.Trajectory`.

    Each row becomes one ``JointState`` with ``positions = chunk[i]``.
    The user's checkpoint dictates what those ``D`` values mean — typically
    ``[joints..., gripper]`` in absolute form. No interpretation happens
    here. This is the payload shape ``robot.execute_trajectory`` expects —
    the execution path for trajectory-capable (joint-space) checkpoints.
    """
    if chunk.ndim != 2:
        raise ValueError(
            f"chunk_to_trajectory: expected 2-D (H, D) array, got shape "
            f"{chunk.shape}"
        )

    waypoints: list[JointState] = []
    for row in chunk:
        waypoints.append({"positions": np.asarray(row, dtype=np.float64)})
    return {"waypoints": waypoints}


# ---------------------------------------------------------------------------
# URL parsing (duplicated here to avoid cross-module import cycle in callers)
# ---------------------------------------------------------------------------


def _parse_ws_url(url: str) -> tuple[str, int]:
    stripped = url
    for scheme in ("ws://", "wss://", "http://", "https://"):
        if stripped.startswith(scheme):
            stripped = stripped[len(scheme):]
            break
    if "/" in stripped:
        stripped = stripped.split("/", 1)[0]
    if ":" not in stripped:
        raise ValueError(f"url {url!r}: expected host:port (missing port)")
    host, port_str = stripped.rsplit(":", 1)
    return host, int(port_str)


__all__ = [
    "PolicyExecutor",
    "encode_obs",
    "chunk_to_trajectory",
    "run_policy_loop",
]
