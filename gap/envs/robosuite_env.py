"""Thin GAP environment surface over an already-constructed robosuite env.

The benchmark owns scene construction and task semantics.  This adapter owns
only the simulator lifecycle expected by :class:`gap.connector.SimConnector`:
reset / OSC stepping / structured robot observations / video.  It contains no
Kitchen, Coffee, object, or robot-name policy.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np


class RobosuiteEnvAdapter:
    """Adapt a robosuite-compatible environment to GAP's pinned env surface."""

    def __init__(
        self,
        env: Any,
        *,
        task_completed: Callable[[Any], bool] | None = None,
        camera_names: list[str] | None = None,
        max_steps: int | None = None,
        gripper_metric_length: float = 0.04,
        video_stride: int = 1,
    ) -> None:
        self.env = env
        self.camera_names = list(camera_names or [])
        self.max_steps = int(max_steps or getattr(env, "horizon", 1000))
        self.gripper_metric_length = float(gripper_metric_length)
        self.video_stride = max(1, int(video_stride))
        self._task_completed_fn = task_completed
        self._current_obs: dict[str, Any] = {}
        self._current_reward = 0.0
        self._current_done = False
        self._current_info: dict[str, Any] = {}
        self._sim_step_count = 0
        self._gripper_fraction = 1.0
        self._record_frames = False
        self._video_frames: list[np.ndarray] = []
        self._step_observers: list[
            Callable[[np.ndarray, dict[str, Any], float, bool], None]
        ] = []

    @property
    def raw_env(self) -> Any:
        """The benchmark environment, for native success and diagnostics."""
        return self.env

    @property
    def raw_observation(self) -> dict[str, Any]:
        """Latest unmodified robosuite observation."""
        return self._current_obs

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        # Robosuite reset consumes numpy's global RNG and does not consistently
        # accept a seed keyword across the benchmark versions GAP supports.
        if seed is not None:
            np.random.seed(int(seed))
            random.seed(int(seed))
        self._current_obs = dict(self.env.reset())
        self._current_reward = 0.0
        self._current_done = False
        self._current_info = {}
        self._sim_step_count = 0
        qpos = np.asarray(self._current_obs.get("robot0_gripper_qpos", [0.04]))
        if qpos.size:
            self._gripper_fraction = float(
                np.clip(abs(float(qpos.flat[0])) / self.gripper_metric_length, 0.0, 1.0)
            )
        if self._record_frames:
            self._video_frames.clear()
            self._capture_frame()
        return self.get_observation(), {}

    def _base_transform(self) -> tuple[np.ndarray, np.ndarray]:
        """Return world-to-robot-base translation and rotation."""
        model, data = self.env.sim.model, self.env.sim.data
        bid = None
        for name in ("robot0_base", "robot0_link0"):
            try:
                bid = int(model.body_name2id(name))
                break
            except Exception:
                continue
        if bid is None:
            return np.zeros(3), np.eye(3)
        from scipy.spatial.transform import Rotation

        base_t = np.asarray(data.body_xpos[bid], dtype=np.float64)
        base_q = np.asarray(data.body_xquat[bid], dtype=np.float64)  # wxyz
        base_Rt = Rotation.from_quat(np.roll(base_q, -1)).as_matrix().T
        return base_t, base_Rt

    def _eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """LIBERO-compatible public TCP pose in the robot-base frame."""
        from scipy.spatial.transform import Rotation

        pos = np.asarray(self._current_obs["robot0_eef_pos"], dtype=np.float64)
        quat_xyzw = np.asarray(self._current_obs["robot0_eef_quat"], dtype=np.float64)
        R = Rotation.from_quat(quat_xyzw).as_matrix()
        # Match FrankaLiberoEnv: robosuite's eef site -> GAP hand/TCP frame.
        tcp_pos = pos + R @ np.array([0.0, 0.0, -0.107])
        tcp_R = R @ Rotation.from_euler("z", np.pi / 2.0).as_matrix()
        base_t, base_Rt = self._base_transform()
        tcp_pos = base_Rt @ (tcp_pos - base_t)
        tcp_R = base_Rt @ tcp_R
        xyzw = Rotation.from_matrix(tcp_R).as_quat()
        return tcp_pos, np.roll(xyzw, 1)  # wxyz

    def get_observation(self) -> dict[str, Any]:
        """Return the same structured boundary used by every GAP connector."""
        from scipy.spatial.transform import Rotation

        obs: dict[str, Any] = {}
        pos, quat_wxyz = self._eef_pose()
        joints = np.asarray(
            self._current_obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float64
        )
        qpos = np.asarray(
            self._current_obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float64
        )
        obs["robot_joint_pos_0"] = np.concatenate([joints, [self._gripper_fraction]])
        obs["robot_cartesian_pos_0"] = np.concatenate(
            [pos, quat_wxyz, [self._gripper_fraction]]
        )
        obs["robot_proprio_pi05_libero_0"] = np.concatenate(
            [
                np.asarray(self._current_obs.get("robot0_eef_pos", np.zeros(3))),
                Rotation.from_quat(
                    np.asarray(
                        self._current_obs.get(
                            "robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]
                        )
                    )
                ).as_rotvec(),
                np.pad(qpos[:2], (0, max(0, 2 - qpos[:2].size))),
            ]
        )
        if self._record_frames:
            for name in self.camera_names:
                frame = self._render(name)
                obs[name] = {"images": {"rgb": frame}}
        return obs

    def _advance(self, action: np.ndarray) -> None:
        if self._sim_step_count >= self.max_steps:
            raise RuntimeError(
                f"benchmark horizon exhausted after {self.max_steps} control steps"
            )
        result = self.env.step(np.asarray(action, dtype=np.float64))
        self._current_obs = dict(result[0])
        self._current_reward = float(result[1])
        self._current_info = dict(result[3] or {}) if len(result) >= 4 else {}
        self._sim_step_count += 1
        self._gripper_fraction = float(np.clip(0.5 - 0.5 * action[-1], 0.0, 1.0))
        self._current_done = self.task_completed()
        for observer in tuple(self._step_observers):
            observer(
                action.copy(),
                self._current_obs,
                self._current_reward,
                self._current_done,
            )
        if self._record_frames and self._sim_step_count % self.video_stride == 0:
            self._capture_frame()

    def add_step_observer(
        self,
        observer: Callable[[np.ndarray, dict[str, Any], float, bool], None],
    ) -> None:
        """Observe raw benchmark steps without changing graph execution."""
        self._step_observers.append(observer)

    def apply_policy_action(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (7,):
            raise ValueError(f"OSC_POSE action must have shape (7,), got {value.shape}")
        if not np.all(np.isfinite(value)) or np.any(np.abs(value) > 1.0):
            raise ValueError("OSC_POSE action must be finite and inside [-1, 1]")
        self._advance(value)

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        if arm_id != 0:
            raise ValueError("this robosuite adapter currently exposes one arm")
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        action = np.zeros(7, dtype=np.float64)
        action[-1] = 1.0 - 2.0 * self._gripper_fraction
        self._advance(action)

    def task_completed(self) -> bool:
        if self._task_completed_fn is not None:
            return bool(self._task_completed_fn(self.env))
        checker = getattr(self.env, "_check_success", None)
        return bool(checker()) if callable(checker) else self._current_done

    def compute_reward(self) -> float:
        return float(self._current_reward)

    def get_current_time_s(self) -> float:
        hz = float(getattr(self.env, "control_freq", 20.0) or 20.0)
        return self._sim_step_count / hz

    def get_latency_info(self) -> dict[str, Any]:
        return {
            "control_steps": self._sim_step_count,
            "control_freq": float(getattr(self.env, "control_freq", 20.0) or 20.0),
        }

    def _render(self, camera: str) -> np.ndarray:
        height_value = getattr(self.env, "camera_heights", 256)
        width_value = getattr(self.env, "camera_widths", 256)
        height = int(height_value if np.isscalar(height_value) else np.asarray(height_value).flat[0])
        width = int(width_value if np.isscalar(width_value) else np.asarray(width_value).flat[0])
        return self.env.sim.render(
            width=width, height=height, camera_name=camera
        )[::-1].copy()

    def _capture_frame(self) -> None:
        if self.camera_names:
            self._video_frames.append(self._render(self.camera_names[0]))

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = bool(enabled)
        if clear:
            self._video_frames.clear()
        if enabled and self._current_obs:
            self._capture_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        result = list(self._video_frames)
        if clear:
            self._video_frames.clear()
        return result

    def close(self) -> None:
        self.env.close()


__all__ = ["RobosuiteEnvAdapter"]
