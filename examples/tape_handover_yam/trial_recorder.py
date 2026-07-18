"""Opt-in trial recorder for run.py / benchmark.py: joint trajectories plus
wrist (left/right D405) and front (agentview) camera feeds.

Not a training data format itself (e.g. LeRobot) — just raw-enough capture
that a later converter (see openpi/scripts/dataset_conversion/yam_tsh_to_lerobot.py
for the shape of such a script) has everything it needs: the actual 14D
commanded action per step (not just observed state, which is what a BC/VLA
policy is trained to predict), and the exact step index each sparse camera
frame lines up with (since cameras are strided for render-cost reasons but
actions are logged every control step).

Isolated from the driver scripts on purpose — it only touches the sim via
``env.on_step`` and public ``get_observation()``/``model``/``data``/``_cmd``,
so wiring it into a driver is a few lines behind a flag with no effect on
default runs:

    recorder = TrialRecorder(env) if record_extra else None
    def on_step(e):
        collision(e)
        grab(e)
        if recorder is not None:
            recorder.on_step(e)
    ...
    if recorder is not None:
        recorder.save(out_dir)
        recorder.close()
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def _encode_verified(path: Path, frames: list, fps: float, attempts: int = 4) -> None:
    """Encode ``frames`` to an mp4 at ``path``, verifying the written frame count.

    Under concurrent offscreen rendering the imageio->ffmpeg raw-video pipe can
    occasionally misalign a frame boundary, so ffmpeg drops/aborts and the mp4
    ends up short or truncated (a silent data-integrity loss — the trial still
    "succeeds"). Frames are already fully in memory here, so we re-encode until
    ffprobe reports the expected frame count, then give up with a loud warning.
    """
    n = len(frames)
    for attempt in range(1, attempts + 1):
        with imageio.get_writer(str(path), fps=fps, codec="h264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        got = _probe_frame_count(path)
        if got == n:
            return
        print(
            f"    [recorder] {path.name}: wrote {got}/{n} frames "
            f"(attempt {attempt}/{attempts}), re-encoding",
            flush=True,
        )
    print(f"    [recorder] WARNING: {path.name} still {got}/{n} frames after "
          f"{attempts} attempts — leaving best effort", flush=True)


def _probe_frame_count(path: Path) -> int:
    """Exact decodable frame count via ffprobe, or -1 if it can't be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        return int(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return -1

# Matches the free-camera renderers in run.py (480x640) so feeds are directly
# comparable; agentview/left/right are named cameras baked into the scene XML
# (libero_yam_tabletop_base_style.xml), not MjvCamera free cameras.
# GAP_CAM_H/GAP_CAM_W override the recorded resolution — lower res renders faster
# and is plenty for training an ACT-style policy (which downsamples anyway).
_CAM_H = int(os.environ.get("GAP_CAM_H", "480"))
_CAM_W = int(os.environ.get("GAP_CAM_W", "640"))
_TRAJ_KEYS = (
    "robot_joint_pos_0", "robot_joint_pos_1",
    "robot_cartesian_pos_0", "robot_cartesian_pos_1",
)


class TrialRecorder:
    """Per-step joint/cartesian/action trajectories + strided wrist/front camera frames."""

    def __init__(self, env, *, camera_stride: int = 1, task_prompt: str = ""):
        self._camera_stride = camera_stride
        self._task_prompt = task_prompt
        self._step_count = 0
        self._traj: dict[str, list] = {k: [] for k in _TRAJ_KEYS}
        self._actions: list = []
        self._camera_step_indices: list = []
        self._front_frames: list = []
        self._left_frames: list = []
        self._right_frames: list = []
        self._renderers: dict[str, mujoco.Renderer] = {}

    def _render(self, env, camera: str):
        """Render one frame from a named camera via a per-camera renderer
        created lazily on first use and reused for the rest of the trial.

        Measured ~2.6s per create-then-close cycle (EGL context churn, not
        the render itself) — with camera_stride=1 over a ~1900-step trial
        that's ~10min of pure overhead. Safe to hold open because
        ``perceive_tape``/``perceive_duct`` (the ones ``camera_frames()``'s
        EGL_BAD_ACCESS warning is about) run once, early, before any
        stepping — this renderer isn't created until the first capture,
        which only happens camera_stride steps into pickup, well after
        perception's own renderers have already closed.
        """
        r = self._renderers.get(camera)
        if r is None:
            r = mujoco.Renderer(env.model, _CAM_H, _CAM_W)
            self._renderers[camera] = r
        r.update_scene(env.data, camera=camera)
        return r.render().copy()

    def on_step(self, env) -> None:
        self._step_count += 1
        obs = env.get_observation()
        for key in _TRAJ_KEYS:
            self._traj[key].append(np.asarray(obs[key], dtype=np.float32))
        # The 14D command actually applied this step (env.step's `action` arg,
        # or whatever a blocking helper like move_to_joints_blocking staged) —
        # distinct from the observed state above, and what a BC/VLA policy is
        # trained to predict.
        self._actions.append(np.asarray(env._cmd, dtype=np.float32).copy())
        # GAP_RECORDER_NO_CAMERAS: log the full-rate trajectory but skip the
        # (expensive) offscreen camera renders — for fast timing/phase analysis
        # runs where only action/joint data is needed.
        if os.environ.get("GAP_RECORDER_NO_CAMERAS"):
            return
        if self._step_count % self._camera_stride != 0:
            return
        # 1-indexed to match len(self._traj[...]) after this step's append above,
        # so camera_step_indices[i] is the row in trajectory.npz this frame pairs with.
        self._camera_step_indices.append(self._step_count)
        self._front_frames.append(self._render(env, "agentview"))
        self._left_frames.append(self._render(env, "left"))
        self._right_frames.append(self._render(env, "right"))

    def save(self, out_dir: Path) -> None:
        """Write trajectory.npz + front.mp4/left_wrist.mp4/right_wrist.mp4 into out_dir.

        trajectory.npz's ``action`` and ``robot_*`` arrays are one row per
        control step (every step, no stride). ``camera_step_indices[i]`` gives
        the 1-indexed step number that camera frame ``i`` (same index into
        front.mp4/left_wrist.mp4/right_wrist.mp4) was captured at, since the
        camera feeds are strided (``camera_stride``) but the trajectory isn't —
        a converter joins on that index rather than assuming 1:1 alignment.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "trajectory.npz",
            action=np.stack(self._actions) if self._actions else np.empty((0, 14)),
            camera_step_indices=np.asarray(self._camera_step_indices, dtype=np.int64),
            task_prompt=np.asarray(self._task_prompt),
            **{k: (np.stack(v) if v else np.empty((0,))) for k, v in self._traj.items()},
        )
        for name, frames in (
            ("front.mp4", self._front_frames),
            ("left_wrist.mp4", self._left_frames),
            ("right_wrist.mp4", self._right_frames),
        ):
            if not frames:
                continue
            # Frames are captured every ``_camera_stride`` control steps at 30 Hz,
            # so encode at 30/stride for real-time playback (encoding strided
            # frames at 30 fps is what made clips look ~stride-x sped up). The
            # full-rate trajectory.npz + camera_step_indices are unaffected, so a
            # LeRobot converter still reconstructs true 30 Hz timing.
            fps = 30 / self._camera_stride
            _encode_verified(out_dir / name, frames, fps)

    def close(self) -> None:
        """Release the per-camera renderers cached lazily by ``_render``."""
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
