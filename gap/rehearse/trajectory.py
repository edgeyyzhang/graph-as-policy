"""Record the privileged world state at every simulator step of a rehearsal.

Node boundaries only show where things ended up. The trajectory shows how
they got there: one sample after every simulator step, tagged with the node
that was running. A sample is the same state summary the rest of the
rehearsal uses (:func:`gap.rehearse.world_state.summarize`): robot
configuration plus every body's pose and contacts.

The sampler wraps the simulator handle's ``step`` for the length of a
rehearsal and only reads state, so the run itself is unchanged. Connectors
without a step handle still get the node-boundary samples.

One ``trajectory.jsonl`` per case, one JSON object per line::

    {"i": 12, "kind": "step", "node": "grasp_sg.descend", "visit": 0,
     "unit": "grasp_sg", "unit_visit": 0, "state": {...}}

``kind`` is ``initial`` / ``final`` (episode boundaries), ``start`` / ``end``
(node boundaries) or ``step`` (after a simulator step). ``node`` is null for
steps outside any node, such as end-node recovery actions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from .world_state import summarize

logger = logging.getLogger(__name__)


def find_step_handle(connector: Any) -> Any | None:
    """The object whose ``step`` advances the simulator, or ``None``."""
    handle = getattr(getattr(connector, "env", None), "handle", None)
    return handle if callable(getattr(handle, "step", None)) else None


class StepSampler:
    """Samples the world after every simulator step while attached."""

    def __init__(self, connector: Any, *, objects: list[str] | None = None):
        self.connector = connector
        self.objects = objects
        self._handle: Any = None
        self._original: Callable[..., Any] | None = None
        self._had_instance_step = False
        self._wrapper: Callable[..., Any] | None = None
        self._file: Any = None
        self._index = 0
        self._tag: dict[str, Any] = {"node": None, "visit": None, "unit": None, "unit_visit": None}
        self._node_steps = 0
        self.steps = 0
        self.errors = 0
        self._video: dict[str, Any] | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def steps_available(self) -> bool:
        return self._handle is not None

    def attach(self) -> bool:
        """Wrap the step handle. Returns False when the connector has none."""
        if self._handle is not None:
            return True
        handle = find_step_handle(self.connector)
        if handle is None:
            return False
        original = handle.step
        self._had_instance_step = "step" in getattr(handle, "__dict__", {})

        def step(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._node_steps += 1
            self.steps += 1
            self._grab_frame(result)
            self._write("step")
            return result

        try:
            handle.step = step
        except Exception:
            logger.debug("step handle cannot be wrapped", exc_info=True)
            return False
        self._handle, self._original, self._wrapper = handle, original, step
        return True

    def detach(self) -> None:
        """Restore the original step call."""
        handle = self._handle
        if handle is None:
            return
        try:
            if getattr(handle, "__dict__", {}).get("step") is self._wrapper:
                if self._had_instance_step:
                    handle.step = self._original
                else:
                    del handle.__dict__["step"]
        except Exception:
            logger.debug("step handle restore failed", exc_info=True)
        self._handle = self._original = self._wrapper = None

    def open(self, path: Path, *, video: Path | None = None, video_every: int = 5,
             step_hz: float = 20.0) -> None:
        """Start a new case file, and optionally a video of the exterior camera.

        With ``video`` set, the image the simulator returns from every
        ``video_every``-th step is kept, and ``close`` writes them as an MP4 at
        ``step_hz / video_every`` frames per second (real time) together with
        ``<video>.frames.json``: the sample index of each frame in the case's
        trajectory file, which is what ties a frame to its node.
        """
        self.close()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._index = 0
        self.steps = 0
        self._node_steps = 0
        self.clear_node()
        self._video = None
        self.video_path: Path | None = None
        if video is not None:
            self._video = {"path": Path(video), "every": max(1, int(video_every)),
                           "fps": float(step_hz) / max(1, int(video_every)), "frames": [], "samples": []}

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        video, self._video = self._video, None
        if video and video["frames"]:
            try:
                import imageio.v3 as iio

                path = video["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(path, np.stack(video["frames"]), fps=video["fps"], codec="libx264")
                path.with_name(path.name + ".frames.json").write_text(
                    json.dumps({"fps": video["fps"], "every": video["every"], "samples": video["samples"]}))
                self.video_path = path
            except Exception:
                self.errors += 1
                logger.warning("video write failed", exc_info=True)

    @property
    def video_written(self) -> Path | None:
        return getattr(self, "video_path", None)

    def _grab_frame(self, result: Any) -> None:
        """Keep the exterior camera image of a step that falls on the video interval."""
        video = self._video
        if not video or self.steps % video["every"] != 0:
            return
        obs = result[0] if isinstance(result, tuple) and result else result
        frame = frame_from_observation(obs)
        if frame is not None:
            video["shape"] = frame.shape[:2]
        else:
            # The connector switches the cameras off during motion segments
            # (GAP_LIBERO_MOTION_RENDER), so the step observation carries no
            # image then; render the exterior camera on demand instead.
            frame = render_exterior(self._handle, video.get("shape"))
        if frame is None:
            return
        video["frames"].append(frame)
        video["samples"].append(self._index)   # the index _write is about to use

    # -- tagging -----------------------------------------------------------

    def set_node(self, node: str, visit: int, unit: str | None, unit_visit: int | None) -> None:
        self._tag = {"node": node, "visit": visit, "unit": unit, "unit_visit": unit_visit}
        self._node_steps = 0

    def clear_node(self) -> None:
        self._tag = {"node": None, "visit": None, "unit": None, "unit_visit": None}

    @property
    def node_steps(self) -> int:
        """Simulator steps since the current node started."""
        return self._node_steps

    def mark(self, kind: str) -> None:
        """Write a boundary sample (``initial``, ``start``, ``end``, ``final``)."""
        self._write(kind)

    # -- sampling ----------------------------------------------------------

    def _write(self, kind: str) -> None:
        if self._file is None:
            return
        try:
            state = summarize(self.connector.world_snapshot(), self.objects)
            row = {"i": self._index, "kind": kind, **self._tag, "state": state}
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._index += 1
        except Exception:
            self.errors += 1
            logger.debug("trajectory sample failed", exc_info=True)


#: Observation keys tried, in order, for the exterior camera image.
IMAGE_KEYS = ("agentview_image", "agentview_rgb", "frontview_image", "rgb", "image")


def frame_from_observation(obs: Any) -> np.ndarray | None:
    """The exterior camera image of a step observation as an HxWx3 uint8 array.

    LIBERO's offscreen renderer returns images upside down; they are flipped
    to the orientation the connector's own frames use.
    """
    if not isinstance(obs, dict):
        return None
    img = None
    for key in IMAGE_KEYS:
        if key in obs:
            img = obs[key]
            break
    if img is None:
        cameras = obs.get("cameras")
        if cameras:
            cam = cameras[0]
            img = cam.get("rgb") if isinstance(cam, dict) else getattr(cam, "rgb", None)
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[::-1])


DEFAULT_CAMERA = "agentview"
DEFAULT_SHAPE = (512, 800)


def render_exterior(handle: Any, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Render the exterior camera of a LIBERO handle directly (about 2 ms)."""
    sim = getattr(getattr(handle, "env", None), "sim", None)
    if sim is None or not callable(getattr(sim, "render", None)):
        return None
    h, w = shape or DEFAULT_SHAPE
    try:
        img = sim.render(camera_name=DEFAULT_CAMERA, width=int(w), height=int(h))
    except Exception:
        logger.debug("exterior render failed", exc_info=True)
        return None
    arr = np.asarray(img)
    if arr.ndim != 3:
        return None
    return np.ascontiguousarray(arr[::-1, :, :3].astype(np.uint8))


def read(path: Path) -> Iterator[dict[str, Any]]:
    """Samples of one case, in order. Malformed lines are skipped."""
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


__all__ = ["StepSampler", "find_step_handle", "read"]
