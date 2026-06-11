"""Shared fakes for the real-robot suite.

``FakeRRClient`` is a plain-socket stand-in for the robots_realtime
client: it speaks the exact wire protocol from
``third_party/robots_realtime/robots_realtime/utils/server_client_utils.py``
— 4-byte big-endian length framing around msgpack(+numpy) payloads, one
action reply per observation request, replies decoded with ``raw=True``
(byte-string keys), exactly like ``MsgpackNumpyClient``.

No hardware anywhere here: everything binds loopback ephemeral ports.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import pytest

msgpack_numpy.patch()


def free_port() -> int:
    """An OS-assigned free TCP port (small race window; fine for tests)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeRRClient:
    """Synchronous msgpack-numpy client speaking the rr wire protocol."""

    def __init__(self, port: int, host: str = "127.0.0.1", connect_timeout: float = 10.0):
        deadline = time.monotonic() + connect_timeout
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError as e:  # server thread may not be listening yet
                last_err = e
                time.sleep(0.05)
        else:
            raise TimeoutError(f"could not connect to msgpack server: {last_err}")

    def send_request(self, obs: dict) -> dict:
        """One exchange: framed observation out, framed action back."""
        payload = msgpack.packb(obs, use_bin_type=True)
        self.sock.sendall(struct.pack("!I", len(payload)) + payload)
        header = self._recv_exact(4)
        (msg_len,) = struct.unpack("!I", header)
        return msgpack.unpackb(self._recv_exact(msg_len), raw=True)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server closed connection")
            buf += chunk
        return buf

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def joints_frame(joints7=None, gripper: float = 1.0, timestamp: float | None = None) -> dict:
    """A joints-only wire observation (camera rate-limited away this tick)."""
    joints7 = np.asarray(
        joints7 if joints7 is not None else [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        dtype=np.float32,
    )
    return {
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "left": {
            "joint_pos": np.concatenate([joints7, [np.float32(gripper)]]),
        },
    }


def full_frame(
    joints7=None,
    gripper: float = 1.0,
    cam_key: str = "camera_top",
    h: int = 8,
    w: int = 8,
    timestamp: float | None = None,
) -> dict:
    """A full wire observation: joints + camera rgb/depth/intrinsics/pose_mat."""
    frame = joints_frame(joints7, gripper, timestamp)
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, 3] = [0.1, 0.2, 0.3]
    frame[cam_key] = {
        "images": {"left_rgb": np.full((h, w, 3), 7, dtype=np.uint8)},
        "depth_data": np.full((h, w), 0.5, dtype=np.float32),
        "intrinsics": {
            "left": {"intrinsics_matrix": np.eye(3, dtype=np.float32) * 100.0},
        },
        "pose_mat": pose_mat,
    }
    return frame


@pytest.fixture
def franka_env():
    """A FrankaRealEnv bound to an ephemeral loopback port (fast heartbeat)."""
    from gap.envs.franka_real_env import FrankaRealEnv

    port = free_port()
    env = FrankaRealEnv(
        camera_names=["cam0"], port=port,
        heartbeat_period=0.05, stale_after_s=0.3,
    )
    env._test_port = port  # convenience for tests
    yield env
    env.close()


class ClientFeeder:
    """Background thread that keeps sending full frames at ~30 Hz."""

    def __init__(self, port: int, **frame_kwargs):
        self.port = port
        self.frame_kwargs = frame_kwargs
        self.replies: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            client = FakeRRClient(self.port)
        except TimeoutError:
            return
        try:
            while not self._stop.is_set():
                self.replies.append(client.send_request(full_frame(**self.frame_kwargs)))
                time.sleep(1 / 30)
        except (ConnectionError, OSError):
            pass
        finally:
            client.close()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
