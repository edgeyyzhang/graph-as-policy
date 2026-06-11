"""Graph-scoped observation stream — background-polled live observation source.

A single ``ObservationStream`` is owned by ``WorkflowExecutor`` for the
duration of one workflow execution. A daemon thread calls a plain ``poll_fn``
(the connector's ``get_observation``) at a fixed period (default 10 Hz) and
caches the latest :class:`~gap.types.Observation`. Skills opt in by declaring
an ``observation_stream: ObservationStream`` parameter and calling
``.latest()`` to read the freshest snapshot — no connector change, no
streaming transport.

The stream handle (``ObservationStreamHandle``) is a per-state wrapper that
forwards ``.latest()`` to the underlying stream and records each consumed
value into the trace, so replay is deterministic.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import StreamUnavailable

if TYPE_CHECKING:
    from ..types import Observation
    from .context import NodeContext

logger = logging.getLogger(__name__)

_DEFAULT_HZ = 10.0
_DEFAULT_PERIOD_S = 0.1
_BACKOFF_INITIAL_S = 0.1
_BACKOFF_MAX_S = 1.0


class ObservationStream:
    """Background-polled observation source.

    The polling thread runs ``poll_fn()`` at a fixed period and caches the
    latest observation under a single lock. ``.latest()`` returns the cached
    value or raises ``StreamUnavailable`` if the first poll has not completed
    within the caller's timeout.

    Lifetime is graph-scoped: the ``WorkflowExecutor`` calls ``start()`` once
    at ``execute()`` entry and ``stop()`` once in the ``finally`` block. The
    thread is a daemon so it does not block process exit.
    """

    def __init__(
        self,
        poll_fn: Callable[[], Observation],
        *,
        period_s: float = _DEFAULT_PERIOD_S,
    ):
        self._poll_fn = poll_fn
        self._period = max(1e-3, float(period_s))
        self._lock = threading.Lock()
        self._latest: tuple[Observation, float] | None = None
        self._err: Exception | None = None
        self._stop = threading.Event()
        self._first_poll = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def period_s(self) -> float:
        return self._period

    @property
    def last_error(self) -> Exception | None:
        return self._err

    def start(self) -> None:
        """Spawn the polling thread. Idempotent: re-start is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._first_poll.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ObservationStream",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the polling thread to exit and join. Idempotent; never raises."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=timeout)
            except Exception:
                logger.debug("ObservationStream join failed", exc_info=True)
        self._thread = None

    def latest(self, timeout: float = 2.0) -> Observation:
        """Return the most-recent :class:`~gap.types.Observation`.

        Blocks up to ``timeout`` seconds for the first poll to complete.
        Raises ``StreamUnavailable`` if the timeout elapses without a
        successful first poll. After the first successful poll, returns
        immediately with whatever was last cached (even if subsequent polls
        have failed — stale-but-good is preferable to no data).
        """
        if not self._first_poll.wait(timeout=max(0.0, float(timeout))):
            raise StreamUnavailable(self._err)
        with self._lock:
            if self._latest is None:
                raise StreamUnavailable(self._err)
            return self._latest[0]

    def latest_with_age(self, timeout: float = 2.0) -> tuple[Observation, float]:
        """Like ``latest()`` but also returns the monotonic sampled-at timestamp."""
        if not self._first_poll.wait(timeout=max(0.0, float(timeout))):
            raise StreamUnavailable(self._err)
        with self._lock:
            if self._latest is None:
                raise StreamUnavailable(self._err)
            return self._latest

    def _run(self) -> None:
        backoff = _BACKOFF_INITIAL_S
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                obs = self._poll_fn()
                with self._lock:
                    self._latest = (obs, t0)
                    self._err = None
                if not self._first_poll.is_set():
                    self._first_poll.set()
                backoff = _BACKOFF_INITIAL_S
            except Exception as e:
                self._err = e
                logger.debug("ObservationStream poll failed: %s", e)
                # Retain previous good value (if any) and back off.
                if self._stop.wait(timeout=backoff):
                    return
                backoff = min(_BACKOFF_MAX_S, backoff * 2)
                continue

            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._period - elapsed)
            if sleep_for > 0 and self._stop.wait(timeout=sleep_for):
                return


class ObservationStreamHandle:
    """Per-state wrapper around an ``ObservationStream``.

    Skills receive a handle (not the raw stream) so each ``.latest()`` call
    is recorded in the per-node trace via ``ctx._stream_read``. This makes
    the consumed snapshots replayable.
    """

    def __init__(
        self,
        stream: ObservationStream,
        ctx: NodeContext,
        name: str,
    ):
        self._stream = stream
        self._ctx = ctx
        self._name = name

    def latest(self, timeout: float = 2.0) -> Observation:
        value, sampled_at = self._stream.latest_with_age(timeout=timeout)
        self._ctx._stream_read(self._name, value, sampled_at)
        return value

    def latest_with_age(self, timeout: float = 2.0) -> tuple[Observation, float]:
        value, sampled_at = self._stream.latest_with_age(timeout=timeout)
        self._ctx._stream_read(self._name, value, sampled_at)
        return value, sampled_at

    @property
    def name(self) -> str:
        return self._name


def start_observation_stream(
    poll_fn: Callable[[], Observation],
    *,
    hz: float = _DEFAULT_HZ,
) -> ObservationStream:
    """Construct and start an ``ObservationStream`` polling at ``hz``."""
    period_s = 1.0 / hz if hz > 0 else _DEFAULT_PERIOD_S
    stream = ObservationStream(poll_fn, period_s=period_s)
    stream.start()
    return stream
