"""Tests for gap.runtime.observation_stream — polling, timeouts, handle tracing."""

from __future__ import annotations

import threading
import time

import pytest

from gap_core.errors import StreamUnavailable
from gap.runtime.observation_stream import (
    ObservationStream,
    ObservationStreamHandle,
    start_observation_stream,
)
from gap.runtime.tracing import DagTrace


class CountingPoller:
    """poll_fn stub returning a fresh observation per call."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            return {"cameras": [], "arms": [], "n": self.calls}


class StubCtx:
    """NodeContext stand-in: forwards _stream_read into a DagTrace like the real one."""

    def __init__(self, trace: DagTrace, node_id: str):
        self._trace = trace
        self._node_id = node_id
        self._stream_seq = 0
        self.reads: list[tuple[int, str, object, float]] = []

    def _stream_read(self, name, value, sampled_at):
        seq = self._stream_seq
        self._stream_seq += 1
        self.reads.append((seq, name, value, sampled_at))
        self._trace.record_stream_read(self._node_id, seq, name, value, sampled_at)


def test_poll_fn_called_and_latest_returns_freshest():
    poller = CountingPoller()
    stream = ObservationStream(poller, period_s=0.01)
    stream.start()
    try:
        first = stream.latest(timeout=2.0)
        assert first["n"] >= 1
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if stream.latest(timeout=2.0)["n"] > first["n"]:
                break
            time.sleep(0.01)
        latest = stream.latest(timeout=2.0)
        assert latest["n"] > first["n"]
        assert poller.calls >= latest["n"]
        assert stream.last_error is None
    finally:
        stream.stop()


def test_latest_raises_stream_unavailable_when_poll_always_fails():
    def failing_poll():
        raise RuntimeError("connector down")

    stream = ObservationStream(failing_poll, period_s=0.01)
    stream.start()
    try:
        with pytest.raises(StreamUnavailable) as excinfo:
            stream.latest(timeout=0.3)
        assert isinstance(excinfo.value.cause, RuntimeError)
    finally:
        stream.stop()


def test_latest_returns_stale_value_after_polls_start_failing():
    state = {"calls": 0}

    def flaky_poll():
        state["calls"] += 1
        if state["calls"] > 1:
            raise RuntimeError("lost connection")
        return {"cameras": [], "arms": [], "n": 1}

    stream = ObservationStream(flaky_poll, period_s=0.01)
    stream.start()
    try:
        assert stream.latest(timeout=2.0)["n"] == 1
        deadline = time.monotonic() + 2.0
        while state["calls"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        # Stale-but-good: cached value still served after failures.
        assert stream.latest(timeout=0.5)["n"] == 1
    finally:
        stream.stop()


def test_stop_joins_thread_and_is_idempotent():
    stream = ObservationStream(CountingPoller(), period_s=0.01)
    stream.start()
    thread = stream._thread
    assert thread is not None and thread.is_alive()
    stream.stop()
    assert not thread.is_alive()
    assert stream._thread is None
    stream.stop()  # idempotent


def test_start_is_idempotent():
    stream = ObservationStream(CountingPoller(), period_s=0.01)
    stream.start()
    try:
        thread = stream._thread
        stream.start()
        assert stream._thread is thread
    finally:
        stream.stop()


def test_handle_records_reads_into_trace(tmp_path):
    trace = DagTrace(tmp_path)
    trace.add_node("servo", {"type": "script", "script": "servo.py"})
    ctx = StubCtx(trace, "servo")

    poller = CountingPoller()
    stream = ObservationStream(poller, period_s=0.01)
    stream.start()
    try:
        handle = ObservationStreamHandle(stream, ctx, name="observation_stream")
        assert handle.name == "observation_stream"

        value = handle.latest(timeout=2.0)
        value2, sampled_at = handle.latest_with_age(timeout=2.0)
        assert value["n"] >= 1
        assert value2["n"] >= value["n"]
        assert sampled_at > 0

        assert [(seq, name) for seq, name, _, _ in ctx.reads] == [
            (0, "observation_stream"),
            (1, "observation_stream"),
        ]
        reads_dir = tmp_path / "node_data" / "servo" / "stream_reads"
        assert sorted(d.name for d in reads_dir.iterdir()) == [
            "000_observation_stream",
            "001_observation_stream",
        ]
    finally:
        stream.stop()


def test_start_observation_stream_factory():
    poller = CountingPoller()
    stream = start_observation_stream(poller, hz=50.0)
    try:
        assert stream.period_s == pytest.approx(0.02)
        assert stream.latest(timeout=2.0)["n"] >= 1
    finally:
        stream.stop()


def test_start_observation_stream_factory_default_hz():
    stream = start_observation_stream(CountingPoller())
    try:
        assert stream.period_s == pytest.approx(0.1)
    finally:
        stream.stop()
