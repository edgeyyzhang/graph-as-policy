"""RRSession lifecycle on a fake script (no robots_realtime, no uv).

The fake session script mimics rr-session's process shape: it forks a
nested child (rr-session forks robot/camera node processes) and loops.
The launcher must tee its output to the log file and kill the whole
process group — including the grandchild — on terminate().
"""

from __future__ import annotations

import os
import signal
import sys
import time

from gap.connector.rr_launcher import DEFAULT_RR_DIR, RRSession

_FAKE_SESSION = """\
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(f"CHILD_PID={child.pid}", flush=True)
print("session started", flush=True)
__EXTRA__
while True:
    time.sleep(0.1)
"""


def _spawn(tmp_path, extra: str = ""):
    script = tmp_path / "fake_rr.py"
    script.write_text(_FAKE_SESSION.replace("__EXTRA__", extra))
    log = tmp_path / "rr.log"
    sess = RRSession(
        "configs/fake.yaml",
        log_path=log,
        command=[sys.executable, "-u", str(script)],
    )
    return sess, log


def _wait_for_log(log, needle: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log.exists() and needle in log.read_text():
            return log.read_text()
        time.sleep(0.05)
    raise AssertionError(f"{needle!r} never appeared in {log}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_spawn_log_tee_and_group_kill(tmp_path):
    sess, log = _spawn(tmp_path)
    try:
        text = _wait_for_log(log, "session started")
        assert sess.poll() is None, "fake session exited prematurely"

        child_pid = int(text.split("CHILD_PID=")[1].splitlines()[0])
        assert _pid_alive(child_pid), "nested child should be running"
        # The child runs in the launcher's own process group (setsid).
        assert os.getpgid(child_pid) == sess.pid

        sess.terminate(grace_s=2.0)
        assert sess.poll() is not None

        # The grandchild dies with the group — not just the direct child.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _pid_alive(child_pid):
            time.sleep(0.05)
        assert not _pid_alive(child_pid), "nested child survived group kill"
    finally:
        sess.terminate(grace_s=0.5)


def test_sigterm_ignoring_child_gets_sigkilled(tmp_path):
    extra = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('TERM_IGNORED', flush=True)"
    sess, log = _spawn(tmp_path, extra=extra)
    try:
        _wait_for_log(log, "TERM_IGNORED")
        t0 = time.monotonic()
        sess.terminate(grace_s=0.5)
        assert sess.poll() is not None
        assert time.monotonic() - t0 < 10.0
        # SIGKILL'ed processes report -SIGKILL.
        assert sess.poll() == -signal.SIGKILL
    finally:
        sess.terminate(grace_s=0.5)


def test_context_manager_terminates(tmp_path):
    with _spawn(tmp_path)[0] as sess:
        _wait_for_log(sess.log_path, "session started")
        pid = sess.pid
        assert _pid_alive(pid)
    assert sess.poll() is not None


def test_terminate_idempotent(tmp_path):
    sess, log = _spawn(tmp_path)
    _wait_for_log(log, "session started")
    sess.terminate(grace_s=1.0)
    sess.terminate(grace_s=1.0)  # second call is a no-op
    assert sess.poll() is not None


def test_default_command_shape():
    """Without the command override, RRSession builds the uv invocation
    rooted at the vendored submodule (never importing robots_realtime)."""
    assert DEFAULT_RR_DIR.name == "robots_realtime"
    assert (DEFAULT_RR_DIR / "pyproject.toml").exists()
    # The default Franka client config ships in the pinned checkout.
    from gap.connector.real import DEFAULT_FRANKA_RR_CONFIG

    assert (DEFAULT_RR_DIR / DEFAULT_FRANKA_RR_CONFIG).exists()
