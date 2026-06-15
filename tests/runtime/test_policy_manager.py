"""Tests for gap.runtime.policy_manager — no real policy server.

Managed entries are exercised with a fake ``python -c`` TCP server so the
``{port}`` substitution, env overrides, readiness preflight, and eviction
teardown all run against a real subprocess without any model weights.

The new entry shape is ``command: list[str]`` (NOT a shell string).
``bundle_dir`` is optional — when set, the manager prepends
``uv run --project <bundle_dir> --`` so the bundle's own venv activates;
when omitted, the command spawns directly (used for tests and any
self-contained user-authored entries).
"""

from __future__ import annotations

import sys

import pytest

from gap.runtime import policy_manager as pm
from gap.runtime.policy_manager import (
    PolicyConfigError,
    PolicyManager,
    PolicyStartupError,
)

# Fake policy server: binds the substituted {port} and idles. Reads its
# lifetime from GAP_TEST_SLEEP with no default, so the managed entry's
# `env:` overrides are load-bearing (a missing override crashes it).
_FAKE_SERVER_CODE = (
    "import os, socket, time; "
    "s = socket.socket(); "
    "s.bind(('127.0.0.1', {port})); "
    "s.listen(5); "
    "time.sleep(float(os.environ['GAP_TEST_SLEEP']))"
)
FAKE_SERVER_CMD = [sys.executable, "-c", _FAKE_SERVER_CODE]

EXIT_EARLY_CMD = [sys.executable, "-c", "import sys; sys.exit(3)"]


# ---------------------------------------------------------------------------
# External (url:) entries
# ---------------------------------------------------------------------------


def test_external_url_passes_preflight_without_socket_probe(monkeypatch):
    probes = []
    monkeypatch.setattr(
        pm, "_tcp_ready", lambda *a, **k: probes.append(a) or True,
    )
    mgr = PolicyManager(entries={"ext": {"url": "ws://127.0.0.1:8123"}})
    mgr.boot_all({"ext"})

    assert mgr.url_for("ext") == "ws://127.0.0.1:8123"
    assert mgr.loaded_ids() == ["ext"]
    assert mgr._managed == {}  # no subprocess spawned
    assert probes == []  # external entries only get URL-shape validation

    # shutdown_all never touches externals: the URL survives.
    mgr.shutdown_all()
    assert mgr.loaded_ids() == ["ext"]


def test_external_url_malformed_fails_preflight():
    mgr = PolicyManager(entries={"ext": {"url": "ws://127.0.0.1"}})
    with pytest.raises(PolicyConfigError, match="missing port"):
        mgr.boot_all({"ext"})

    mgr = PolicyManager(entries={"ext": {"url": "ws://127.0.0.1:nope"}})
    with pytest.raises(PolicyConfigError, match="not an integer"):
        mgr.boot_all({"ext"})


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------


def test_unknown_policy_id_fails_preflight():
    mgr = PolicyManager(entries={})
    with pytest.raises(PolicyConfigError, match="'ghost'"):
        mgr.boot_all({"ghost"})


def test_entry_with_url_and_command_rejected():
    mgr = PolicyManager(
        entries={"both": {"url": "ws://h:1", "command": ["x", "--port", "{port}"]}},
    )
    with pytest.raises(PolicyConfigError, match="not both"):
        mgr.boot_all({"both"})


def test_entry_with_neither_url_nor_command_rejected():
    mgr = PolicyManager(entries={"empty": {}})
    with pytest.raises(PolicyConfigError, match="must declare"):
        mgr.boot_all({"empty"})


def test_url_for_before_boot_raises():
    mgr = PolicyManager(entries={"ext": {"url": "ws://127.0.0.1:8123"}})
    with pytest.raises(PolicyConfigError, match="not loaded"):
        mgr.url_for("ext")


# ---------------------------------------------------------------------------
# Managed (command:) entries
# ---------------------------------------------------------------------------


def test_managed_spawns_with_port_substitution_and_tears_down_on_evict():
    mgr = PolicyManager(
        entries={
            "srv": {
                "command": FAKE_SERVER_CMD,
                "env": {"GAP_TEST_SLEEP": "60"},
            },
        },
        startup_timeout_s=30.0,
        evict_grace_s=5.0,
    )
    mgr.boot_all({"srv"})
    try:
        url = mgr.url_for("srv")
        host, port = pm._parse_ws_url(url)
        assert host == "127.0.0.1"
        # The fake server really is listening on the substituted port.
        assert pm._tcp_ready(host, port, timeout=2.0)
        proc = mgr._managed["srv"].proc
        assert proc.poll() is None
        assert mgr._managed["srv"].port == port
    finally:
        mgr.shutdown_all()

    assert proc.poll() is not None  # evicted: process group terminated
    assert mgr.loaded_ids() == []
    mgr.shutdown_all()  # idempotent


def test_managed_subprocess_early_exit_fails_boot():
    mgr = PolicyManager(
        entries={"bad": {"command": EXIT_EARLY_CMD}},
        startup_timeout_s=10.0,
    )
    with pytest.raises(PolicyStartupError, match="exited early"):
        mgr.boot_all({"bad"})
    assert mgr.loaded_ids() == []


def test_managed_env_must_be_mapping():
    mgr = PolicyManager(
        entries={"bad": {"command": FAKE_SERVER_CMD, "env": "GAP_TEST_SLEEP=60"}},
    )
    with pytest.raises((PolicyConfigError, PolicyStartupError), match="mapping"):
        mgr.boot_all({"bad"})


def test_managed_command_must_be_list(tmp_path):
    mgr = PolicyManager(
        entries={"bad": {"command": "python -c 'pass' --port {port}"}},
    )
    with pytest.raises((PolicyConfigError, PolicyStartupError), match="non-empty list"):
        mgr.boot_all({"bad"})


def test_managed_command_must_be_non_empty():
    mgr = PolicyManager(entries={"bad": {"command": []}})
    with pytest.raises((PolicyConfigError, PolicyStartupError), match="non-empty list"):
        mgr.boot_all({"bad"})


def test_boot_all_with_no_required_ids_is_noop():
    mgr = PolicyManager(entries={})
    mgr.boot_all(set())
    assert mgr.loaded_ids() == []


# ---------------------------------------------------------------------------
# bundle_dir wiring → `uv run --project` prefix
# ---------------------------------------------------------------------------


def test_bundle_dir_prepends_uv_run_project(monkeypatch, tmp_path):
    """A bundle_dir entry wraps the command in ``uv run --project ... --``."""
    spawn_calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self):
            self.pid = 99999
            self._exited = False

        def poll(self):
            return 0 if self._exited else None

        def wait(self, timeout=None):
            self._exited = True
            return 0

    def fake_popen(argv, **kwargs):
        spawn_calls.append(list(argv))
        # Fake an immediately-listening server: skip the readiness probe.
        monkeypatch.setattr(pm, "_tcp_ready", lambda *a, **k: True)
        return _FakeProc()

    monkeypatch.setattr(pm.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pm, "_allocate_free_port", lambda: 47000)

    bundle = tmp_path / "policies" / "fake-policy"
    bundle.mkdir(parents=True)
    mgr = PolicyManager(
        entries={
            "fake": {
                "command": ["python", "-m", "fake.server", "--port", "{port}"],
                "bundle_dir": bundle,
            },
        },
        startup_timeout_s=2.0,
    )
    mgr.boot_all({"fake"})

    assert len(spawn_calls) == 1
    argv = spawn_calls[0]
    # uv run --project <bundle_dir> -- <command…> with {port} substituted.
    assert argv == [
        "uv", "run", "--project", str(bundle), "--",
        "python", "-m", "fake.server", "--port", "47000",
    ]
    # shell=False is non-negotiable — no env-var expansion in argv.


def test_no_bundle_dir_spawns_command_directly(monkeypatch, tmp_path):
    """Without bundle_dir, the command is spawned as-is (no uv run prefix)."""
    spawn_calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self):
            self.pid = 99998

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        spawn_calls.append(list(argv))
        monkeypatch.setattr(pm, "_tcp_ready", lambda *a, **k: True)
        return _FakeProc()

    monkeypatch.setattr(pm.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pm, "_allocate_free_port", lambda: 48000)

    mgr = PolicyManager(
        entries={
            "raw": {
                "command": ["/abs/path/serve", "--port", "{port}"],
            },
        },
        startup_timeout_s=2.0,
    )
    mgr.boot_all({"raw"})

    assert spawn_calls == [["/abs/path/serve", "--port", "48000"]]


def test_spawn_uses_shell_false(monkeypatch, tmp_path):
    """The new spawn path NEVER uses shell=True — no env-var expansion."""
    captured: dict = {}

    class _FakeProc:
        def __init__(self):
            self.pid = 99997

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["shell"] = kwargs.get("shell", False)
        monkeypatch.setattr(pm, "_tcp_ready", lambda *a, **k: True)
        return _FakeProc()

    monkeypatch.setattr(pm.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pm, "_allocate_free_port", lambda: 49000)

    mgr = PolicyManager(
        entries={"x": {"command": ["python", "--version"]}},
        startup_timeout_s=2.0,
    )
    mgr.boot_all({"x"})

    assert captured["shell"] is False
    assert captured["argv"] == ["python", "--version"]
