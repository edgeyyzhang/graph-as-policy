"""Tests for gap.runtime.policy_manager + policy_presets — no real policy server.

Managed entries are exercised with a fake ``python -c`` TCP server so the
``{port}`` substitution, env overrides, readiness preflight, and eviction
teardown all run against a real subprocess without any model weights.
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
from gap.runtime.policy_presets import PRESETS, resolve_policies

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
FAKE_SERVER_CMD = f'{sys.executable} -c "{_FAKE_SERVER_CODE}"'

EXIT_EARLY_CMD = f"{sys.executable} -c 'import sys; sys.exit(3)'"


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


def test_entry_with_url_and_start_cmd_rejected():
    mgr = PolicyManager(
        entries={"both": {"url": "ws://h:1", "start_cmd": "x --port {port}"}},
    )
    with pytest.raises(PolicyConfigError, match="not both"):
        mgr.boot_all({"both"})


def test_entry_with_neither_url_nor_start_cmd_rejected():
    mgr = PolicyManager(entries={"empty": {}})
    with pytest.raises(PolicyConfigError, match="must declare"):
        mgr.boot_all({"empty"})


def test_url_for_before_boot_raises():
    mgr = PolicyManager(entries={"ext": {"url": "ws://127.0.0.1:8123"}})
    with pytest.raises(PolicyConfigError, match="not loaded"):
        mgr.url_for("ext")


# ---------------------------------------------------------------------------
# Managed (start_cmd:) entries
# ---------------------------------------------------------------------------


def test_managed_spawns_with_port_substitution_and_tears_down_on_evict():
    mgr = PolicyManager(
        entries={
            "srv": {"start_cmd": FAKE_SERVER_CMD, "env": {"GAP_TEST_SLEEP": "60"}},
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
        entries={"bad": {"start_cmd": EXIT_EARLY_CMD}},
        startup_timeout_s=10.0,
    )
    with pytest.raises(PolicyStartupError, match="exited early"):
        mgr.boot_all({"bad"})
    assert mgr.loaded_ids() == []


def test_managed_env_must_be_mapping():
    mgr = PolicyManager(
        entries={"bad": {"start_cmd": FAKE_SERVER_CMD, "env": "GAP_TEST_SLEEP=60"}},
    )
    with pytest.raises((PolicyConfigError, PolicyStartupError), match="mapping"):
        mgr.boot_all({"bad"})


def test_boot_all_with_no_required_ids_is_noop():
    mgr = PolicyManager(entries={})
    mgr.boot_all(set())
    assert mgr.loaded_ids() == []


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_presets_shape_is_minimal_and_factual():
    assert set(PRESETS) == {"pi05-libero", "molmoact-libero"}
    for preset in PRESETS.values():
        assert set(preset) == {"checkpoint_uri", "start_cmd", "env", "notes"}
        assert "--port {port}" in preset["start_cmd"]
        assert preset["env"] == {}

    pi = PRESETS["pi05-libero"]
    assert pi["checkpoint_uri"] == "s3://openpi-assets/checkpoints/pi05_libero"
    assert "serve_policy.py" in pi["start_cmd"]

    molmo = PRESETS["molmoact-libero"]
    assert "$GAP_OPENPI_DIR" in molmo["start_cmd"]
    assert "GAP_OPENPI_DIR" in molmo["notes"]


def test_preset_expansion_produces_managed_entry():
    resolved = resolve_policies({"pi": {"preset": "pi05-libero"}})
    entry = resolved["pi"]
    assert "url" not in entry and "preset" not in entry
    assert entry["start_cmd"] == PRESETS["pi05-libero"]["start_cmd"]
    assert "{port}" in entry["start_cmd"]
    assert entry["env"] == {}


def test_preset_entry_env_overrides_merge():
    resolved = resolve_policies(
        {"pi": {"preset": "pi05-libero", "env": {"JAX_PLATFORMS": "cuda"}}},
    )
    assert resolved["pi"]["env"] == {"JAX_PLATFORMS": "cuda"}
    # The shared preset table is not mutated.
    assert PRESETS["pi05-libero"]["env"] == {}


def test_non_preset_entries_pass_through_unchanged():
    entries = {
        "ext": {"url": "ws://127.0.0.1:8123"},
        "managed": {"start_cmd": "serve --port {port}"},
    }
    assert resolve_policies(entries) == entries
    assert resolve_policies(None) == {}


def test_unknown_preset_fails():
    with pytest.raises(PolicyConfigError, match="unknown preset"):
        resolve_policies({"pi": {"preset": "pi99-nowhere"}})


def test_preset_with_explicit_start_cmd_rejected():
    with pytest.raises(PolicyConfigError, match="not both"):
        resolve_policies(
            {"pi": {"preset": "pi05-libero", "start_cmd": "x --port {port}"}},
        )


def test_policy_manager_resolves_presets_at_construction():
    mgr = PolicyManager(entries={"pi": {"preset": "pi05-libero"}})
    entry = mgr._entries["pi"]
    assert entry["start_cmd"] == PRESETS["pi05-libero"]["start_cmd"]
    assert "preset" not in entry
