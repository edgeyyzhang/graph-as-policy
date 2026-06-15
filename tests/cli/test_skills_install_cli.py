"""CLI plumbing for ``gap skills install`` — the per-bundle venv sync verb.

The verb wraps `uv sync --project <bundle_dir>`. These tests mock
``subprocess.run`` so the install path is exercised end-to-end without
running ``uv`` (which would download model deps and take minutes).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gap.cli import main as cli_main


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["gap", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    return int(exc.value.code or 0)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_SKILLS_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _bundle(root: Path, kind_dir: str, name: str, *, pyproject: bool) -> Path:
    """Create a minimal bundle directory. When `pyproject=True`, the bundle
    owns its venv (gap skills install will `uv sync` it)."""
    d = root / kind_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture. Use when testing.\n"
        f"metadata: {{tags: [test]}}\n"
        f"gap:\n  tools:\n    - {name}.run: A fixture tool.\n"
        f"  exit_conditions: {{ok: done}}\n---\n"
    )
    if pyproject:
        (d / "pyproject.toml").write_text(
            f'[project]\nname = "{name}-bundle"\nversion = "0.0.0"\n'
            f'dependencies = []\n'
            f'[build-system]\nrequires = ["hatchling"]\n'
            f'build-backend = "hatchling.build"\n'
            f'[tool.hatch.build.targets.wheel]\nbypass-selection = true\n'
        )
    return d


_REAL_RUN = subprocess.run


class _UvSyncRecorder:
    """Capture `uv sync --project <dir>` invocations, pass other calls through."""

    def __init__(self, returncodes: list[int] | None = None):
        self.calls: list[list[str]] = []
        self._returncodes = list(returncodes or [])

    def __call__(self, cmd, *args, **kwargs):
        argv = list(cmd) if not isinstance(cmd, (str, bytes)) else [cmd]
        if len(argv) >= 2 and argv[0] == "uv" and argv[1] == "sync":
            self.calls.append(argv)
            rc = self._returncodes.pop(0) if self._returncodes else 0
            return subprocess.CompletedProcess(argv, rc)
        return _REAL_RUN(cmd, *args, **kwargs)


def test_install_named_bundle_runs_uv_sync(monkeypatch, tmp_path, capsys):
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "fake-policy", pyproject=True)

    rec = _UvSyncRecorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "install", "--skills", str(reg), "fake-policy"],
    )
    assert code == 0
    assert len(rec.calls) == 1
    bundle_dir = reg / "policies" / "fake-policy"
    assert rec.calls[0] == ["uv", "sync", "--project", str(bundle_dir)]
    out = capsys.readouterr().out
    assert "[fake-policy] OK" in out


def test_install_skips_bundles_without_pyproject(monkeypatch, tmp_path, capsys):
    """In-process bundles (no pyproject.toml) inherit gap's venv — install
    them by skipping with a note, not erroring."""
    reg = tmp_path / "reg"
    _bundle(reg, "skills", "no-deps-skill", pyproject=False)

    rec = _UvSyncRecorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "install", "--skills", str(reg), "no-deps-skill"],
    )
    assert code == 0
    assert rec.calls == []
    out = capsys.readouterr().out
    assert "[no-deps-skill] skipped" in out
    assert "no pyproject.toml" in out


def test_install_all_iterates_every_bundle_with_pyproject(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "with-deps-a", pyproject=True)
    _bundle(reg, "tools", "with-deps-b", pyproject=True)
    _bundle(reg, "skills", "no-deps-c", pyproject=False)

    rec = _UvSyncRecorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "install", "--skills", str(reg), "--all"],
    )
    assert code == 0
    # uv sync runs only for the two bundles with their own pyproject.
    synced_dirs = sorted(argv[3] for argv in rec.calls)
    assert synced_dirs == sorted([
        str(reg / "policies" / "with-deps-a"),
        str(reg / "tools" / "with-deps-b"),
    ])


def test_install_nothing_to_do_errors(monkeypatch, tmp_path, capsys):
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "x", pyproject=True)

    code = _run_cli(monkeypatch, ["skills", "install", "--skills", str(reg)])
    assert code == 2
    out = capsys.readouterr().out
    assert "nothing to install" in out


def test_install_unknown_bundle_errors(monkeypatch, tmp_path, capsys):
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "real", pyproject=True)

    code = _run_cli(
        monkeypatch,
        ["skills", "install", "--skills", str(reg), "ghost"],
    )
    assert code == 2
    out = capsys.readouterr().out
    assert "unknown bundle(s)" in out


def test_install_uv_sync_failure_propagates(monkeypatch, tmp_path):
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "fail-policy", pyproject=True)

    rec = _UvSyncRecorder(returncodes=[1])
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "install", "--skills", str(reg), "fail-policy"],
    )
    assert code == 1
    assert len(rec.calls) == 1


def test_check_reports_venv_state(monkeypatch, tmp_path, capsys):
    """`gap skills check` annotates each bundle with its venv-ready state
    so users see what `gap skills install` would have to do."""
    reg = tmp_path / "reg"
    _bundle(reg, "policies", "needs-install", pyproject=True)
    ready = _bundle(reg, "policies", "already-installed", pyproject=True)
    (ready / ".venv").mkdir()
    _bundle(reg, "skills", "no-pyproject", pyproject=False)

    code = _run_cli(
        monkeypatch,
        ["skills", "check", "--skills", str(reg)],
    )
    # The validation may FAIL on these synthetic bundles (e.g. allowed_tools
    # missing) — we only care about the venv-note column here.
    out = capsys.readouterr().out
    assert "(venv missing — run `gap skills install needs-install`)" in out
    assert "(venv-ready)" in out
    # Bundles without pyproject get no venv note.
    no_pyproject_line = next(
        line for line in out.splitlines() if "no-pyproject:" in line
    )
    assert "(venv" not in no_pyproject_line
    # Suppress unused warning on `code` — exit value isn't load-bearing here.
    _ = code
