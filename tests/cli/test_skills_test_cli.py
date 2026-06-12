"""CLI plumbing for ``gap skills test`` and the ``gap skills new`` scaffolds."""

from __future__ import annotations

import subprocess
import sys
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


def _make_registry(root: Path, bundles: tuple[str, ...], *, tests: bool = True) -> Path:
    for name in bundles:
        d = root / "tools" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Fixture. Use when testing.\n"
            f"gap:\n  tools:\n    - {name}.run: A fixture tool.\n---\n"
        )
    if tests:
        (root / "tests").mkdir()
    return root


_REAL_RUN = subprocess.run


class _Recorder:
    """Intercept the pytest invocations; pass anything else through
    (stdlib helpers like platform.* shell out via subprocess.run too)."""

    def __init__(self, returncodes: list[int] | None = None):
        self.calls: list[tuple[list[str], Path]] = []
        self._returncodes = list(returncodes or [])

    def __call__(self, cmd, *args, cwd=None, **kwargs):
        argv = list(cmd) if not isinstance(cmd, (str, bytes)) else [cmd]
        if len(argv) >= 3 and argv[1:3] == ["-m", "pytest"]:
            self.calls.append((argv, Path(cwd)))
            rc = self._returncodes.pop(0) if self._returncodes else 0
            return subprocess.CompletedProcess(argv, rc)
        return _REAL_RUN(cmd, *args, cwd=cwd, **kwargs)


def test_exact_file_targeting_and_fallback(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "reg", ("rgt-st-filed", "rgt-st-loose"))
    (reg / "tests" / "test_rgt_st_filed.py").write_text("def test_ok(): pass\n")

    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "test", "--skills", str(reg), "rgt-st-filed", "rgt-st-loose"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "falling back to -k 'rgt_st_loose'" in out

    assert len(rec.calls) == 2
    file_cmd, file_cwd = rec.calls[0]
    assert file_cmd[:3] == [sys.executable, "-m", "pytest"]
    assert file_cmd[3:] == ["tests/test_rgt_st_filed.py"]
    assert file_cwd == reg
    k_cmd, _ = rec.calls[1]
    assert k_cmd[3:] == ["tests", "-k", "rgt_st_loose"]


def test_no_args_runs_every_registry(monkeypatch, capsys, tmp_path):
    a = _make_registry(tmp_path / "reg-a", ("rgt-st-a",))
    b = _make_registry(tmp_path / "reg-b", ("rgt-st-b",))
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch, ["skills", "test", "--skills", str(a), "--skills", str(b)],
    )
    assert code == 0
    assert [c[1] for c in rec.calls] == [a, b]
    assert all(c[0][3:] == ["tests"] for c in rec.calls)


def test_passthrough_after_double_dash(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "reg", ("rgt-st-pass",))
    (reg / "tests" / "test_rgt_st_pass.py").write_text("def test_ok(): pass\n")
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    code = _run_cli(
        monkeypatch,
        ["skills", "test", "--skills", str(reg), "rgt-st-pass", "--", "-m", "gpu", "-x"],
    )
    assert code == 0
    cmd, _ = rec.calls[0]
    assert cmd[3:] == ["tests/test_rgt_st_pass.py", "-m", "gpu", "-x"]


def test_exit_code_mapping(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "reg", ("rgt-st-fail",))
    (reg / "tests" / "test_rgt_st_fail.py").write_text("def test_no(): assert False\n")

    monkeypatch.setattr(subprocess, "run", _Recorder([1]))
    assert _run_cli(
        monkeypatch, ["skills", "test", "--skills", str(reg), "rgt-st-fail"],
    ) == 1

    # pytest exit 5 ("no tests collected"): error when bundles were named...
    monkeypatch.setattr(subprocess, "run", _Recorder([5]))
    assert _run_cli(
        monkeypatch, ["skills", "test", "--skills", str(reg), "rgt-st-fail"],
    ) == 1
    # ...but only a note in run-everything mode.
    monkeypatch.setattr(subprocess, "run", _Recorder([5]))
    assert _run_cli(monkeypatch, ["skills", "test", "--skills", str(reg)]) == 0


def test_unknown_bundle_and_missing_tests_dir(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "reg", ("rgt-st-known",), tests=False)
    assert _run_cli(
        monkeypatch, ["skills", "test", "--skills", str(reg), "nope"],
    ) == 2
    assert "unknown bundle(s) nope" in capsys.readouterr().out

    # Explicitly requested bundle in a registry without tests/ -> failure.
    assert _run_cli(
        monkeypatch, ["skills", "test", "--skills", str(reg), "rgt-st-known"],
    ) == 1
    assert "no tests/ directory" in capsys.readouterr().out

    # Run-everything mode just skips it.
    assert _run_cli(monkeypatch, ["skills", "test", "--skills", str(reg)]) == 0


# ---------------------------------------------------------------------------
# Scaffold round trip: registry init -> skills new -> generated tests pass
# ---------------------------------------------------------------------------


def test_scaffold_round_trip_runs_real_pytest(monkeypatch, capsys, tmp_path):
    reg = tmp_path / "scratch-skills"
    assert _run_cli(monkeypatch, ["registry", "init", str(reg)]) == 0

    assert _run_cli(
        monkeypatch,
        ["skills", "new", "rgt-waver", "--kind", "skill", "--skills", str(reg)],
    ) == 0
    out = capsys.readouterr().out
    assert "tests/test_rgt_waver.py" in out
    assert (reg / "tests" / "test_rgt_waver.py").is_file()
    assert (reg / "tests" / "conftest.py").is_file()

    assert _run_cli(
        monkeypatch,
        ["skills", "new", "rgt-wavetool", "--kind", "tool", "--skills", str(reg)],
    ) == 0
    capsys.readouterr()

    # Scaffolds pass the format check...
    assert _run_cli(monkeypatch, ["skills", "check", "--skills", str(reg)]) == 0
    capsys.readouterr()

    # ...and their generated unit tests pass under a real pytest run.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=reg, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
