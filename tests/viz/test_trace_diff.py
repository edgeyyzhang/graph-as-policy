"""trace_diff tests — two real traces of the same workflow, one diverging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.runtime.trace_diff import (
    diff_trace_dirs,
    format_summary,
    to_json_dict,
)

from .test_trial_api import _run_trial


def _boom(x: int) -> dict:
    raise RuntimeError("sensor drift")


@pytest.fixture()
def trace_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Same workflow executed twice; the second run's compute node fails."""
    root = tmp_path / "outputs"
    good = _run_trial(root, "rehearsal")
    bad = _run_trial(root, "real", compute=_boom)
    return good, bad


def test_identical_traces_agree(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    a = _run_trial(root, "a")
    b = _run_trial(root, "b")

    diff = diff_trace_dirs(a, b)
    assert diff.matched_count > 0
    assert diff.verdict_agreement_rate == 1.0
    assert diff.status_agreement_rate == 1.0
    assert diff.first_divergence is None
    assert diff.rehearsal_only == []
    assert diff.real_only == []


def test_diverging_node_reported(trace_pair: tuple[Path, Path]) -> None:
    good, bad = trace_pair
    diff = diff_trace_dirs(good, bad)

    assert diff.matched_count > 0
    assert diff.verdict_agreement_rate < 1.0

    fd = diff.first_divergence
    assert fd is not None
    assert fd.name == "work_sg.compute"
    assert fd.rehearsal_status == "ok"
    assert fd.real_status == "error"
    assert not fd.verdict_agree and not fd.status_agree


def test_accepts_file_path_and_dir(trace_pair: tuple[Path, Path]) -> None:
    good, bad = trace_pair
    via_dir = diff_trace_dirs(good, bad)
    via_file = diff_trace_dirs(good / "dag_trace.json", bad / "dag_trace.json")
    assert via_dir.matched_count == via_file.matched_count
    assert via_dir.verdict_agreement_rate == via_file.verdict_agreement_rate


def test_json_and_summary_renderings(trace_pair: tuple[Path, Path]) -> None:
    good, bad = trace_pair
    diff = diff_trace_dirs(good, bad)

    payload = to_json_dict(diff)
    json.dumps(payload)  # must be serializable
    assert payload["first_divergence"]["name"] == "work_sg.compute"
    assert 0.0 < payload["verdict_agreement_rate"] < 1.0

    summary = format_summary(diff)
    assert "work_sg.compute" in summary
    assert "DIFFER" in summary
    assert "first divergence: work_sg.compute" in summary


def test_cli_trace_diff_exit_codes(trace_pair: tuple[Path, Path], capsys) -> None:
    """`gap trace-diff A B` returns 0 on full agreement, 1 on divergence."""
    import argparse

    from gap.cli.trace_diff import register

    good, bad = trace_pair

    def run(a: Path, b: Path, out: Path | None = None) -> int:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        register(sub)
        argv = ["trace-diff", str(a), str(b)]
        if out is not None:
            argv += ["--out", str(out)]
        args = parser.parse_args(argv)
        return args.func(args)

    assert run(good, good) == 0
    out_json = good.parent / "diff.json"
    assert run(good, bad, out=out_json) == 1
    assert json.loads(out_json.read_text())["first_divergence"]["name"] == "work_sg.compute"
    assert "DIFFER" in capsys.readouterr().out
