"""Grid loop + report wiring + gate/resume semantics, no GPU/LLM.

Stubs the one engine seam (``modes.score_suite`` for grid mode,
``gap.agent.launcher.launch`` for suites mode) so the harness logic runs
in CI.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import gap.benchmark.modes as modes_mod
from gap.agent.config import PipelineConfig, SuiteSpec, TrialConfig
from gap.agent.launcher import LaunchResult, SuiteLaunchResult, TaskResult
from gap.benchmark.config import BenchmarkConfig
from gap.benchmark.harness import run_benchmark


def _grid_cfg(tmp_path, **kw) -> BenchmarkConfig:
    defaults = dict(
        source_yaml=Path("x.yaml"),
        families=["posvar"],
        variations=["pos_var"],
        modes=["llm_generation"],
        task_ids=[0, 1, 2],
        n_seeds=2,
        num_workers=1,
        output_dir=tmp_path,
    )
    defaults.update(kw)
    cfg = BenchmarkConfig(**defaults)
    cfg.pipeline_config = PipelineConfig()
    return cfg


def _fake_score(calls, success_for=lambda t: t == 0):
    async def fake(*, suite_name, task_ids, n_seeds, workflow_dir_map=None,
                   record_video=False, **_kw):
        calls.append({
            "suite": suite_name,
            "task_ids": list(task_ids),
            "wf_map": (None if workflow_dir_map is None
                       else sorted(workflow_dir_map)),
            "record_video": record_video,
        })
        trs = [
            TaskResult(
                task_id=t,
                success_count=(n_seeds if success_for(t) else 0),
                total_trials=n_seeds,
                success_rate=(1.0 if success_for(t) else 0.0),
                completion_rate=(1.0 if success_for(t) else 0.0),
                avg_reward=0.0,
                trial_results=[],
            )
            for t in task_ids
        ]
        return trs, 1.0
    return fake


# --------------------------------------------------------------------------
# Grid mode: one score_suite per cell, wf-map shape, summary wiring
# --------------------------------------------------------------------------


def test_grid_offline(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(modes_mod, "score_suite", _fake_score(calls))
    # Keep the template mode CI-safe: no LIBERO metadata import.
    import gap.benchmark.workflow_materialize as wm_mod

    monkeypatch.setattr(
        wm_mod, "resolve_task_prompt",
        lambda suite, tid: "pick up the milk and place it in the basket",
    )

    template = tmp_path / "template"
    template.mkdir()
    (template / "workflow.json").write_text(json.dumps(
        {"version": 3, "meta": {"policy": "{{policy_id}}"},
         "nodes": {}, "edges": [], "conditional_edges": {}, "subgraphs": {}},
    ))
    from gap.benchmark.config import BenchmarkModeOverride

    cfg = _grid_cfg(
        tmp_path,
        modes=["llm_generation", "policy_only"],
        record_video=True,
        mode_overrides={
            "policy_only": BenchmarkModeOverride(workflow_dir=str(template)),
        },
    )
    summary = asyncio.run(run_benchmark(cfg))

    # One score_suite per scored mode-cell.
    assert len(calls) == 2
    for c in calls:
        assert c["task_ids"] == [0, 1, 2]
        assert c["record_video"] is True
    wf_shapes = {tuple(c["wf_map"]) if c["wf_map"] else None for c in calls}
    # llm_generation -> None ; policy_only -> per-task map {0,1,2}
    assert None in wf_shapes
    assert (0, 1, 2) in wf_shapes

    mat = summary.summary["matrix"]
    col = "posvar/pos_var"
    assert mat["llm_generation"][col]["error"] is None
    assert mat["llm_generation"][col]["n_trials"] == 3 * cfg.n_seeds
    # policy axis off + no explicit policy_id -> plain mode row key.
    assert mat["policy_only"][col]["error"] is None

    run_dir = summary.run_dir
    assert run_dir is not None and run_dir.parent == tmp_path
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.tsv").is_file()
    assert (run_dir / "aaa_done_flag.txt").is_file()
    loaded = json.loads((run_dir / "summary.json").read_text())
    assert loaded["grid"]["families"] == ["posvar"]
    assert loaded["grid"]["task_ids"] == [0, 1, 2]

    # Pooled metrics: 6 trials/mode-cell, task 0 fully passing.
    assert summary.n_trials == 12
    assert summary.n_success == 4
    assert summary.success_rate == pytest.approx(4 / 12)
    assert summary.ok is True  # not gated


def test_template_mode_without_workflow_dir_is_error_cell(
    tmp_path, monkeypatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(modes_mod, "score_suite", _fake_score(calls))
    cfg = _grid_cfg(tmp_path, modes=["llm_plus_policy"])
    summary = asyncio.run(run_benchmark(cfg))
    assert calls == []  # produce failed before scoring
    [cell] = summary.cells
    assert cell.error and "workflow_dir" in cell.error


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


def test_gate_pass_and_fail(tmp_path, monkeypatch) -> None:
    # All tasks pass -> 100% >= 0.90 -> ok.
    monkeypatch.setattr(
        modes_mod, "score_suite", _fake_score([], success_for=lambda t: True),
    )
    cfg = _grid_cfg(tmp_path / "pass")
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.gated is True
    assert summary.success_rate == 1.0
    assert summary.ok is True

    # 1/3 tasks pass -> 33% < 0.90 -> gate fails.
    monkeypatch.setattr(modes_mod, "score_suite", _fake_score([]))
    cfg = _grid_cfg(tmp_path / "fail")
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.ok is False
    assert summary.gate_threshold == pytest.approx(0.90)

    # Same numbers, lower bar -> passes.
    cfg = _grid_cfg(tmp_path / "low", gate_threshold=0.30)
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.ok is True


def test_gate_fails_on_error_cell_or_zero_trials(tmp_path, monkeypatch) -> None:
    # Error cell (template mode without workflow_dir) -> gate fails even
    # though the scored cells are perfect.
    monkeypatch.setattr(
        modes_mod, "score_suite", _fake_score([], success_for=lambda t: True),
    )
    cfg = _grid_cfg(
        tmp_path / "err", modes=["llm_generation", "llm_plus_policy"],
    )
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.success_rate == 1.0
    assert summary.ok is False

    # Zero trials -> gate fails.
    async def empty_score(**_kw):
        return [], 0.0
    monkeypatch.setattr(modes_mod, "score_suite", empty_score)
    cfg = _grid_cfg(tmp_path / "zero")
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.n_trials == 0
    assert summary.ok is False


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_resume_skips_completed_cells_and_rebuilds_summary(
    tmp_path, monkeypatch,
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        modes_mod, "score_suite",
        _fake_score(calls, success_for=lambda t: True),
    )
    cfg = _grid_cfg(tmp_path, variations=["pos_var", "all"])
    first = asyncio.run(run_benchmark(cfg))
    assert len(calls) == 2  # two cells scored
    run_dir = first.run_dir
    assert (run_dir / "llm_generation" / "posvar" / "pos_var"
            / "cell_result.json").is_file()

    # Simulate an interrupted run: delete one cell's result.
    (run_dir / "llm_generation" / "posvar" / "all" / "cell_result.json").unlink()

    calls.clear()
    cfg2 = _grid_cfg(tmp_path, variations=["pos_var", "all"])
    second = asyncio.run(run_benchmark(cfg2, resume=True))

    # Same run dir reused; only the missing cell re-scored.
    assert second.run_dir == run_dir
    assert len(calls) == 1
    assert calls[0]["suite"] == "libero_object_all_variance"

    # Merged summary covers BOTH cells (old + new).
    assert len(second.cells) == 2
    assert second.n_trials == first.n_trials
    rebuilt = json.loads((run_dir / "summary.json").read_text())
    assert len(rebuilt["cells"]) == 2


def test_no_resume_creates_fresh_run_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(modes_mod, "score_suite", _fake_score([]))
    import time

    first = asyncio.run(run_benchmark(_grid_cfg(tmp_path)))
    time.sleep(1.1)  # run dirs are second-resolution timestamps
    second = asyncio.run(run_benchmark(_grid_cfg(tmp_path)))
    assert first.run_dir != second.run_dir


# --------------------------------------------------------------------------
# Suites mode (grocery acceptance shape)
# --------------------------------------------------------------------------


def _suites_cfg(tmp_path) -> BenchmarkConfig:
    pc = PipelineConfig(
        task="auto",
        suites=[
            SuiteSpec(suite_name="dup", task_ids=[0], task_prompts={0: "a"}),
            SuiteSpec(suite_name="dup", task_ids=[1], task_prompts={1: "b"}),
        ],
        trials=TrialConfig(
            trials_per_generation=2, num_workers=1, output_dir=tmp_path,
        ),
    )
    cfg = BenchmarkConfig(
        source_yaml=Path("x.yaml"),
        suites_mode=True,
        output_dir=tmp_path,
        gate_threshold=0.90,
    )
    cfg.pipeline_config = pc
    return cfg


def _fake_launch(calls, success: bool = True):
    async def fake(config, **_kw):
        suite = config.suites[0]
        calls.append({
            "suite": suite.suite_name,
            "task_ids": suite.task_ids,
            "output_dir": str(config.trials.output_dir),
        })
        n = config.trials.trials_per_generation
        succ = n if success else 0
        tr = TaskResult(
            task_id=suite.task_ids[0],
            success_count=succ,
            total_trials=n,
            success_rate=1.0 if success else 0.0,
            completion_rate=1.0 if success else 0.0,
        )
        sr = SuiteLaunchResult(
            suite_name=suite.suite_name,
            task_results=[tr],
            success_rate=tr.success_rate,
            total_trials=n,
        )
        return LaunchResult(
            suite_results=[sr],
            task_results=[tr],
            success_rate=tr.success_rate,
            total_trials=n,
        )
    return fake


def test_suites_mode_one_cell_per_suite(tmp_path, monkeypatch) -> None:
    import gap.agent.launcher as launcher_mod

    calls: list[dict] = []
    monkeypatch.setattr(launcher_mod, "launch", _fake_launch(calls))

    cfg = _suites_cfg(tmp_path)
    summary = asyncio.run(run_benchmark(cfg, gate=True))

    # One single-suite launch per cell; dup names disambiguated.
    assert len(calls) == 2
    assert {c["task_ids"][0] for c in calls} == {0, 1}
    dirs = {Path(c["output_dir"]).name for c in calls}
    assert dirs == {"dup", "dup_02"}

    assert summary.n_trials == 4
    assert summary.success_rate == 1.0
    assert summary.ok is True
    run_dir = summary.run_dir
    assert (run_dir / "suites" / "dup" / "cell_result.json").is_file()
    assert (run_dir / "suites" / "dup_02" / "cell_result.json").is_file()
    assert (run_dir / "summary.tsv").is_file()


def test_suites_mode_gate_fail_and_resume(tmp_path, monkeypatch) -> None:
    import gap.agent.launcher as launcher_mod

    calls: list[dict] = []
    monkeypatch.setattr(
        launcher_mod, "launch", _fake_launch(calls, success=False),
    )
    cfg = _suites_cfg(tmp_path)
    summary = asyncio.run(run_benchmark(cfg, gate=True))
    assert summary.success_rate == 0.0
    assert summary.ok is False

    # Resume: both cells already persisted -> no launches re-run.
    calls.clear()
    cfg2 = _suites_cfg(tmp_path)
    resumed = asyncio.run(run_benchmark(cfg2, resume=True))
    assert calls == []
    assert resumed.run_dir == summary.run_dir
    assert resumed.n_trials == summary.n_trials


# --------------------------------------------------------------------------
# Video collation
# --------------------------------------------------------------------------


def test_collect_videos_into_run_dir(tmp_path) -> None:
    """Videos collate into ``<run>/videos/`` — the run's own output tree,
    co-located with summary.tsv — never an ad-hoc external folder."""
    from gap.benchmark.harness import _collect_videos

    run = tmp_path / "20260101_000000"
    trials = [
        ("llm_generation/posvar/pos_var/task_00",
         "trial_01_rc0_reward1.000_pass"),
        ("llm_generation/posvar/pos_var/task_01",
         "trial_01_rc1_reward0.000_fail"),
        ("suites/dup/task_00", "trial_02_rc0_reward1.000_pass"),
    ]
    for parent, trial in trials:
        d = run / parent / trial
        d.mkdir(parents=True)
        (d / "video.mp4").write_bytes(b"\x00mp4")

    _collect_videos(run)

    vids = sorted(p.name for p in (run / "videos").glob("*.mp4"))
    assert vids == [
        "llm_generation__posvar__pos_var__task_00__trial_01__pass.mp4",
        "llm_generation__posvar__pos_var__task_01__trial_01__fail.mp4",
        "suites__dup__task_00__trial_02__pass.mp4",
    ]
    # content preserved (hard-link or copy)
    assert (run / "videos" / vids[0]).read_bytes() == b"\x00mp4"
