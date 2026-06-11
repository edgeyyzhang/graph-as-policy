"""Pure (no-GPU, CI-safe) tests for benchmark result normalization."""

from __future__ import annotations

import csv
import json

from gap.agent.launcher import TaskResult
from gap.agent.parallel import TrialResult
from gap.benchmark.report import (
    ModeResult,
    build_summary,
    error_cell,
    merge_cells,
    normalize_task_results,
)


def _task(
    task_id: int, n: int, succ: int, reward: float,
    completion: float | None = None,
) -> TaskResult:
    trials = [
        TrialResult(trial_id=i, task_id=task_id, task_completed=(i < succ),
                    reward=reward)
        for i in range(n)
    ]
    return TaskResult(
        task_id=task_id,
        success_count=succ,
        total_trials=n,
        success_rate=succ / n if n else 0.0,
        completion_rate=(
            completion if completion is not None
            else (succ / n if n else 0.0)
        ),
        avg_reward=reward,
        trial_results=trials,
    )


def test_normalize_pools_trial_weighted() -> None:
    # task0: 2/4 @ r=1.0 ; task1: 1/2 @ r=0.0
    trs = [_task(0, 4, 2, 1.0), _task(1, 2, 1, 0.0)]
    mr = normalize_task_results(
        mode="llm_generation",
        variation="pos_var",
        suite_name="libero_object_target_pos_var20x20",
        task_results=trs,
        produce_wall_s=10.0,
        eval_wall_s=30.0,
    )
    assert mr.n_trials == 6
    assert mr.n_success == 3
    assert abs(mr.success_rate - 0.5) < 1e-9
    # trial-weighted reward: (1.0*4 + 0.0*2)/6
    assert abs(mr.avg_reward - (4.0 / 6.0)) < 1e-9
    assert mr.wall_clock_s == 40.0
    assert len(mr.per_task) == 2
    assert mr.per_task[0]["task_id"] == 0


def test_normalize_completion_rate_trial_weighted() -> None:
    """Partial credit pools trial-weighted, distinct from success_rate."""
    # task0: 0/4 success but 0.5 completion ; task1: 1/2 success, 0.5 comp.
    trs = [
        _task(0, 4, 0, 0.5, completion=0.5),
        _task(1, 2, 1, 0.5, completion=0.5),
    ]
    mr = normalize_task_results(
        mode="m", variation="permutation", suite_name="permutation_packing",
        task_results=trs,
    )
    assert mr.success_rate == 1 / 6
    assert abs(mr.completion_rate - 0.5) < 1e-9
    assert mr.per_task[0]["completion_rate"] == 0.5


def test_normalize_empty_is_zero_not_div0() -> None:
    mr = normalize_task_results(
        mode="m", variation="all", suite_name="s", task_results=[]
    )
    assert mr.n_trials == 0
    assert mr.success_rate == 0.0
    assert mr.completion_rate == 0.0
    assert mr.avg_reward == 0.0


def test_error_cell_keeps_grid_completing() -> None:
    c = error_cell(
        mode="llm_plus_policy",
        variation="pos_var",
        suite_name="libero_object_target_pos_var20x20",
        error="ValueError: llm_plus_policy requires workflow_dir",
    )
    assert c.error and "ValueError" in c.error
    assert c.n_trials == 0 and c.success_rate == 0.0


def test_merge_cells_pools_and_keeps_partial_failures() -> None:
    scored = [
        normalize_task_results(
            mode="m", variation="v", suite_name="s",
            task_results=[_task(0, 4, 4, 1.0)],
        ),
        normalize_task_results(
            mode="m", variation="v", suite_name="s",
            task_results=[_task(1, 4, 0, 0.0)],
        ),
        error_cell(mode="m", variation="v", suite_name="s", error="boom"),
    ]
    merged = merge_cells(scored)
    assert merged.n_trials == 8 and merged.n_success == 4
    assert merged.success_rate == 0.5
    assert merged.error is None  # partial failure stays scored
    all_err = merge_cells(
        [error_cell(mode="m", variation="v", suite_name="s", error="boom")]
    )
    assert all_err.error == "boom"


def test_build_summary_writes_json_and_tsv(tmp_path) -> None:
    cells = [
        normalize_task_results(
            mode="llm_generation", variation="pos_var", suite_name="sA",
            family="posvar", task_results=[_task(0, 2, 2, 1.0)],
        ),
        normalize_task_results(
            mode="llm_plus_policy", variation="pos_var", suite_name="sA",
            family="posvar", policy_id="libero_pi05",
            task_results=[_task(0, 2, 1, 0.5)],
        ),
        error_cell(mode="policy_only", family="posvar", variation="pos_var",
                   suite_name="sA", error="no workflow_dir"),
    ]
    grid = {"modes": ["llm_generation", "llm_plus_policy", "policy_only"],
            "families": ["posvar"], "n_tasks": 1, "n_seeds": 2}
    summary = build_summary(cells, out_dir=tmp_path, grid=grid)

    # JSON round-trips and has the matrix (keyed by family/variation;
    # policy cells disambiguated as mode@policy_id).
    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert loaded["grid"] == grid
    assert len(loaded["cells"]) == 3
    col = "posvar/pos_var"
    assert loaded["matrix"]["llm_generation"][col]["success_rate"] == 1.0
    assert loaded["matrix"]["llm_plus_policy@libero_pi05"][col]["success_rate"] == 0.5
    assert loaded["matrix"]["policy_only"][col]["error"] == "no workflow_dir"
    assert summary["matrix"] == loaded["matrix"]

    # TSV: header + 3 data rows + pivots (success + completion).
    rows = list(csv.reader((tmp_path / "summary.tsv").open(), delimiter="\t"))
    assert rows[0][0] == "mode"
    assert len(rows[1:4]) == 3
    pivot_idx = next(
        i for i, r in enumerate(rows) if r and r[0] == "success_rate_pivot"
    )
    assert col in rows[pivot_idx]
    pivot_rows = rows[pivot_idx + 1:]
    err_row = next(r for r in pivot_rows if r and r[0] == "policy_only")
    assert "ERR" in err_row
    gen_row = next(r for r in pivot_rows if r and r[0] == "llm_generation")
    assert "1.0000" in gen_row
    comp_idx = next(
        i for i, r in enumerate(rows) if r and r[0] == "completion_rate_pivot"
    )
    assert comp_idx > pivot_idx


def test_moderesult_is_dataclass_serializable() -> None:
    from dataclasses import asdict

    mr = ModeResult(mode="m", variation="all", suite_name="s")
    d = asdict(mr)
    assert d["mode"] == "m" and d["per_task"] == []
    # Round-trips through json + kwargs (the resume path).
    rt = ModeResult(**json.loads(json.dumps(d)))
    assert rt == mr
