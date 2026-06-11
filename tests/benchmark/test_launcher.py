"""Offline tests for the gap.agent.launcher pipeline (no sim, no LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import gap.agent.multi_agent as multi_agent_mod
import gap.agent.parallel as parallel_mod
from gap.agent.config import PipelineConfig, SuiteSpec, TrialConfig
from gap.agent.launcher import (
    _suite_dir_names,
    codegen_prompt,
    launch,
)
from gap.agent.multi_agent import PipelineResult

from .conftest import stub_connector_factory, stub_workflow_dict

# --------------------------------------------------------------------------
# objects-block plumbing into codegen
# --------------------------------------------------------------------------


def test_codegen_prompt_plain_suite() -> None:
    suite = SuiteSpec(
        suite_name="s", task_prompts={0: "pick up the milk"},
    )
    assert codegen_prompt(suite, 0) == "pick up the milk"


def test_codegen_prompt_appends_objects_context() -> None:
    suite = SuiteSpec(
        suite_name="libero_object_all_variance",
        task_prompts={0: "Pick the milk carton and place it in the basket."},
        objects={
            "target": "milk carton",
            "expected_label": "Milk",
            "shape_hint": "rectangular carton, white with colored print",
        },
    )
    prompt = codegen_prompt(suite, 0)
    assert prompt.startswith(
        "Pick the milk carton and place it in the basket."
    )
    assert "Object context" in prompt
    assert "- target: milk carton" in prompt
    assert "- expected_label: Milk" in prompt
    assert "- shape_hint: rectangular carton" in prompt


def test_codegen_prompt_missing_prompt_returns_empty() -> None:
    suite = SuiteSpec(suite_name="s", objects={"target": "x"})
    assert codegen_prompt(suite, 3, "auto") == ""
    # Explicit fallback (config.task) is used when no per-task prompt.
    assert codegen_prompt(suite, 3, "do the thing").startswith("do the thing")


# --------------------------------------------------------------------------
# suite output dir naming
# --------------------------------------------------------------------------


def test_suite_dir_names_dedup() -> None:
    suites = [
        SuiteSpec(suite_name="a"),
        SuiteSpec(suite_name="b"),
        SuiteSpec(suite_name="a"),
        SuiteSpec(suite_name="a"),
    ]
    assert _suite_dir_names(suites) == ["a", "b", "a_02", "a_03"]


# --------------------------------------------------------------------------
# end-to-end launch() with stubbed codegen + stub connector
# --------------------------------------------------------------------------


def _stub_codegen(captured: list[dict]):
    async def fake_run_codegen(*, task_id, task_prompt, config, output_dir):
        captured.append({"task_id": task_id, "task_prompt": task_prompt})
        wf_dir = Path(output_dir) / f"task_{task_id:02d}"
        wf_dir.mkdir(parents=True, exist_ok=True)
        (wf_dir / "workflow.json").write_text(json.dumps(stub_workflow_dict()))
        return PipelineResult(
            success=True, workflow_json="{}", workflow_dir=wf_dir,
        )
    return fake_run_codegen


def test_launch_codegen_to_trials(tmp_path, monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        multi_agent_mod, "run_codegen", _stub_codegen(captured),
    )
    monkeypatch.setattr(
        parallel_mod, "default_connector_factory", stub_connector_factory,
    )
    monkeypatch.setenv("GAP_PARALLEL_INPROC", "1")

    config = PipelineConfig(
        task="fallback",
        suites=[SuiteSpec(
            suite_name="stub_suite",
            task_prompts={0: "Pick the milk and place it in the basket."},
            objects={"target": "milk", "expected_label": "Milk"},
        )],
        trials=TrialConfig(
            trials_per_generation=2,
            task_ids=[0],
            num_workers=1,
            output_dir=tmp_path / "run",
        ),
    )
    result = asyncio.run(launch(config))

    # Codegen saw the objects context (load-bearing for G1).
    assert len(captured) == 1
    assert captured[0]["task_id"] == 0
    assert "- expected_label: Milk" in captured[0]["task_prompt"]

    # 1 task x 2 trials, all passing through the stub connector.
    assert result.total_trials == 2
    assert result.success_rate == 1.0
    assert result.completion_rate == 1.0
    assert len(result.task_results) == 1
    assert result.task_results[0].success_count == 2

    # Artifact layout: renamed trial dirs + per-task results.json +
    # suite aggregate + done flag.
    task_dir = tmp_path / "run" / "task_00"
    trial_dirs = sorted(
        p.name for p in task_dir.iterdir()
        if p.is_dir() and p.name.startswith("trial_")
    )
    assert trial_dirs == [
        "trial_01_rc0_reward1.000_pass",
        "trial_02_rc0_reward1.000_pass",
    ]
    assert (task_dir / trial_dirs[0] / "result.json").is_file()
    results_json = json.loads((task_dir / "results.json").read_text())
    assert results_json["success_count"] == 2
    agg = json.loads((tmp_path / "run" / "aggregate_results.json").read_text())
    assert agg["success_rate"] == 1.0
    assert (tmp_path / "run" / "aaa_done_flag" / "aaa_done_flag.txt").is_file()


def test_launch_prebuilt_workflow_with_objects_templating(
    tmp_path, monkeypatch,
) -> None:
    """workflow_dir + suite.objects -> templated copy, no codegen."""
    monkeypatch.setattr(
        parallel_mod, "default_connector_factory", stub_connector_factory,
    )
    monkeypatch.setenv("GAP_PARALLEL_INPROC", "1")

    def _boom(**kwargs):  # codegen must NOT run
        raise AssertionError("codegen should be skipped")

    monkeypatch.setattr(multi_agent_mod, "run_codegen", _boom)

    template = tmp_path / "template"
    template.mkdir()
    wf = stub_workflow_dict()
    wf["meta"]["description"] = "grab the {{target}}"
    (template / "workflow.json").write_text(json.dumps(wf))

    config = PipelineConfig(
        task="t",
        suites=[SuiteSpec(
            suite_name="stub_suite",
            task_prompts={0: "p"},
            objects={"target": "ketchup"},
        )],
        trials=TrialConfig(
            trials_per_generation=1,
            task_ids=[0],
            num_workers=1,
            output_dir=tmp_path / "run",
        ),
    )
    result = asyncio.run(launch(config, workflow_dir=str(template)))
    assert result.total_trials == 1 and result.success_rate == 1.0

    materialized = json.loads(
        (tmp_path / "run" / "workflow" / "workflow.json").read_text()
    )
    assert materialized["meta"]["description"] == "grab the ketchup"


def test_launch_workflow_dir_map(tmp_path, monkeypatch) -> None:
    """Per-task pre-built workflows (the benchmark template-mode path)."""
    monkeypatch.setattr(
        parallel_mod, "default_connector_factory", stub_connector_factory,
    )
    monkeypatch.setenv("GAP_PARALLEL_INPROC", "1")
    wf0 = tmp_path / "wf0"
    wf0.mkdir()
    (wf0 / "workflow.json").write_text(json.dumps(stub_workflow_dict()))

    config = PipelineConfig(
        task="t",
        suites=[SuiteSpec(suite_name="stub_suite", task_prompts={0: "p"})],
        trials=TrialConfig(
            trials_per_generation=1,
            task_ids=[0],
            num_workers=1,
            output_dir=tmp_path / "run",
        ),
    )
    result = asyncio.run(launch(config, workflow_dir_map={0: str(wf0)}))
    assert result.total_trials == 1
    assert result.success_rate == 1.0


def test_launch_multi_suite_cross_aggregate(tmp_path, monkeypatch) -> None:
    captured: list[dict] = []
    monkeypatch.setattr(
        multi_agent_mod, "run_codegen", _stub_codegen(captured),
    )
    monkeypatch.setattr(
        parallel_mod, "default_connector_factory", stub_connector_factory,
    )
    monkeypatch.setenv("GAP_PARALLEL_INPROC", "1")
    # Two suites with the SAME name (the grocery acceptance shape).
    config = PipelineConfig(
        task="t",
        suites=[
            SuiteSpec(suite_name="dup", task_ids=[0], task_prompts={0: "a"}),
            SuiteSpec(suite_name="dup", task_ids=[1], task_prompts={1: "b"}),
        ],
        trials=TrialConfig(
            trials_per_generation=1,
            task_ids=[0],
            num_workers=1,
            output_dir=tmp_path / "run",
        ),
    )
    result = asyncio.run(launch(config))
    assert result.total_trials == 2
    assert (tmp_path / "run" / "dup" / "task_00").is_dir()
    assert (tmp_path / "run" / "dup_02" / "task_01").is_dir()
    cross = json.loads(
        (tmp_path / "run" / "cross_suite_results.json").read_text()
    )
    assert cross["total_trials"] == 2
    assert len(cross["suites"]) == 2
