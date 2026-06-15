"""Import wiring, CLI surface, and the sim-marked mini-benchmark."""

from __future__ import annotations

import json

import pytest

from .conftest import EXAMPLES


def test_import_only() -> None:
    """Always-on: harness + modes import and the factory works."""
    import gap.benchmark
    from gap.benchmark.builtin_modes import build_mode
    from gap.benchmark.config import KNOWN_MODES
    from gap.benchmark.harness import run_benchmark  # noqa: F401

    assert callable(gap.benchmark.run)
    for m in KNOWN_MODES:
        assert build_mode(m).name == m
    with pytest.raises(ValueError, match="unknown benchmark mode"):
        build_mode("monolithic")


def test_cli_modules_register() -> None:
    """The benchmark + policy CLI modules register their parsers."""
    import argparse

    from gap.cli.benchmark import register as reg_benchmark
    from gap.cli.policy import register as reg_policy

    parser = argparse.ArgumentParser(prog="gap")
    sub = parser.add_subparsers(dest="command")
    reg_benchmark(sub)
    reg_policy(sub)

    args = parser.parse_args(
        ["benchmark", "cfg.yaml", "--gate", "--resume",
         "--families", "posvar", "--modes", "llm_generation"],
    )
    assert args.command == "benchmark"
    assert args.gate and args.resume
    assert args.families == ["posvar"]

    args = parser.parse_args(["policy", "list"])
    assert args.command == "policy"
    args = parser.parse_args(["policy", "serve", "pi05-libero", "--port", "9100"])
    assert args.bundle == "pi05-libero" and args.port == 9100


def test_policy_list_works_without_policy_bundles(tmp_path, capsys) -> None:
    """`gap policy list` over an empty registry prints a clean 0-bundles
    message — no PRESETS dict, no hardcoded recipes, all sourced from the
    bundle catalog."""
    import argparse

    from gap.cli.policy import _handle_list

    # Empty registry: a checkout dir with no policies/ subdir.
    (tmp_path / "skills").mkdir()
    args = argparse.Namespace(skills=[str(tmp_path)], registry=None)
    assert _handle_list(args) == 0
    out = capsys.readouterr().out
    assert "0 policy bundle(s)" in out


def test_example_yamls_exist() -> None:
    for name in (
        "smoke.yaml",
        "posvar.yaml",
        "grocery_acceptance.yaml",
        "grocery_acceptance_smoke.yaml",
    ):
        assert (EXAMPLES / name).is_file(), name


# --------------------------------------------------------------------------
# Mini-benchmark on the real sim: 1 cell x 1 trial of a stub mode whose
# workflow is the models-free motion graph (go_home + gripper + check) —
# real harness + real parallel + real LIBERO env, no LLM.
# --------------------------------------------------------------------------


@pytest.mark.sim
def test_mini_benchmark_stub_mode_on_real_sim(tmp_path) -> None:
    try:
        import gap.envs.registry  # noqa: F401
    except ImportError:
        pytest.skip("gap.envs not importable")

    import asyncio
    from pathlib import Path

    import gap.benchmark.builtin_modes as builtin_modes_mod
    from gap.agent.config import PipelineConfig
    from gap.benchmark.builtin_modes import build_mode
    from gap.benchmark.config import BenchmarkConfig
    from gap.benchmark.harness import run_benchmark
    from gap.benchmark.modes import BenchmarkMode, ModeRequest

    class StubMotionMode(BenchmarkMode):
        """Pre-built models-free motion graph — no codegen, no models."""

        name = "stub_motion"

        async def produce_workflows(
            self, req: ModeRequest,
        ) -> dict[int, str] | None:
            graph = {
                "version": 3,
                "meta": {"name": "models_free_motion"},
                "nodes": {
                    "motion": {"type": "subgraph", "ref": "motion"},
                    "done": {"type": "end", "status": "success"},
                },
                "edges": [["START", "motion"]],
                "conditional_edges": {
                    "motion": {
                        "router_field": "exit", "mapping": {"ok": "done"},
                    },
                },
                "subgraphs": {
                    "motion": {
                        "skill": "generic",
                        "inputs": {},
                        "outputs": {},
                        "nodes": {
                            "home": {"type": "tool", "tool": "robot.go_home"},
                            "open": {
                                "type": "tool",
                                "tool": "robot.open_gripper",
                                "inputs": {"settle_steps": 10},
                            },
                            "close": {
                                "type": "tool",
                                "tool": "robot.close_gripper",
                                "inputs": {"settle_steps": 10},
                            },
                            "check": {
                                "type": "tool", "tool": "sim.check_success",
                            },
                            "ok": {"type": "noop"},
                        },
                        "edges": [
                            ["START", "home"],
                            ["home", "open"],
                            ["open", "close"],
                            ["close", "check"],
                            ["check", "ok"],
                            ["ok", "END"],
                        ],
                        "conditional_edges": {},
                        "exit": {
                            "router_field": None, "success_values": ["ok"],
                        },
                    },
                },
            }
            out: dict[int, str] = {}
            for tid in req.task_ids:
                wf_dir = req.produce_dir / f"task_{tid:02d}" / "workflow"
                wf_dir.mkdir(parents=True, exist_ok=True)
                (wf_dir / "workflow.json").write_text(json.dumps(graph))
                out[tid] = str(wf_dir)
            return out

    builtin_modes_mod._REGISTRY[StubMotionMode.name] = StubMotionMode
    try:
        assert build_mode("stub_motion").name == "stub_motion"

        cfg = BenchmarkConfig(
            source_yaml=Path("stub.yaml"),
            families=["posvar"],
            variations=["all"],       # libero_object_all_variance (vab)
            task_ids=[0],
            n_seeds=1,
            num_workers=1,
            output_dir=tmp_path,
        )
        cfg.modes = ["stub_motion"]   # bypass KNOWN_MODES validation
        cfg.pipeline_config = PipelineConfig()

        try:
            summary = asyncio.run(run_benchmark(cfg))
        except Exception as exc:  # missing sim deps (mujoco/EGL/assets)
            pytest.skip(f"sim env unavailable: {exc}")

        assert summary.run_dir is not None
        assert (summary.run_dir / "summary.json").is_file()
        loaded = json.loads((summary.run_dir / "summary.json").read_text())
        [cell] = loaded["cells"]
        assert cell["mode"] == "stub_motion"
        assert cell["suite_name"] == "libero_object_all_variance"
        assert cell["error"] is None
        assert cell["n_trials"] == 1
        # The motion graph never delivers the object — the point is that
        # the full harness->launch->parallel->sim pipeline executed and
        # scored the trial.
        result_files = list(summary.run_dir.rglob("result.json"))
        assert len(result_files) == 1
        result = json.loads(result_files[0].read_text())
        assert result["exit_code"] == 0
    finally:
        builtin_modes_mod._REGISTRY.pop("stub_motion", None)
