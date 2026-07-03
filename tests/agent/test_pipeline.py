"""Canned-response pipeline tests — NO live LLM anywhere.

A scripted coordinator → subgraph_agent → checkpoint_agent conversation
(authored against gap.builder; see conftest) drives the real
generate_workflow / run_codegen machinery against the real open-robot-skills
checkout. Asserts the written workflow dir loads, validates clean, and
the checkpoint sidecars round-trip through gap.runtime.verify.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from gap_core.errors import ValidationIssue

from gap.agent.config import PipelineConfig

from .conftest import (
    CHECKPOINT_RESPONSE_GOOD,
    COORDINATOR_RESPONSE,
    FIXED_MAKE_OBB_RESPONSE,
    GRASP_SUBGRAPH_RESPONSE,
    PERCEIVE_SUBGRAPH_RESPONSE,
    PIPELINE_RESPONSES,
)


def _config(skills_root) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.skills = skills_root
    return cfg


@pytest.fixture
def pipeline_run(skills_root, stub_llm, tmp_path):
    """Run the full canned pipeline once; returns (result, stub, wf_dir)."""
    from gap.agent.multi_agent import run_codegen

    stub = stub_llm(list(PIPELINE_RESPONSES))
    result = asyncio.run(run_codegen(
        task_id=0,
        task_prompt="Pick up the alphabet soup",
        config=_config(skills_root),
        output_dir=tmp_path,
    ))
    assert result.success, result.execution_stderr
    return result, stub, result.workflow_dir


class TestPipeline:
    def test_workflow_dir_contents(self, pipeline_run):
        result, stub, wf_dir = pipeline_run
        assert (wf_dir / "workflow.json").is_file()
        assert (wf_dir / "multi_agent_meta.json").is_file()
        # Inline script written; canonical scripts materialized from bundles.
        assert (wf_dir / "scripts/perceive_target/make_obb.py").is_file()
        assert (wf_dir / "scripts/perceive_target/perceive_dino_vlm.py").is_file()
        assert (wf_dir / "scripts/grasp_target/compute_align_pose.py").is_file()
        # Checkpoint sidecars.
        assert (wf_dir / "checkpoints/perceive_target.py").is_file()
        assert (wf_dir / "checkpoints/grasp_target.py").is_file()

        wf = json.loads((wf_dir / "workflow.json").read_text())
        assert set(wf["subgraphs"]) == {"perceive_target", "grasp_target"}
        assert wf["subgraphs"]["grasp_target"]["stage"] == "grasp"
        assert wf["subgraphs"]["grasp_target"]["skill"] == "grasping-direct-ik"

    def test_canonical_script_matches_bundle(self, pipeline_run, skills_root):
        _, _, wf_dir = pipeline_run
        bundle_src = (
            skills_root / "skills/grasping-direct-ik/scripts/compute_align_pose.py"
        ).read_text()
        written = (wf_dir / "scripts/grasp_target/compute_align_pose.py").read_text()
        assert written == bundle_src

    def test_workflow_loads_and_validates(self, pipeline_run, skills_root):
        _, _, wf_dir = pipeline_run
        from gap.agent._catalog import load_codegen_registries
        from gap.runtime.validate import validate_workflow
        from gap.runtime.workflow import load_workflow

        wf = load_workflow(wf_dir / "workflow.json")
        skills, tools = load_codegen_registries(skills_root)
        issues = validate_workflow(
            wf, agent_registry=None, skill_registry=skills, tool_registry=tools,
        )
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []

    def test_checkpoint_sidecars_load_via_verify(self, pipeline_run):
        _, _, wf_dir = pipeline_run
        from gap.runtime.verify import load_checkpoints

        perceive_cps = load_checkpoints(wf_dir / "checkpoints/perceive_target.py")
        grasp_cps = load_checkpoints(wf_dir / "checkpoints/grasp_target.py")
        assert [c.name for c in perceive_cps] == ["target_obb_sane"]
        assert [c.name for c in grasp_cps] == ["grasp_pose_above_table", "target_held"]
        assert all(c.validate for c in grasp_cps)
        assert all(c.rationale for c in grasp_cps)
        assert grasp_cps[0].subgraph == "grasp_target"

        # The 2-arg dict-subscript predicates evaluate against TypedDict outputs.
        z_cp = grasp_cps[0]
        assert z_cp.predicate(None, {"grasp_pose": {"position": {"z": 0.05}}}) is True
        assert z_cp.predicate(None, {"grasp_pose": {"position": {"z": -0.02}}}) is False
        obb_cp = perceive_cps[0]
        assert obb_cp.predicate(None, {"target_obb": {"extent": {"x": 0.04}}}) is True

    def test_checkpoint_retry_loop_ran(self, pipeline_run):
        _, stub, _ = pipeline_run
        # coordinator + 2 subgraphs + checkpoint (bad) + checkpoint (good)
        assert len(stub.calls) == 5
        retry_msg = stub.calls[4]["messages"][-1]["content"]
        assert "validation errors" in retry_msg
        assert "grasp_pose" in retry_msg  # the z-rule re-prompt names the output

    # ------------------------------------------------------------------
    # Prompt-assembly snapshots
    # ------------------------------------------------------------------

    def test_coordinator_prompt_snapshot(self, pipeline_run):
        _, stub, _ = pipeline_run
        system = stub.calls[0]["system"]
        assert "# Coordinator" in system
        assert "## Available Skills" in system
        # Skill-kind bundles ONLY in the skills catalog.
        assert "| `perceiving-objects` |" in system
        assert "| `grasping-direct-ik` |" in system
        assert "| `sam3` |" not in system
        assert "| `geometry` |" not in system
        # Tool bundles surface through the flat tool catalog, with schemas.
        assert "## Available Tools (flat catalog)" in system
        assert "`sam3.segment_text`" in system
        assert "`robot.go_to_pose`" in system
        assert "pose: Se3Pose" in system  # UnitSchema field rendering
        # Task appended at the end.
        assert "Pick up the alphabet soup" in system
        # Codegen meta-tools bound through the provider tool loop.
        assert {t["name"] for t in stub.calls[0]["tools"]} == {
            "read_skill_reference", "read_skill_example", "report_missing_capability",
        }

    def test_subgraph_prompt_snapshot(self, pipeline_run):
        _, stub, _ = pipeline_run
        system = stub.calls[1]["system"]
        # Skill body inlined verbatim.
        assert "## Skill in scope: `perceiving-objects`" in system
        assert "# perceiving-objects" in system  # SKILL.md body heading
        # Canonical scripts table with introspected schemas.
        assert "### Canonical scripts" in system
        assert "`scripts/perceive_dino_vlm.py`" in system
        assert "object_name: str" in system
        # Allowed-tools catalog includes a connector tool with its schema...
        assert "`robot.get_observation`" in system
        assert "cameras: list[CameraFrame]" in system
        # ...and bundle tools from the whitelist.
        assert "`grounding-dino.detect`" in system
        assert "`sam3.segment_box`" in system
        # Tools outside the whitelist are not in the subgraph catalog
        # (the name may appear in shared-doc prose, but never as an entry).
        assert "- `curobo.plan_to_grasp_poses`" not in system
        # Per-call context.
        assert "## Your subgraph: `perceive_target`" in system
        assert "### Required outputs" in system
        assert "target_obb" in system

    def test_grasp_subgraph_prompt_has_upstream_outputs(self, pipeline_run):
        _, stub, _ = pipeline_run
        system = stub.calls[2]["system"]
        assert "## Skill in scope: `grasping-direct-ik`" in system
        assert "### Upstream outputs" in system
        assert "| perceive_target | target_obb |" in system
        assert "### Bound inputs" in system

    def test_checkpoint_prompt_snapshot(self, pipeline_run):
        _, stub, _ = pipeline_run
        call = stub.calls[3]
        assert "# Checkpoint agent" in call["system"]
        assert "## Canonical checkpoint shapes by skill" in call["system"]
        assert "`grasping-direct-ik`" in call["system"]
        # Dict-subscript guidance, not proto attribute access.
        assert "o['grasp_pose']['position']['z']" in call["system"]
        user = call["messages"][0]["content"]
        assert "Builder source" in user
        assert "Bound outputs" in user
        assert "`grasp_target`  (skill: `grasping-direct-ik`)" in user


class TestScriptFixLoop:
    def test_fix_loop_rewrites_script(self, skills_root, stub_llm, tmp_path, monkeypatch):
        """A canned bad script followed by a canned fix: the validation
        loop attributes the error to the script node, asks the LLM for a
        fix, and rewrites the file."""
        from gap.agent import multi_agent

        validation_calls = {"n": 0}

        def fake_validation(wf_dir, config):
            validation_calls["n"] += 1
            if validation_calls["n"] == 1:
                return [ValidationIssue(
                    severity="error",
                    node_id="subgraphs.perceive_target.nodes.make_obb",
                    field=None,
                    message="run() output 'obb' does not match declared type",
                )]
            return []

        monkeypatch.setattr(multi_agent, "_run_graph_validation", fake_validation)

        stub = stub_llm([
            COORDINATOR_RESPONSE,
            PERCEIVE_SUBGRAPH_RESPONSE,
            GRASP_SUBGRAPH_RESPONSE,
            CHECKPOINT_RESPONSE_GOOD,
            FIXED_MAKE_OBB_RESPONSE,
        ])
        result = asyncio.run(multi_agent.run_codegen(
            task_id=0,
            task_prompt="Pick up the alphabet soup",
            config=_config(skills_root),
            output_dir=tmp_path,
        ))
        assert result.success, result.execution_stderr
        assert validation_calls["n"] == 2

        written = (result.workflow_dir / "scripts/perceive_target/make_obb.py").read_text()
        assert "FIXED_MARKER = True" in written
        # The fix request was a plain completion carrying the bad script + error.
        fix_call = stub.calls[-1]
        assert fix_call["kind"] == "complete"
        assert "make_obb" in fix_call["messages"][0]["content"]
        assert "does not match declared type" in fix_call["messages"][0]["content"]

    def test_coordinator_retry_on_unknown_skill(self, skills_root, stub_llm, tmp_path):
        """A coordinator response naming an unknown skill is re-prompted."""
        from gap.agent.multi_agent import run_codegen

        bad_coordinator = COORDINATOR_RESPONSE.replace(
            'skill="perceiving-objects"', 'skill="perception_single"', 1,
        )
        stub = stub_llm([
            bad_coordinator,
            COORDINATOR_RESPONSE,
            PERCEIVE_SUBGRAPH_RESPONSE,
            GRASP_SUBGRAPH_RESPONSE,
            CHECKPOINT_RESPONSE_GOOD,
        ])
        result = asyncio.run(run_codegen(
            task_id=0,
            task_prompt="Pick up the alphabet soup",
            config=_config(skills_root),
            output_dir=tmp_path,
        ))
        assert result.success, result.execution_stderr
        retry_msg = stub.calls[1]["messages"][-1]["content"]
        assert "unknown skill 'perception_single'" in retry_msg


def _config_no_checkpoints(skills_root) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.skills = skills_root
    cfg.composition.checkpoint_agent = False
    return cfg


# A grasp subgraph that references `in.missing_thing` without declaring it
# (and the coordinator doesn't declare it either) — a genuine S5 structural
# error that the shallow parse-check misses but the full per-subgraph
# validator (Part 2) catches and feeds back to the subgraph_agent.
GRASP_SUBGRAPH_RESPONSE_BAD = '''\
```python
from gap.builder import Subgraph, Ref, START, END

sg = Subgraph(name="grasp_target", skill="grasping-direct-ik")
sg.add_input("target_obb", type_name="OrientedBoundingBox")

sg.add_node("compute_grasp", type="tool",
            tool="geometry.top_down_grasp_candidates",
            inputs={"obb": Ref("in.missing_thing")})
sg.add_node("close", type="tool", tool="robot.close_gripper")

sg.add_exit("grasped")
sg.add_edge(START, "compute_grasp")
sg.add_edge("compute_grasp", "close")
sg.add_edge("close", "grasped")
sg.add_edge("grasped", END)

sg.set_outputs(grasp_pose=Ref("compute_grasp.candidates.poses.0"))
sg.set_on_error("failed")
```
'''


class TestStructuralFeedback:
    def test_structural_error_regenerates_at_authoring_agent(
        self, skills_root, stub_llm, tmp_path,
    ):
        """A subgraph with a real structural error (S5: `in.missing_thing`
        not declared) is caught at authoring time and fed back to the SAME
        subgraph_agent, which regenerates. (Part 2.)"""
        from gap.agent.multi_agent import run_codegen

        stub = stub_llm([
            COORDINATOR_RESPONSE,
            PERCEIVE_SUBGRAPH_RESPONSE,
            GRASP_SUBGRAPH_RESPONSE_BAD,   # rejected by per-subgraph structural check
            GRASP_SUBGRAPH_RESPONSE,       # the regenerated (valid) subgraph
        ])
        result = asyncio.run(run_codegen(
            task_id=0,
            task_prompt="Pick up the alphabet soup",
            config=_config_no_checkpoints(skills_root),
            output_dir=tmp_path,
        ))
        assert result.success, result.execution_stderr
        # coordinator + perceive + grasp(bad) + grasp(retry) = 4 calls.
        assert len(stub.calls) == 4
        # The retry carried the S5 structural error back to the agent.
        retry_msg = stub.calls[3]["messages"][-1]["content"]
        assert "missing_thing" in retry_msg
        assert "S5" in retry_msg or "not declared in subgraph inputs" in retry_msg


class TestHonestReporting:
    def test_residual_errors_flip_success_false(
        self, skills_root, stub_llm, tmp_path, monkeypatch,
    ):
        """An unfixable residual validation error (workflow-level, no script
        node to route to) flips success to False and is surfaced on the
        result, instead of being reported as a successful run. (Part 3.)"""
        from gap.agent import multi_agent

        def always_failing(wf_dir, config):
            return [ValidationIssue(
                severity="error",
                node_id="workflow.edges",
                field=None,
                message="bogus unfixable workflow error (W3)",
            )]

        monkeypatch.setattr(multi_agent, "_run_graph_validation", always_failing)

        stub_llm([
            COORDINATOR_RESPONSE,
            PERCEIVE_SUBGRAPH_RESPONSE,
            GRASP_SUBGRAPH_RESPONSE,
        ])
        result = asyncio.run(multi_agent.run_codegen(
            task_id=0,
            task_prompt="Pick up the alphabet soup",
            config=_config_no_checkpoints(skills_root),
            output_dir=tmp_path,
        ))
        assert result.success is False
        assert result.validation_errors
        assert "workflow.edges" in result.execution_stderr
