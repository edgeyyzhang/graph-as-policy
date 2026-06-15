"""Canned-response tests for LLM-invented ("generated") skills.

A generated skill is a subgraph the coordinator declares with
``generated=True`` and a fresh ``skill`` name that is NOT a registered
bundle. The subgraph_agent implements it from scratch with ``type:script``
nodes (no SKILL.md, no canonical scripts). These tests drive the real
codegen machinery with a stubbed LLM and assert the invented skill flows
end-to-end: coordinator validation accepts it, the subgraph prompt
assembles without a registry lookup, and the written workflow validates
clean against the runtime validator.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from gap.agent._catalog import load_codegen_registries
from gap.agent._registry import default_agent_registry
from gap.agent.config import PipelineConfig
from gap.agent.prompt_assembler import PromptAssembler
from gap.agent.subgraph_runner import _validate_workflow_spec

from .conftest import PERCEIVE_SUBGRAPH_RESPONSE

# ---------------------------------------------------------------------------
# Canned agent responses — a single invented-skill subgraph workflow
# ---------------------------------------------------------------------------

GEN_COORDINATOR_RESPONSE = '''\
No catalog skill counts widgets, so I'll invent one.

```python
from gap.builder import WorkflowSpec, START

spec = WorkflowSpec(name="measure_task", description="Count the widgets")

spec.declare_subgraph(
    "measure",
    skill="detect-and-count-widgets",   # invented — not in the catalog
    generated=True,
    description="Detect and count the widgets visible on the table",
    outputs={"count": "int"},
    exit_success_values=["measured"],
    on_error="failed",
)

spec.add_subgraph_node("measure", ref="measure")
spec.add_end("done", status="success")
spec.add_end(
    "abort",
    status="failure",
    recovery=[{"tool": "robot.go_home", "inputs": {}}],
)

spec.add_edge(START, "measure")
spec.add_conditional_edges(
    "measure",
    {"measured": "done", "failed": "abort"},
    router_field="exit",
)
```
'''

COUNT_SCRIPT = '''\
"""Count widgets from an observation (invented-skill implementation)."""

from typing import TypedDict

from gap import NodeContext
from gap.types import Observation


class Output(TypedDict):
    count: int


def run(ctx: NodeContext, observation: Observation) -> Output:
    return {"count": len(observation["cameras"])}
'''

GEN_SUBGRAPH_RESPONSE = f'''\
```python
from gap.builder import Subgraph, Ref, START, END

sg = Subgraph(name="measure", skill="detect-and-count-widgets")

sg.add_node("observe", type="tool", tool="robot.get_observation")
sg.add_node("count", type="script",
            script="scripts/measure/count_widgets.py",
            inputs={{"observation": Ref("observe")}})

sg.add_exit("measured")
sg.add_edge(START, "observe")
sg.add_edge("observe", "count")
sg.add_edge("count", "measured")
sg.add_edge("measured", END)

sg.set_outputs(count=Ref("count.count"))
sg.set_on_error("failed")
```

```python:scripts/measure/count_widgets.py
{COUNT_SCRIPT}
```
'''

GEN_PIPELINE_RESPONSES = [GEN_COORDINATOR_RESPONSE, GEN_SUBGRAPH_RESPONSE]


def _config(skills_root) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.skills = skills_root
    # Focus on the generated-skill codegen path; the checkpoint agent is
    # an orthogonal follow-on pass.
    cfg.composition.checkpoint_agent = False
    return cfg


@pytest.fixture
def gen_run(skills_root, stub_llm, tmp_path):
    from gap.agent.multi_agent import run_codegen

    stub = stub_llm(list(GEN_PIPELINE_RESPONSES))
    result = asyncio.run(run_codegen(
        task_id=0,
        task_prompt="Count the widgets on the table",
        config=_config(skills_root),
        output_dir=tmp_path,
    ))
    return result, stub


class TestGeneratedSkillPipeline:
    def test_pipeline_succeeds(self, gen_run):
        result, _ = gen_run
        assert result.success, result.execution_stderr

    def test_invented_skill_and_script_written(self, gen_run):
        result, _ = gen_run
        wf_dir = result.workflow_dir
        wf = json.loads((wf_dir / "workflow.json").read_text())
        # The invented skill name survives into the SubgraphDef as metadata.
        assert wf["subgraphs"]["measure"]["skill"] == "detect-and-count-widgets"
        # The LLM-authored implementation script was written (and not
        # clobbered by canonical-wins, since the skill has no bundle).
        script = wf_dir / "scripts/measure/count_widgets.py"
        assert script.is_file()
        assert "def run(ctx" in script.read_text()

    def test_runtime_validation_clean(self, gen_run):
        result, _ = gen_run
        from gap.runtime.validate import validate_workflow
        from gap.runtime.workflow import load_workflow

        wf = load_workflow(result.workflow_dir / "workflow.json")
        issues = validate_workflow(wf)
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, errors


class TestCoordinatorSpecValidation:
    def test_accepts_generated_unregistered_skill(self, skills_root):
        skills, _ = load_codegen_registries(skills_root)
        spec = {
            "nodes": {"m": {"type": "subgraph", "ref": "m"}},
            "edges": [["START", "m"]],
            "subgraphs": {
                "m": {
                    "skill": "totally-invented-skill",
                    "generated": True,
                    "description": "do a brand new thing",
                    "exit": {"router_field": None, "success_values": ["ok"]},
                    "on_error": "failed",
                },
            },
        }
        assert _validate_workflow_spec(spec, skills) == ""

    def test_rejects_unregistered_skill_without_generated_flag(self, skills_root):
        skills, _ = load_codegen_registries(skills_root)
        spec = {
            "nodes": {"m": {"type": "subgraph", "ref": "m"}},
            "edges": [["START", "m"]],
            "subgraphs": {
                "m": {
                    "skill": "perceiving-objct",  # typo of a real skill
                    "description": "perceive",
                    "exit": {"router_field": None, "success_values": ["found"]},
                    "on_error": "not_found",
                },
            },
        }
        err = _validate_workflow_spec(spec, skills)
        assert "unknown skill" in err
        assert "generated=True" in err  # the message points at the fix

    def test_generated_requires_description(self, skills_root):
        skills, _ = load_codegen_registries(skills_root)
        spec = {
            "nodes": {"m": {"type": "subgraph", "ref": "m"}},
            "edges": [["START", "m"]],
            "subgraphs": {
                "m": {
                    "skill": "invented",
                    "generated": True,
                    "description": "   ",  # blank
                    "exit": {"router_field": None, "success_values": ["ok"]},
                    "on_error": "failed",
                },
            },
        }
        assert "description" in _validate_workflow_spec(spec, skills)


class TestGeneratedSubgraphPromptAssembly:
    def test_assembles_without_registry_lookup(self, skills_root):
        skills, tools = load_codegen_registries(skills_root)
        assembler = PromptAssembler(default_agent_registry(), skills, tools)
        spec = {
            "name": "measure",
            "description": "Detect and count the widgets",
            "inputs": {},
            "outputs": {"count": "int"},
            "exit": {"router_field": None, "success_values": ["measured"]},
            "on_error": "failed",
            "generated": True,
        }
        # Must NOT raise KeyError despite the skill being unregistered.
        prompt = assembler.assemble_subgraph_agent("detect-and-count-widgets", spec)
        text = prompt.system_prompt
        assert "detect-and-count-widgets" in text
        assert "invented skill" in text.lower()
        # Empty allowed_tools => the full runtime catalog is exposed; spot
        # check a couple of connector tools that every generated skill may use.
        assert "robot.get_observation" in text


# ---------------------------------------------------------------------------
# Part 1 (input-clobber fix): a generated subgraph that consumes an upstream
# output. The coordinator declares the generated subgraph with inputs={}; the
# subgraph_agent declares the real input via add_input(...). The pipeline must
# PRESERVE the agent's declaration (not overwrite it with the coordinator's
# empty dict) so the cross-subgraph wiring validates.
# ---------------------------------------------------------------------------

TWIST_COORDINATOR_RESPONSE = '''\
```python
from gap.builder import WorkflowSpec, START

spec = WorkflowSpec(name="twist_task", description="Perceive then twist the thing")

spec.declare_subgraph(
    "perceive_target",
    skill="perceiving-objects",
    description="Locate the thing",
    outputs={
        "target_obb": "OrientedBoundingBox",
        "target_mask": "Mask",
        "target_cloud": "PointCloud",
    },
    exit_success_values=["found"],
    on_error="not_found",
)
spec.declare_subgraph(
    "twist_target",
    skill="twist-in-place",          # invented
    generated=True,
    description="Twist the thing in place using its perceived OBB",
    inputs={},                        # coordinator declares NONE...
    outputs={},
    exit_success_values=["twisted"],
    on_error="failed",
)

spec.add_subgraph_node("perceive_target", ref="perceive_target")
spec.add_subgraph_node("twist_target", ref="twist_target")
spec.add_end("done", status="success")
spec.add_end(
    "abort", status="failure",
    recovery=[{"tool": "robot.go_home", "inputs": {}}],
)

spec.add_edge(START, "perceive_target")
spec.add_conditional_edges(
    "perceive_target",
    {"found": "twist_target", "not_found": "abort"},
    router_field="exit",
)
spec.add_conditional_edges(
    "twist_target",
    {"twisted": "done", "failed": "abort"},
    router_field="exit",
)
```
'''

COMPUTE_TWIST_SCRIPT = '''\
"""Compute a twist target pose from the object's OBB center."""

from typing import TypedDict

from gap import NodeContext
from gap.types import OrientedBoundingBox, Se3Pose


class Output(TypedDict):
    pose: Se3Pose


def run(ctx: NodeContext, obb: OrientedBoundingBox) -> Output:
    c = obb["center"]
    return {"pose": {
        "position": {"x": c["x"], "y": c["y"], "z": c["z"]},
        "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }}
'''

TWIST_SUBGRAPH_RESPONSE = f'''\
```python
from gap.builder import Subgraph, Ref, START, END

sg = Subgraph(name="twist_target", skill="twist-in-place")
# ...but the subgraph_agent declares the input it actually needs:
sg.add_input("target_obb", type_name="OrientedBoundingBox")

sg.add_node("compute_twist", type="script",
            script="scripts/twist_target/compute_twist.py",
            inputs={{"obb": Ref("in.target_obb")}})
sg.add_node("move", type="tool", tool="robot.go_to_pose",
            inputs={{"pose": Ref("compute_twist.pose")}})

sg.add_exit("twisted")
sg.add_edge(START, "compute_twist")
sg.add_edge("compute_twist", "move")
sg.add_edge("move", "twisted")
sg.add_edge("twisted", END)
sg.set_on_error("failed")
```

```python:scripts/twist_target/compute_twist.py
{COMPUTE_TWIST_SCRIPT}
```
'''


class TestGeneratedSkillInputPreserved:
    @pytest.fixture
    def twist_run(self, skills_root, stub_llm, tmp_path):
        from gap.agent.multi_agent import run_codegen

        cfg = PipelineConfig()
        cfg.skills = skills_root
        cfg.composition.checkpoint_agent = False
        stub = stub_llm([
            TWIST_COORDINATOR_RESPONSE,
            PERCEIVE_SUBGRAPH_RESPONSE,
            TWIST_SUBGRAPH_RESPONSE,
        ])
        result = asyncio.run(run_codegen(
            task_id=0,
            task_prompt="Twist the thing",
            config=cfg,
            output_dir=tmp_path,
        ))
        return result, stub

    def test_agent_declared_input_survives(self, twist_run):
        result, _ = twist_run
        assert result.success, result.execution_stderr
        wf = json.loads((result.workflow_dir / "workflow.json").read_text())
        # The coordinator declared inputs={}, but the subgraph_agent's
        # add_input("target_obb") must be preserved (not clobbered away).
        assert wf["subgraphs"]["twist_target"]["inputs"] == {
            "target_obb": "OrientedBoundingBox",
        }

    def test_cross_subgraph_wiring_validates(self, twist_run):
        result, _ = twist_run
        from gap.runtime.validate import validate_workflow
        from gap.runtime.workflow import load_workflow

        wf = load_workflow(result.workflow_dir / "workflow.json")
        errors = [i for i in validate_workflow(wf) if i.severity == "error"]
        assert not errors, errors
