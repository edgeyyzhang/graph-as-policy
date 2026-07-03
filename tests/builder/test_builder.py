"""Tests for gap.builder.

Round-trip strategy: builder ↔ runtime dataclasses compared by value, since
v3 workflow.json files freely mix ``"inputs": {}`` with omitted inputs and a
byte-equal round-trip is not achievable (the parser collapses both forms).

The runtime (``gap.runtime``) ports separately; tests that need its parser
or validator ``importorskip`` it so the pure authoring surface stays
covered either way.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from gap_core.errors import GraphValidationError

from gap.builder import (
    END,
    START,
    BuilderError,
    Ref,
    Subgraph,
    Workflow,
    WorkflowSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow_paths() -> list[Path]:
    """Every v3 workflow.json checked into the repo.

    Filters out legacy schema files (the builder targets v3 only).
    """
    roots = [
        REPO_ROOT / "examples",
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("workflow.json")):
            try:
                raw = json.loads(p.read_text())
            except Exception:
                continue
            if isinstance(raw, dict) and raw.get("version") == 3:
                out.append(p)
    return out


def _normalize(wf) -> tuple:
    """Reduce a runtime Workflow to a hashable, comparison-friendly form.

    Strips ``workflow_dir`` (irrelevant for value equality) and turns
    everything else into nested tuples/dicts via ``dataclasses.asdict``.
    """
    def _walk(value):
        if is_dataclass(value):
            return tuple(
                (f.name, _walk(getattr(value, f.name)))
                for f in dataclasses.fields(value)
                if f.name != "workflow_dir"
            )
        if isinstance(value, dict):
            return tuple(sorted((k, _walk(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(_walk(v) for v in value)
        return value
    return _walk(wf)


# ---------------------------------------------------------------------------
# Round-trip: load existing JSON → builder → to_dict → parsed equals original
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_workflow_round_trip(path: Path) -> None:
    rw = pytest.importorskip("gap.runtime.workflow")
    original = rw.load_workflow(path)
    wf = Workflow.load(path)
    rebuilt = rw._parse_workflow(wf.to_dict(), path.parent)
    assert _normalize(rebuilt) == _normalize(original)


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_workflow_save_reload(tmp_path: Path, path: Path) -> None:
    rw = pytest.importorskip("gap.runtime.workflow")
    wf = Workflow.load(path)
    out = tmp_path / "workflow.json"
    wf.save(out, validate=False)        # skip structural validation here; covered by round-trip
    reloaded = Workflow.load(out)
    original = rw.load_workflow(path)
    assert _normalize(rw._parse_workflow(reloaded.to_dict(), path.parent)) == _normalize(original)


# ---------------------------------------------------------------------------
# Authoring: re-author the graph_obb_policy-style workflow from scratch
# ---------------------------------------------------------------------------


def _build_setup_sg() -> Subgraph:
    sg = Subgraph(name="setup_sg", skill="generic")
    sg.add_node("open_gripper", type="tool", tool="robot.open_gripper",
                inputs={"settle_steps": 40})
    sg.add_exit("done")
    sg.add_edge(START, "open_gripper")
    sg.add_edge("open_gripper", "done")
    sg.add_edge("done", END)
    sg.set_on_error("failed")
    return sg


def _build_target_sg() -> Subgraph:
    sg = Subgraph(name="target_sg", skill="perceiving-objects")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node("perceive_vlm", type="script",
                script="scripts/target/perceive_dino_vlm.py",
                inputs={
                    "cameras": Ref("observe.cameras"),
                    "object_name": "{{target_full}}",
                    "text_prompts": [],
                })
    sg.add_node("merge", type="script",
                script="scripts/target/merge.py",
                inputs={
                    "cameras": Ref("observe.cameras"),
                    "object_name": "{{target_full}}",
                    "vlm_result": Ref("perceive_vlm"),
                })
    sg.add_exit("found")
    sg.add_edge(START, "observe")
    sg.add_edge("observe", "perceive_vlm")
    sg.add_edge("perceive_vlm", "merge")
    sg.add_edge("merge", "found")
    sg.add_edge("found", END)
    sg.set_outputs(target_obb=Ref("merge.obb"), target_mask=Ref("merge.mask"))
    sg.set_on_error("not_found")
    return sg


def _build_approach_sg() -> Subgraph:
    sg = Subgraph(name="approach_sg", skill="generic")
    sg.add_input("target_obb", type_name="OrientedBoundingBox")
    sg.add_node("approach_above", type="script",
                script="scripts/approach/approach_above_target.py",
                inputs={
                    "target_obb": Ref("in.target_obb"),
                    "target_z": 0.3,
                    "obb_clearance": 0.12,
                })
    sg.add_exit("ready")
    sg.add_edge(START, "approach_above")
    sg.add_edge("approach_above", "ready")
    sg.add_edge("ready", END)
    sg.set_on_error("failed")
    return sg


def _build_run_sg() -> Subgraph:
    sg = Subgraph(name="run_sg", skill="generic")
    sg.add_input("observation_stream", type_name="ObservationStream")
    sg.add_node("run_policy", type="tool", tool="pi05-libero.run",
                inputs={
                    "observation_stream": Ref("in.observation_stream"),
                    "prompt": "pick the {{target}} and place it in the basket",
                    "termination_prompt": "",
                    "max_windows": 60,
                    "replan_every": 5,
                    "term_period": 6,
                    "arm_id": 0,
                    "vlm_camera": 0,
                    "settle_steps": 10,
                })
    sg.add_exit("succeeded")
    sg.add_edge(START, "run_policy")
    sg.add_edge("run_policy", "succeeded")
    sg.add_edge("succeeded", END)
    sg.set_on_error("failed")
    return sg


def _build_graph_obb_policy() -> Workflow:
    wf = Workflow(
        name="graph_obb_policy",
        description=(
            "Hybrid OBB-approach + Pi05 policy workflow for LIBERO. OBB "
            "perception localizes the target object, a Cartesian approach "
            "translates the end-effector to above it, then the libero_pi05 "
            "closed-loop VLA takes over to do the dexterous grasp + place."
        ),
    )
    wf.add_subgraph(_build_setup_sg())
    wf.add_subgraph(_build_target_sg())
    wf.add_subgraph(_build_approach_sg())
    wf.add_subgraph(_build_run_sg())

    wf.add_node("setup",    type="subgraph", ref="setup_sg")
    wf.add_node("target",   type="subgraph", ref="target_sg")
    wf.add_node("approach", type="subgraph", ref="approach_sg")
    wf.add_node("run",      type="subgraph", ref="run_sg")
    wf.add_node("done",     type="end", status="success")
    wf.add_node("abort",    type="end", status="failure",
                recovery=[{
                    "tool": "robot.open_gripper",
                    "inputs": {"settle_steps": 40},
                }])

    wf.add_edge(START, "setup")
    wf.add_conditional_edges("setup",
        {"done": "target", "failed": "abort"}, router_field="exit")
    wf.add_conditional_edges("target",
        {"found": "approach", "not_found": "abort"}, router_field="exit")
    wf.add_conditional_edges("approach",
        {"ready": "run", "failed": "abort"}, router_field="exit")
    wf.add_conditional_edges("run",
        {"succeeded": "done", "failed": "abort"}, router_field="exit")
    return wf


def test_authored_workflow_to_dict_shape() -> None:
    """The authored workflow serializes to the expected v3 JSON shape."""
    d = _build_graph_obb_policy().to_dict()
    assert d["version"] == 3
    assert d["meta"]["name"] == "graph_obb_policy"
    assert set(d["nodes"]) == {"setup", "target", "approach", "run", "done", "abort"}
    assert set(d["subgraphs"]) == {"setup_sg", "target_sg", "approach_sg", "run_sg"}
    assert d["edges"] == [[START, "setup"]]
    assert d["conditional_edges"]["run"] == {
        "router_field": "exit",
        "mapping": {"succeeded": "done", "failed": "abort"},
    }
    target = d["subgraphs"]["target_sg"]
    assert target["skill"] == "perceiving-objects"
    assert target["outputs"]["target_obb"] == {"$ref": "merge.obb"}
    assert target["exit"] == {"router_field": None, "success_values": ["found"]}
    assert target["on_error"] == "not_found"


def test_authored_workflow_parses_and_validates(tmp_path: Path) -> None:
    """Authored workflow passes the runtime parser when it is available."""
    rw = pytest.importorskip("gap.runtime.workflow")
    authored = _build_graph_obb_policy()
    wf = rw._parse_workflow(authored.to_dict(), tmp_path)
    assert wf.version == 3


# ---------------------------------------------------------------------------
# Recovery: end nodes emit {"tool", "inputs"} entries
# ---------------------------------------------------------------------------


def test_end_node_recovery_emits_tool_calls() -> None:
    wf = Workflow(name="recovery_shape")
    wf.add_node("abort", type="end", status="failure",
                recovery=[
                    {"tool": "robot.open_gripper", "inputs": {"settle_steps": 40}},
                    SimpleNamespace(tool="robot.go_home", inputs={}),  # ToolCall-shaped
                ])
    rec = wf.to_dict()["nodes"]["abort"]["recovery"]
    assert rec == [
        {"tool": "robot.open_gripper", "inputs": {"settle_steps": 40}},
        {"tool": "robot.go_home", "inputs": {}},
    ]


def test_recovery_rejects_non_tool_call() -> None:
    wf = Workflow(name="bad_recovery")
    with pytest.raises(BuilderError, match="ToolCall or dict"):
        wf.add_node("abort", type="end", status="failure", recovery=["robot.go_home"])


# ---------------------------------------------------------------------------
# WorkflowSpec — coordinator output
# ---------------------------------------------------------------------------


def test_workflow_spec_to_dict() -> None:
    spec = WorkflowSpec(name="grocery", description="Coordinator scaffold.")
    spec.declare_subgraph(
        "find_sg",
        skill="perceiving-objects",
        description="Find the soup can.",
        exit_success_values=["found"],
        on_error="not_found",
        outputs={"target_obb": "OrientedBoundingBox"},
    )
    spec.declare_subgraph(
        "grasp_sg",
        skill="grasping-direct-ik",
        description="Grasp the soup can.",
        exit_success_values=["grasped"],
        on_error="failed",
        inputs={"target_obb": "OrientedBoundingBox"},
        stage="grasp",
    )
    spec.add_subgraph_node("find", ref="find_sg")
    spec.add_subgraph_node("grasp", ref="grasp_sg")
    spec.add_end("done", status="success")
    spec.add_end("abort", status="failure",
                 recovery=[{"tool": "robot.open_gripper", "inputs": {}}])
    spec.add_edge(START, "find")
    spec.add_conditional_edges("find",
        {"found": "grasp", "not_found": "abort"}, router_field="exit")
    spec.add_conditional_edges("grasp",
        {"grasped": "done", "failed": "abort"}, router_field="exit")

    d = spec.to_dict()
    assert d["version"] == 3
    assert d["subgraphs"]["find_sg"] == {
        "skill": "perceiving-objects",
        "inputs": {},
        "outputs": {"target_obb": "OrientedBoundingBox"},
        "description": "Find the soup can.",
        "exit": {"router_field": None, "success_values": ["found"]},
        "on_error": "not_found",
    }
    assert d["subgraphs"]["grasp_sg"]["stage"] == "grasp"
    assert d["nodes"]["abort"]["recovery"] == [{"tool": "robot.open_gripper", "inputs": {}}]


def test_workflow_spec_guards() -> None:
    spec = WorkflowSpec()
    spec.declare_subgraph(
        "sg", skill="generic", description="d",
        exit_success_values=["ok"], on_error="failed",
    )
    with pytest.raises(BuilderError, match="already declared"):
        spec.declare_subgraph(
            "sg", skill="generic", description="d",
            exit_success_values=["ok"], on_error="failed",
        )
    with pytest.raises(BuilderError, match="exit_success_values"):
        spec.declare_subgraph(
            "sg2", skill="generic", description="d",
            exit_success_values=[], on_error="failed",
        )
    with pytest.raises(BuilderError, match="stage"):
        spec.declare_subgraph(
            "sg3", skill="generic", description="d",
            exit_success_values=["ok"], on_error="failed", stage="hover",
        )
    with pytest.raises(BuilderError, match="requires ref="):
        spec.add_subgraph_node("n", ref="")
    with pytest.raises(BuilderError, match="status"):
        spec.add_end("e", status="maybe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Subgraph standalone
# ---------------------------------------------------------------------------


def test_subgraph_standalone_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("gap.runtime.workflow")
    pytest.importorskip("gap.runtime.validate")
    sg = _build_target_sg()
    p = tmp_path / "target_sg.json"
    sg.save(p)
    loaded = Subgraph.load(p)
    assert loaded.name == sg.name
    assert loaded.to_dict() == sg.to_dict()


def test_subgraph_save_runs_validation(tmp_path: Path) -> None:
    """A subgraph missing add_exit fails S7 (empty success_values)."""
    pytest.importorskip("gap.runtime.workflow")
    pytest.importorskip("gap.runtime.validate")
    sg = Subgraph(name="broken_sg", skill="generic")
    sg.add_node("step", type="tool", tool="robot.open_gripper")
    sg.add_edge(START, "step")
    sg.add_edge("step", END)
    with pytest.raises(GraphValidationError) as exc:
        sg.save(tmp_path / "broken_sg.json")
    assert any("success_values" in i.message for i in exc.value.issues)


# ---------------------------------------------------------------------------
# BuilderError surface (authoring-time guards)
# ---------------------------------------------------------------------------


def test_duplicate_node_name_raises() -> None:
    sg = Subgraph(name="sg", skill="generic")
    sg.add_node("step", type="tool", tool="robot.open_gripper")
    with pytest.raises(BuilderError, match="already exists"):
        sg.add_node("step", type="noop")


def test_reserved_node_name_raises() -> None:
    sg = Subgraph(name="sg", skill="generic")
    with pytest.raises(BuilderError, match="reserved"):
        sg.add_node("START", type="noop")


def test_invalid_node_type_raises() -> None:
    sg = Subgraph(name="sg", skill="generic")
    with pytest.raises(BuilderError, match="invalid subgraph node type"):
        sg.add_node("end1", type="end")


def test_streaming_only_on_tool_or_script() -> None:
    sg = Subgraph(name="sg", skill="generic")
    with pytest.raises(BuilderError, match="streaming"):
        sg.add_node("rt", type="router", script="r.py", streaming=True)
    sg.add_node("r1", type="tool", tool="x.y", streaming=True)
    assert sg._nodes["r1"]["streaming"] is True


def test_add_exit_conflicts_with_router() -> None:
    sg = Subgraph(name="sg", skill="generic")
    sg.set_exit_router(router_field="exit", success_values=["ok"])
    with pytest.raises(BuilderError, match="set_exit_router"):
        sg.add_exit("ok")


def test_workflow_save_runs_validation(tmp_path: Path) -> None:
    """A workflow with an unreachable node should fail W4 on save()."""
    pytest.importorskip("gap.runtime.workflow")
    pytest.importorskip("gap.runtime.validate")
    wf = Workflow(name="bad")
    wf.add_node("orphan", type="end", status="success")
    # No edge from START → orphan, no end-target either.
    with pytest.raises(GraphValidationError):
        wf.save(tmp_path / "bad.json")


def test_subgraph_node_inputs_with_refs() -> None:
    """Refs round-trip cleanly through to_dict()."""
    sg = Subgraph(name="t", skill="generic")
    sg.add_node("a", type="tool", tool="x.y")
    sg.add_node("b", type="script", script="b.py",
                inputs={"x": Ref("a.out"), "y": 1})
    d = sg.to_dict()
    assert d["nodes"]["b"]["inputs"] == {"x": {"$ref": "a.out"}, "y": 1}
