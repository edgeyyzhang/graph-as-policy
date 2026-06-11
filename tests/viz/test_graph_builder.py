"""graph_builder tests — golden v3 graphs → WorkflowGraph JSON snapshots.

Port schemas must surface gap.schema type names ("PointCloud", "Se3Pose",
...) via tool ``UnitSchema``s and script type hints — no proto descriptors.
"""

from __future__ import annotations

from pathlib import Path

from gap.runtime.workflow import load_workflow
from gap.viz.graph_builder import build_workflow_graph

from .conftest import streaming_graph, two_stage_graph, write_workflow


def _build(tmp_path: Path, raw: dict, registry, *, check_script: bool = False):
    wf_dir = write_workflow(tmp_path, raw, check_script=check_script)
    wf = load_workflow(wf_dir / "workflow.json")
    return build_workflow_graph(wf, wf_dir, tool_registry=registry)


# ---------------------------------------------------------------------------
# Golden graph 1: two subgraphs, cross-subgraph binding, conditional edges
# ---------------------------------------------------------------------------


def test_two_stage_structure(tmp_path, golden_tool_registry) -> None:
    graph = _build(tmp_path, two_stage_graph(), golden_tool_registry, check_script=True)

    assert graph.begin == "perceive_sg"
    assert graph.meta == {"name": "two_stage"}

    # Subgraph schemas: two real subgraphs + two end pseudo-subgraphs.
    by_id = {sg.subgraph_id: sg for sg in graph.subgraphs}
    assert set(by_id) == {"perceive_sg", "grasp_sg", "done", "abort"}
    assert by_id["perceive_sg"].agent == "perception"
    assert by_id["perceive_sg"].outputs == {"target_cloud": "detect.cloud"}
    assert by_id["perceive_sg"].begin_state == "detect"
    assert by_id["grasp_sg"].inputs == {"target_cloud": "PointCloud"}
    assert by_id["done"].is_end and by_id["done"].end_status == "success"
    assert by_id["abort"].is_end and by_id["abort"].end_status == "failure"
    assert [r.tool for r in by_id["abort"].recovery] == ["robot.open_gripper"]

    # State schemas (the on_error symbol becomes a synthetic end state).
    states = {s.state_id: s for s in graph.states}
    assert set(states) == {
        "perceive_sg.detect", "perceive_sg.check", "perceive_sg.found",
        "perceive_sg.not_found",
        "grasp_sg.plan", "grasp_sg.move", "grasp_sg.grasped",
        "grasp_sg.failed",
    }
    assert states["perceive_sg.detect"].state_type == "tool"
    assert states["perceive_sg.detect"].skill == "vision.detect"
    assert states["perceive_sg.check"].state_type == "script"
    assert states["perceive_sg.check"].script == "scripts/check.py"
    assert states["perceive_sg.found"].state_type == "end"
    assert states["perceive_sg.not_found"].state_type == "end"
    assert states["perceive_sg.check"].inputs_def == {"cloud": "detect.cloud"}


def test_two_stage_control_edges(tmp_path, golden_tool_registry) -> None:
    graph = _build(tmp_path, two_stage_graph(), golden_tool_registry, check_script=True)

    edges = {(e.source, e.target, e.edge_type, e.label) for e in graph.control_edges}
    assert edges == {
        # intra-subgraph success edges
        ("perceive_sg.detect", "perceive_sg.check", "success", ""),
        ("grasp_sg.plan", "grasp_sg.move", "success", ""),
        ("grasp_sg.move", "grasp_sg.grasped", "success", ""),
        # intra-subgraph conditional transition
        ("perceive_sg.check", "perceive_sg.found", "transition", "yes"),
        # top-level transitions (exit value → next subgraph / end node)
        ("perceive_sg.found", "grasp_sg", "transition", "found"),
        ("perceive_sg.not_found", "abort", "transition", "not_found"),
        ("grasp_sg.grasped", "done", "transition", "grasped"),
        ("grasp_sg.failed", "abort", "transition", "failed"),
    }


def test_two_stage_data_edges_and_ports(tmp_path, golden_tool_registry) -> None:
    graph = _build(tmp_path, two_stage_graph(), golden_tool_registry, check_script=True)

    data = sorted(
        (d.model_dump() for d in graph.data_edges),
        key=lambda d: (d["source"], d["target"]),
    )
    assert data == [
        {
            "source": "grasp_sg.plan", "source_field": "pose",
            "target": "grasp_sg.move", "target_field": "pose",
            "ref_path": "plan.pose", "cross_subgraph": False,
            "type_label": "Se3Pose",
            "source_sg_port": "", "target_sg_port": "",
        },
        {
            "source": "perceive_sg.detect", "source_field": "cloud",
            "target": "grasp_sg.plan", "target_field": "cloud",
            "ref_path": "in.target_cloud", "cross_subgraph": True,
            "type_label": "PointCloud",
            "source_sg_port": "target_cloud", "target_sg_port": "target_cloud",
        },
        {
            "source": "perceive_sg.detect", "source_field": "cloud",
            "target": "perceive_sg.check", "target_field": "cloud",
            "ref_path": "detect.cloud", "cross_subgraph": False,
            "type_label": "PointCloud",
            "source_sg_port": "", "target_sg_port": "",
        },
    ]

    # Port schemas: gap.schema names from tool UnitSchemas and script hints.
    states = {s.state_id: s for s in graph.states}
    detect_out = {p.name: p for p in states["perceive_sg.detect"].output_ports}
    assert detect_out["cloud"].type_label == "PointCloud"
    assert detect_out["cloud"].is_message and not detect_out["cloud"].is_repeated
    assert detect_out["label"].type_label == "str"
    assert not detect_out["label"].is_message

    detect_in = {p.name: p for p in states["perceive_sg.detect"].input_ports}
    assert set(detect_in) == {"camera"}  # only *used* inputs become ports
    assert detect_in["camera"].type_label == "str"

    check_in = {p.name: p for p in states["perceive_sg.check"].input_ports}
    assert check_in["cloud"].type_label == "PointCloud"
    check_out = {p.name: p for p in states["perceive_sg.check"].output_ports}
    assert check_out["verdict"].type_label == "str"

    plan_out = {p.name: p for p in states["grasp_sg.plan"].output_ports}
    assert plan_out["pose"].type_label == "Se3Pose"
    assert plan_out["pose"].is_message
    assert plan_out["score"].type_label == "float"


def test_unregistered_tool_degrades_to_portless_state(tmp_path) -> None:
    """No tool registry → states still build, just without port schemas."""
    graph = _build(tmp_path, two_stage_graph(), None, check_script=True)
    states = {s.state_id: s for s in graph.states}
    detect = states["perceive_sg.detect"]
    assert detect.state_type == "tool"
    # No registry: only ports synthesized from data edges remain.
    assert {p.name for p in detect.output_ports} == {"cloud"}
    assert {p.type_label for p in detect.output_ports} == {"?"}
    # Script schemas don't need the registry — they introspect run() hints.
    assert {p.name for p in states["perceive_sg.check"].output_ports} == {"verdict"}


# ---------------------------------------------------------------------------
# Golden graph 2: streaming source node
# ---------------------------------------------------------------------------


def test_streaming_graph(tmp_path, golden_tool_registry) -> None:
    graph = _build(tmp_path, streaming_graph(), golden_tool_registry)

    assert graph.begin == "servo_sg"
    states = {s.state_id: s for s in graph.states}
    assert states["servo_sg.track"].state_type == "tool"
    assert states["servo_sg.track"].skill == "tracker.stream"

    # The whole-node $ref ("track") becomes a data edge with the synthetic
    # "_out" port on the streaming source.
    [edge] = graph.data_edges
    assert edge.model_dump() == {
        "source": "servo_sg.track", "source_field": "_out",
        "target": "servo_sg.servo", "target_field": "pose",
        "ref_path": "track", "cross_subgraph": False,
        "type_label": "",
        "source_sg_port": "", "target_sg_port": "",
    }
    track_ports = {p.name for p in states["servo_sg.track"].output_ports}
    assert "_out" in track_ports

    # Streaming nodes have no outgoing control edges.
    sources = {e.source for e in graph.control_edges}
    assert "servo_sg.track" not in sources
    assert ("servo_sg.servo", "servo_sg.ok") in {
        (e.source, e.target) for e in graph.control_edges
    }


def test_workflow_graph_serializes_to_json(tmp_path, golden_tool_registry) -> None:
    """The pydantic surface the API returns must be JSON-serializable."""
    graph = _build(tmp_path, two_stage_graph(), golden_tool_registry, check_script=True)
    payload = graph.model_dump()
    assert payload["begin"] == "perceive_sg"
    assert isinstance(payload["states"], list)
    assert isinstance(payload["control_edges"], list)
    import json
    json.dumps(payload)  # must not raise
