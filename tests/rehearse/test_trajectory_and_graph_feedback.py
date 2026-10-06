"""Per-step trajectories and the per-graph feedback files, against a fake
connector whose stub tools step a fake simulator handle (no simulator)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gap_core.tools import ToolRegistry

from gap.rehearse import rehearse
from gap.rehearse import trajectory as traj
from gap.rehearse.graph_feedback import SECTIONS
from gap.rehearse.trajectory import StepSampler
from gap.rehearse.values import compact, mapping_text, text
from gap.rehearse.world_state import changed_bodies, pose_changed

STEPS_PER_CLOSE = 7


def _subgraph_workflow() -> dict:
    return {
        "version": 3, "meta": {},
        "nodes": {
            "grasp": {"type": "subgraph", "ref": "grasp_sg"},
            "done": {"type": "end", "status": "success"},
            "abort": {"type": "end", "status": "failure"},
        },
        "edges": [["START", "grasp"]],
        "conditional_edges": {"grasp": {"router_field": "exit", "mapping": {"grasped": "done", "failed": "abort"}}},
        "subgraphs": {"grasp_sg": {
            "skill": "stub", "inputs": {}, "outputs": {"width": {"$ref": "close.width"}},
            "nodes": {"close": {"type": "tool", "tool": "stub.close", "inputs": {}}, "grasped": {"type": "noop"}},
            "edges": [["START", "close"], ["close", "grasped"], ["grasped", "END"]],
            "conditional_edges": {},
            "exit": {"router_field": None, "success_values": ["grasped"]}, "on_error": "failed",
        }},
    }


def _flat_workflow() -> dict:
    return {
        "version": 3, "meta": {},
        "nodes": {
            "close": {"type": "tool", "tool": "stub.close", "inputs": {}},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "close"], ["close", "done"]],
        "conditional_edges": {}, "subgraphs": {},
    }


def _write(root: Path, workflow: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text(json.dumps(workflow))
    return root


class _World:
    robot_link_prefixes = ("robot0_",)
    robot_body_name = "robot0"
    tabletop_body_name = "table_top"

    def __init__(self, soup_x: float, held: bool, time_s: float):
        def body(name, position, contacts):
            b = SimpleNamespace(name=name, position=np.array(position), is_region=False,
                                quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), contacts=frozenset(contacts))
            b.is_settled = lambda: True
            return b
        self.bodies = {
            "soup": body("soup", [soup_x, 0.0, 0.05], {"robot0_leftfinger"} if held else {"table_top"}),
            "basket": body("basket", [0.7, 0.2, 0.0], {"table_top"}),
            "table_top": body("table_top", [0.0, 0.0, 0.0], set()),
        }
        self.robot_view = SimpleNamespace(
            ee_position=np.array([soup_x, 0.0, 0.3]), ee_quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
            gripper_open_fraction=0.2 if held else 1.0, joint_pos=np.zeros(7))
        self.time_s = time_s
        self._held = held

    def held_body(self):
        return self.bodies["soup"] if self._held else None


class _Handle:
    """Stand-in for the simulator handle: each step moves the soup 1 cm."""

    def __init__(self, owner):
        self.owner = owner

    def step(self, action=None):
        self.owner.ticks += 1
        self.owner.soup_x += 0.01
        return None, 0.0, False, {}


class _Connector:
    def __init__(self, with_handle: bool = True):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_callable("stub.close", self._close, summary="stub close")
        self.ticks = 0
        self.soup_x = 0.1
        self.held = False
        self.case = None
        if with_handle:
            self.env = SimpleNamespace(handle=_Handle(self))

    def _close(self) -> dict:
        handle = getattr(getattr(self, "env", None), "handle", None)
        for _ in range(STEPS_PER_CLOSE):
            if handle is not None:
                handle.step(None)
            else:
                self.soup_x += 0.01
        self.held = True
        return {"width": 0.03}

    def reset(self, seed=None):
        self.case, self.ticks, self.soup_x, self.held = seed, 0, 0.1, False

    def world_snapshot(self):
        return _World(self.soup_x, self.held, self.ticks * 0.05)

    def check_success(self):
        return (self.case == 1, 1.0 if self.case == 1 else 0.0)

    def close(self):
        pass


def _headings(markdown: str) -> list[str]:
    return re.findall(r"^## (.+)$", markdown, flags=re.MULTILINE)


# --- values and state ------------------------------------------------------


def test_compact_reduces_by_structure_only():
    big = np.zeros((512, 800), dtype=np.uint8)
    assert compact(big) == "<array shape=(512, 800) dtype=uint8>"
    assert compact(np.array([1.0, 2.0, 3.00004])) == [1.0, 2.0, 3.0]
    assert compact(list(range(20))) == {"count": 20, "first": [0, 1, 2]}
    assert compact({"a": {"x": 0.123456, "y": 1, "z": float("nan")}}) == {"a": {"x": 0.1235, "y": 1, "z": "nan"}}


def test_text_renders_vectors_and_mappings_on_one_line():
    assert text({"x": 0.4721, "y": 0.001, "z": 0.3168}) == "(x 0.472, y 0.001, z 0.317)"
    assert text([0.1, 0.2, 0.3]) == "(0.100, 0.200, 0.300)"
    assert text({"count": 12, "first": [1, 2, 3]}) == "list of 12, first: 1; 2; 3"
    assert mapping_text({}) == "none"
    assert mapping_text({"obb": {"center": {"x": 1.0, "y": 2.0, "z": 3.0}}}) == "obb: center (x 1.000, y 2.000, z 3.000)"


def test_change_tests_use_fixed_tolerances():
    a = {"position": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    assert not pose_changed(a, {"position": [0.0005, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    assert pose_changed(a, {"position": [0.002, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]})
    assert pose_changed(a, {"position": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.9990, 0.0436, 0.0, 0.0]})  # 5 degrees
    s0 = {"objects": {"soup": {"position": [0.1, 0, 0], "contacts": ["table_top"]},
                      "basket": {"position": [0.7, 0, 0], "contacts": []}}}
    s1 = {"objects": {"soup": {"position": [0.1, 0, 0], "contacts": ["robot0_leftfinger"]},
                      "basket": {"position": [0.7, 0, 0], "contacts": []}}}
    assert changed_bodies([s0, s1]) == ["soup"]  # contacts changed, pose did not


# --- sampler ---------------------------------------------------------------


def test_sampler_records_every_step_and_restores_the_handle(tmp_path: Path):
    conn = _Connector()
    handle = conn.env.handle
    sampler = StepSampler(conn)
    assert sampler.attach() and "step" in handle.__dict__
    sampler.open(tmp_path / "trajectory.jsonl")
    sampler.mark("initial")
    sampler.set_node("grasp_sg.close", 0, "grasp_sg", 0)
    sampler.mark("start")
    for _ in range(STEPS_PER_CLOSE):
        handle.step(None)
    sampler.mark("end")
    assert sampler.node_steps == STEPS_PER_CLOSE
    sampler.clear_node()
    handle.step(None)  # a step outside any node
    sampler.close()
    sampler.detach()

    assert "step" not in handle.__dict__ and handle.step.__func__ is _Handle.step
    rows = list(traj.read(tmp_path / "trajectory.jsonl"))
    assert [r["i"] for r in rows] == list(range(len(rows)))
    assert [r["kind"] for r in rows] == ["initial", "start"] + ["step"] * STEPS_PER_CLOSE + ["end", "step"]
    assert rows[0]["node"] is None and rows[-1]["node"] is None
    assert {r["node"] for r in rows[1:-1]} == {"grasp_sg.close"}
    assert rows[2]["state"]["objects"]["soup"]["position"][0] == pytest.approx(0.11)
    assert rows[2]["state"]["objects"]["soup"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert len(rows[2]["state"]["joints"]) == 7


def test_sampler_restores_the_handle_after_a_failing_step(tmp_path: Path):
    conn = _Connector()
    handle = conn.env.handle

    def boom(action=None):
        raise RuntimeError("sim crashed")

    handle.step = boom  # an instance-level step is put back as it was
    sampler = StepSampler(conn)
    assert sampler.attach()
    sampler.open(tmp_path / "t.jsonl")
    with pytest.raises(RuntimeError):
        handle.step(None)
    sampler.close()
    sampler.detach()
    assert handle.step is boom


def test_sampler_without_a_step_handle_is_unavailable():
    assert StepSampler(_Connector(with_handle=False)).attach() is False


# --- rehearse end to end ---------------------------------------------------


def test_rehearse_writes_trajectories_and_one_file_per_graph(tmp_path: Path):
    wf = _write(tmp_path / "wf", _subgraph_workflow())
    out = tmp_path / "r1"
    conn = _Connector()
    rehearse(wf, sim="fake/0", cases="1-2", out=out, connector=conn, trajectory_interval=3)
    assert "step" not in conn.env.handle.__dict__  # restored after the rehearsal

    record = json.loads((out / "cases" / "case_0001" / "case.json").read_text())
    assert record["trajectory"]["steps"] == STEPS_PER_CLOSE and record["trajectory"]["per_step"] is True
    visit = record["visits"][0]
    assert (visit["node"], visit["unit"], visit["unit_visit"], visit["steps"]) == ("grasp_sg.close", "grasp_sg", 0, 7)
    assert visit["outputs"] == {"width": 0.03} and visit["action"] == {}
    assert record["exits"][0]["outputs"] == {"width": 0.03}

    rows = list(traj.read(out / "cases" / "case_0001" / "trajectory.jsonl"))
    assert [r["kind"] for r in rows] == ["initial", "start"] + ["step"] * STEPS_PER_CLOSE + ["end", "final"]

    main = (out / "feedback" / "main.md").read_text()
    sub = (out / "feedback" / "subgraphs" / "grasp_sg.md").read_text()
    expected = [f"{i + 1}. {name}" for i, name in enumerate(SECTIONS)]
    assert _headings(main) == expected and _headings(sub) == expected
    assert "Task success, judged by the simulator: 1 of 2 cases." in main
    assert "Task success, judged by the simulator: 1 of 2 cases." in sub
    assert "#### grasp (grasp_sg), visit 0: grasped" in main
    assert "#### close, visit 0: ok" in sub
    assert "outputs: width: 0.030" in main and "outputs: width: 0.030" in sub
    assert "soup position: (0.100, 0.000, 0.050) -> (0.170, 0.000, 0.050)" in sub
    assert "basket position" not in sub  # unchanged bodies are left out of a state change
    # The start of a case lists every body.
    assert "basket position: (0.700, 0.200, 0.000)" in main

    # Every trajectory path written in a feedback file exists.
    pointers = set(re.findall(r"`(trajectories/[^`]+)`", main + sub))
    assert pointers == {"trajectories/case_0001/main.md", "trajectories/case_0002/main.md",
                        "trajectories/case_0001/grasp_sg.md", "trajectories/case_0002/grasp_sg.md"}
    assert all((out / p).exists() for p in pointers)
    assert (out / "feedback" / "subgraphs" / "grasp_sg.json").exists()

    view = (out / "trajectories" / "case_0001" / "grasp_sg.md").read_text()
    assert "### close (visit 0)" in view and "7 simulator steps" in view
    assert "soup position" in view and "basket" not in view
    body = [ln for ln in view.splitlines() if ln.startswith("| ") and not ln.startswith("| i ")]
    # start, steps 3 and 6, the last step (7), end
    assert [ln.split("|")[2].strip() for ln in body] == ["start", "step", "step", "step", "end"]


def test_flat_graph_gets_the_main_file_only(tmp_path: Path):
    wf = _write(tmp_path / "wf", _flat_workflow())
    out = tmp_path / "r1"
    rehearse(wf, sim="fake/0", cases="1", out=out, connector=_Connector())
    main = (out / "feedback" / "main.md").read_text()
    assert _headings(main) == [f"{i + 1}. {name}" for i, name in enumerate(SECTIONS)]
    assert "#### close, visit 0: ok" in main
    assert not (out / "feedback" / "subgraphs").exists()
    assert sorted(p.name for p in (out / "trajectories" / "case_0001").iterdir()) == ["main.md"]


def test_connector_without_a_step_handle_keeps_node_boundaries(tmp_path: Path):
    wf = _write(tmp_path / "wf", _subgraph_workflow())
    out = tmp_path / "r1"
    rehearse(wf, sim="fake/0", cases="1", out=out, connector=_Connector(with_handle=False))
    record = json.loads((out / "cases" / "case_0001" / "case.json").read_text())
    assert record["trajectory"]["per_step"] is False and record["trajectory"]["steps"] == 0
    rows = list(traj.read(out / "cases" / "case_0001" / "trajectory.jsonl"))
    assert [r["kind"] for r in rows] == ["initial", "start", "end", "final"]
    sub = (out / "feedback" / "subgraphs" / "grasp_sg.md").read_text()
    assert "soup position: (0.100, 0.000, 0.050) -> (0.170, 0.000, 0.050)" in sub


def test_no_trajectory_still_writes_the_graph_files(tmp_path: Path):
    wf = _write(tmp_path / "wf", _subgraph_workflow())
    out = tmp_path / "r1"
    conn = _Connector()
    rehearse(wf, sim="fake/0", cases="1", out=out, connector=conn, trajectory=False)
    assert not (out / "cases" / "case_0001" / "trajectory.jsonl").exists()
    assert not (out / "trajectories").exists()
    sub = (out / "feedback" / "subgraphs" / "grasp_sg.md").read_text()
    assert "trajectories/" not in sub and "#### close, visit 0: ok" in sub


def test_changes_go_to_the_graph_they_belong_to(tmp_path: Path):
    wf = _write(tmp_path / "wf", _subgraph_workflow())
    rehearse(wf, sim="fake/0", cases="1-2", out=tmp_path / "r1", connector=_Connector())
    data = json.loads((wf / "workflow.json").read_text())
    data["subgraphs"]["grasp_sg"]["nodes"]["close"]["inputs"] = {"settle_steps": 90}
    (wf / "workflow.json").write_text(json.dumps(data))
    out = tmp_path / "r2"
    rehearse(wf, sim="fake/0", cases="1-2", out=out, previous=tmp_path / "r1", connector=_Connector())

    main = (out / "feedback" / "main.md").read_text()
    sub = json.loads((out / "feedback" / "subgraphs" / "grasp_sg.json").read_text())
    assert "No change to this graph." in main
    assert "Previous round: 1 of 2. Fixed: none. Broken: none." in main
    assert sub["changes"]["nodes_changed"] == {
        "grasp_sg.close": {"parameters": {"settle_steps": {"from": None, "to": 90}}}}
    assert "settle_steps" in (out / "feedback" / "subgraphs" / "grasp_sg.md").read_text()


def test_joint_moves_count_as_changes_and_are_rendered():
    from gap.rehearse.graph_feedback import state_change
    from gap.rehearse.trajectory_view import table
    from gap.rehearse.world_state import joints_changed

    closed = {"position": [0.6, 0.3, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0], "contacts": [],
              "joints": {"cabinet_bottom_level": 0.0, "cabinet_top_level": 0.0}}
    opened = dict(closed, joints={"cabinet_bottom_level": 0.12, "cabinet_top_level": 0.0})
    assert not joints_changed(closed, dict(closed, joints={"cabinet_bottom_level": 0.0005, "cabinet_top_level": 0.0}))
    assert joints_changed(closed, opened)
    s0 = {"objects": {"cabinet": closed}, "ee": {"position": [0, 0, 0.3]}}
    s1 = {"objects": {"cabinet": opened}, "ee": {"position": [0, 0, 0.3]}}
    assert changed_bodies([s0, s1]) == ["cabinet"]
    change = state_change(s0, s1)
    assert change == {"cabinet joints": [closed["joints"], opened["joints"]]}
    rows = table([{"i": 0, "kind": "start", "state": s0}, {"i": 1, "kind": "step", "state": s1}], interval=1)
    assert "cabinet joints" in rows.splitlines()[0]
    assert "bottom_level=0.120, top_level=0.000" in rows
