"""execute(on_executor=...), checkpoint results in the trace, and rehearse()
end to end against a fake connector (stub tools, no simulator)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gap_core.tools import ToolRegistry

import gap
from gap.rehearse import rehearse
from gap.runtime.executor import WorkflowExecutor


def _workflow() -> dict:
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


_SIDECAR = textwrap.dedent('''
    from gap.runtime.verify import Checkpoint

    def _held(world) -> bool:
        return world.held_body() is not None

    CHECKPOINTS = [Checkpoint(name="target_held", subgraph="grasp_sg", predicate=_held, validate=True,
                              rationale="the gripper must hold the target after close")]
''')


def _write_workflow(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text(json.dumps(_workflow()))
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "checkpoints" / "grasp_sg.py").write_text(_SIDECAR)
    return root


class _FakeWorld:
    """Minimal stand-in for gap.runtime.verify.World."""

    robot_link_prefixes = ("robot0_",)
    robot_body_name = "robot0"
    tabletop_body_name = "table_top"

    def __init__(self, held: bool, soup_x: float):
        soup = SimpleNamespace(name="soup", position=np.array([soup_x, 0.0, 0.05]), is_region=False,
                               contacts=frozenset({"robot0_leftfinger"} if held else {"table_top"}))
        soup.is_settled = lambda: True
        table = SimpleNamespace(name="table_top", position=np.zeros(3), is_region=False, contacts=frozenset())
        table.is_settled = lambda: True
        self.bodies = {"soup": soup, "table_top": table}
        self.robot_view = SimpleNamespace(ee_position=np.array([0.1, 0.0, 0.3]),
                                          ee_quaternion_wxyz=np.array([0.0, 1.0, 0.0, 0.0]),
                                          gripper_open_fraction=0.2 if held else 0.0)
        self._held = held

    def held_body(self):
        return self.bodies["soup"] if self._held else None


class _FakeConnector:
    """Case 1 grasps the can; case 2 closes on air. Both report exit 'grasped'."""

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_callable("stub.close", self._close, summary="stub close")
        self.case = None
        self.closed = False
        self.resets: list[int] = []

    def _close(self) -> dict:
        self.moved = True
        return {"width": 0.03 if self.case == 1 else 0.0}

    def reset(self, seed=None):
        self.case = seed
        self.moved = False
        self.resets.append(seed)

    def world_snapshot(self):
        held = self.case == 1 and self.moved
        return _FakeWorld(held=held, soup_x=0.4 if held else 0.1)

    def check_success(self):
        return (self.case == 1, 1.0 if self.case == 1 else 0.0)

    def close(self):
        self.closed = True


def test_on_executor_fires_and_checkpoints_land_in_trace(tmp_path: Path):
    wf = _write_workflow(tmp_path / "wf")
    conn = _FakeConnector()
    conn.reset(seed=1)
    seen: list[WorkflowExecutor] = []
    boundaries: list[tuple[str, object]] = []

    def attach(ex: WorkflowExecutor) -> None:
        seen.append(ex)
        ex.trace.on_node_end = lambda name, ok: boundaries.append((name, ok))

    result = gap.execute(wf, conn, trace_dir=tmp_path / "trace", checkpoints="warn", on_executor=attach)
    assert result.success and result.exit_status == "success"
    assert len(seen) == 1 and isinstance(seen[0], WorkflowExecutor)
    # Tracer boundaries fire for tool/script nodes; the subgraph node itself is
    # covered by the subgraph exit hook rather than by start_node/end_node.
    assert boundaries == [("grasp_sg.close", True)]
    assert [cp.name for cp in result.checkpoint_results] == ["target_held"]

    trace = json.loads((tmp_path / "trace" / "dag_trace.json").read_text())
    grasp_node = next(n for n in trace["nodes"] if n["name"] == "grasp")
    assert grasp_node["checkpoints"][0]["name"] == "target_held"
    assert grasp_node["checkpoints"][0]["passed"] is True
    assert grasp_node["checkpoints"][0]["visit"] == 0
    assert any(e["event_type"] == "checkpoint_evaluated" for e in trace["events"])


def test_rehearse_end_to_end_with_fake_connector(tmp_path: Path):
    wf = _write_workflow(tmp_path / "wf")
    conn = _FakeConnector()
    out1 = tmp_path / "r1"
    fb = rehearse(wf, sim="fake/0", cases="1-2", out=out1, connector=conn, checkpoints="warn")

    assert conn.resets == [1, 2] and conn.closed is False  # injected connectors are not closed
    assert fb["summary"]["successes"] == 1 and fb["summary"]["cases"] == 2
    assert [c["success"] for c in fb["cases"]] == [True, False]
    # Both cases exit 'grasped'; only case 2 contradicts the world.
    assert fb["units"]["grasp_sg"]["verdicts"] == {"grasped": 2}
    assert fb["units"]["grasp_sg"]["checkpoints"]["target_held"] == {"passed": 1, "failed": 1}
    assert fb["disagreements"] == [{"unit": "grasp_sg", "verdict": "grasped", "claims": "held", "count": 1, "cases": [2]}]
    path1 = fb["per_case"][0]["path"]
    assert path1[0]["unit"] == "grasp_sg" and path1[0]["held"] == "soup"
    assert path1[0]["moved_since_start_m"] == {"soup": 0.3}
    assert (out1 / "cases" / "case_0001" / "case.json").exists()
    assert (out1 / "cases" / "case_0001" / "trace" / "dag_trace.json").exists()
    assert (out1 / "workflow.snapshot.json").exists() and (out1 / "feedback.md").exists()

    rec = json.loads((out1 / "cases" / "case_0002" / "case.json").read_text())
    nodes = [v["node"] for v in rec["visits"]]
    assert nodes == ["grasp_sg.close"]
    assert rec["visits"][0]["world_before"]["held"] is None
    assert rec["exits"][0]["exit"] == "grasped" and rec["exits"][0]["world"]["held"] is None

    # Round two with a changed parameter: the diff and the per-case changes show up.
    data = json.loads((wf / "workflow.json").read_text())
    data["subgraphs"]["grasp_sg"]["nodes"]["close"]["inputs"] = {"settle_steps": 90}
    (wf / "workflow.json").write_text(json.dumps(data))
    fb2 = rehearse(wf, sim="fake/0", cases=[1, 2], out=tmp_path / "r2", previous=out1, connector=_FakeConnector())
    assert fb2["summary"]["previous_successes"] == 1
    assert fb2["summary"]["fixed"] == [] and fb2["summary"]["broken"] == []
    assert fb2["diff"]["workflow"]["nodes_changed"]["grasp_sg.close"]["parameters"] == {
        "settle_steps": {"from": None, "to": 90}}
    assert "Changes since previous round" in (tmp_path / "r2" / "feedback.md").read_text()
