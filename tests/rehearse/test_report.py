"""gap.rehearse.report / collect on synthetic case records (no simulator)."""

from __future__ import annotations

import json
from pathlib import Path

from gap.rehearse.collect import unit_rows
from gap.rehearse.report import DEFAULT_CLAIMS, build_feedback, render_markdown, write_feedback


def _world(held=None, gripper=0.5, soup=(0.0, 0.0, 0.0), contacts=()):
    return {
        "ee": {"position": [0.3, 0.0, 0.4], "quaternion_wxyz": [0, 1, 0, 0]},
        "gripper_open_fraction": gripper,
        "held": held,
        "objects": {"soup": {"position": list(soup), "contacts": list(contacts), "settled": True}},
        "robot_contacts": ["soup"] if held == "soup" else [],
    }


def _subgraph_record(case, *, success, grasp_exit, held_after_grasp, error=None):
    initial = _world()
    return {
        "case": case, "success": success, "reward": 1.0 if success else 0.0,
        "graph_success": True, "exit_status": "success", "error": error, "duration_s": 3.0,
        "trace_dir": f"/tmp/none/case_{case:04d}/trace",
        "initial_world": initial,
        "final_world": _world(held=None, soup=(0.2, 0.1, 0.05)),
        "visits": [
            {"node": "grasp", "visit": 0, "seq": 0, "t_start": 1.0, "t_end": 2.0, "success": True,
             "world_before": initial, "world_after": _world(held=held_after_grasp)},
            {"node": "grasp.close", "visit": 0, "seq": 1, "t_start": 1.5, "t_end": 1.9, "success": True,
             "world_before": initial, "world_after": _world(held=held_after_grasp)},
        ],
        "exits": [
            {"subgraph": "grasp_sg", "visit": 0, "exit": grasp_exit, "error_path": False, "elapsed_s": 1.0,
             "seq": 1, "world": _world(held=held_after_grasp, gripper=0.2)},
            {"subgraph": "transport_sg", "visit": 0, "exit": "placed", "error_path": False, "elapsed_s": 2.0,
             "seq": 2, "world": _world(held=None, soup=(0.2, 0.1, 0.05))},
        ],
        "checkpoints": [
            {"name": "target_held", "subgraph": "grasp_sg", "passed": held_after_grasp is not None,
             "eval_error": None, "diagnostics": {"held_body": held_after_grasp}},
        ],
    }


def _flat_record(case, *, success):
    initial = _world()
    return {
        "case": case, "success": success, "graph_success": True, "exit_status": "success",
        "error": None, "trace_dir": f"/tmp/none/flat_{case}", "initial_world": initial,
        "final_world": _world(soup=(0.3, 0, 0)),
        "visits": [
            {"node": "object0_grasp", "visit": 0, "seq": 0, "t_start": 0.0, "t_end": 0.1, "success": True,
             "world_before": initial, "world_after": initial},
            {"node": "object0_transport", "visit": 0, "seq": 1, "t_start": 0.1, "t_end": 2.0, "success": success,
             "world_before": _world(held="soup"), "world_after": _world(held="soup", soup=(0.3, 0, 0))},
        ],
        "exits": [], "checkpoints": [],
    }


def test_unit_rows_prefer_subgraph_exits_and_fall_back_to_top_level_nodes():
    sub = _subgraph_record(1, success=True, grasp_exit="grasped", held_after_grasp="soup")
    rows = unit_rows(sub)
    assert [r["unit"] for r in rows] == ["grasp_sg", "transport_sg"]
    assert rows[0]["verdict"] == "grasped"
    assert rows[1]["since_start"]["displacement_m"]["soup"] > 0.2

    flat = _flat_record(1, success=False)
    rows = unit_rows(flat)
    assert [r["unit"] for r in rows] == ["object0_grasp", "object0_transport"]
    assert rows[1]["verdict"] == "error"
    assert rows[1]["effect"]["displacement_m"]["soup"] == 0.3


def test_feedback_counts_disagreements_units_and_changes(tmp_path: Path):
    meta = {"workflow_dir": "wf", "sim": "libero_10/0", "cases": [1, 2, 3], "manifest": {"files": {"workflow.json": "a"}}}
    # Round one: case 2 claims grasped while nothing is held; case 3 fails outright.
    round1 = [
        _subgraph_record(1, success=True, grasp_exit="grasped", held_after_grasp="soup"),
        _subgraph_record(2, success=False, grasp_exit="grasped", held_after_grasp=None),
        _subgraph_record(3, success=False, grasp_exit="missed", held_after_grasp=None, error="PipelineError: boom"),
    ]
    fb1 = build_feedback(round1, meta)
    assert fb1["summary"] == {"cases": 3, "successes": 1, "previous_successes": None, "fixed": [], "broken": [],
                              "graph_has_subgraphs": True}
    dis = fb1["disagreements"]
    assert len(dis) == 1 and dis[0]["unit"] == "grasp_sg" and dis[0]["verdict"] == "grasped"
    assert dis[0]["claims"] == "held" and dis[0]["cases"] == [2]
    grasp = fb1["units"]["grasp_sg"]
    assert grasp["visits"] == 3 and grasp["verdicts"] == {"grasped": 2, "missed": 1}
    assert grasp["checkpoints"]["target_held"] == {"passed": 1, "failed": 2}
    assert grasp["held_at_exit"] == {"soup": 1, "None": 2}
    assert fb1["errors"][0]["message"].startswith("PipelineError")
    assert fb1["diff"] is None

    prev_dir = tmp_path / "r1"
    write_feedback(prev_dir, fb1)
    (prev_dir / "run.json").write_text(json.dumps({"manifest": {"files": {"workflow.json": "a"}}}))
    (prev_dir / "workflow.snapshot.json").write_text(json.dumps({
        "version": 3, "nodes": {"grasp": {"type": "subgraph", "ref": "grasp_sg"}}, "edges": [["START", "grasp"]],
        "subgraphs": {"grasp_sg": {"nodes": {"close": {"type": "tool", "tool": "robot.close_gripper",
                                                        "inputs": {"settle_steps": 60}}}}}}))
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps({
        "version": 3, "nodes": {"grasp": {"type": "subgraph", "ref": "grasp_sg"}}, "edges": [["START", "grasp"]],
        "subgraphs": {"grasp_sg": {"nodes": {"close": {"type": "tool", "tool": "robot.close_gripper",
                                                        "inputs": {"settle_steps": 90}}}}}}))

    # Round two: case 2 fixed, case 1 broken, case 3 unchanged.
    round2 = [
        _subgraph_record(1, success=False, grasp_exit="missed", held_after_grasp=None),
        _subgraph_record(2, success=True, grasp_exit="grasped", held_after_grasp="soup"),
        _subgraph_record(3, success=False, grasp_exit="missed", held_after_grasp=None),
    ]
    meta2 = dict(meta, manifest={"files": {"workflow.json": "b"}})
    fb2 = build_feedback(round2, meta2, previous=prev_dir, workflow_dir=wf_dir)
    assert fb2["summary"]["previous_successes"] == 1
    assert fb2["summary"]["fixed"] == [2] and fb2["summary"]["broken"] == [1]
    assert fb2["disagreements"] == []
    changed = fb2["diff"]["workflow"]["nodes_changed"]["grasp_sg.close"]["parameters"]["settle_steps"]
    assert changed == {"from": 60, "to": 90}
    assert fb2["diff"]["files"]["modified"] == ["workflow.json"]

    md = render_markdown(fb2)
    assert "fixed [2], broken [1]" in md
    assert "settle_steps" in md and "grasp_sg" in md
    out = tmp_path / "r2"
    write_feedback(out, fb2)
    assert json.loads((out / "feedback.json").read_text())["summary"]["successes"] == 1
    assert (out / "feedback.md").read_text().startswith("# Rehearsal feedback")


def test_flat_graph_feedback_uses_top_level_nodes():
    meta = {"workflow_dir": "wf", "sim": "libero_10/0", "cases": [1, 2], "manifest": {"files": {}}}
    fb = build_feedback([_flat_record(1, success=True), _flat_record(2, success=False)], meta)
    assert fb["summary"]["graph_has_subgraphs"] is False
    assert set(fb["units"]) == {"object0_grasp", "object0_transport"}
    assert fb["units"]["object0_transport"]["verdicts"] == {"ok": 1, "error": 1}
    assert fb["units"]["object0_transport"]["errors"] == 1
    assert any(e["node"] == "object0_transport" for e in fb["errors"])
    assert fb["disagreements"] == []  # ok/error assert nothing about the world
    assert DEFAULT_CLAIMS["grasped"] == "held"
