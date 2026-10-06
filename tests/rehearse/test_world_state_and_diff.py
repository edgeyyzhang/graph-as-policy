"""gap.rehearse.world_state and gap.rehearse.diff on hand-built inputs."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gap.rehearse.diff import is_empty, manifest_diff, workflow_diff
from gap.rehearse.world_state import diff, object_names, summarize


def _body(name, pos, contacts=(), is_region=False):
    b = SimpleNamespace(name=name, position=np.asarray(pos, float), contacts=frozenset(contacts),
                        is_region=is_region)
    b.is_settled = lambda: True
    return b


def _fake_world(held=None):
    bodies = {
        "soup": _body("soup", (0.1, 0.2, 0.05), contacts=("robot0_leftfinger",) if held else ("table_top",)),
        "basket": _body("basket", (0.3, 0.0, 0.05)),
        "table_top": _body("table_top", (0, 0, 0)),
        "robot0_link7": _body("robot0_link7", (0.1, 0.2, 0.3)),
        "basket_region": _body("basket_region", (0.3, 0.0, 0.05), is_region=True),
    }
    w = SimpleNamespace(
        bodies=bodies,
        robot_view=SimpleNamespace(ee_position=np.array([0.1, 0.2, 0.3]),
                                   ee_quaternion_wxyz=np.array([0, 1, 0, 0]), gripper_open_fraction=0.25),
        robot_link_prefixes=("robot0_",), robot_body_name="robot0", tabletop_body_name="table_top",
    )
    w.held_body = lambda: bodies["soup"] if held else None
    return w


def test_summarize_keeps_objects_only_and_reports_held_and_contacts():
    w = _fake_world(held=True)
    assert object_names(w) == ["basket", "soup"]
    s = summarize(w)
    assert s["held"] == "soup"
    assert s["gripper_open_fraction"] == 0.25
    assert s["ee"]["position"] == [0.1, 0.2, 0.3]
    assert set(s["objects"]) == {"basket", "soup"}
    assert s["objects"]["soup"]["contacts"] == ["robot0_leftfinger"]
    assert s["robot_contacts"] == ["soup"]
    assert summarize(w, objects=["soup"])["objects"].keys() == {"soup"}
    assert summarize(None)["held"] is None


def test_diff_reports_displacement_and_transitions():
    before = summarize(_fake_world(held=False))
    after_world = _fake_world(held=True)
    after_world.bodies["soup"].position = np.array([0.4, 0.2, 0.05])
    after = summarize(after_world)
    d = diff(before, after)
    assert d["held_before"] is None and d["held_after"] == "soup"
    assert abs(d["displacement_m"]["soup"] - 0.3) < 1e-6
    assert d["displacement_m"]["basket"] == 0.0


def test_workflow_diff_flattens_subgraphs_and_separates_parameters_from_wiring():
    a = {"version": 3,
         "nodes": {"grasp": {"type": "subgraph", "ref": "g"}, "done": {"type": "end", "status": "success"}},
         "edges": [["START", "grasp"]],
         "conditional_edges": {"grasp": {"router_field": "exit", "mapping": {"grasped": "done", "failed": "done"}}},
         "subgraphs": {"g": {"nodes": {"cands": {"type": "tool", "tool": "geometry.top_down_grasp_candidates",
                                                  "inputs": {"obb": {"$ref": "in.obb"}, "z_offset": -0.04}},
                                        "close": {"type": "tool", "tool": "robot.close_gripper", "inputs": {}}},
                             "edges": [["START", "cands"], ["cands", "close"]],
                             "exit": {"router_field": None, "success_values": ["grasped"]}, "on_error": "failed"}}}
    b = {"version": 3,
         "nodes": {"grasp": {"type": "subgraph", "ref": "g"}, "done": {"type": "end", "status": "success"},
                   "abort": {"type": "end", "status": "failure"}},
         "edges": [["START", "grasp"]],
         "conditional_edges": {"grasp": {"router_field": "exit", "mapping": {"grasped": "done", "failed": "abort"}}},
         "subgraphs": {"g": {"nodes": {"cands": {"type": "tool", "tool": "geometry.top_down_grasp_candidates",
                                                  "inputs": {"obb": {"$ref": "in.obb"}, "z_offset": -0.02}},
                                        "close": {"type": "tool", "tool": "robot.close_gripper", "inputs": {}},
                                        "check": {"type": "script", "script": "scripts/check.py", "inputs": {}}},
                             "edges": [["START", "cands"], ["cands", "close"], ["close", "check"]],
                             "exit": {"router_field": None, "success_values": ["grasped", "missed"]},
                             "on_error": "failed"}}}
    d = workflow_diff(a, b)
    assert d["nodes_added"] == ["abort", "g.check"]
    assert d["nodes_removed"] == []
    assert d["nodes_changed"] == {"g.cands": {"parameters": {"z_offset": {"from": -0.04, "to": -0.02}}}}
    assert ["g.close", "g.check"] in d["edges_added"]
    assert "grasp" in d["routing_changed"] and "g#exit" in d["routing_changed"]
    assert not is_empty(d)
    assert is_empty(workflow_diff(a, a))
    assert manifest_diff({"x": "1", "y": "2"}, {"y": "3", "z": "4"}) == {"added": ["z"], "removed": ["x"], "modified": ["y"]}
