"""Unit tests for scripts/migrate_tool_names.py (fixture rewrites + idempotency)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_tool_names.py"


@pytest.fixture(scope="module")
def mig():
    spec = importlib.util.spec_from_file_location("migrate_tool_names", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["migrate_tool_names"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GRAPH = {
    "version": 3,
    "nodes": {
        "done": {"type": "end", "status": "success"},
        "abort": {
            "type": "end",
            "status": "failure",
            "recovery": [
                {"service": "gripper.v1.Gripper", "method": "Open", "inputs": {}},
                {"service": "robot_control.v1.RobotControl", "method": "GoHome", "inputs": {}},
            ],
        },
    },
    "subgraphs": {
        "sg": {
            "nodes": {
                "observe": {"type": "tool", "tool": "observation.GetObservation"},
                "seg": {
                    "type": "tool",
                    "tool": "sam3.SegmentBox",
                    "inputs": {"box": {"$ref": "detect.box"}},
                },
                "detect": {"type": "tool", "tool": "grounding_dino.Detect"},
                "drop": {"type": "tool", "tool": "geometry_svc.ComputeDropPosition"},
                "plan": {"type": "tool", "tool": "curobo.PlanLinear"},
                "track": {"type": "tool", "tool": "sam3_tracker.StartTracking"},
                "qualified": {
                    "type": "tool",
                    "tool": "observation.v1.Observation.GetObservation",
                },
            },
        },
    },
}

_SCRIPT_PY = '''\
def run(ctx, target):
    obs = ctx.call("observation.v1.Observation", "GetObservation")
    ee = ctx.call("robot_control.v1.RobotControl", "GetEEPose", arm_id=0)
    ctx.call("robot_control.v1.RobotControl", "GoToPose", pose=target)
    pts = ctx.call(
        "geometry_svc.v1.GeometrySvc",
        "MaskToWorldPoints",
        cameras=obs.cameras,
    )
    ctx.call("gripper.v1.Gripper", "Open", settle_steps=60)
    ans = ctx.call("vlm.v1.VLM", "QueryYesNo", question="done?")
    return pts, ans
'''

_SKILL_MD = """\
---
name: fixture
description: Test fixture.
allowed-tools:
  - robot_control.GoToPose
  - robot_control.GoHome
  - gripper.Open
  - geometry_svc.MaskToWorldPoints
  - observation.GetObservation
  - sam3.SegmentBox
  - molmo.QueryYesNo
gap:
  allowed_tools: [sim_bridge.CheckTaskCompletion, gemini_er.Detect]
---

# fixture

Calls robot_control.GoToPose then gripper.Open.
"""


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "graph").mkdir()
    (tmp_path / "graph" / "workflow.json").write_text(json.dumps(_GRAPH, indent=2))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "do_thing.py").write_text(_SCRIPT_PY)
    (tmp_path / "SKILL.md").write_text(_SKILL_MD)
    return tmp_path


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------


def test_explicit_map_coverage(mig):
    cases = {
        ("observation", "GetObservation"): "robot.get_observation",
        ("observation", "GetCameraPose"): "robot.get_camera_pose",
        ("robot_control", "GoToPose"): "robot.go_to_pose",
        ("robot_control", "GoToPoseCartesian"): "robot.go_to_pose_cartesian",
        ("robot_control", "ExecuteJointTrajectory"): "robot.execute_trajectory",
        ("robot_control", "MoveToJoints"): "robot.move_to_joints",
        ("robot_control", "GoHome"): "robot.go_home",
        ("robot_control", "GetEEPose"): "robot.get_ee_pose",
        ("gripper", "Open"): "robot.open_gripper",
        ("gripper", "Close"): "robot.close_gripper",
        ("gripper", "GetPosition"): "robot.get_gripper",
        ("gripper", "GetPose"): "robot.get_gripper_pose",
        ("sim_bridge", "Reset"): "sim.reset",
        ("sim_bridge", "StepOnce"): "sim.step",
        ("sim_bridge", "CheckTaskCompletion"): "sim.check_success",
        ("sim_bridge", "ApplyPolicyAction"): "sim.apply_policy_action",
        ("sim_bridge", "EnableVideoCapture"): "sim.enable_video",
        ("sim_bridge", "SaveVideo"): "sim.save_video",
        ("sam3", "SegmentText"): "sam3.segment_text",
        ("sam3", "SegmentPoint"): "sam3.segment_point",
        ("sam3", "SegmentBox"): "sam3.segment_box",
        ("sam3_tracker", "StartTracking"): "sam3.tracker_start_tracking",
        ("grounding_dino", "Detect"): "grounding-dino.detect",
        ("molmo", "PointPrompt"): "molmo.point_prompt",
        ("molmo", "Query"): "molmo.query",
        ("molmo", "QueryYesNo"): "molmo.query_yes_no",
        ("vlm", "Query"): "vlm.query",
        ("vlm", "QueryYesNo"): "vlm.query_yes_no",
        ("gemini_er", "Detect"): "gemini-er.detect",
        ("geometry_svc", "ComputeDropPosition"): "geometry.compute_drop_position",
        ("geometry_svc", "MaskToWorldPoints"): "geometry.mask_to_world_points",
        ("curobo", "SolveIK"): "curobo.solve_ik",
        ("curobo", "PlanLinear"): "curobo.plan_linear",
    }
    for (svc, method), expected in cases.items():
        assert mig.map_tool(svc, method) == expected, (svc, method)


def test_unknown_service_untouched(mig):
    assert mig.map_tool("mystery_svc", "DoThing") is None
    assert mig.map_short_string("robot.go_home") is None  # already migrated
    assert mig.map_short_string("not_a_tool") is None


# ---------------------------------------------------------------------------
# Graph rewrites
# ---------------------------------------------------------------------------


def test_graph_tool_fields_and_recovery(mig, tree):
    path = tree / "graph" / "workflow.json"
    count, _ = mig.migrate_file(path)
    assert count == 9  # 7 tool fields + 2 recovery blocks
    data = json.loads(path.read_text())
    nodes = data["subgraphs"]["sg"]["nodes"]
    assert nodes["observe"]["tool"] == "robot.get_observation"
    assert nodes["seg"]["tool"] == "sam3.segment_box"
    assert nodes["seg"]["inputs"] == {"box": {"$ref": "detect.box"}}
    assert nodes["detect"]["tool"] == "grounding-dino.detect"
    assert nodes["drop"]["tool"] == "geometry.compute_drop_position"
    assert nodes["plan"]["tool"] == "curobo.plan_linear"
    assert nodes["track"]["tool"] == "sam3.tracker_start_tracking"
    assert nodes["qualified"]["tool"] == "robot.get_observation"
    recovery = data["nodes"]["abort"]["recovery"]
    assert recovery == [
        {"tool": "robot.open_gripper", "inputs": {}},
        {"tool": "robot.go_home", "inputs": {}},
    ]


# ---------------------------------------------------------------------------
# Python script rewrites
# ---------------------------------------------------------------------------


def test_script_ctx_call_to_ctx_tool(mig, tree):
    path = tree / "scripts" / "do_thing.py"
    count, _ = mig.migrate_file(path)
    assert count == 6
    text = path.read_text()
    assert 'ctx.tool("robot.get_observation")' in text
    assert 'ctx.tool("robot.get_ee_pose", arm_id=0)' in text
    assert 'ctx.tool("robot.go_to_pose", pose=target)' in text
    assert 'ctx.tool("robot.open_gripper", settle_steps=60)' in text
    assert 'ctx.tool("vlm.query_yes_no", question="done?")' in text
    # Multi-line call collapsed onto the mapped tool name.
    assert '"geometry.mask_to_world_points"' in text
    assert "ctx.call(" not in text


# ---------------------------------------------------------------------------
# SKILL.md rewrites
# ---------------------------------------------------------------------------


def test_skill_md_allowed_tools(mig, tree):
    path = tree / "SKILL.md"
    count, _ = mig.migrate_file(path)
    assert count == 11  # 7 allowed-tools + 2 gap.allowed_tools + 2 in body
    text = path.read_text()
    assert "- robot.go_to_pose" in text
    assert "- robot.go_home" in text
    assert "- robot.open_gripper" in text
    assert "- geometry.mask_to_world_points" in text
    assert "- robot.get_observation" in text
    assert "- sam3.segment_box" in text
    assert "- molmo.query_yes_no" in text
    assert "[sim.check_success, gemini-er.detect]" in text
    assert "Calls robot.go_to_pose then robot.open_gripper." in text


# ---------------------------------------------------------------------------
# Idempotency + --check
# ---------------------------------------------------------------------------


def test_idempotent(mig, tree):
    for path in (
        tree / "graph" / "workflow.json",
        tree / "scripts" / "do_thing.py",
        tree / "SKILL.md",
    ):
        first, _ = mig.migrate_file(path)
        assert first > 0
        snapshot = path.read_text()
        second, _ = mig.migrate_file(path)
        assert second == 0
        assert path.read_text() == snapshot


def test_check_mode(mig, tree, capsys):
    # Before migration: hits found, exit 1, files untouched.
    before = (tree / "SKILL.md").read_text()
    rc = mig.main([str(tree), "--check"])
    assert rc == 1
    assert (tree / "SKILL.md").read_text() == before
    out = capsys.readouterr().out
    assert "observation.GetObservation" in out or "GetObservation" in out

    # Migrate, then --check is clean.
    rc = mig.main([str(tree)])
    assert rc == 0
    rc = mig.main([str(tree), "--check"])
    assert rc == 0
    assert "clean" in capsys.readouterr().out
