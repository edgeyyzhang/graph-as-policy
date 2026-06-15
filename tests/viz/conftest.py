"""Shared fixtures for the viz test suite.

NOTE: no ``from __future__ import annotations`` in this file — the stub
tools' signatures must carry *real* type objects so that
``gap.tools.schema.extract_schema`` can introspect them without a module
globals namespace.
"""

import json
from pathlib import Path
from typing import TypedDict

import pytest

from gap_core.tools import ToolRegistry
from gap_core.types import CameraFrame, PointCloud, Se3Pose

# ---------------------------------------------------------------------------
# Stub tool I/O types (gap.schema-registered names where it matters)
# ---------------------------------------------------------------------------


class DetectOut(TypedDict):
    cloud: PointCloud
    label: str


class PlanOut(TypedDict):
    pose: Se3Pose
    score: float


class MoveOut(TypedDict):
    ok: bool


def _detect(camera: str) -> DetectOut:  # pragma: no cover - schema only
    raise NotImplementedError


def _plan(cloud: PointCloud) -> PlanOut:  # pragma: no cover - schema only
    raise NotImplementedError


def _move(pose: Se3Pose) -> MoveOut:  # pragma: no cover - schema only
    raise NotImplementedError


def _stream(camera: str) -> CameraFrame:  # pragma: no cover - schema only
    raise NotImplementedError


@pytest.fixture()
def golden_tool_registry() -> ToolRegistry:
    """Typed stub tools backing the golden graphs' `type: tool` nodes."""
    reg = ToolRegistry()
    reg.register_callable("vision.detect", _detect, summary="detect object")
    reg.register_callable("grasp.plan", _plan, summary="plan grasp")
    reg.register_callable("robot.move_to_pose", _move, summary="move arm")
    reg.register_callable("tracker.stream", _stream, summary="stream frames")
    return reg


# ---------------------------------------------------------------------------
# Golden v3 graphs
# ---------------------------------------------------------------------------

CHECK_SCRIPT = '''\
from typing import TypedDict

from gap_core.types import PointCloud


class CheckOut(TypedDict):
    verdict: str


def run(ctx, cloud: PointCloud) -> CheckOut:
    return {"verdict": "yes"}
'''


def two_stage_graph() -> dict:
    """Two subgraphs with cross-subgraph data binding + conditional routing."""
    return {
        "version": 3,
        "meta": {"name": "two_stage"},
        "nodes": {
            "perceive": {"type": "subgraph", "ref": "perceive_sg"},
            "grasp": {"type": "subgraph", "ref": "grasp_sg"},
            "done": {"type": "end", "status": "success"},
            "abort": {
                "type": "end", "status": "failure",
                "recovery": [{"tool": "robot.open_gripper"}],
            },
        },
        "edges": [["START", "perceive"]],
        "conditional_edges": {
            "perceive": {
                "router_field": "exit",
                "mapping": {"found": "grasp", "not_found": "abort"},
            },
            "grasp": {
                "router_field": "exit",
                "mapping": {"grasped": "done", "failed": "abort"},
            },
        },
        "subgraphs": {
            "perceive_sg": {
                "skill": "perception",
                "inputs": {},
                "outputs": {"target_cloud": {"$ref": "detect.cloud"}},
                "nodes": {
                    "detect": {
                        "type": "tool", "tool": "vision.detect",
                        "inputs": {"camera": "wrist"},
                    },
                    "check": {
                        "type": "script", "script": "scripts/check.py",
                        "inputs": {"cloud": {"$ref": "detect.cloud"}},
                    },
                    "found": {"type": "noop"},
                },
                "edges": [["START", "detect"], ["detect", "check"], ["found", "END"]],
                "conditional_edges": {
                    "check": {"router_field": "verdict", "mapping": {"yes": "found"}},
                },
                "exit": {"router_field": None, "success_values": ["found"]},
                "on_error": "not_found",
            },
            "grasp_sg": {
                "skill": "grasping",
                "inputs": {"target_cloud": "PointCloud"},
                "outputs": {},
                "nodes": {
                    "plan": {
                        "type": "tool", "tool": "grasp.plan",
                        "inputs": {"cloud": {"$ref": "in.target_cloud"}},
                    },
                    "move": {
                        "type": "tool", "tool": "robot.move_to_pose",
                        "inputs": {"pose": {"$ref": "plan.pose"}},
                    },
                    "grasped": {"type": "noop"},
                },
                "edges": [
                    ["START", "plan"], ["plan", "move"],
                    ["move", "grasped"], ["grasped", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["grasped"]},
                "on_error": "failed",
            },
        },
    }


def streaming_graph() -> dict:
    """One subgraph with a streaming source node consumed by a servo node."""
    return {
        "version": 3,
        "meta": {"name": "streaming"},
        "nodes": {
            "servo_stage": {"type": "subgraph", "ref": "servo_sg"},
            "done": {"type": "end", "status": "success"},
            "failed_end": {"type": "end", "status": "failure"},
        },
        "edges": [["START", "servo_stage"]],
        "conditional_edges": {
            "servo_stage": {
                "router_field": "exit",
                "mapping": {"ok": "done", "failed": "failed_end"},
            },
        },
        "subgraphs": {
            "servo_sg": {
                "skill": "servoing",
                "inputs": {},
                "outputs": {},
                "nodes": {
                    "track": {
                        "type": "tool", "tool": "tracker.stream",
                        "streaming": True,
                        "inputs": {"camera": "wrist"},
                    },
                    "servo": {
                        "type": "tool", "tool": "robot.move_to_pose",
                        "inputs": {"pose": {"$ref": "track"}},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "track"], ["START", "servo"],
                    ["servo", "ok"], ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
                "on_error": "failed",
            },
        },
    }


def go_home_graph() -> dict:
    """Graph with `go_home` reset nodes that the static renderer must hide."""
    return {
        "version": 3,
        "meta": {"name": "go_home_demo"},
        "nodes": {
            "work": {"type": "subgraph", "ref": "work_sg"},
            "go_home": {"type": "tool", "tool": "robot.go_home", "inputs": {}},
            "done": {"type": "end", "status": "success"},
            "abort": {"type": "end", "status": "failure"},
        },
        "edges": [["START", "work"], ["go_home", "done"]],
        "conditional_edges": {
            "work": {
                "router_field": "exit",
                "mapping": {"ok": "go_home", "failed": "abort"},
            },
        },
        "subgraphs": {
            "work_sg": {
                "skill": "worker",
                "inputs": {},
                "outputs": {},
                "nodes": {
                    "act": {"type": "tool", "tool": "robot.move_to_pose", "inputs": {}},
                    "go_home_reset": {"type": "tool", "tool": "robot.go_home", "inputs": {}},
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "act"], ["act", "go_home_reset"],
                    ["go_home_reset", "ok"], ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
                "on_error": "failed",
            },
        },
    }


def write_workflow(tmp_path: Path, raw: dict, *, check_script: bool = False) -> Path:
    """Materialize a workflow dict (and optional script) into *tmp_path*."""
    (tmp_path / "workflow.json").write_text(json.dumps(raw, indent=2))
    if check_script:
        scripts = tmp_path / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "check.py").write_text(CHECK_SCRIPT)
    return tmp_path
