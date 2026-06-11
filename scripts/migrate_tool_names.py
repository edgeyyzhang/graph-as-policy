#!/usr/bin/env python3
"""Migrate legacy gRPC service/method tool names to flat gap tool names.

Rewrites three artifact kinds in place:

1. **Graphs** (``workflow.json``): ``"tool": "<service>.<Method>"`` fields
   and end-node ``recovery`` blocks in the retired
   ``{"service": "x.v1.X", "method": "M"}`` form (rewritten to
   ``{"tool": "<new>"}``).
2. **SKILL.md** frontmatter: entries under ``allowed-tools:`` /
   ``gap.allowed_tools`` (any ``service.Method`` token in the file is
   rewritten — tool names are unambiguous tokens).
3. **Python scripts**: ``ctx.call("svc.v1.Svc", "Method", ...)`` →
   ``ctx.tool("<mapped>", ...)``. Handles both fully-qualified service
   names (``"observation.v1.Observation"``) and shortnames
   (``"observation"``).

The mapping is idempotent — new-style names are never sources, so running
the script twice is a no-op. ``--check`` lists unmigrated hits without
writing and exits 1 when any are found.

Usage:
    python scripts/migrate_tool_names.py PATH [PATH ...] [--check]

PATH may be a file or a directory (recursed for ``*.json``, ``*.py``,
``SKILL.md``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Mapping: (service_shortname, Method) -> new flat tool name
# ---------------------------------------------------------------------------

EXPLICIT_MAP: dict[tuple[str, str], str] = {
    # observation.v1.Observation
    ("observation", "GetObservation"): "robot.get_observation",
    ("observation", "GetCameraPose"): "robot.get_camera_pose",
    # robot_control.v1.RobotControl
    ("robot_control", "GoToPose"): "robot.go_to_pose",
    ("robot_control", "GoToPoseCartesian"): "robot.go_to_pose_cartesian",
    ("robot_control", "ExecuteJointTrajectory"): "robot.execute_trajectory",
    ("robot_control", "MoveToJoints"): "robot.move_to_joints",
    ("robot_control", "GoHome"): "robot.go_home",
    ("robot_control", "GetEEPose"): "robot.get_ee_pose",
    # ApplyPolicyAction was an RPC on RobotControl but is a sim capability.
    ("robot_control", "ApplyPolicyAction"): "sim.apply_policy_action",
    # gripper.v1.Gripper
    ("gripper", "Open"): "robot.open_gripper",
    ("gripper", "Close"): "robot.close_gripper",
    ("gripper", "GetPosition"): "robot.get_gripper",
    ("gripper", "GetPose"): "robot.get_gripper_pose",
    # sim_bridge.v1.SimBridge
    ("sim_bridge", "Reset"): "sim.reset",
    ("sim_bridge", "StepOnce"): "sim.step",
    ("sim_bridge", "CheckTaskCompletion"): "sim.check_success",
    ("sim_bridge", "ApplyPolicyAction"): "sim.apply_policy_action",
    ("sim_bridge", "EnableVideoCapture"): "sim.enable_video",
    ("sim_bridge", "SaveVideo"): "sim.save_video",
    # sam3.v1.SAM3
    ("sam3", "SegmentText"): "sam3.segment_text",
    ("sam3", "SegmentPoint"): "sam3.segment_point",
    ("sam3", "SegmentBox"): "sam3.segment_box",
    # grounding_dino.v1.GroundingDino
    ("grounding_dino", "Detect"): "grounding-dino.detect",
    # molmo.v1.Molmo
    ("molmo", "PointPrompt"): "molmo.point_prompt",
    ("molmo", "Query"): "molmo.query",
    ("molmo", "QueryYesNo"): "molmo.query_yes_no",
    # vlm.v1.VLM
    ("vlm", "Query"): "vlm.query",
    ("vlm", "QueryYesNo"): "vlm.query_yes_no",
    # gemini_er.v1.GeminiER
    ("gemini_er", "Detect"): "gemini-er.detect",
}


def _snake(name: str) -> str:
    """CamelCase → snake_case (``MaskToWorldPoints`` → ``mask_to_world_points``)."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def map_tool(service: str, method: str) -> str | None:
    """Map a (service shortname, Method) pair to the new flat tool name.

    Returns ``None`` for unknown services (leave untouched).
    """
    explicit = EXPLICIT_MAP.get((service, method))
    if explicit is not None:
        return explicit
    if service == "sam3_tracker":
        return f"sam3.tracker_{_snake(method)}"
    if service == "geometry_svc":
        return f"geometry.{_snake(method)}"
    if service == "curobo":
        return f"curobo.{_snake(method)}"
    return None


def _service_shortname(qualified: str) -> str:
    """``"robot_control.v1.RobotControl"`` → ``"robot_control"``."""
    return qualified.split(".", 1)[0]


def map_short_string(value: str) -> str | None:
    """Map a graph-style ``service.Method`` shortname string.

    Handles both ``"sam3.SegmentBox"`` and fully-qualified
    ``"observation.v1.Observation.GetObservation"``.
    """
    if not isinstance(value, str) or "." not in value:
        return None
    parts = value.split(".")
    service = parts[0]
    method = parts[-1]
    if not method or not method[0].isupper():
        return None  # already a flat lowercase tool name (idempotency)
    return map_tool(service, method)


# ---------------------------------------------------------------------------
# Graph (workflow.json) migration
# ---------------------------------------------------------------------------


def migrate_graph_obj(obj) -> tuple[object, int]:
    """Recursively rewrite ``tool:`` fields and retired recovery blocks."""
    count = 0
    if isinstance(obj, dict):
        # Retired recovery form: {"service": ..., "method": ..., "inputs": ...}
        if "service" in obj and "method" in obj and "tool" not in obj:
            new = map_tool(_service_shortname(str(obj["service"])), str(obj["method"]))
            if new is not None:
                rebuilt = {"tool": new}
                if "inputs" in obj:
                    rebuilt["inputs"] = obj["inputs"]
                extra = {
                    k: v for k, v in obj.items()
                    if k not in ("service", "method", "inputs")
                }
                rebuilt.update(extra)
                inner, inner_count = migrate_graph_obj(rebuilt)
                return inner, inner_count + 1
        out = {}
        for key, value in obj.items():
            if key == "tool" and isinstance(value, str):
                new = map_short_string(value)
                if new is not None:
                    out[key] = new
                    count += 1
                    continue
            new_value, c = migrate_graph_obj(value)
            out[key] = new_value
            count += c
        return out, count
    if isinstance(obj, list):
        out_list = []
        for item in obj:
            new_item, c = migrate_graph_obj(item)
            out_list.append(new_item)
            count += c
        return out_list, count
    return obj, count


def check_graph_obj(obj, path: str = "$") -> list[str]:
    """List unmigrated tool references in a graph dict."""
    hits: list[str] = []
    if isinstance(obj, dict):
        if "service" in obj and "method" in obj and "tool" not in obj:
            hits.append(f"{path}: recovery service/method {obj.get('service')}.{obj.get('method')}")
        for key, value in obj.items():
            if key == "tool" and isinstance(value, str) and map_short_string(value):
                hits.append(f"{path}.{key}: {value}")
            else:
                hits.extend(check_graph_obj(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(check_graph_obj(item, f"{path}[{i}]"))
    return hits


# ---------------------------------------------------------------------------
# Python script migration: ctx.call("svc", "Method", ...) -> ctx.tool("new", ...)
# ---------------------------------------------------------------------------

_CTX_CALL_RX = re.compile(
    r"ctx\.call\(\s*\"(?P<service>[\w.]+)\"\s*,\s*\"(?P<method>\w+)\"\s*(?P<sep>,\s*|\))",
    re.DOTALL,
)


def migrate_python_text(text: str) -> tuple[str, int]:
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        new = map_tool(_service_shortname(m.group("service")), m.group("method"))
        if new is None:
            return m.group(0)
        count += 1
        sep = m.group("sep")
        if sep.startswith(","):
            return f'ctx.tool("{new}"{sep}'
        return f'ctx.tool("{new}")'

    return _CTX_CALL_RX.sub(_sub, text), count


def check_python_text(text: str) -> list[str]:
    hits = []
    for m in _CTX_CALL_RX.finditer(text):
        new = map_tool(_service_shortname(m.group("service")), m.group("method"))
        if new is not None:
            hits.append(f'ctx.call("{m.group("service")}", "{m.group("method")}")')
    return hits


# ---------------------------------------------------------------------------
# SKILL.md migration: rewrite service.Method tokens (allowed-tools entries)
# ---------------------------------------------------------------------------

_TOKEN_RX = re.compile(r"\b([a-z][\w]*(?:\.v\d+\.[A-Za-z_]\w*)?\.[A-Z]\w*)\b")


def migrate_skill_md_text(text: str) -> tuple[str, int]:
    count = 0

    def _sub(m: re.Match) -> str:
        nonlocal count
        new = map_short_string(m.group(1))
        if new is None:
            return m.group(0)
        count += 1
        return new

    return _TOKEN_RX.sub(_sub, text), count


def check_skill_md_text(text: str) -> list[str]:
    return [
        m.group(1)
        for m in _TOKEN_RX.finditer(text)
        if map_short_string(m.group(1)) is not None
    ]


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------


def _iter_targets(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            yield from sorted(
                p for p in path.rglob("*")
                if p.is_file() and (
                    p.suffix == ".json" or p.suffix == ".py" or p.name == "SKILL.md"
                )
            )
        elif path.is_file():
            yield path


def migrate_file(path: Path, *, check: bool = False) -> tuple[int, list[str]]:
    """Migrate (or check) one file. Returns (num_rewrites, check_hits)."""
    text = path.read_text()
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return 0, []
        if check:
            return 0, check_graph_obj(data)
        migrated, count = migrate_graph_obj(data)
        if count:
            path.write_text(json.dumps(migrated, indent=2) + "\n")
        return count, []
    if path.name == "SKILL.md":
        if check:
            return 0, check_skill_md_text(text)
        migrated_text, count = migrate_skill_md_text(text)
        if count:
            path.write_text(migrated_text)
        return count, []
    if path.suffix == ".py":
        if check:
            return 0, check_python_text(text)
        migrated_text, count = migrate_python_text(text)
        if count:
            path.write_text(migrated_text)
        return count, []
    return 0, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Files or directories to migrate")
    parser.add_argument("--check", action="store_true",
                        help="List unmigrated hits without rewriting; exit 1 if any")
    args = parser.parse_args(argv)

    total = 0
    any_hits = False
    for target in _iter_targets(args.paths):
        rewrites, hits = migrate_file(target, check=args.check)
        if args.check:
            for hit in hits:
                any_hits = True
                print(f"{target}: {hit}")
        elif rewrites:
            total += rewrites
            print(f"{target}: {rewrites} rewrite(s)")

    if args.check:
        if any_hits:
            print("unmigrated tool names found", file=sys.stderr)
            return 1
        print("clean: no legacy tool names found")
        return 0
    print(f"done: {total} rewrite(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
