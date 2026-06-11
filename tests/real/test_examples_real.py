"""Real-robot examples: v2→v3 migration correctness + validation.

The jello graph was hand-migrated from the dev tree's v2 schema
(states/transitions); the cable graph was already v3 but carried legacy
tool names. Both must load, validate against the open-robot-skills checkout
with zero errors, reference only known tools, and reference only
scripts that exist on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = {
    "real_franka_pick_place": REPO_ROOT / "examples" / "real_franka_pick_place" / "graph",
    "cable_ur": REPO_ROOT / "examples" / "cable_ur" / "graph",
}

# robot.* connector tools registered at runtime — not in the static @tool
# registry at validate time. NO sim.* names here on purpose: real-robot
# graphs must never reference sim tools.
CONNECTOR_TOOLS = frozenset({
    "robot.close_gripper", "robot.execute_trajectory", "robot.get_camera_pose",
    "robot.get_ee_pose", "robot.get_gripper", "robot.get_gripper_pose",
    "robot.get_observation", "robot.go_home", "robot.go_to_pose",
    "robot.go_to_pose_cartesian", "robot.move_to_joints", "robot.open_gripper",
    "robot.solve_ik",
})

# ur_zed registers ONLY these (perception-only connector).
UR_ZED_TOOLS = frozenset({
    "robot.get_observation", "robot.get_camera_pose", "robot.get_ee_pose",
    "robot.get_gripper", "robot.get_gripper_pose",
})


def _skills_root() -> Path:
    candidate = REPO_ROOT.parent / "open-robot-skills"
    if not (candidate / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {candidate}")
    return candidate


def _walk_tools(obj, tools: set[str]) -> None:
    if isinstance(obj, dict):
        t = obj.get("tool")
        if isinstance(t, str):
            tools.add(t)
        for v in obj.values():
            _walk_tools(v, tools)
    elif isinstance(obj, list):
        for v in obj:
            _walk_tools(v, tools)


def _graph_tools(graph_dir: Path) -> set[str]:
    raw = json.loads((graph_dir / "workflow.json").read_text())
    tools: set[str] = set()
    _walk_tools(raw, tools)
    return tools


def _graph_scripts(graph_dir: Path) -> set[str]:
    raw = json.loads((graph_dir / "workflow.json").read_text())
    scripts: set[str] = set()

    def _walk(obj):
        if isinstance(obj, dict):
            s = obj.get("script")
            if isinstance(s, str):
                scripts.add(s)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(raw)
    return scripts


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_graph_validates_zero_errors(name: str) -> None:
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    wf = load_workflow(EXAMPLES[name] / "workflow.json")
    assert wf.version == 3
    skill_registry = load_skills(_skills_root())
    issues = validate_workflow(wf, skill_registry=skill_registry)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], f"{name}: {[str(e) for e in errors]}"


@pytest.fixture(scope="module")
def bundle_tool_registry():
    """A ToolRegistry holding every open-robot-skills bundle @tool.

    The @tool decorator pushes onto a process-global pending queue only
    at module import, and any earlier test that builds a registry drains
    that queue — so this fixture re-imports each bundle's tools module
    (re-firing the decorators) before draining into a fresh registry,
    then restores the queue for suites that assert its membership.
    """
    import importlib

    from gap.skills import load_skills
    from gap.tools import ToolRegistry
    from gap.tools import _registry as tool_registry_mod

    skills = load_skills(_skills_root())
    snapshot = list(tool_registry_mod._PENDING_TOOLS)
    tool_registry_mod._PENDING_TOOLS.clear()
    for info in skills.list_skills():
        if info.tools_module is not None:
            importlib.reload(info.tools_module)
    reg = ToolRegistry()
    reg.discover_pending()
    tool_registry_mod._PENDING_TOOLS.extend(snapshot)
    return reg


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_graph_tools_known(name: str, bundle_tool_registry) -> None:
    """Every tool name resolves: connector surface or a bundle @tool."""
    unknown = {
        t for t in _graph_tools(EXAMPLES[name])
        if t not in CONNECTOR_TOOLS and t not in bundle_tool_registry
    }
    assert unknown == set(), f"{name}: unknown tools {sorted(unknown)}"


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_no_sim_tools_in_real_graphs(name: str) -> None:
    sim_tools = {t for t in _graph_tools(EXAMPLES[name]) if t.startswith("sim.")}
    assert sim_tools == set()


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_scripts_exist(name: str) -> None:
    graph_dir = EXAMPLES[name]
    missing = [s for s in _graph_scripts(graph_dir) if not (graph_dir / s).exists()]
    assert missing == []


def test_cable_graph_is_perception_only() -> None:
    """The ur_zed connector registers only observation tools — the cable
    graph must not reference anything outside that surface."""
    robot_tools = {
        t for t in _graph_tools(EXAMPLES["cable_ur"]) if t.startswith("robot.")
    }
    assert robot_tools <= UR_ZED_TOOLS, (
        f"cable graph uses motion tools unavailable on ur_zed: "
        f"{sorted(robot_tools - UR_ZED_TOOLS)}"
    )


class TestJelloMigration:
    """Spot-checks that the v2→v3 hand-migration preserved the semantics."""

    @pytest.fixture
    def raw(self) -> dict:
        return json.loads(
            (EXAMPLES["real_franka_pick_place"] / "workflow.json").read_text()
        )

    def test_v2_states_became_subgraph_nodes(self, raw):
        assert set(raw["nodes"]) == {
            "perceive_container", "perceive_target", "grasp", "transport",
            "done", "abort",
        }

    def test_transitions_became_conditional_edges(self, raw):
        ce = raw["conditional_edges"]
        # The v2 transitions table, including the repeat-until-clean loop.
        assert ce["transport"]["mapping"]["placed"] == "perceive_target"
        assert ce["perceive_target"]["mapping"]["table_clean"] == "done"
        assert ce["grasp"]["mapping"]["grasped"] == "transport"
        assert ce["grasp"]["mapping"]["collision"] == "abort"

    def test_on_failure_collapsed_to_on_error(self, raw):
        sgs = raw["subgraphs"]
        assert sgs["perceive_container_sg"]["on_error"] == "not_found"
        assert sgs["perceive_target_sg"]["on_error"] == "table_clean"
        assert sgs["grasp_sg"]["on_error"] == "collision"
        assert sgs["transport_sg"]["on_error"] == "blocked"

    def test_recovery_migrated_to_tool_form(self, raw):
        recovery = raw["nodes"]["abort"]["recovery"]
        assert [r["tool"] for r in recovery] == [
            "robot.open_gripper", "robot.go_home",
        ]

    def test_grasp_refs_migrated_to_candidates_shape(self, raw):
        """v2 `compute_grasp.position` / `.poses.0` → v3 candidates result."""
        text = json.dumps(raw)
        assert "compute_grasp.candidates.poses.0" in text
        assert "compute_grasp.position" not in text
        assert "top_down_grasp_poses_from_obb" not in text
