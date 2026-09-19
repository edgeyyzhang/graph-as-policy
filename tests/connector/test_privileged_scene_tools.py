"""Portable privileged provider tests, independent of any simulator SDK."""

from types import SimpleNamespace

import numpy as np
import pytest

from gap.connector.privileged import PrivilegedSimTools
from gap.runtime.verify import Body, World

_Q = np.array([1.0, 0.0, 0.0, 0.0])


class PortableAdapter:
    def __init__(self):
        self._boxes = {
            "mug": (np.array([0.20, 0.00, 0.10]), np.array([0.03, 0.04, 0.05]), _Q),
            "cabinet": (np.array([0.50, 0.00, 0.20]), np.array([0.20, 0.20, 0.20]), _Q),
            "table": (np.array([0.30, 0.00, 0.00]), np.array([0.50, 0.50, 0.02]), _Q),
        }
        self.drawer_position = 0.15

    def reference_frame_name(self):
        return "robot_base"

    def object_names(self):
        return list(self._boxes)

    def task_object_names(self):
        return ["mug", "cabinet"]

    def movable_object_names(self):
        return ["mug"]

    def part_names(self, root):
        return ["drawer_link"] if root == "cabinet" else []

    def object_pose(self, name):
        center, _half, quat = self._boxes[name]
        return center, quat

    def object_box(self, name):
        if name == "drawer_link":
            return np.array([0.50, 0.12, 0.18]), np.array([0.18, 0.08, 0.07]), _Q
        return self._boxes.get(name)

    def feature_names(self, root):
        return ["drawer_handle"] if root == "cabinet" else []

    def feature_box(self, root, feature):
        if (root, feature) != ("cabinet", "drawer_handle"):
            return None
        return np.array([0.50, 0.25, 0.20]), np.array([0.08, 0.01, 0.01]), _Q

    def articulation_names(self, root=None):
        names = ["drawer_slide"]
        return names if root in (None, "cabinet") else []

    def articulation_state(self, name):
        if name != "drawer_slide":
            return None
        return {
            "name": name,
            "parent": "cabinet",
            "child": "drawer_link",
            "kind": "prismatic",
            "axis": np.array([0.0, 1.0, 0.0]),
            "pivot": np.array([0.50, 0.00, 0.18]),
            "position": self.drawer_position,
            "lower": 0.0,
            "upper": 0.2,
            "progress": self.drawer_position / 0.2,
        }

    def contact_pairs(self):
        return [("mug", "robot0_finger"), ("mug", "table")]

    def collision_feature_contact_pairs(self):
        return [("mug_geom", "left_pad"), ("mug_geom", "table_collision")]

    def named_frame_pose(self, name):
        frames = {
            "mug_site": (np.array([0.52, 0.01, 0.10]), _Q),
            "cabinet_frame": (np.array([0.50, 0.00, 0.10]), _Q),
            "table_collision": (np.array([0.30, 0.00, 0.00]), _Q),
        }
        if name in frames:
            return frames[name]
        return self.object_pose(name) if name in self._boxes else None

    def snapshot(self):
        bodies = {}
        contacts = {
            "mug": frozenset({"robot0_finger", "table"}),
            "table": frozenset({"mug"}),
            "cabinet": frozenset(),
        }
        for name, (center, half, quat) in self._boxes.items():
            bodies[name] = Body(
                name=name,
                position=center.copy(),
                quaternion_wxyz=quat.copy(),
                aabb_lower=center - half,
                aabb_upper=center + half,
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
                contacts=contacts[name],
            )
        return World(env_id=0, bodies=bodies, robot_link_prefixes=("robot0_",))


@pytest.fixture
def tools():
    adapter = PortableAdapter()
    definitions = {
        "ready": {
            "all": [
                {"flag": "cooking_latched"},
                {"joint": {"name": "drawer_slide", "at_least": 0.14}},
                {"touching": {"a": ["mug_geom"], "b": ["left_pad"]}},
                {"in_box": {
                    "body": "mug_site",
                    "frame": "cabinet_frame",
                    "x": [0.019, 0.021],
                    "y": [0.009, 0.011],
                    "z": [-0.001, 0.001],
                }},
                {"offset": {
                    "body": "mug_site",
                    "frame": "cabinet_frame",
                    "axis": "x",
                    "at_least": 0.019,
                }},
                {"not": {"flag": "failed"}},
            ]
        },
        "alias": {"ref": "ready"},
    }
    connector = SimpleNamespace(
        _world_adapter=adapter,
        env=None,
        get_task_flags=lambda: {
            "native_success": False,
            "cooking_latched": True,
            "failed": False,
        },
        get_condition_definitions=lambda: definitions,
    )
    return PrivilegedSimTools(connector)


def test_entity_payload_is_backend_neutral(tools):
    listing = tools.list_entities()
    mug = next(item for item in listing["entities"] if item["name"] == "mug")
    assert mug["kind"] == "object"
    assert mug["movable"] is True

    state = tools.get_entity_state("mug")["entity"]
    assert state["source"] == "privileged"
    assert state["frame"] == "world"
    assert state["obb"]["extent"]["z"] == pytest.approx(0.05)


def test_feature_can_feed_existing_functional_feature_skills(tools):
    feature = tools.get_feature(
        "cabinet", "handle", kind="handle", axis="long"
    )["feature"]
    assert feature["kind"] == "handle"
    assert feature["functional"]["kind"] == "shaft"
    assert feature["functional"]["axis"] == {"x": 1.0, "y": 0.0, "z": 0.0}
    assert feature["functional"]["radius_outer"] == pytest.approx(0.01)


def test_articulation_hides_backend_joint_vocabulary(tools):
    state = tools.get_articulation("cabinet")["articulation"]
    assert state["kind"] == "prismatic"
    assert state["progress"] == pytest.approx(0.75)
    assert state["axis"] == {"x": 0.0, "y": 1.0, "z": 0.0}
    assert state["source"] == "privileged"


def test_relative_pose_is_expressed_in_reference_entity_frame(tools):
    result = tools.relative_pose("mug", "cabinet")
    assert result["frame"] == "entity:cabinet"
    assert result["pose"]["position"]["x"] == pytest.approx(-0.30)


def test_condition_tree_uses_same_relations_as_graph_milestones(tools):
    result = tools.evaluate_condition({
        "all": [
            {"holding": {"entity": "mug"}},
            {"touching": {"a": "mug", "b": "table"}},
            {"articulation": {
                "entity": "cabinet",
                "progress": 0.70,
                "comparison": "at_least",
            }},
        ]
    })
    assert result["satisfied"] is True
    assert len(result["diagnostics"]["children"]) == 3


def test_task_flags_and_complete_benchmark_condition_contract(tools):
    flags = tools.get_task_flags()
    assert flags["flags"]["cooking_latched"] is True

    result = tools.evaluate_condition({"ref": "ready"})
    assert result["satisfied"] is True
    resolved = result["diagnostics"]["resolved"]
    assert [child["operator"] for child in resolved["children"]] == [
        "flag", "joint", "touching", "in_box", "offset", "not",
    ]
    contact = resolved["children"][2]
    assert contact["level"] == "collision_feature"
    assert contact["matching_pairs"] == [("mug_geom", "left_pad")]


def test_describe_condition_expands_named_refs_without_evaluation(tools):
    described = tools.describe_condition("alias")
    assert described["name"] == "alias"
    assert described["source"] == "task_adapter"
    assert described["condition"] == {"ref": "ready"}
    assert described["expanded"]["ref"] == "ready"
    assert described["expanded"]["expanded"]["all"][0] == {
        "flag": "cooking_latched"
    }
    assert described["available"] == ["alias", "ready"]


def test_condition_reports_failed_local_axis_and_flag_value(tools):
    box = tools.evaluate_condition({
        "in_box": {
            "body": "mug_site",
            "frame": "cabinet_frame",
            "x": [-0.01, 0.01],
        }
    })
    assert box["satisfied"] is False
    assert box["diagnostics"]["axes"]["x"]["actual"] == pytest.approx(0.02)
    assert box["diagnostics"]["axes"]["x"]["passed"] is False

    flag = tools.evaluate_condition({
        "flag": {"name": "native_success", "equals": True}
    })
    assert flag["satisfied"] is False
    assert flag["diagnostics"]["actual"] is False


def test_simconnector_accepts_non_mujoco_scene_provider():
    from gap.connector import SimConnector

    adapter = PortableAdapter()
    connector = SimConnector(
        None,
        SimpleNamespace(),
        scene_adapter=adapter,
        task_flags_provider=lambda: {"native_success": True},
        condition_definitions={"done": {"flag": "native_success"}},
    )
    assert connector.capabilities.world_state is True
    assert connector._ensure_world_adapter() is adapter
    result = connector.tool_registry.invoke(
        "sim.get_entity_state", None, entity="mug"
    )
    assert result["entity"]["name"] == "mug"
    flags = connector.tool_registry.invoke("sim.get_task_flags", None)
    assert flags["flags"]["native_success"] is True
    done = connector.tool_registry.invoke(
        "sim.evaluate_condition", None, condition={"ref": "done"}
    )
    assert done["satisfied"] is True
