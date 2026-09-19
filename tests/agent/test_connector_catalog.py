"""Parity test: the safe codegen-time connector catalog must match the
always-available surface of a real SimConnector. Joint-position commands are
capability-gated and therefore absent from the environment-free catalog.

The codegen catalog is derived from a throwaway ``SimConnector(None,
SimpleNamespace())`` — this test pins the assumption that connector
construction and tool registration never touch the env, by comparing
against a connector built around the connector suite's FakeEnv.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gap.agent._catalog import connector_tool_descriptors

_CONNECTOR_CONFTEST = Path(__file__).resolve().parents[1] / "connector" / "conftest.py"


@pytest.fixture(scope="module")
def real_connector_registry():
    spec = importlib.util.spec_from_file_location(
        "_connector_conftest_for_catalog", _CONNECTOR_CONFTEST,
    )
    mod = importlib.util.module_from_spec(spec)
    # dataclasses' string-annotation resolution requires the module to be
    # registered in sys.modules during exec.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)

    from gap.connector import SimConnector

    conn = SimConnector(mod.FakeEnv(), mod.FakeEnvConfig(), ik=mod.FakeIK())
    return conn.tool_registry


def test_same_tool_names(real_connector_registry):
    catalog = connector_tool_descriptors()
    optional_joint_motion = {
        "robot.move_to_joints", "robot.execute_trajectory", "robot.go_home",
    }
    assert set(catalog) == set(real_connector_registry._tools) - optional_joint_motion
    # The safe connector surface from the task contract is present.
    expected_robot = {
        "robot.get_observation", "robot.get_camera_pose", "robot.get_ee_pose",
        "robot.go_to_pose", "robot.go_to_pose_cartesian", "robot.solve_ik",
        "robot.open_gripper", "robot.close_gripper", "robot.get_gripper",
        "robot.get_gripper_pose", "robot.get_joint_state",
        # Embodiment description + partial grip / settle.
        "robot.describe_gripper", "robot.describe_arm", "robot.describe_workspace",
        "robot.grasp_frame", "robot.set_grip", "robot.wait_steps",
    }
    expected_sim = {
        "sim.reset", "sim.step", "sim.check_success", "sim.apply_policy_action",
        "sim.enable_video", "sim.save_video",
        # Privileged scene geometry — present in the catalog even without a
        # reachable MuJoCo, so codegen can see them; see SimConnector.
        "sim.list_objects", "sim.get_object_pose", "sim.get_object_obb",
        "sim.get_part_obb",
        # Portable privileged provider: skills consume these rather than
        # MuJoCo body / geom / joint vocabulary.
        "sim.list_entities", "sim.resolve_entity", "sim.get_entity_state",
        "sim.list_features", "sim.get_feature", "sim.list_articulations",
        "sim.get_articulation", "sim.get_contacts", "sim.get_task_flags",
        "sim.relative_pose", "sim.evaluate_condition",
        # Zero-step introspection: expands a named condition and its `ref`
        # relations without evaluating it.
        "sim.describe_condition",
    }
    assert expected_robot | expected_sim == set(catalog.keys())


def test_schemas_match(real_connector_registry):
    catalog = connector_tool_descriptors()
    for name, codegen_desc in catalog.items():
        real_desc = real_connector_registry._tools[name]
        assert codegen_desc.summary == real_desc.summary, name
        assert codegen_desc.tags == real_desc.tags, name

        real_in = real_desc.schema.inputs
        gen_in = codegen_desc.schema.inputs
        assert set(gen_in) == set(real_in), name
        for fname, real_field in real_in.items():
            gen_field = gen_in[fname]
            assert gen_field.type_str == real_field.type_str, (name, fname)
            assert gen_field.required == real_field.required, (name, fname)
            assert gen_field.default == real_field.default, (name, fname)

        real_out = real_desc.schema.outputs
        gen_out = codegen_desc.schema.outputs
        assert set(gen_out) == set(real_out), name
        for fname, real_field in real_out.items():
            assert gen_out[fname].type_str == real_field.type_str, (name, fname)
