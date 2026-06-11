"""Parity test: the codegen-time connector tool catalog must match the
schemas a REAL SimConnector registers at execution time.

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
    assert set(catalog.keys()) == set(real_connector_registry._tools.keys())
    # The full connector surface from the task contract is present.
    expected_robot = {
        "robot.get_observation", "robot.get_camera_pose", "robot.get_ee_pose",
        "robot.go_to_pose", "robot.go_to_pose_cartesian", "robot.move_to_joints",
        "robot.execute_trajectory", "robot.go_home", "robot.solve_ik",
        "robot.open_gripper", "robot.close_gripper", "robot.get_gripper",
        "robot.get_gripper_pose",
    }
    expected_sim = {
        "sim.reset", "sim.step", "sim.check_success", "sim.apply_policy_action",
        "sim.enable_video", "sim.save_video",
    }
    assert expected_robot | expected_sim == set(catalog.keys())


def test_schemas_match(real_connector_registry):
    catalog = connector_tool_descriptors()
    for name, real_desc in real_connector_registry._tools.items():
        codegen_desc = catalog[name]
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
