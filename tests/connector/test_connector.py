"""Connector contract tests on a FakeEnv implementing the env surface."""

from __future__ import annotations

import numpy as np
import pytest
from gap_core.errors import ToolError

from gap.connector import Capabilities, SimConnector

from .conftest import FakeEnv, FakeEnvConfig, FakeIK

EXPECTED_TOOLS = {
    # robot.* — getters
    "robot.get_observation",
    "robot.get_camera_pose",
    "robot.get_ee_pose",
    "robot.get_gripper",
    "robot.get_gripper_pose",
    # robot.* — control
    "robot.go_to_pose",
    "robot.go_to_pose_cartesian",
    "robot.move_to_joints",
    "robot.execute_trajectory",
    "robot.go_home",
    "robot.open_gripper",
    "robot.close_gripper",
    # robot.* — planning
    "robot.solve_ik",
    # sim.*
    "sim.reset",
    "sim.step",
    "sim.check_success",
    "sim.apply_policy_action",
    "sim.enable_video",
    "sim.save_video",
}


class TestToolRegistration:
    def test_all_tool_names_registered(self, connector):
        reg = connector.tool_registry
        for name in EXPECTED_TOOLS:
            assert name in reg, f"missing tool {name}"

    def test_guard_tags(self, connector):
        reg = connector.tool_registry
        sim_step = (
            "robot.go_to_pose", "robot.go_to_pose_cartesian",
            "robot.move_to_joints", "robot.execute_trajectory",
            "robot.go_home", "robot.open_gripper", "robot.close_gripper",
            "sim.step", "sim.apply_policy_action",
        )
        for name in sim_step:
            assert "sim_step" in reg.get(name).tags, name
        assert "planning" in reg.get("robot.solve_ik").tags
        # Getters carry no guard tag.
        for name in (
            "robot.get_observation", "robot.get_ee_pose",
            "robot.get_gripper", "robot.get_gripper_pose",
            "robot.get_camera_pose", "sim.check_success", "sim.reset",
        ):
            assert reg.get(name).tags == (), name

    def test_registry_built_once(self, connector):
        assert connector.tool_registry is connector.tool_registry


class TestObservationAssembly:
    def test_shapes_and_types(self, connector):
        obs = connector.get_observation()
        assert set(obs.keys()) == {"cameras", "arms"}
        assert len(obs["cameras"]) == 1
        cam = obs["cameras"][0]
        assert cam["name"] == "cam0"
        assert cam["rgb"].shape == (8, 8, 3) and cam["rgb"].dtype == np.uint8
        assert cam["depth"].shape == (8, 8) and cam["depth"].dtype == np.float32
        assert cam["intrinsics"].shape == (3, 3)

    def test_rgb_not_reencoded(self, connector, fake_env):
        raw = fake_env.get_observation()["cam0"]["images"]["rgb"]
        obs = connector._build_observation(fake_env.get_observation())
        # numpy buffer passes through (no byte packing / copies)
        assert obs["cameras"][0]["rgb"].dtype == raw.dtype
        assert obs["cameras"][0]["rgb"].shape == raw.shape

    def test_camera_pose_wxyz(self, connector):
        obs = connector.get_observation()
        rot = obs["cameras"][0]["pose"]["rotation"]
        # env pose array was [x,y,z, qw,qx,qy,qz] = [.1,.2,.3, 0,0,1,0]
        assert rot == {"w": 0.0, "x": 0.0, "y": 1.0, "z": 0.0}
        pos = obs["cameras"][0]["pose"]["position"]
        assert (pos["x"], pos["y"], pos["z"]) == (0.1, 0.2, 0.3)

    def test_arm_state(self, connector, fake_env):
        obs = connector.get_observation()
        assert len(obs["arms"]) == 1
        arm = obs["arms"][0]
        assert arm["joint_state"]["positions"].shape == (7,)
        assert arm["gripper_fraction"] == pytest.approx(fake_env._gripper_pos)
        rot = arm["ee_pose"]["rotation"]
        # cart[3:7] = (0,1,0,0) wxyz
        assert (rot["w"], rot["x"], rot["y"], rot["z"]) == (0.0, 1.0, 0.0, 0.0)

    def test_arm_base_offset_applied(self, fake_env):
        config = FakeEnvConfig(arm_bases=((1.0, 2.0, 3.0),))
        conn = SimConnector(fake_env, config, ik=FakeIK())
        obs = conn.get_observation()
        pos = obs["arms"][0]["ee_pose"]["position"]
        assert pos["x"] == pytest.approx(0.4 + 1.0)
        assert pos["y"] == pytest.approx(0.05 + 2.0)
        assert pos["z"] == pytest.approx(0.3 + 3.0)

    def test_get_camera_pose_tool(self, connector):
        out = connector.get_camera_pose("cam0")
        assert out["pose"]["rotation"]["y"] == 1.0
        with pytest.raises(ToolError):
            connector.get_camera_pose("nope")


class TestGripper:
    def test_open_settle_steps_counted(self, connector, fake_env):
        before = fake_env._sim_step_count
        out = connector.open_gripper()
        assert fake_env._sim_step_count - before == 40
        assert out["position"] == pytest.approx(1.0)

    def test_close_settle_steps_counted(self, connector, fake_env):
        before = fake_env._sim_step_count
        out = connector.close_gripper()
        assert fake_env._sim_step_count - before == 60
        # 60 settle steps at 0.05/step closes fully from 1.0
        assert out["position"] == pytest.approx(0.0)

    def test_settle_steps_override(self, connector, fake_env):
        before = fake_env._sim_step_count
        connector.open_gripper(settle_steps=7)
        assert fake_env._sim_step_count - before == 7

    def test_gripper_fraction_clipped(self, connector, fake_env):
        connector.set_gripper(2.5)
        assert fake_env._gripper_fraction == 1.0
        connector.set_gripper(-1.0)
        assert fake_env._gripper_fraction == 0.0


class TestMoveToJoints:
    def test_velocity_mode_action_synthesis(self, connector, fake_env):
        target = [0.5] * 7
        connector.move_to_joints(target, tolerance=0.01, max_steps=200)
        first = fake_env.step_actions[0]
        # velocity_joints: delta = (target - current) * control_freq, plus
        # the gripper command 1 - 2*fraction (fraction=1 → -1).
        expected = (np.array(target) - 0.0) * 20.0
        np.testing.assert_allclose(first[:7], expected)
        assert first[7] == pytest.approx(-1.0)

    def test_converges_within_tolerance(self, connector, fake_env):
        target = np.array([0.4, -0.2, 0.1, 0.0, 0.3, -0.1, 0.2])
        connector.move_to_joints(list(target), tolerance=0.01, max_steps=200)
        assert np.linalg.norm(fake_env._joints - target) < 0.01
        # 50%-tracking dynamics need multiple loop iterations.
        assert len(fake_env.step_actions) > 1

    def test_blocking_env_path_with_settle(self, fake_config):
        calls = {}

        class BlockingEnv(FakeEnv):
            def move_to_joints_blocking(self, target, *, tolerance, max_steps, arm_id):
                calls["target"] = np.asarray(target)
                calls["tolerance"] = tolerance
                calls["max_steps"] = max_steps
                self._joints = np.asarray(target, dtype=np.float64)

        env = BlockingEnv()
        conn = SimConnector(env, fake_config, ik=FakeIK())
        before = env._sim_step_count
        conn.move_to_joints([0.2] * 7, tolerance=0.02, max_steps=50)
        assert calls["tolerance"] == 0.02 and calls["max_steps"] == 50
        # Panda (dof=7) settle default is 0 extra steps.
        assert env._sim_step_count == before

    def test_tool_requires_joint_config(self, connector):
        with pytest.raises(ToolError):
            connector._tool_move_to_joints(None)

    def test_tool_accepts_joint_state_dict(self, connector, fake_env):
        connector._tool_move_to_joints({"positions": [0.2] * 7}, max_steps=100)
        assert np.linalg.norm(fake_env._joints - 0.2) < 0.01


class TestGoToPose:
    def test_calls_ik_then_steps_to_convergence(self, connector, fake_env):
        ik = connector.ik
        ik.joints = [0.25] * 7
        pose = {
            "position": {"x": 0.4, "y": 0.0, "z": 0.3},
            "rotation": {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0},
        }
        connector.go_to_pose(pose)
        assert len(ik.calls) == 1
        # IK is seeded with the arm's current joints.
        assert ik.calls[0]["seed_joints"] is not None
        # Default tcp offset (-0.1 z) forwarded to the backend.
        np.testing.assert_allclose(
            ik.calls[0]["tcp_offset"], [0.0, 0.0, -0.1]
        )
        assert np.linalg.norm(fake_env._joints - 0.25) < 0.01

    def test_z_approach_solves_twice(self, connector):
        ik = connector.ik
        pose = {
            "position": {"x": 0.4, "y": 0.0, "z": 0.3},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        connector.go_to_pose(pose, z_approach=0.1)
        assert len(ik.calls) == 2
        approach = ik.calls[0]["pose"]["position"]
        assert approach["z"] == pytest.approx(0.4)  # 0.3 + 0.1

    def test_ik_failure_raises_tool_error(self, connector):
        connector.ik.solve_ik = lambda *a, **k: None
        with pytest.raises(ToolError, match="IK failed for target pose"):
            connector.go_to_pose({
                "position": {"x": 9.0, "y": 9.0, "z": 9.0},
                "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            })

    def test_cartesian_plans_then_executes(self, connector, fake_env):
        pose = {
            "position": {"x": 0.4, "y": 0.1, "z": 0.3},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        connector.go_to_pose_cartesian(pose)
        assert len(connector.ik.plan_calls) == 1
        # Both linear-plan waypoints were executed via move_to_joints.
        assert len(fake_env.step_actions) > 0


class TestGoHome:
    def test_moves_to_config_home(self, connector, fake_env, fake_config):
        connector.go_home()
        assert np.linalg.norm(
            fake_env._joints - np.array(fake_config.home_joints)
        ) < 0.02

    def test_skipped_on_real(self, fake_env):
        conn = SimConnector(fake_env, FakeEnvConfig(is_real=True), ik=FakeIK())
        conn.go_home()
        assert fake_env.step_actions == []


class TestSimTools:
    def test_capabilities_flags(self, connector):
        caps = connector.capabilities
        assert isinstance(caps, Capabilities)
        assert caps.reset is True
        assert caps.success_check is True
        assert caps.video is True
        assert caps.world_state is False  # FakeEnv has no mujoco sim

    def test_reset_returns_observation(self, connector, fake_env):
        obs = connector.reset(seed=7)
        assert fake_env.reset_seeds == [7]
        assert "cameras" in obs and "arms" in obs

    def test_pending_seed_consumed_once(self, fake_env, fake_config):
        conn = SimConnector(fake_env, fake_config, ik=FakeIK(), seed=42)
        conn.reset()
        conn.reset()
        assert fake_env.reset_seeds == [42, None]

    def test_tool_reset_seed_zero_is_unseeded(self, connector, fake_env):
        connector._tool_reset(seed=0)
        assert fake_env.reset_seeds == [None]

    def test_check_success(self, connector, fake_env):
        fake_env._success = True
        fake_env._reward = 1.0
        completed, reward = connector.check_success()
        assert completed is True and reward == 1.0
        out = connector._tool_check_success()
        assert out == {"task_completed": True, "reward": 1.0}

    def test_step_tool_sets_gripper(self, connector, fake_env):
        out = connector._tool_step(gripper_fraction=0.5)
        assert fake_env._gripper_fraction == 0.5
        assert {"observation", "reward", "terminated", "truncated"} <= set(out)

    def test_apply_policy_action_failed_precondition(self, connector):
        with pytest.raises(ToolError) as exc:
            connector._tool_apply_policy_action([0.0] * 7)
        assert "ApplyPolicyAction not supported by env" in str(exc.value)
        assert "apply_policy_action(action) method" in str(exc.value)

    def test_apply_policy_action_dispatches(self, fake_config):
        class PolicyEnv(FakeEnv):
            def __init__(self):
                super().__init__()
                self.policy_actions = []

            def apply_policy_action(self, action):
                self.policy_actions.append(np.asarray(action))

        env = PolicyEnv()
        conn = SimConnector(env, fake_config, ik=FakeIK())
        conn._tool_apply_policy_action([0.1] * 7)
        assert len(env.policy_actions) == 1

    def test_apply_policy_action_requires_action(self, connector):
        with pytest.raises(ToolError, match="'action' field is required"):
            connector._tool_apply_policy_action([])

    def test_video_roundtrip(self, connector, fake_env, tmp_path):
        connector.start_video()
        for _ in range(3):
            connector.step_once()
        out = connector.save_video(str(tmp_path / "v.mp4"), fps=5)
        assert out["success"] is True
        assert out["num_frames"] == 3
        assert (tmp_path / "v.mp4").exists()

    def test_context_manager_closes_env(self, fake_env, fake_config):
        with SimConnector(fake_env, fake_config, ik=FakeIK()) as conn:
            assert conn.capabilities.reset
        assert fake_env.closed is True


class TestExecuteIntegration:
    def test_gap_execute_uses_connector_tools(self, connector):
        """gap.execute duck-types the connector (registry + observation)."""
        from gap.runtime.execute import execute

        graph = {
            "version": 3,
            "meta": {"name": "gripper_cycle"},
            "nodes": {
                "motion": {"type": "subgraph", "ref": "motion"},
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "motion"]],
            "conditional_edges": {
                "motion": {"router_field": "exit", "mapping": {"ok": "done"}},
            },
            "subgraphs": {
                "motion": {
                    "skill": "generic",
                    "inputs": {},
                    "outputs": {},
                    "nodes": {
                        "open": {
                            "type": "tool",
                            "tool": "robot.open_gripper",
                            "inputs": {"settle_steps": 3},
                        },
                        "close": {
                            "type": "tool",
                            "tool": "robot.close_gripper",
                            "inputs": {"settle_steps": 3},
                        },
                        "check": {"type": "tool", "tool": "sim.check_success"},
                        "ok": {"type": "noop"},
                    },
                    "edges": [
                        ["START", "open"],
                        ["open", "close"],
                        ["close", "check"],
                        ["check", "ok"],
                        ["ok", "END"],
                    ],
                    "conditional_edges": {},
                    "exit": {"router_field": None, "success_values": ["ok"]},
                },
            },
        }
        result = execute(graph, connector, trace_dir=None)
        assert result.error is None
        assert result.success is True
