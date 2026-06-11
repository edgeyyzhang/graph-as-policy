"""Sim-marked integration: real LIBERO env behind the connector.

Runs only with the [libero] extra (``pytest -m sim``); skips cleanly when
gap.envs (or the sim stack underneath it) is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.sim


def _envs_available() -> bool:
    try:
        import gap.envs.registry  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def conn():
    if not _envs_available():
        pytest.skip("gap.envs not importable")
    import gap.connector

    try:
        connector = gap.connector.sim(
            "libero", task="libero_object_all_variance/0", seed=0
        )
    except Exception as exc:  # missing sim deps (mujoco/EGL/task assets)
        pytest.skip(f"sim env unavailable: {exc}")
    yield connector
    connector.close()


class TestConnectorContract:
    def test_capabilities(self, conn):
        caps = conn.capabilities
        assert caps.reset and caps.success_check
        assert caps.world_state  # MuJoCo sim reachable → ground truth

    def test_reset_and_observation(self, conn):
        obs = conn.reset()
        assert obs["cameras"], "no cameras in observation"
        cam = obs["cameras"][0]
        assert cam["rgb"].dtype == np.uint8 and cam["rgb"].ndim == 3
        arm = obs["arms"][0]
        assert arm["joint_state"]["positions"].shape == (7,)
        assert 0.0 <= arm["gripper_fraction"] <= 1.5

    def test_ee_pose_unit_quaternion(self, conn):
        pose = conn.get_ee_pose()
        q = pose["rotation"]
        norm = (q["w"] ** 2 + q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

    def test_world_snapshot(self, conn):
        world = conn.world_snapshot()
        assert world.body_names(), "no ground-truth bodies"
        robot = world.robot()
        assert robot.joint_pos.shape == (7,)


class TestModelsFreeGraphE2E:
    def test_motion_graph_via_gap_execute(self, conn):
        """go_home → move_to_joints → open/close gripper → sim.check_success."""
        from gap.runtime.execute import execute

        joints = list(np.asarray(conn._home_joints) + 0.05)
        graph = {
            "version": 3,
            "meta": {"name": "models_free_motion"},
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
                        "home": {"type": "tool", "tool": "robot.go_home"},
                        "move": {
                            "type": "tool",
                            "tool": "robot.move_to_joints",
                            "inputs": {
                                "joint_config": {"positions": joints},
                                "tolerance": 0.02,
                                "max_steps": 150,
                            },
                        },
                        "open": {
                            "type": "tool",
                            "tool": "robot.open_gripper",
                            "inputs": {"settle_steps": 10},
                        },
                        "close": {
                            "type": "tool",
                            "tool": "robot.close_gripper",
                            "inputs": {"settle_steps": 10},
                        },
                        "check": {"type": "tool", "tool": "sim.check_success"},
                        "ok": {"type": "noop"},
                    },
                    "edges": [
                        ["START", "home"],
                        ["home", "move"],
                        ["move", "open"],
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
        conn.reset()
        result = execute(graph, conn, trace_dir=None)
        assert result.error is None, result.error
        assert result.success is True

        # The graph's moves actually drove the sim arm.
        joint_pos = conn.get_observation()["arms"][0]["joint_state"]["positions"]
        assert np.linalg.norm(joint_pos - np.asarray(joints)) < 0.1
