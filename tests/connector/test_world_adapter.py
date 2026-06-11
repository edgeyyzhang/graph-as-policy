"""LiberoWorldAdapter on a hand-built mujoco-like FakeEnv state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gap.connector.world_adapter import LiberoWorldAdapter, find_mujoco_sim
from gap.runtime.verify import World

# Body table (id: name, parent):
#   0 world        (parent 0)
#   1 table        (parent 0)
#   2 robot0_base  (parent 0)
#   3 gripper0_eef (parent 2)
#   4 cube_main    (parent 0)   object root, free joint
#   5 bowl_main    (parent 0)   object root, free joint
_BODY_NAMES = ["world", "table", "robot0_base", "gripper0_eef", "cube_main", "bowl_main"]

_GEOM_BOX = 6


class FakeModel:
    def __init__(self):
        self.nbody = 6
        self.body_parentid = np.array([0, 0, 0, 2, 0, 0])
        self.body_pos = np.zeros((6, 3))
        self.body_pos[3] = [0.0, 0.0, 0.2]  # eef offset from base
        self.body_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (6, 1))

        # One box geom per body 1..5 (none on world).
        #               table          base           eef            cube             bowl
        self.geom_bodyid = np.array([1, 2, 3, 4, 5])
        self.geom_type = np.array([_GEOM_BOX] * 5)
        self.geom_size = np.array([
            [0.4, 0.4, 0.02],    # table top slab
            [0.05, 0.05, 0.1],   # robot base
            [0.02, 0.02, 0.03],  # gripper
            [0.02, 0.02, 0.02],  # cube
            [0.08, 0.08, 0.04],  # bowl
        ])
        self.geom_pos = np.zeros((5, 3))
        self.geom_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (5, 1))
        self.geom_rbound = np.linalg.norm(self.geom_size, axis=1)

        self._joint_addrs = {f"robot0_joint{i}": i - 1 for i in range(1, 8)}
        self._joint_addrs["gripper0_finger_joint1"] = 7

    def body_name2id(self, name):
        return _BODY_NAMES.index(name)

    def body_id2name(self, bid):
        return _BODY_NAMES[bid]

    def get_joint_qpos_addr(self, name):
        return self._joint_addrs[name]


class FakeData:
    def __init__(self):
        # World poses: table slab top at z=0.40; cube resting on it.
        self.body_xpos = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.38],    # table center (half 0.02 → top 0.40)
            [-0.4, 0.0, 0.0],
            [0.1, 0.0, 0.5],     # gripper above cube
            [0.1, 0.0, 0.42],    # cube center (half 0.02 → bottom 0.40)
            [0.3, 0.2, 0.44],    # bowl
        ])
        self.body_xquat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (6, 1))
        self.qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02])
        self.cvel = np.zeros((6, 6))
        self.cvel[4, 3:6] = [0.0, 0.0, 0.001]  # cube nearly settled
        # Contacts: cube-table, cube-gripper (geom indices).
        self.contact = [
            SimpleNamespace(geom1=3, geom2=0),  # cube vs table
            SimpleNamespace(geom1=3, geom2=2),  # cube vs gripper0_eef
        ]
        self.ncon = len(self.contact)


def make_fake_env():
    sim = SimpleNamespace(model=FakeModel(), data=FakeData())
    inner = SimpleNamespace(sim=sim, obj_body_id={"cube": 4, "bowl": 5})
    handle = SimpleNamespace(env=inner)
    return SimpleNamespace(handle=handle, _gripper_fraction=0.5)


@pytest.fixture
def world() -> World:
    return LiberoWorldAdapter(make_fake_env()).snapshot()


def test_find_mujoco_sim_walks_wrappers():
    env = make_fake_env()
    sim = find_mujoco_sim(env)
    assert sim is env.handle.env.sim
    assert find_mujoco_sim(SimpleNamespace()) is None


def test_bodies_present(world):
    assert set(world.body_names()) == {"cube", "bowl", "table"}


def test_object_pose_from_sim_state(world):
    cube = world.body("cube")
    np.testing.assert_allclose(cube.position, [0.1, 0.0, 0.42])
    np.testing.assert_allclose(cube.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_aabbs_from_model_geoms(world):
    cube = world.body("cube")
    assert cube.bottom_z == pytest.approx(0.40)
    assert cube.top_z == pytest.approx(0.44)
    table = world.body("table")
    assert table.top_z == pytest.approx(0.40)
    assert table.left_x == pytest.approx(-0.4)


def test_predicates_evaluate(world):
    cube, table, bowl = world.body("cube"), world.body("table"), world.body("bowl")
    assert cube.is_on(table)
    assert not cube.is_in(bowl)
    assert cube.is_settled()
    assert bowl.is_above(table, require_xy_overlap=True)


def test_contacts_canonicalized_and_filtered(world):
    cube = world.body("cube")
    assert "table" in cube.contacts
    assert "gripper0_eef" in cube.contacts
    # symmetric entry on the table side
    assert "cube" in world.body("table").contacts
    assert world.body("bowl").contacts == frozenset()


def test_is_grasped_via_gripper_prefix(world):
    assert world.body("cube").is_grasped()
    assert world.held_body().name == "cube"
    assert not world.body("bowl").is_grasped()


def test_robot_view(world):
    robot = world.robot()
    np.testing.assert_allclose(
        robot.joint_pos, [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    )
    assert robot.joint_names[0] == "robot0_joint1"
    # finger qpos 0.02 / 0.04 stroke → half open
    assert robot.gripper_open_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(robot.ee_position, [0.1, 0.0, 0.5])


def test_refresh_after_reset_rebuilds(world):
    env = make_fake_env()
    adapter = LiberoWorldAdapter(env)
    adapter.snapshot()
    # Simulate a hard reset swapping the sim instance.
    env.handle.env.sim = SimpleNamespace(model=FakeModel(), data=FakeData())
    adapter.refresh()
    w2 = adapter.snapshot()
    assert set(w2.body_names()) == {"cube", "bowl", "table"}


def test_obs_dict_fallback_without_sim():
    """No reachable MuJoCo sim → cube_poses (GetState source) fallback."""

    class ObsOnlyEnv:
        def get_observation(self):
            return {
                "cube_poses": {
                    "soup_can": np.array([0.2, 0.1, 0.45, 1.0, 0.0, 0.0, 0.0]),
                }
            }

    world = LiberoWorldAdapter(ObsOnlyEnv()).snapshot()
    can = world.body("soup_can")
    np.testing.assert_allclose(can.position, [0.2, 0.1, 0.45])
    assert can.aabb_upper[2] > can.aabb_lower[2]


def test_simconnector_world_snapshot_wired():
    from gap.connector import SimConnector

    from .conftest import FakeEnvConfig, FakeIK

    env = make_fake_env()
    # Wire the minimal env surface bits SimConnector construction needs.
    conn = SimConnector(env, FakeEnvConfig(), ik=FakeIK())
    world = conn.world_snapshot()
    assert world.body("cube").is_on(world.body("table"))
    assert conn.capabilities.world_state is True
