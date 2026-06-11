"""In-process pyroki IK parity tests (CPU JAX; default suite, no sim).

Parity strategy: ground truth comes from pyroki's own forward kinematics of
the ``panda_description`` URDF. Solving back to a known FK pose must land
within millimeters/centiradians — the same costs/iterations/tolerances as
the source service, so any numerical drift from the port shows up here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gap.connector import ik as gik
from gap.types import make_pose

_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


@pytest.fixture(scope="module")
def robot():
    return gik.load_robot("panda_description")


def _fk_panda_hand(robot, joints7: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(position, wxyz) of panda_hand at the given 7-dof arm config."""
    import jax.numpy as jnp

    cfg = np.zeros(robot.joints.num_actuated_joints)
    cfg[:7] = joints7
    Ts = np.asarray(robot.forward_kinematics(jnp.array(cfg)))
    row = Ts[robot.links.names.index("panda_hand")]
    return row[4:7].copy(), row[:4].copy()  # wxyz_xyz layout


def _pose_error(robot, joints7, target_pos, target_wxyz) -> tuple[float, float]:
    pos, wxyz = _fk_panda_hand(robot, joints7)
    pos_err = float(np.linalg.norm(pos - target_pos))
    # angle between quaternions (sign-insensitive)
    dot = abs(float(np.dot(wxyz, target_wxyz)))
    ang_err = 2.0 * float(np.arccos(min(dot, 1.0)))
    return pos_err, ang_err


def test_robot_model_cached(robot):
    assert gik.load_robot("panda_description") is robot
    assert robot.joints.num_actuated_joints == 8  # 7 arm + finger


def test_solve_home_pose_near_zero_error(robot):
    """Solve back to the FK pose of the Franka home config."""
    target_pos, target_wxyz = _fk_panda_hand(robot, _HOME)
    pose = make_pose(target_pos, target_wxyz)

    joints = gik.solve_ik(pose, seed_joints=list(_HOME))
    assert joints is not None and len(joints) == 7

    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3, f"position error {pos_err:.4f} m"
    assert ang_err < 0.05, f"orientation error {ang_err:.4f} rad"
    # Seeded with home, the solution stays on the home branch.
    assert np.linalg.norm(np.asarray(joints) - np.asarray(_HOME)) < 0.2


def test_solve_without_seed_basic_path(robot):
    target_pos, target_wxyz = _fk_panda_hand(robot, _HOME)
    joints = gik.solve_ik(make_pose(target_pos, target_wxyz))
    assert joints is not None and len(joints) == 7
    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3
    assert ang_err < 0.05


def test_fk_solve_roundtrip(robot):
    """FK a perturbed config, solve back, FK again → same pose."""
    perturbed = list(np.asarray(_HOME) + [0.2, -0.1, 0.15, 0.1, -0.2, 0.1, -0.15])
    target_pos, target_wxyz = _fk_panda_hand(robot, perturbed)

    joints = gik.solve_ik(make_pose(target_pos, target_wxyz), seed_joints=list(_HOME))
    assert joints is not None
    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3, f"position error {pos_err:.4f} m"
    assert ang_err < 0.05, f"orientation error {ang_err:.4f} rad"


def test_backend_tcp_offset_shifts_target(robot):
    """A configured TCP offset solves for the link pose shifted along the
    tool axis — verified against FK of the returned joints."""
    offset = np.array([0.0, 0.0, -0.107])
    backend = gik.PyRokiBackend(tcp_offset=offset)
    # Tool-tip target: hand pose displaced by R @ offset (downward-facing).
    hand_pos, hand_wxyz = _fk_panda_hand(robot, _HOME)
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(
        [hand_wxyz[1], hand_wxyz[2], hand_wxyz[3], hand_wxyz[0]]
    ).as_matrix()
    tip_pos = hand_pos - R @ offset  # link target == tip + R@offset == hand
    joints = backend.solve_ik(
        make_pose(tip_pos, hand_wxyz), seed_joints=list(_HOME)
    )
    assert joints is not None
    pos_err, ang_err = _pose_error(robot, joints, hand_pos, hand_wxyz)
    assert pos_err < 5e-3
    assert ang_err < 0.05


def test_plan_linear_tracks_endpoints(robot):
    start_pos, start_wxyz = _fk_panda_hand(robot, _HOME)
    end_pos = start_pos + np.array([0.0, 0.0, 0.08])

    backend = gik.PyRokiBackend()
    traj = backend.plan_linear(
        make_pose(start_pos, start_wxyz),
        make_pose(end_pos, start_wxyz),
        num_waypoints=5,
        ik_refinement_iters=3,
    )
    assert traj is not None
    waypoints = traj["waypoints"]
    assert len(waypoints) == 5
    # Endpoint FK error (waypoints are 8-dof backend-native; arm = first 7).
    last = list(np.asarray(waypoints[-1]["positions"])[:7])
    pos_err, ang_err = _pose_error(robot, last, end_pos, start_wxyz)
    assert pos_err < 1e-2
    assert ang_err < 0.1
    # Smoothness: no waypoint-to-waypoint joint jump beyond the source's
    # 0.5 rad threshold.
    arrs = [np.asarray(w["positions"])[:7] for w in waypoints]
    for a, b in zip(arrs, arrs[1:], strict=False):
        assert np.max(np.abs(b - a)) < 0.5
