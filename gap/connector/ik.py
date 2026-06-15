"""In-process PyRoKI inverse kinematics for the connector layer.

This module merges two pieces of the source research codebase into one
plain-Python unit (no gRPC):

- the PyRoKI *solver* setup from the dev tree's pyroki service (and its
  ``pyroki_snippets``): basic IK, velocity-cost IK seeded with the previous
  configuration, and linear Cartesian planning by waypoint-interpolated IK;
- the *frame/TCP math* from ``services/sim_bridge/ik_backend.py``'s
  ``PyRoKIBackend``: the Franka path (apply the configured TCP offset and
  optional TCP rotation, solve for ``panda_hand``) plus the 6-DOF path's
  joint-order reversal and base-frame conversion quirks.

The numerics are kept identical to the source: same cost weights
(pos 50 / ori 10), same trust-region config, same refinement iteration
counts and jump threshold.

pyroki is CPU-JAX — the first solve per robot model JIT-compiles (a few
seconds); subsequent solves are milliseconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gap_core.types import Se3Pose, Trajectory, make_pose

logger = logging.getLogger(__name__)

# YAM TCP offset from link_6 to gripper tip (meters, in link_6 frame).
# Matches MuJoCo's `grasp_site` (pos="0 0 0.1347") on link_6 — the +Z axis
# points toward the gripper fingers, so the tip is at +0.1347 m along link_6 z.
_YAM_TCP_OFFSET = np.array([0.0, 0.0, 0.1347], dtype=np.float64)


# ---------------------------------------------------------------------------
# Robot model loading (cached)
# ---------------------------------------------------------------------------

_ROBOT_CACHE: dict[str, object] = {}


def load_robot(
    robot_urdf: str = "panda_description",
    robot_urdf_path: str | None = None,
):
    """Load (and cache) a ``pyroki.Robot`` model.

    Mirrors ``PyRoKIServicer._load_solver``: a local URDF path wins when it
    exists on disk; otherwise the named ``robot_descriptions`` entry is
    loaded (default ``panda_description``).
    """
    import pyroki as pk

    key = robot_urdf_path if robot_urdf_path and Path(robot_urdf_path).exists() else robot_urdf
    cached = _ROBOT_CACHE.get(str(key))
    if cached is not None:
        return cached

    if robot_urdf_path and Path(robot_urdf_path).exists():
        import yourdfpy

        urdf_path = Path(robot_urdf_path)
        mesh_dir = str(urdf_path.parent / "assets")
        logger.info("Loading robot URDF from path %r with PyRoKI...", robot_urdf_path)
        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            mesh_dir=mesh_dir,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
    else:
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        logger.info("Loading robot URDF %r with PyRoKI...", robot_urdf)
        urdf = load_robot_description(robot_urdf)
    robot = pk.Robot.from_urdf(urdf)
    logger.info(
        "PyRoKI loaded: robot=%r, actuated_joints=%d",
        robot_urdf_path or robot_urdf,
        robot.joints.num_actuated_joints,
    )
    _ROBOT_CACHE[str(key)] = robot
    return robot


# ---------------------------------------------------------------------------
# JAX solves (verbatim from the dev tree's pyroki service snippets)
# ---------------------------------------------------------------------------


# The JAX solves live in gap.connector._ik_jax (verbatim from the source
# pyroki_snippets — @jdc.jit needs module-level jax imports for annotation
# resolution). Imported lazily so `import gap.connector` stays light.


def solve_ik_basic(
    robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Solve the basic IK problem (no seed).

    Returns the joint configuration, shape ``(num_actuated_joints,)``.
    """
    from gap.connector import _ik_jax

    return _ik_jax.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_wxyz=np.asarray(target_wxyz, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
    )


def solve_ik_vel_cost(
    robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    prev_cfg: np.ndarray,
    initial_cfg: np.ndarray | None = None,
) -> np.ndarray:
    """Velocity-cost IK: prefer solutions near ``prev_cfg``.

    Keeps the solver on the IK branch closest to the arm's current
    configuration (no elbow flips between solves).
    """
    from gap.connector import _ik_jax

    return _ik_jax.solve_ik_vel_cost(
        robot=robot,
        target_link_name=target_link_name,
        target_wxyz=np.asarray(target_wxyz, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
        prev_cfg=np.asarray(prev_cfg, dtype=np.float64),
        initial_cfg=initial_cfg,
    )


# ---------------------------------------------------------------------------
# Linear Cartesian planning (verbatim from the dev tree's pyroki service)
# ---------------------------------------------------------------------------


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two wxyz quaternions.

    Returns an array of shape ``(num_steps, 4)`` in wxyz format.
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 20,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
    ik_refinement_iters: int = 15,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Returns an array of shape ``(num_waypoints, num_joints)``.
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations, strict=False)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(ik_refinement_iters):
                cfg = solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = solve_ik_basic(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped "
                        f"{np.degrees(joint_diff[joint_idx]):.1f} deg "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    if jump_warnings:
        logger.warning(
            "%d large joint jump(s) detected (threshold: %.1f deg):\n%s",
            len(jump_warnings),
            np.degrees(jump_threshold),
            "\n".join(jump_warnings),
        )

    return np.array(trajectory)


# ---------------------------------------------------------------------------
# Pose helpers (gap.types Se3Pose <-> numpy)
# ---------------------------------------------------------------------------


def _pose_to_numpy(pose: Se3Pose) -> tuple[np.ndarray, np.ndarray]:
    """Extract (position, wxyz quaternion) arrays from an Se3Pose."""
    p = pose["position"]
    r = pose["rotation"]
    position = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    wxyz = np.array([r["w"], r["x"], r["y"], r["z"]], dtype=np.float64)
    return position, wxyz


def _trajectory_from_array(arr: np.ndarray) -> Trajectory:
    return {
        "waypoints": [
            {"positions": np.asarray(row, dtype=np.float64)} for row in np.asarray(arr)
        ]
    }


# ---------------------------------------------------------------------------
# Backend — PyRoKIBackend's frame/TCP math, in-process
# ---------------------------------------------------------------------------


class PyRokiBackend:
    """In-process IK backend with the source ``PyRoKIBackend`` semantics.

    For **Panda** (7-DOF): pre-applies the configured TCP offset (optionally
    respecting a TCP-frame rotation, e.g. Robotiq mounted at pi/4) and solves
    for the ``panda_hand`` link. The per-call ``tcp_offset`` argument is
    accepted for interface parity but — exactly as in the source backend —
    the *configured* offset governs the Panda path (``None`` for LIBERO:
    targets are panda_hand-frame poses already).

    For **6-DOF (YAM)**: converts world-frame TCP target to base-frame
    ``link_6`` target, reverses seed joint order before the solve, and
    reverses returned joint order (PyRoKI parses the YAM URDF in
    joint6→joint1 order).
    """

    def __init__(
        self,
        *,
        arm_dof: int = 7,
        tcp_offset: np.ndarray | None = None,
        tcp_rotation: Rotation | None = None,
        home_joints: list[float] | None = None,
        arm_bases: list[tuple[float, float, float]] | None = None,
        robot_urdf: str = "panda_description",
        robot_urdf_path: str | None = None,
        target_link: str | None = None,
    ) -> None:
        self._arm_dof = int(arm_dof)
        self._tcp_offset = (
            np.asarray(tcp_offset, dtype=np.float64) if tcp_offset is not None else None
        )
        self._tcp_rotation = tcp_rotation
        self._home_joints = list(home_joints) if home_joints is not None else None
        self._arm_bases = list(arm_bases) if arm_bases is not None else None
        self._robot_urdf = robot_urdf
        self._robot_urdf_path = robot_urdf_path
        self._target_link = target_link or ("panda_hand" if self._arm_dof == 7 else "link_6")
        self._robot = None  # lazy

    # -- model -------------------------------------------------------------

    @property
    def robot(self):
        if self._robot is None:
            self._robot = load_robot(self._robot_urdf, self._robot_urdf_path)
        return self._robot

    @property
    def trajectory_needs_joint_reverse(self) -> bool:
        return self._arm_dof == 6

    def _is_yam(self) -> bool:
        return self._arm_dof == 6

    # -- frames ------------------------------------------------------------

    def world_pose_to_base_frame(self, pose: Se3Pose, arm_id: int) -> Se3Pose:
        """Transform a world-frame Se3Pose into the specified arm's base frame.

        For YAM the arms are mounted vertically (identity base rotation), so
        only a translational offset is needed.
        """
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pose
        bx, by, bz = self._arm_bases[arm_id]
        p = pose["position"]
        return {
            "position": {"x": p["x"] - bx, "y": p["y"] - by, "z": p["z"] - bz},
            "rotation": dict(pose["rotation"]),
        }

    # -- IK ------------------------------------------------------------------

    def solve_ik(
        self,
        target_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        seed_joints: list[float] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> list[float] | None:
        """Solve IK. Returns joints in simulator-native order, or None on failure."""
        if self._is_yam():
            return self._solve_yam(target_world_pose, arm_id, seed_joints)
        return self._solve_panda(target_world_pose, tcp_offset, seed_joints)

    # --- Panda path --------------------------------------------------------

    def _solve_panda(
        self,
        pose: Se3Pose,
        tcp_offset: np.ndarray | None,
        seed_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Panda: rotate panda_hand orientation if a TCP rotation is configured
        (Robotiq at pi/4), shift target from tool tip to panda_hand link, and
        solve.

        ``seed_joints`` becomes the velocity-cost solve's ``prev_cfg`` so the
        solver stays on the IK branch closest to the arm's current
        configuration; without it the basic solve runs from the internal
        default seed (which can jump branches between phases).
        """
        solve_pose = pose
        if self._tcp_offset is not None:
            q = pose["rotation"]
            R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
            if self._tcp_rotation is not None:
                R_link = R @ self._tcp_rotation.inv().as_matrix()
                link_quat_xyzw = Rotation.from_matrix(R_link).as_quat()
                link_rot = {
                    "x": float(link_quat_xyzw[0]),
                    "y": float(link_quat_xyzw[1]),
                    "z": float(link_quat_xyzw[2]),
                    "w": float(link_quat_xyzw[3]),
                }
            else:
                R_link = R
                link_rot = dict(pose["rotation"])
            tcp_world = R_link @ self._tcp_offset
            p = pose["position"]
            solve_pose = {
                "position": {
                    "x": p["x"] + float(tcp_world[0]),
                    "y": p["y"] + float(tcp_world[1]),
                    "z": p["z"] + float(tcp_world[2]),
                },
                "rotation": link_rot,
            }

        target_position, target_wxyz = _pose_to_numpy(solve_pose)
        n_actuated = self.robot.joints.num_actuated_joints
        try:
            if seed_joints is not None:
                # The ``panda_description`` URDF exposes 8 actuated joints
                # (panda_joint1..7 + panda_finger_joint1). The simulator only
                # tracks the 7 arm joints, so pad with a neutral finger qpos
                # to match the velocity-cost solver's expected prev_cfg shape.
                seed = list(seed_joints)[: self._arm_dof]
                seed = seed + [0.0] * (n_actuated - len(seed))
                cfg = solve_ik_vel_cost(
                    robot=self.robot,
                    target_link_name=self._target_link,
                    target_wxyz=target_wxyz,
                    target_position=target_position,
                    prev_cfg=np.asarray(seed, dtype=np.float64),
                )
            else:
                cfg = solve_ik_basic(
                    robot=self.robot,
                    target_link_name=self._target_link,
                    target_wxyz=target_wxyz,
                    target_position=target_position,
                )
        except Exception:
            logger.exception("IK solve failed")
            return None
        joints = [float(v) for v in np.asarray(cfg)]
        # Drop the finger joint pyroki returns; the arm controller only
        # consumes the arm joints.
        return joints[: self._arm_dof]

    # --- YAM path ----------------------------------------------------------

    def _solve_yam(
        self,
        pose: Se3Pose,
        arm_id: int,
        seed_joints: list[float] | None,
    ) -> list[float] | None:
        """YAM: world→base frame, TCP→link_6 subtraction, reversed seed."""
        base_frame_pose = self.world_pose_to_base_frame(pose, arm_id)
        q = base_frame_pose["rotation"]
        R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        tcp_in_base = R @ _YAM_TCP_OFFSET
        bp = base_frame_pose["position"]
        link6_pose: Se3Pose = {
            "position": {
                "x": bp["x"] - float(tcp_in_base[0]),
                "y": bp["y"] - float(tcp_in_base[1]),
                "z": bp["z"] - float(tcp_in_base[2]),
            },
            "rotation": dict(base_frame_pose["rotation"]),
        }
        # Seed IK in reversed order (PyRoKI parses YAM URDF as joint6→joint1)
        if seed_joints is not None:
            s = list(seed_joints)[: self._arm_dof]
            s.reverse()
            seed = s
        else:
            home = self._home_joints or [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
            seed = list(reversed(home))
        target_position, target_wxyz = _pose_to_numpy(link6_pose)
        try:
            cfg = solve_ik_vel_cost(
                robot=self.robot,
                target_link_name=self._target_link,
                target_wxyz=target_wxyz,
                target_position=target_position,
                prev_cfg=np.asarray(seed, dtype=np.float64),
            )
        except Exception:
            logger.exception("IK solve failed")
            return None
        joints = [float(v) for v in np.asarray(cfg)]
        joints.reverse()  # to joint1→joint6 (simulator order)
        return joints

    # --- Linear plan -------------------------------------------------------

    def plan_linear(
        self,
        start_world_pose: Se3Pose,
        end_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        tcp_offset: np.ndarray | None = None,
        num_waypoints: int = 40,
        ik_refinement_iters: int = 40,
        jump_threshold: float = 0.5,
    ) -> Trajectory | None:
        """Plan a Cartesian straight-line trajectory.

        Waypoints are in backend-native joint order; see
        ``trajectory_needs_joint_reverse``. Defaults match the source's
        ``PyRoKIPlanRequest(num_waypoints=40, ik_refinement_iters=40)``.
        """
        if self._is_yam():
            start_pose = self.world_pose_to_base_frame(start_world_pose, arm_id)
            end_pose = self.world_pose_to_base_frame(end_world_pose, arm_id)
            adjusted = []
            for bp in (start_pose, end_pose):
                q = bp["rotation"]
                R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
                off = R @ _YAM_TCP_OFFSET
                p = bp["position"]
                adjusted.append(
                    make_pose(
                        (p["x"] - off[0], p["y"] - off[1], p["z"] - off[2]),
                        (q["w"], q["x"], q["y"], q["z"]),
                    )
                )
            start_pose, end_pose = adjusted
        else:
            start_pose = start_world_pose
            end_pose = end_world_pose

        start_position, start_wxyz = _pose_to_numpy(start_pose)
        end_position, end_wxyz = _pose_to_numpy(end_pose)
        try:
            sol_traj = plan_trajectory_linear_ik(
                robot=self.robot,
                target_link_name=self._target_link,
                start_pos=start_position,
                start_wxyz=start_wxyz,
                end_pos=end_position,
                end_wxyz=end_wxyz,
                num_waypoints=num_waypoints,
                jump_threshold=jump_threshold,
                ik_refinement_iters=ik_refinement_iters,
            )
        except Exception:
            logger.exception("Linear planning failed")
            return None
        return _trajectory_from_array(np.asarray(sol_traj))

    def supports_world_aware_plan(self) -> bool:
        return False

    def plan_to_pose(self, *args, **kwargs):
        raise NotImplementedError(
            "PyRokiBackend does not support collision-aware single-pose planning; "
            "use the open-robot-skills curobo bundle for world-aware planning."
        )


# ---------------------------------------------------------------------------
# Standalone conveniences (default Franka backend)
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND: PyRokiBackend | None = None


def _default_backend() -> PyRokiBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = PyRokiBackend()
    return _DEFAULT_BACKEND


def solve_ik(
    target_pose: Se3Pose,
    seed_joints: list[float] | None = None,
    tcp_offset: np.ndarray | None = None,
    *,
    robot_urdf_path: str | None = None,
    arm_id: int = 0,
) -> list[float] | None:
    """Solve IK against the default Franka model (or a custom URDF path).

    ``target_pose`` is a world-frame :class:`gap.types.Se3Pose` for the
    ``panda_hand`` link (no TCP offset is configured on the default
    backend — matching the source LIBERO setup).
    """
    if robot_urdf_path:
        backend = PyRokiBackend(robot_urdf_path=robot_urdf_path)
    else:
        backend = _default_backend()
    return backend.solve_ik(
        target_pose, arm_id=arm_id, seed_joints=seed_joints, tcp_offset=tcp_offset
    )


def plan_linear(
    start_pose: Se3Pose,
    end_pose: Se3Pose,
    *,
    num_waypoints: int = 40,
    ik_refinement_iters: int = 40,
    robot_urdf_path: str | None = None,
    arm_id: int = 0,
) -> Trajectory | None:
    """Linear Cartesian plan against the default Franka model."""
    if robot_urdf_path:
        backend = PyRokiBackend(robot_urdf_path=robot_urdf_path)
    else:
        backend = _default_backend()
    return backend.plan_linear(
        start_pose,
        end_pose,
        arm_id=arm_id,
        num_waypoints=num_waypoints,
        ik_refinement_iters=ik_refinement_iters,
    )


__all__ = [
    "PyRokiBackend",
    "load_robot",
    "plan_linear",
    "plan_trajectory_linear_ik",
    "slerp_quaternions",
    "solve_ik",
    "solve_ik_basic",
    "solve_ik_vel_cost",
]
