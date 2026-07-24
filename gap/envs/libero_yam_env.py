"""LiberoYamEnv — GaP env adapter for the bimanual YAM LIBERO-YAM scenes.

Wraps ``libero_yam.envs.ControlEnv`` and exposes the
:class:`gap.envs.base_env.BaseEnv` surface the GaP connector reads, so a workflow
graph can drive the YAM arms with the connector's motion primitives
(``robot.go_to_pose`` / ``robot.open_gripper`` / ...).

ALL dynamics come from LIBERO-YAM: every motion is executed by
``ControlEnv.step_single`` (MuJoCo ``<position>`` actuators, ``gravcomp=1``, the
deployed-YAML kp/kd, joint-limit clipping, 17 substeps @ ``physics_dt=0.002``).
This adapter only maps the connector's per-arm contract onto LIBERO-YAM's 14-D
bimanual action ``[L j1..6, L grip, R j1..6, R grip]``.

Per-arm TCP poses are surfaced in the arm's **base frame** (via
``libero_yam.sim.kinematics.arm_tcp_in_base``); the connector adds the base
offset from ``EnvConfig.arm_bases`` to recover world frame.
"""

from __future__ import annotations

import os

# MuJoCo latches its GL backend at import; default to EGL before mujoco loads.
os.environ.setdefault("MUJOCO_GL", "egl")

from typing import Any

import numpy as np

from .base_env import BaseEnv
from .registry import EnvConfig

# link_6 -> gripper-tip offset in the link_6 frame (matches the grasp_site and
# LIBERO-YAM/scripts/yam_ik.py).
YAM_TCP_OFFSET = (0.0, 0.0, 0.1347) # calculated myself

# 14-D layout indices.
_GRIP = {0: 6, 1: 13}      # gripper slot per arm
_J0 = {0: 0, 1: 7}         # first arm-joint slot per arm

class LiberoYamEnv(BaseEnv):
    """GaP-facing wrapper over the bimanual YAM ``ControlEnv``."""

    def __init__(self, bddl_file: str, *, max_steps: int = 6000,
                 camera_names: tuple = ()) -> None:
        super().__init__()
        from libero_yam.envs import ControlEnv, YamControlConfig
        from libero_yam.sim.kinematics import (
            arm_joint_limits_from_model,
            relax_actuator_forcerange,
        )

        # Use scripted-motion gains (stiffer wrists) so IK targets are tracked.
        self._control = ControlEnv(bddl_file=str(bddl_file), render_cameras=False,
                                   chunk_dim=1,
                                   control_config=YamControlConfig.scripted_motion())
        # Replace YAML soft-limits with the URDF mechanical range read from the
        # model — this is what cuRobo solves within, so IK solutions never get
        # clipped. Also relax forcerange so kp=250 doesn't saturate actuators.
        self._control.cfg.arm_joint_limits = arm_joint_limits_from_model(
            self._control
        )
        relax_actuator_forcerange(self._control)
        self.max_steps = int(max_steps)
        self._cmd = np.zeros(14, dtype=np.float32)  # last full command; idle arm holds it
        self._sim_step_count = 0
        self._current_done = False
        self.on_step = None  # optional callback(env) fired each control step (e.g. video)
        # GaP object key == MJCF body name (yellow_tape_1, duct_tape_1, ...).
        self._objects = [nm for v in self._control.problem.get("objects", {}).values() for nm in v]

        # Perception cameras (RGB-D). Rendered ON DEMAND via camera_frames() —
        # NOT on the per-step get_observation() hot path — since perception is
        # only needed at the grasp instant. Empty tuple => no rendering, so
        # non-vision callers pay nothing. The renderers are created LAZILY on the
        # first camera_frames() call, not here: the graph executor runs nodes on
        # ThreadPoolExecutor workers, and MuJoCo's EGL context is thread-affine,
        # so a main-thread-created renderer fails eglMakeCurrent from a worker.
        # Creating them on the rendering thread avoids that.
        self._percept_cams = tuple(camera_names)
        self._cam_hw = (self._control.env._camera_height,
                        self._control.env._camera_width)

        # Video capture (driven by SimConnector.start_video / save_video).
        self._video_enabled = False
        self._video_frames: list[np.ndarray] = []
        self._video_renderer = None
        self._video_cam = None
        self._video_every = 8  # capture every Nth sim step

    # --- MuJoCo handles -----------------------------------------------------
    @property
    def model(self):
        return self._control.env.model

    @property
    def data(self):
        return self._control.env.data

    def _qpos_arm(self, arm_id: int) -> list[int]:
        return self._control.env._qpos_indices[_J0[arm_id]:_J0[arm_id] + 6]

    # --- BaseEnv surface ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = self._control.reset()
        self._cmd = np.asarray(obs["state"], dtype=np.float32)
        self._apply_physics_tuning()
        self._apply_tape_overrides()  # eval: re-place tapes per env vars, then settle
        for _ in range(25):  # let the authored scene settle, holding the reset pose
            self._step_once()
        self._sim_step_count = 0
        self._current_done = False
        return self.get_observation(), {"task": "libero_yam"}

    def _apply_physics_tuning(self) -> None:
        """MuJoCo contact/actuator overrides required for stable tape grasping.

        Without these the default gains are too weak to clamp the ring and the
        default contact params cause instability. Ported from run.py so that
        ``gap run`` gets the same physics as the custom driver.
        """
        import mujoco

        m = self.model
        ### some mujoco overrides, directly taken from LIBERO_YAM repo
        # Soften globally stiff contacts (same fix as oracle_pick_cream_cheese).
        m.geom_solref[m.geom_solref[:, 0] < 0.02, 0] = 0.02
        m.geom_solimp[:, 0] = np.minimum(m.geom_solimp[:, 0], 0.9)
        m.geom_solimp[:, 1] = np.minimum(m.geom_solimp[:, 1], 0.95)

        # Stiffer contacts for tape convex-decomp geoms so fingers don't phase
        # through the ring wall. geom_priority=1 forces MuJoCo to use ONLY the
        # tape's solref for finger-tape contacts.

        # ----------------------------------------------------------
        # TAPE OVERRIDES
        ### Mujoco Overrides specific to this task, ensure no "magical" grasp
        tape_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "yellow_tape_1__object")
        if tape_bid >= 0:
            for gi in range(m.ngeom):
                if int(m.geom_bodyid[gi]) == tape_bid and int(m.geom_contype[gi]) > 0:
                    m.geom_solref[gi, 0] = 0.005
                    m.geom_solimp[gi, 0] = 0.998
                    m.geom_solimp[gi, 1] = 0.9999
                    m.geom_priority[gi] = 1
                    m.geom_condim[gi] = 6
                    m.geom_friction[gi, 2] = 0.01
            # Realistic tape rotational inertia (the arena compiler floors it too high).
            m.body_inertia[tape_bid] = np.array([4.7e-5, 4.7e-5, 8.9e-5]) * 1.69
            # Damp the tape's free joint. With zero damping the 50 g, low-inertia ring
            # tumbles freely off the near-elastic tape contacts when the receiver
            # releases it (multiple flips before it settles), and picks up spurious
            # momentum from contact forces during other low-velocity moments too
            # (e.g. the exchange's retract) — damping force scales with velocity, so
            # it's near-zero while the ring is rigidly gripped and stationary, but NOT
            # negligible whenever the tape is moving without being firmly held.
            # The free joint is on the PARENT body (this "__object" body holds the
            # tape's real mass/geoms but has no joint of its own).
            jid = m.body_jntadr[m.body_parentid[tape_bid]]
            if jid >= 0 and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = m.jnt_dofadr[jid]
                m.dof_damping[adr:adr + 3] = 0.05    # translational (m/s drag)
                m.dof_damping[adr + 3:adr + 6] = 0.005  # rotational (kills the flip)
        # ----------------------------------------------------------

        # Stiffen gripper PD so fingers clamp the ring during reorientation
        # and resist the receiver's contact force at exchange.
        ### Not entirely sure if this is faithful
        for side, kp in (("left", 2000.0), ("right", 800.0)):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_gripper")
            if gid >= 0:
                m.actuator_gainprm[gid, 0] = kp
                m.actuator_biasprm[gid, 1] = -kp
                m.actuator_biasprm[gid, 2] = -5.0
        


    ## This is the way I have it setup: the tape is teleported to a new (x, y) position, 
    ##  then the simulator is advanced 25 steps so gravity and contacts let it come to rest.
    def _apply_tape_overrides(self) -> None:
        """Generalization eval: re-place tape free bodies from env vars before the
        settle. ``GAP_YELLOW_XY`` / ``GAP_DUCT_XY`` = "x,y" in LIBERO world metres
        (see the openpi->LIBERO mapping in the eval driver). No-op when unset, so
        normal runs are unaffected. Writes the free-joint qpos (x,y), leaving z +
        orientation at the authored resting values; the settle loop lets it rest.
        """
        import os
        import mujoco

        for name, var in (("yellow_tape_1", "GAP_YELLOW_XY"),
                          ("duct_tape_1", "GAP_DUCT_XY")):
            val = os.environ.get(var)
            if not val:
                continue
            x, y = (float(v) for v in val.split(","))
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            qadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            self.data.qpos[qadr] = x
            self.data.qpos[qadr + 1] = y
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self._cmd = np.asarray(action, dtype=np.float32).reshape(14)
        return self._advance()

    def compute_reward(self) -> float:
        return 1.0 if self._current_done else 0.0

    def task_completed(self) -> bool:
        return self._current_done

    # --- Connector control hooks -------------------------------------------
    def _advance(self):
        _o, _r, done, _t, info = self._control.step_single(self._cmd)
        self._sim_step_count += 1
        self._current_done = bool(done)
        self._grab_video_frame()
        if self.on_step is not None:
            self.on_step(self)
        return self.get_observation(), self.compute_reward(), bool(done), self._sim_step_count >= self.max_steps, info

    def _step_once(self) -> None:
        self._advance()

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        # 1=open, 0=closed — identical convention to LIBERO-YAM.
        self._cmd[_GRIP[arm_id]] = float(np.clip(fraction, 0.0, 1.0))

    def move_to_joints_blocking(self, target_joints, tolerance=0.03, max_steps=300, arm_id=0):
        """Drive ``arm_id`` to ``target_joints`` under the YAM PD; idle arm holds.

        Two-phase motion matching the oracle's approach:

        Phase 1 — interpolation: linearly ramp the command from the current
        joint position to target over ~20 steps per radian of max-joint error.
        This prevents the correction formula from being applied while the arm
        is far from target, which would command past joint limits and cause
        oscillation.

        Phase 2 — hold with correction: once near the target, apply
        ``q_target + (q_target − q_actual)`` every few steps to compensate
        the PD steady-state error caused by wrist joint frictionloss.
        Mirrors LIBERO-YAM/scripts/oracle_pick_cream_cheese.py::step_with_correction.
        """
        j0 = _J0[arm_id]
        arm_q = self._qpos_arm(arm_id)
        target = np.asarray(target_joints, dtype=np.float32).reshape(-1)[:6]

        # --- Phase 1: interpolate to approach the target smoothly ---
        current = self.data.qpos[arm_q].astype(np.float32)
        dist = float(np.max(np.abs(target - current)))
        # ~20 steps per radian so each step is ≤0.05 rad (PD can track this)
        n_interp = min(int(dist * 20) + 1, max_steps // 2)
        for i in range(1, n_interp + 1):
            alpha = i / n_interp
            self._cmd[j0:j0 + 6] = current + alpha * (target - current)
            self._step_once()

        # --- Phase 2: hold at target with frictionloss correction ---
        self._cmd[j0:j0 + 6] = target
        remaining = max(1, max_steps - n_interp)
        for step in range(remaining):
            if step and step % 6 == 0:
                actual = self.data.qpos[arm_q].astype(np.float32)
                if float(np.linalg.norm(actual - target)) < tolerance:
                    break
                self._cmd[j0:j0 + 6] = target + (target - actual)
            self._step_once()
        # Reset to clean target so settle steps don't command the overcorrected value.
        self._cmd[j0:j0 + 6] = target

    # --- Observation --------------------------------------------------------
    def get_observation(self) -> dict[str, Any]:
        """Per-arm joint + base-frame TCP state, plus ground-truth object poses.

        No camera keys (the handover skill is non-vision; the driver renders
        video separately — the connector tolerates absent cameras).
        """
        from libero_yam.sim.kinematics import arm_tcp_in_base

        obs: dict[str, Any] = {}
        for arm in (0, 1):
            ### gets joint pose
            joints = self.data.qpos[self._qpos_arm(arm)].astype(np.float64)
            ### gripper pos
            grip = float(self._cmd[_GRIP[arm]])
            obs[f"robot_joint_pos_{arm}"] = np.concatenate([joints, [grip]])
            pos, quat = arm_tcp_in_base(self._control, arm, YAM_TCP_OFFSET)
            obs[f"robot_cartesian_pos_{arm}"] = np.concatenate([pos, quat, [grip]])
        obs["cube_poses"] = {nm: self._body_pose(nm) for nm in self._objects}
        return obs

    def camera_frames(self) -> dict[str, Any]:
        """Render the perception cameras as RGB-D + calibration, on demand.

        Returns ``{cam: {images:{rgb,depth}, intrinsics, pose}}`` in the shape the
        connector's ``_build_camera_frame`` expects. ``pose`` is the OpenCV
        camera-to-world (``[x,y,z, qw,qx,qy,qz]``): MuJoCo's camera frame is
        OpenGL (+Y up, -Z forward) but ``geometry.mask_to_world_points`` assumes
        OpenCV (+Y down, +Z forward), so we apply the diag(1,-1,-1) flip here —
        validated to back-project to ~2 mm (scratchpad/depth_check2.py). Depth is
        metric metres straight from MuJoCo's depth renderer (no conversion).
        """
        if not self._percept_cams:
            return {}
        import mujoco
        from libero_yam.sim import cameras as camlib

        H, W = self._cam_hw
        # Create the renderers FRESH per call and close them before returning.
        # The graph executor renders run.py's video cameras (grab()) on the same
        # worker threads; a perception renderer left holding a live/current EGL
        # context contends with those (eglMakeCurrent -> EGL_BAD_ACCESS on a later
        # grab). Building + closing within one call leaves no lingering context.
        # Perception runs once (at the grasp), so the per-call cost is negligible.
        if os.environ.get("GAP_NO_SHADOW"):
            self._control.env.model.light_castshadow[:] = 0
        rgb_r = mujoco.Renderer(self._control.env.model, H, W)
        depth_r = mujoco.Renderer(self._control.env.model, H, W)
        depth_r.enable_depth_rendering()
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        frames: dict[str, Any] = {}
        try:
            for cam in self._percept_cams:
                rgb_r.update_scene(self.data, camera=cam)
                rgb = rgb_r.render().copy()
                depth_r.update_scene(self.data, camera=cam)
                depth = depth_r.render().copy().astype(np.float32)
                c2w = camlib.extrinsics(self._control.env, cam) @ flip
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, np.ascontiguousarray(c2w[:3, :3]).flatten())
                frames[cam] = {
                    "images": {"rgb": rgb, "depth": depth},
                    "intrinsics": camlib.intrinsics(self._control.env, cam, H, W),
                    "pose": np.concatenate([c2w[:3, 3], quat]),
                }
        finally:
            rgb_r.close()
            depth_r.close()
        return frames

    def _body_pose(self, name: str) -> np.ndarray:
        import mujoco

        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.concatenate([self.data.xpos[bid], self.data.xquat[bid]])  # world pos+wxyz

    def get_object_pose(self, object_name: str) -> dict[str, Any]:
        """World-frame pose of a scene object (backs the sim.object_pose tool).

        ``object_name`` (not ``name``) avoids colliding with ToolRegistry.invoke's
        own ``name`` argument when dispatched via ctx.tool.
        """
        p = self._body_pose(object_name)
        return {"object_name": object_name, "position": p[:3].tolist(),
                "quaternion_wxyz": p[3:].tolist()}

    # --- Video capture ------------------------------------------------------
    def enable_video_capture(self, enabled: bool = True, clear: bool = True) -> None:
        self._video_enabled = enabled
        if clear:
            self._video_frames.clear()

    def get_video_frames(self, clear: bool = True) -> list[np.ndarray]:
        frames = list(self._video_frames)
        if clear:
            self._video_frames.clear()
        return frames

    def _grab_video_frame(self) -> None:
        if not self._video_enabled or self._sim_step_count % self._video_every != 0:
            return
        import mujoco
        if self._video_renderer is None:
            self._video_renderer = mujoco.Renderer(self.model, 480, 640)
            self._video_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._video_cam)
            self._video_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._video_cam.lookat[:] = [0.56, 0.02, 0.84]
            self._video_cam.distance = 0.94
            self._video_cam.azimuth = 10.0
            self._video_cam.elevation = -34.0
        self._video_renderer.update_scene(self.data, camera=self._video_cam)
        self._video_frames.append(self._video_renderer.render().copy())

    def close(self):
        if self._video_renderer is not None:
            self._video_renderer.close()
            self._video_renderer = None
        self._control.env.close()


def make_env(suite_name, task_id, camera_names=None, enable_render=False, **extra):
    """Registry factory → ``(env, EnvConfig)``. ``task_id`` indexes the sorted
    ``*.bddl`` in ``BDDL_FILES_PATH/<suite_name>/``; pass ``bddl_file=`` to override."""
    from libero_yam import BDDL_FILES_PATH
    from libero_yam.sim.kinematics import arm_base_world_pose

    bddl_file = extra.get("bddl_file")
    if bddl_file is None:
        bddls = sorted((BDDL_FILES_PATH / suite_name).glob("*.bddl"))
        bddl_file = str(bddls[int(task_id) % len(bddls)])

    env = LiberoYamEnv(bddl_file, max_steps=int(extra.get("max_steps", 6000)),
                       camera_names=tuple(camera_names or ()))
    env.reset()  # settle so base-body world poses are valid
    arm_bases = tuple(tuple(map(float, arm_base_world_pose(env._control, a)[0])) for a in (0, 1))

    config = EnvConfig(
        arm_dof=6, num_arms=2, action_mode="absolute_joints", control_freq=30.0,
        tcp_offset=YAM_TCP_OFFSET, arm_bases=arm_bases,
        default_cameras=tuple(camera_names or ()), is_real=False,
    )
    return env, config
