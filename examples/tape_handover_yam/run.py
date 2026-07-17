#!/usr/bin/env python
"""Run a bimanual tape-handover graph on the LIBERO-YAM env (non-policy).

Authoring is done by GaP's agent (see GENERATE.md) or by ``build_graph.py``;
this driver only EXECUTES a graph dir, adding the example's evaluation extras
on top of what ``gap run <graph> --sim libero_yam_tabletop/2`` already does:
two free-camera renders (front + opposite) and an arm-arm collision report.
All sim physics tuning lives in the env adapter (``gap.envs.libero_yam_env``),
so both entry points run the identical scene.

    python examples/tape_handover_yam/run.py GRAPH_DIR [--record-extra]

GRAPH_DIR is REQUIRED — point it at an agent-authored graph, e.g.
.../generated/task_00 (produce one with ``gap generate``; see GENERATE.md). The
hand-authored reference graph (build_graph.py) still builds, but is no longer the
implicit default.

``--record-extra`` opts into ``trial_recorder.TrialRecorder`` (isolated helper,
see that module): per-step joint/cartesian trajectories plus the front
(agentview) and wrist (left/right D405) camera feeds, written to
``renders/extra/`` alongside the existing front/opposite videos. Off by
default — no effect on a plain run.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from gap.connector.libero_yam import libero_yam as _libero_yam
from gap.runtime.execute import execute

HERE = Path(__file__).resolve().parent
# Passing skills= explicitly REPLACES sibling auto-discovery, so open-robot-skills
# must be listed too — perception + planning use its grounding-dino / sam3 /
# curobo bundles (base first, tsh-skills layered on top).
OPEN_ROBOT_SKILLS = HERE.parents[2] / "open-robot-skills"
TSH_SKILLS = HERE.parents[1] / "tsh-skills"   # in-repo (moved from the sibling checkout)
BDDL = HERE.parents[2] / "LIBERO-YAM" / "libero_yam" / "bddl_files" / \
    "libero_yam_tabletop" / "two_tape_handover.bddl"


class ArmCollisionMonitor:
    """Detect left-arm vs right-arm collisions from MuJoCo contacts each step.

    A contact counts as an arm-arm collision when one geom belongs to a
    ``left_*`` body and the other to a ``right_*`` body. Finger-finger contacts
    (expected during the two-fingers-in-one-hole exchange) are tagged separately
    from LINK-BODY contacts (the wrist/arm collisions that mean the arms crashed).
    Call it as the env step hook; read ``.report()`` after the run.
    """

    def __init__(self, model):
        self.geom_arm: dict[int, str] = {}
        self.geom_body: dict[int, str] = {}
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bn.startswith("left"):
                self.geom_arm[gid], self.geom_body[gid] = "left", bn
            elif bn.startswith("right"):
                self.geom_arm[gid], self.geom_body[gid] = "right", bn
        self.events: dict[tuple, dict] = {}
        self.step = 0

    def __call__(self, env) -> None:
        self.step += 1
        d = env.data
        for i in range(d.ncon):
            c = d.contact[i]
            a1, a2 = self.geom_arm.get(int(c.geom1)), self.geom_arm.get(int(c.geom2))
            if a1 is None or a2 is None or a1 == a2:
                continue  # not a cross-arm contact
            key = tuple(sorted((self.geom_body[int(c.geom1)], self.geom_body[int(c.geom2)])))
            pen = max(0.0, -float(c.dist))
            e = self.events.get(key)
            if e is None:
                self.events[key] = {"count": 1, "max_pen": pen, "first": self.step, "last": self.step}
            else:
                e["count"] += 1
                e["last"] = self.step
                e["max_pen"] = max(e["max_pen"], pen)

    def report(self) -> str:
        if not self.events:
            return "ARM-COLLISION CHECK: no left/right arm contacts detected ✓"
        link_hits = [k for k in self.events if not all("finger" in b for b in k)]
        head = (
            f"ARM-COLLISION CHECK: {'LINK-BODY COLLISION' if link_hits else 'finger contact only'} "
            f"({len(self.events)} body-pairs)"
        )
        lines = [head]
        for key in sorted(self.events, key=lambda k: -self.events[k]["max_pen"]):
            e = self.events[key]
            tag = "LINK-BODY" if not all("finger" in b for b in key) else "finger"
            lines.append(
                f"  [{tag}] {key[0]} <-> {key[1]}: {e['count']} contacts, "
                f"max_pen={e['max_pen'] * 1000:.1f}mm, steps {e['first']}-{e['last']}"
            )
        return "\n".join(lines)


def _free_camera(model, azimuth):
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.56, 0.02, 0.84]
    cam.distance = 0.94
    cam.azimuth = azimuth
    cam.elevation = -34.0
    return renderer, cam


def main(argv=None) -> int:
    t_start = time.perf_counter()
    argv = list(argv or [])
    record_extra = "--record-extra" in argv
    if record_extra:
        argv.remove("--record-extra")
    if not argv:
        print(
            "run.py: a graph dir is required. Generate one with `gap generate` "
            "(see GENERATE.md), then:\n"
            "  python examples/tape_handover_yam/run.py "
            "examples/tape_handover_yam/generated/task_00 [--record-extra]",
            file=sys.stderr)
        return 2
    graph_dir = Path(argv[0])

    # agentview = external perception primary; left/right = wrist backups. These
    # surface as RGB-D CameraFrames on robot.get_observation (rendered on demand).
    conn = _libero_yam(
        bddl=str(BDDL), cameras=["agentview", "left", "right"])
    conn.reset()
    env = conn.env

    # Front + opposite free-camera renders (the back wall is hidden for the
    # opposite view so it looks back toward the robot bases). Skipped under
    # --record-extra: TrialRecorder's front(agentview)+wrist feeds already
    # cover it, and these two more renders every 8 steps roughly double wall
    # time on top of the recorder's own overhead.
    renderer = renderer_opp = cam = cam_opp = back_wall_gid = None
    if not record_extra:
        renderer, cam = _free_camera(env.model, azimuth=10.0)
        renderer_opp, cam_opp = _free_camera(env.model, azimuth=190.0)
        back_wall_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "back_wall")

    frames: list = []
    frames_opp: list = []
    n = [0]

    def grab(e):
        if renderer is None:
            return
        n[0] += 1
        if n[0] % 8 == 0:
            renderer.update_scene(e.data, camera=cam)
            frames.append(renderer.render().copy())
            wall_alpha = env.model.geom_rgba[back_wall_gid, 3]
            env.model.geom_rgba[back_wall_gid, 3] = 0.0
            renderer_opp.update_scene(e.data, camera=cam_opp)
            frames_opp.append(renderer_opp.render().copy())
            env.model.geom_rgba[back_wall_gid, 3] = wall_alpha

    collision = ArmCollisionMonitor(env.model)
    recorder = None
    if record_extra:
        from trial_recorder import TrialRecorder
        recorder = TrialRecorder(env, task_prompt=BDDL.stem.replace("_", " "))

    def on_step(e):
        collision(e)
        grab(e)
        if recorder is not None:
            recorder.on_step(e)

    env.on_step = on_step

    t_setup = time.perf_counter()
    print(f"executing {graph_dir} ... (setup took {t_setup - t_start:.1f}s)")
    result = execute(graph_dir, conn,
                     skills=[str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)])
    t_exec = time.perf_counter()

    done = bool(env.task_completed())
    print(f"\nworkflow success: {result.success}")
    print(f"execution took {t_exec - t_setup:.1f}s")
    print(f"task_completed (BDDL goal): {done}")
    if result.error:
        print(f"error: {result.error}")
    print("\n" + collision.report())

    renders = HERE / "renders"
    renders.mkdir(exist_ok=True)
    for name, buf in (("handover.mp4", frames), ("handover_opposite.mp4", frames_opp)):
        if buf:
            out = renders / name
            with imageio.get_writer(str(out), fps=30, codec="h264", quality=8) as w:
                for f in buf:
                    w.append_data(f)
            print(f"wrote {out} ({len(buf)} frames)")

    if renderer is not None:
        renderer.close()
        renderer_opp.close()
    if recorder is not None:
        extra_dir = renders / "extra"
        recorder.save(extra_dir)
        recorder.close()
        print(f"wrote trajectories + camera feeds -> {extra_dir}")
    conn.close()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
