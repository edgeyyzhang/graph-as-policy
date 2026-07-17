#!/usr/bin/env python
"""Phase-localize the ~18mm -Y place error.

Logs, every sim step, the tape centre expressed in the RECEIVER grasp-site frame
(offset_local) — the same frame the exchange measures receiver_offset in. If
offset_local is constant after the receiver grabs, the grip is rigid and any
place error is a measurement/compose bug (compare to the stored receiver_offset).
If it JUMPS, we see exactly which phase (giver-release vs face-on->down reorient)
moves the ring. The gripper approach axis (grasp-site frame +Z, world) marks the
reorientation; the giver/receiver gripper openings mark grab/release.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["RECORD_VIDEO"] = "0"
os.environ["GAP_RECV_SPLAY_DEG"] = "0.0"

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import gap
from gap.connector.libero_yam import libero_yam as _libero_yam
from run import BDDL, OPEN_ROBOT_SKILLS, TSH_SKILLS

_SKILLS = [str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)]
YELLOW_XY = (0.40, 0.25)
DUCT_XY = (0.40, -0.25)
HERE = Path(__file__).resolve().parent


def main():
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "generated" / "task_00")
    os.environ["GAP_YELLOW_XY"] = f"{YELLOW_XY[0]:.4f},{YELLOW_XY[1]:.4f}"
    os.environ["GAP_DUCT_XY"] = f"{DUCT_XY[0]:.4f},{DUCT_XY[1]:.4f}"

    conn = _libero_yam(bddl=str(BDDL), cameras=["agentview", "left", "right"])
    conn.reset()
    env = conn.env
    m, d = env.model, env.data

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_grasp_site")
    tape_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "yellow_tape_1")
    duct_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "duct_tape_1")
    # gripper joint qpos addresses (14D action: right gripper is the last DOF)
    log = []

    def on_step(e):
        R = np.array(d.site_xmat[sid]).reshape(3, 3)
        p = np.array(d.site_xpos[sid])
        tape = np.array(d.xpos[tape_bid])
        off_local = R.T @ (tape - p)          # tape centre in grasp-site frame
        approach = R[:, 2]                     # grasp-site +Z in world (gripper approach)
        log.append((len(log), off_local.copy(), approach.copy(), tape.copy(), p.copy()))

    env.on_step = on_step
    result = gap.execute(graph_dir, conn, skills=_SKILLS)

    tape = np.array(d.xpos[tape_bid])[:2]
    duct = np.array(d.xpos[duct_bid])[:2]
    err = (tape - duct) * 1000
    print(f"\ngraph_ok={result.success}  task_done={env.task_completed()}")
    print(f"FINAL place err = {np.linalg.norm(err):.1f} mm  vec(mm)={[round(v,1) for v in err]}")

    # stored receiver_offset from this run
    nd = Path(graph_dir) / "node_data" / "handover_tape.bimanual_exchange" / "output.json"
    ro = None
    if nd.exists():
        ro = json.load(open(nd)).get("receiver_offset")
        print(f"stored receiver_offset (grab, grasp-site frame) mm = "
              f"{[round(v*1000,1) for v in ro]}")

    # Downsample the trace; flag big jumps in off_local.
    print("\nstep | approachXYZ            | off_local(mm) xyz          | |off|mm  d|off|mm")
    prev = None
    n = len(log)
    idxs = sorted(set(list(range(0, n, max(1, n // 60))) + [n - 1]))
    for i in idxs:
        _, ol, ap, _, _ = log[i]
        olmm = ol * 1000
        dnorm = 0.0 if prev is None else np.linalg.norm(ol - prev) * 1000
        prev = ol
        print(f"{i:4d} | [{ap[0]:+.2f} {ap[1]:+.2f} {ap[2]:+.2f}] | "
              f"[{olmm[0]:+7.1f} {olmm[1]:+7.1f} {olmm[2]:+7.1f}] | "
              f"{np.linalg.norm(olmm):6.1f}  {dnorm:6.1f}")

    conn.close()


if __name__ == "__main__":
    main()
