#!/usr/bin/env python
"""Splay isolation A/B: run the SAME scene twice, differing only in the receiver
splay (derived 0 deg vs the tuned ~8.5 deg via GAP_RECV_SPLAY_DEG), and report
the GT concentricity of the placed yellow tape vs the duct.

Everything else (tape/duct position, meet point, perception) is identical, so the
delta in place error is attributable to the receiver splay alone. GT is read from
the MuJoCo bodies yellow_tape_1 / duct_tape_1 after each rollout.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["RECORD_VIDEO"] = "0"  # no video; we only need the GT numbers

import sys
from pathlib import Path

import mujoco
import numpy as np

import gap
from gap.connector.libero_yam import libero_yam as _libero_yam
from run import BDDL, OPEN_ROBOT_SKILLS, TSH_SKILLS

_SKILLS = [str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)]

# Fixed scene (clean anchors from the benchmark grid): giver-side yellow, receiver-side duct.
YELLOW_XY = (0.40, 0.25)
DUCT_XY = (0.40, -0.25)


def _body_xy(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


def _run(graph_dir: str, splay_deg: float):
    os.environ["GAP_YELLOW_XY"] = f"{YELLOW_XY[0]:.4f},{YELLOW_XY[1]:.4f}"
    os.environ["GAP_DUCT_XY"] = f"{DUCT_XY[0]:.4f},{DUCT_XY[1]:.4f}"
    os.environ["GAP_RECV_SPLAY_DEG"] = f"{splay_deg}"

    conn = _libero_yam(bddl=str(BDDL), cameras=["agentview", "left", "right"])
    conn.reset()
    env = conn.env
    try:
        result = gap.execute(graph_dir, conn, skills=_SKILLS)
        graph_ok = bool(result.success)
        done = bool(env.task_completed())
        tape_xy = _body_xy(env.model, env.data, "yellow_tape_1")
        duct_xy = _body_xy(env.model, env.data, "duct_tape_1")
        err = float(np.linalg.norm(tape_xy - duct_xy))
        err_vec = (tape_xy - duct_xy).tolist()
        note = str(result.error) if result.error else ""
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return dict(splay=splay_deg, graph_ok=graph_ok, done=done,
                tape_xy=tape_xy.tolist(), duct_xy=duct_xy.tolist(),
                err_mm=err * 1000.0, err_vec_mm=[v * 1000.0 for v in err_vec],
                note=note)


def main():
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "generated" / "task_00")
    rows = []
    for splay in (0.0, 8.5):
        print(f"\n===== SPLAY = {splay} deg =====", flush=True)
        rows.append(_run(graph_dir, splay))
    print("\n\n======== SPLAY ISOLATION RESULT ========")
    for r in rows:
        print(f"splay={r['splay']:>4} deg  graph_ok={r['graph_ok']}  task_done={r['done']}"
              f"  place_err={r['err_mm']:.1f} mm  vec(mm)={[round(v,1) for v in r['err_vec_mm']]}"
              f"  {('note='+r['note']) if r['note'] else ''}")


if __name__ == "__main__":
    main()
