#!/usr/bin/env python
"""Drop-height sweep: same scene, derived splay (0 deg), vary the place release
height (GAP_PLACE_DROP_CLEARANCE) to test whether the free-fall drop causes the
~18mm -Y place roll. Reports GT concentricity (yellow_tape_1 vs duct_tape_1).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["RECORD_VIDEO"] = "0"
os.environ["GAP_RECV_SPLAY_DEG"] = "0.0"  # derived splay, held fixed

import sys
from pathlib import Path

import mujoco
import numpy as np

import gap
from gap.connector.libero_yam import libero_yam as _libero_yam
from run import BDDL, OPEN_ROBOT_SKILLS, TSH_SKILLS

_SKILLS = [str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)]
YELLOW_XY = (0.40, 0.25)
DUCT_XY = (0.40, -0.25)
DROPS = (0.05, 0.02, 0.01)  # m; 0.05 is the current baseline


def _body_xy(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


def _run(graph_dir: str, drop: float):
    os.environ["GAP_YELLOW_XY"] = f"{YELLOW_XY[0]:.4f},{YELLOW_XY[1]:.4f}"
    os.environ["GAP_DUCT_XY"] = f"{DUCT_XY[0]:.4f},{DUCT_XY[1]:.4f}"
    os.environ["GAP_PLACE_DROP_CLEARANCE"] = f"{drop}"

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
    return dict(drop=drop, graph_ok=graph_ok, done=done,
                err_mm=err * 1000.0, err_vec_mm=[v * 1000.0 for v in err_vec], note=note)


def main():
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "generated" / "task_00")
    rows = []
    for drop in DROPS:
        print(f"\n===== DROP_CLEARANCE = {drop*1000:.0f} mm =====", flush=True)
        rows.append(_run(graph_dir, drop))
    print("\n\n======== DROP SWEEP RESULT (splay=0) ========")
    for r in rows:
        print(f"drop={r['drop']*1000:>3.0f} mm  graph_ok={r['graph_ok']}  task_done={r['done']}"
              f"  place_err={r['err_mm']:.1f} mm  vec(mm)={[round(v,1) for v in r['err_vec_mm']]}"
              f"  {('note='+r['note']) if r['note'] else ''}")


if __name__ == "__main__":
    main()
