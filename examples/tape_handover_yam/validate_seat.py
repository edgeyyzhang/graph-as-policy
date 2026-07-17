#!/usr/bin/env python
"""Validate a DERIVED settled-seat formula for receiver_offset.

Measures the TRUE body-fixed tape offset in the exact get_ee_pose frame the place
skill uses (R_ee^-1 . (tape_GT - TCP)) — constant during the rigid hold, so it's
exactly what receiver_offset SHOULD be. Then compares against:
  - the stored receiver_offset (pre-settle FK measurement place currently uses)
  - candidate geometric seats from perceived radii + gripper geometry.
Also prints the truth vector re-expressed in the recv_quat grab frame so its
components (radial / vertical / axial) are interpretable.
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
ARM = 1  # receiver


def main():
    graph_dir = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "generated" / "task_00")
    os.environ["GAP_YELLOW_XY"] = f"{YELLOW_XY[0]:.4f},{YELLOW_XY[1]:.4f}"
    os.environ["GAP_DUCT_XY"] = f"{DUCT_XY[0]:.4f},{DUCT_XY[1]:.4f}"

    conn = _libero_yam(bddl=str(BDDL), cameras=["agentview", "left", "right"])
    conn.reset()
    env = conn.env
    m, d = env.model, env.data
    tape_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "yellow_tape_1")

    samples = []  # (step, offset_local in get_ee_pose frame, |tape-TCP|, gripper_frac)

    def on_step(e):
        ee = conn.get_ee_pose(arm_id=ARM)
        p = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
        r = ee["rotation"]
        R = Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])
        tape = np.array(d.xpos[tape_bid])
        off_local = R.inv().apply(tape - p)
        gf = conn.get_gripper_fraction(arm_id=ARM)
        samples.append((len(samples), off_local, float(np.linalg.norm(tape - p)), gf))

    env.on_step = on_step
    result = gap.execute(graph_dir, conn, skills=_SKILLS)

    # Save the full trace for offline analysis (no need to re-run the sim).
    np.save(HERE / "seat_trace.npy",
            np.array([np.concatenate([[s[0]], s[1], [s[2], s[3]]]) for s in samples]))

    # Held phase = receiver gripper closed (gf<0.5) AND actually holding the tape
    # (0.03m < |tape-TCP| < 0.12m, excludes the far-idle and the post-release drop).
    # The body-fixed offset is constant while held; take the median of the LAST
    # contiguous held block (just before the place release).
    held = [s for s in samples if s[3] < 0.5 and 0.03 < s[2] < 0.12]
    truth = (np.median(np.array([s[1] for s in held[-150:]]), axis=0)
             if held else samples[-1][1])
    print(f"\ngraph_ok={result.success} task_done={env.task_completed()}  held_samples={len(held)}")

    nd = Path(graph_dir) / "node_data"
    def load(p):
        f = nd / p / "output.json"
        return json.load(open(f)) if f.exists() else {}
    exch = load("handover_tape.bimanual_exchange")
    perc = load("perceive_tape.perceive_tape")
    grip = load("gripper_geometry.derive_gripper_geometry")

    stored = np.array(exch.get("receiver_offset", [np.nan]*3))
    recv_tcp = exch.get("receiver_tcp")
    hole_r = perc.get("hole_radius"); rim_r = perc.get("rim_radius")
    fa = grip.get("fingertip_axial"); fhg = grip.get("finger_half_gap")

    print("\n--- TRUTH (what place SHOULD use), get_ee_pose frame ---")
    print(f"  truth_offset (mm)  = {(truth*1000).round(1)}   |.|={np.linalg.norm(truth)*1000:.1f}")
    print(f"  stored offset (mm) = {(stored*1000).round(1)}   |.|={np.linalg.norm(stored)*1000:.1f}")
    print(f"  stored - truth (mm)= {((stored-truth)*1000).round(1)}   "
          f"err={np.linalg.norm(stored-truth)*1000:.1f}")

    # Re-express truth in the recv_quat grab frame so components are interpretable.
    if recv_tcp:
        rq = recv_tcp[3:7]  # wxyz
        Rgrab = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]])
        # truth is in the live get_ee_pose frame; convert to grab frame is identity if
        # R_ee==Rgrab. We instead show truth rotated by (Rgrab^-1 R_ee)~I; print raw.
    ring_dy = (hole_r + rim_r) / 2 if hole_r and rim_r else None
    print("\n--- geometry we have ---")
    print(f"  hole_r={hole_r and round(hole_r*1000,1)}mm rim_r={rim_r and round(rim_r*1000,1)}mm"
          f"  ring_dy=(hole+rim)/2={ring_dy and round(ring_dy*1000,1)}mm")
    print(f"  fingertip_axial={fa and round(fa*1000,1)}mm  finger_half_gap={fhg and round(fhg*1000,1)}mm")
    print(f"  |truth|={np.linalg.norm(truth)*1000:.1f}mm  wall_mid(ring_dy)={ring_dy and round(ring_dy*1000,1)}mm"
          f"  sqrt(ring_dy^2+fa^2)="
          f"{(np.hypot(ring_dy, fa)*1000):.1f}mm" if ring_dy and fa else "")

    # dump raw held trace tail for inspection
    print("\n  held offset samples (mm), last 8:")
    for s in held[-8:]:
        print("   ", s[0], (s[1]*1000).round(1), round(s[2]*1000,1))
    conn.close()


if __name__ == "__main__":
    main()
