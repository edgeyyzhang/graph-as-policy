"""Derive the YAM gripper's grasp offsets from the robot model via FK.

The grasp offsets in ``constants.py`` (``GRASP_RING_DY``, ``GRIPPER_FINGERTIP_BEHIND``,
and the pad z-span the ``GRASP_HOOK_DZ`` seat depth reasons about) are pure
gripper geometry — the comments even say they were "measured from yam.xml FK" by
hand. The finger pads are nested five bodies deep under ``link_6`` with a
per-body quaternion at each level (``link_6 → link_left_finger[prismatic] →
lf_rot → lf_down → sphere geoms``) and the ``grasp_site`` (TCP) frame is itself
rotated (``quat="1 0 0 -1"``), so hand-derivation is error-prone. This module
recomputes them from the model instead: load the arm XML, set the gripper
aperture, run FK, and read the fingertip contact geoms in the ``grasp_site``
frame. Drop in a different gripper URDF and the offsets self-configure — no
re-tuning.

Model location: derived from ``GAP_CUROBO_ROBOT_CONFIGS`` (the connector sets it
to ``yam.yml=<LIBERO-YAM>/assets/curobo_configs/yam.yml``), from which the arm
XML at ``<LIBERO-YAM>/assets/sim_models/i2rt_yam/yam.xml`` is found — the same
wiring the curobo bundle already relies on, so no new hardcoded path.

Frame convention (grasp_site / TCP frame, matching how the grasp offsets are used):
  * site-x — the tool approach axis; the fingertips sit ``fingertip_axial`` ahead
             of the TCP along it (the ``GRIPPER_FINGERTIP_BEHIND`` / ring_dx offset).
  * site-y — the finger open/close axis; the pair straddles the rim with
             ``finger_half_gap`` half-spread at the open aperture (``GRASP_RING_DY``).
  * site-z — the tool axis; the pads span ``[pad_z_min, pad_z_max]`` about the TCP
             (drives the ``GRASP_HOOK_DZ`` seat depth vs the table).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


def _find_arm_xml() -> str:
    """Locate the YAM arm MuJoCo XML from the curobo-config wiring (no new path).

    ``GAP_CUROBO_ROBOT_CONFIGS`` is ``name=<...>/assets/curobo_configs/yam.yml``;
    the arm XML is a sibling under ``assets/sim_models/i2rt_yam/yam.xml``.
    """
    env = os.environ.get("GAP_CUROBO_ROBOT_CONFIGS", "")
    cfg = env.split("=", 1)[1] if "=" in env else env
    if cfg:
        assets = Path(cfg).resolve().parent.parent  # .../assets
        xml = assets / "sim_models" / "i2rt_yam" / "yam.xml"
        if xml.exists():
            return str(xml)
    raise FileNotFoundError(
        "gripper-geometry: could not locate yam.xml. Set GAP_CUROBO_ROBOT_CONFIGS "
        "(the connector does this) so the arm model can be found for FK.")


@lru_cache(maxsize=1)
def grip_geometry(xml_path: str | None = None) -> dict:
    """Derive the gripper grasp offsets from the arm model via FK (cached).

    Returns a dict of metres, all in the ``grasp_site`` (TCP) frame:
      ``finger_half_gap``   half the open-aperture spread of the finger pair (site-y)
      ``fingertip_axial``   fingertip offset ahead of the TCP along the approach (site-x)
      ``pad_z_min/pad_z_max`` pad contact z-span about the TCP (site-z)
      ``aperture_open/closed`` the finger joint range endpoints (m)
    """
    import mujoco

    xml = xml_path or _find_arm_xml()
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)

    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
    qadr = m.jnt_qposadr[jid]
    lo, hi = (float(v) for v in m.jnt_range[jid])  # closed, open

    # The grip pads are the sphere collision geoms on the fingertip bodies
    # (``lf_down`` / ``rf_down``) — the points that actually contact the ring.
    tips = []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if bn.endswith("_down") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            tips.append(g)
    if not tips:
        raise RuntimeError("gripper-geometry: no fingertip sphere geoms found")

    def tips_in_site(q: float) -> np.ndarray:
        d.qpos[:] = 0.0
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        sp = d.site_xpos[site].copy()
        sR = d.site_xmat[site].reshape(3, 3).copy()
        return np.array([sR.T @ (d.geom_xpos[g] - sp) for g in tips])

    open_tips = tips_in_site(hi)
    return {
        "finger_half_gap": float((open_tips[:, 1].max() - open_tips[:, 1].min()) / 2.0),
        "fingertip_axial": float(np.median(open_tips[:, 0])),
        "pad_z_min": float(open_tips[:, 2].min()),
        "pad_z_max": float(open_tips[:, 2].max()),
        "aperture_open": hi,
        "aperture_closed": lo,
    }


if __name__ == "__main__":
    # Derived-from-FK gripper geometry (the tuned constants it replaced are
    # archived in skills/deprecated/tuned_fallback_constants.py).
    g = grip_geometry()
    print("=== YAM gripper geometry: derived (FK) ===")
    print(f"  finger_half_gap = {g['finger_half_gap']*1000:6.1f} mm")
    print(f"  fingertip_axial = {g['fingertip_axial']*1000:6.1f} mm")
    print(f"  pad z-span      = [{g['pad_z_min']*1000:.1f}, {g['pad_z_max']*1000:.1f}] mm "
          f"about TCP  (drives GRASP_HOOK_DZ seat depth)")
    print(f"  aperture open/closed = {g['aperture_open']*1000:.1f} / "
          f"{g['aperture_closed']*1000:.1f} mm")
