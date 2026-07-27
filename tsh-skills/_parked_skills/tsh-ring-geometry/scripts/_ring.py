"""Shared ring geometry for the tape spool: radii estimation + grasp pose.

Two consumers must agree bit-for-bit on the ring-grasp geometry:

  * ``tsh-pickup`` executes the grasp,
  * ``tsh-route`` probes the SAME poses (``execute=False``) to decide which
    arm picks and whether a handover is needed.

so both derive their poses from :func:`ring_grasp_poses` here. The radii
estimator is the same top-face-slab math the perception core used to carry
(``_perceive.estimate_ring_radii``), moved here because it is a *ring*
derivation, not a perception step — the generic ``perceive_object`` emits
only object-agnostic geometry (cloud / top face / half thickness).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from .constants import (
    DOWN_QUAT,
    GRASP_HOOK_DZ,
    GRASP_LIFT_CLEARANCE,
    GRASP_PRE_DZ,
    GRASP_RING_ANGLE_DEG,
    PERCEIVE_TOP_SLAB,
    RING_HOLE_MAX,
    RING_HOLE_MIN,
    RING_HOLE_PCTILE,
    RING_RIM_MARGIN,
    RING_RIM_PCTILE,
)


def ring_radii(pts, center_xy, top_z=None, top_slab=PERCEIVE_TOP_SLAB):
    """Estimate ``(hole_radius, rim_radius)`` from a ring-shaped cloud.

    Uses the 2nd/98th percentiles of the radial distance distribution for
    robustness. When ``top_z`` is given, the radii are measured on the
    TOP-FACE SLAB only (``z > top_z - top_slab``) — from an angled view the
    camera sees the table *through* the hole, and those low-z points would
    otherwise collapse the hole-radius estimate.
    """
    pts = np.asarray(pts)
    if top_z is not None:
        pts = pts[pts[:, 2] > top_z - top_slab]
    dists = np.linalg.norm(pts[:, :2] - np.asarray(center_xy), axis=1)
    hole_r = float(np.percentile(dists, RING_HOLE_PCTILE))
    rim_r = float(np.percentile(dists, RING_RIM_PCTILE))
    return hole_r, rim_r


def radii_ok(hole_r, rim_r) -> bool:
    """Sanity band on the perceived radii — reject noisy/degenerate clouds."""
    return bool(RING_HOLE_MIN < hole_r < RING_HOLE_MAX
                and rim_r > hole_r + RING_RIM_MARGIN)


def ring_grasp_poses(ctx: NodeContext, arm_id: int, center_xyz, *,
                     hole_radius: float, rim_radius: float,
                     fingertip_axial: float, finger_half_gap: float) -> dict:
    """The ring-grasp geometry for ``arm_id`` at the perceived tape centre.

    Centres the TCP on the wall midpoint ``(hole_r + rim_r)/2`` so both
    fingers meet their wall with the same travel; the fingertip trails the
    TCP along the approach axis (``fingertip_axial``). The grasp revolves
    ``GRASP_RING_ANGLE_DEG`` clockwise about the vertical hole axis, plus an
    extra 90° CW for a -Y-side arm so its fingers sit on the mirrored arc
    (side read from the arm base, not tuned).

    Returns a dict with the exact world poses pickup executes (and route
    probes)::

        grasp_xy     (gx, gy) — TCP centre over the wall midpoint
        grasp_quat   wxyz — DOWN_QUAT revolved with the ring angle
        hover_z / seat_z / lift_z — world TCP heights for the three legs
        held_offset  predicted tape centre in the TCP frame at the seat
                     (what ``tape_in_giver`` will measure) — lets the route
                     probe a place hover for this arm BEFORE any grasp.
    """
    if isinstance(center_xyz, dict):
        center_xyz = [center_xyz["x"], center_xyz["y"], center_xyz["z"]]
    x, y, z = (float(v) for v in center_xyz)
    ring_dy = (float(hole_radius) + float(rim_radius)) / 2
    ring_dx = float(fingertip_axial)

    Rz = Rotation.from_rotvec([0.0, 0.0, -np.radians(GRASP_RING_ANGLE_DEG)])
    base_y = ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"][1]
    if float(base_y) < 0.0:
        Rz = Rotation.from_rotvec([0.0, 0.0, -np.radians(90.0)]) * Rz
    off = Rz.apply([ring_dx, ring_dy, 0.0])
    gx, gy = x + off[0], y + off[1]
    qw0, qx0, qy0, qz0 = DOWN_QUAT
    R_grasp = Rz * Rotation.from_quat([qx0, qy0, qz0, qw0])
    rx, ry, rz, rw = R_grasp.as_quat()

    # Tape centre relative to the TCP at the seat pose, expressed in the TCP
    # frame: TCP sits at centre + off (world), so tape-in-TCP = R^-1 · (-off).
    held_offset = R_grasp.inv().apply(-off).tolist()

    return {
        "grasp_xy": (float(gx), float(gy)),
        "grasp_quat": (float(rw), float(rx), float(ry), float(rz)),
        "hover_z": z + GRASP_PRE_DZ,
        "seat_z": z + GRASP_HOOK_DZ,
        "lift_z": z + GRASP_LIFT_CLEARANCE,
        "held_offset": [float(v) for v in held_offset],
    }
