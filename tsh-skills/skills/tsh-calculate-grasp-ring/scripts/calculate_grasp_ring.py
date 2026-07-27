"""Ring radii + grasp poses for ONE arm — the shared grasp geometry.

Measures the ring's hole/rim radii from its perceived point cloud (top-face
slab, 2nd/98th radial percentile — was tsh-ring-geometry) and derives the
grasp from them: the TCP centres on the wall midpoint ``(hole_r + rim_r)/2``
so both fingers meet their wall with the same travel, with the fingertip
trailing the TCP along the approach axis (``fingertip_axial``). The grasp
revolves ``GRASP_RING_ANGLE_DEG`` clockwise about the vertical hole axis, plus
an extra 90° CW for a −Y-side arm so its fingers sit on the mirrored arc (side
read from the arm base, not tuned).

This node is the SINGLE producer of that geometry: ``tsh-route-arms-bimanual``
probes these exact poses (``execute=False``) to decide which arm picks, and
``tsh-pickup`` executes them — so the route's feasibility answer and the grasp
can never disagree. It only DECIDES the poses; it never plans or moves. It
raises when the radii fall outside the sanity band (degenerate cloud, no tuned
fallback).

Emits the three grasp legs as ready-to-plan world poses, the predicted
held-tape offset (tape centre in the TCP frame at the seat — what
``tape_in_giver`` will later measure), and the radii themselves (relayed to
tsh-route-arms-bimanual's receiver probe and tsh-handover's thread offset).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import Quaternion, Vec3

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


class Output(TypedDict):
    hover_xyz: Vec3       # approach hover over the wall midpoint
    seat_xyz: Vec3        # descended seat (near finger in the hole)
    lift_xyz: Vec3        # post-grasp lift, relative to grasp height
    grasp_quat: Quaternion  # shared TCP orientation for all three legs
    held_offset: Vec3     # predicted tape centre in the TCP frame at the seat
    hole_radius: float    # inner hole radius (m)
    rim_radius: float     # outer rim radius (m)


def _as3(v) -> list:
    return [float(v["x"]), float(v["y"]), float(v["z"])]


def _ring_radii(pts, center_xy, top_z, top_slab=PERCEIVE_TOP_SLAB):
    """(hole_r, rim_r) from the TOP-FACE SLAB of a ring cloud (2nd/98th pctile).

    The slab cut (``z > top_z - top_slab``) drops the table points the camera
    sees through the hole from an angled view, which would otherwise collapse
    the hole-radius estimate.
    """
    pts = np.asarray(pts)
    pts = pts[pts[:, 2] > top_z - top_slab]
    dists = np.linalg.norm(pts[:, :2] - np.asarray(center_xy), axis=1)
    return (float(np.percentile(dists, RING_HOLE_PCTILE)),
            float(np.percentile(dists, RING_RIM_PCTILE)))


def _radii_ok(hole_r, rim_r) -> bool:
    """Sanity band on the perceived radii — reject noisy/degenerate clouds."""
    return bool(RING_HOLE_MIN < hole_r < RING_HOLE_MAX
                and rim_r > hole_r + RING_RIM_MARGIN)


def run(ctx: NodeContext, *, target_xyz: list, target_cloud, target_half_z: float,
        arm_id: int, fingertip_axial: float, finger_half_gap: float) -> Output:
    """Ring radii + grasp poses for ``arm_id`` at the perceived tape centre.

    target_xyz:    world grasp point (body-centroid height), from perception.
    target_cloud:  world-frame ring point cloud, from perception.
    target_half_z: perceived half-thickness — locates the top face
                   (``top_z = center_z + half_z``) for the radii slab cut.
    arm_id:        the arm these poses are for (either side works — the geometry
                   mirrors from the arm base). Instantiate once per arm needed.
    fingertip_axial / finger_half_gap: FK-derived gripper offsets from this
                   subgraph's own derive_gripper_geometry node (the fingertip trails the TCP by
                   ``fingertip_axial`` along the approach axis).
    """
    x, y, z = _as3(target_xyz)

    # Ring radii from the perceived cloud's top-face slab (was tsh-ring-geometry).
    raw = target_cloud.get("points") if isinstance(target_cloud, dict) else target_cloud
    pts = np.asarray(raw, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        raise RuntimeError("calculate_grasp_ring: cloud missing or degenerate (<4 points)")
    hole_radius, rim_radius = _ring_radii(pts, (x, y), z + float(target_half_z))
    if not _radii_ok(hole_radius, rim_radius):
        raise RuntimeError(
            f"calculate_grasp_ring: radii hole={hole_radius*1000:.1f}mm "
            f"rim={rim_radius*1000:.1f}mm outside sanity band — degenerate cloud, "
            "no tuned fallback")

    # Grasp geometry from the radii + gripper offsets.
    ring_dy = (hole_radius + rim_radius) / 2
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
    quat: Quaternion = {"w": float(rw), "x": float(rx), "y": float(ry), "z": float(rz)}

    # Tape centre relative to the TCP at the seat, in the TCP frame: the TCP
    # sits at centre + off (world), so tape-in-TCP = R^-1 · (-off).
    held_offset = R_grasp.inv().apply(-off).tolist()

    def _xyz(zval: float) -> Vec3:
        return {"x": float(gx), "y": float(gy), "z": float(zval)}

    return {
        "hover_xyz": _xyz(z + GRASP_PRE_DZ),
        "seat_xyz": _xyz(z + GRASP_HOOK_DZ),
        "lift_xyz": _xyz(z + GRASP_LIFT_CLEARANCE),
        "grasp_quat": quat,
        "held_offset": {"x": float(held_offset[0]), "y": float(held_offset[1]),
                        "z": float(held_offset[2])},
        "hole_radius": hole_radius,
        "rim_radius": rim_radius,
    }
