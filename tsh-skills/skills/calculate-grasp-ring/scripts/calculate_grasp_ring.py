"""Ring radii + grasp poses for ONE arm — the shared grasp geometry.

Measures the ring's hole/rim radii from its perceived point cloud (top-face
slab, 2nd/98th radial percentile — was tsh-ring-geometry) and derives the
grasp from them: the TCP centres on the wall midpoint ``(hole_r + rim_r)/2``
so both fingers meet their wall with the same travel, with the fingertip
trailing the TCP along the approach axis (``fingertip_axial``). The grasp
revolves ``GRASP_RING_ANGLE_DEG`` clockwise about the vertical hole axis, plus
an extra 90° CW for a −Y-side arm so its fingers sit on the mirrored arc (side
read from the arm base, not tuned).

This node is the SINGLE producer of that geometry: ``bimanual-route-arms``
probes these exact poses (``execute=False``) to decide which arm picks, and
``pickup`` executes them — so the route's feasibility answer and the grasp
can never disagree. It only DECIDES the poses; it never plans or moves. It
raises when the radii fall outside the sanity band (degenerate cloud, no tuned
fallback).

Emits the three grasp legs as ready-to-plan world poses, the predicted
held-tape offset (tape centre in the TCP frame at the seat — what
``tape_in_giver`` will later measure), and the radii themselves (relayed to
bimanual-route-arms's receiver probe and bimanual-handover's thread offset).
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
    # Both arms in one pass: bimanual-route-arms needs BOTH arms' legs to
    # decide which arm can grasp, so emitting them from a single node keeps the
    # names plain (W8-checkable) instead of a per-instance <arm0|arm1>_ prefix.
    arm0_hover_xyz: Vec3      # approach hover over the wall midpoint
    arm0_seat_xyz: Vec3       # descended seat (near finger in the hole)
    arm0_lift_xyz: Vec3       # post-grasp lift, relative to grasp height
    arm0_grasp_quat: Quaternion  # shared TCP orientation for that arm's legs
    arm0_held_offset: Vec3    # predicted tape centre in the TCP frame at the seat
    arm1_hover_xyz: Vec3
    arm1_seat_xyz: Vec3
    arm1_lift_xyz: Vec3
    arm1_grasp_quat: Quaternion
    arm1_held_offset: Vec3
    hole_radius: float    # inner hole radius (m) — the ring, not the arm
    rim_radius: float     # outer rim radius (m) — the ring, not the arm


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


def _arm_legs(ctx: NodeContext, arm_id: int, x: float, y: float, z: float,
              hole_radius: float, rim_radius: float, fingertip_axial: float) -> dict:
    """The three grasp legs + predicted held offset for ONE arm.

    The grasp revolves ``GRASP_RING_ANGLE_DEG`` clockwise about the vertical hole
    axis, plus an extra 90° CW when the arm's base sits on −Y so its fingers land
    on the mirrored arc. That side flip is read from the arm base, so the two
    arms genuinely differ — same ring, mirrored approach.
    """
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

    # Tape centre relative to the TCP at the seat, in the TCP frame: the TCP
    # sits at centre + off (world), so tape-in-TCP = R^-1 · (-off).
    held_offset = R_grasp.inv().apply(-off).tolist()

    def _xyz(zval: float) -> Vec3:
        return {"x": float(gx), "y": float(gy), "z": float(zval)}

    return {
        "hover_xyz": _xyz(z + GRASP_PRE_DZ),
        "seat_xyz": _xyz(z + GRASP_HOOK_DZ),
        "lift_xyz": _xyz(z + GRASP_LIFT_CLEARANCE),
        "grasp_quat": {"w": float(rw), "x": float(rx), "y": float(ry), "z": float(rz)},
        "held_offset": {"x": float(held_offset[0]), "y": float(held_offset[1]),
                        "z": float(held_offset[2])},
    }


def run(ctx: NodeContext, *, target_xyz: list, target_cloud, target_half_z: float,
        fingertip_axial: float, finger_half_gap: float) -> Output:
    """Ring radii + grasp poses for BOTH arms at the perceived tape centre.

    Emits arm0_* and arm1_* legs from one node because bimanual-route-arms
    consumes both to decide which arm grasps — there is no point at which only
    one arm's geometry is wanted, and one node keeps the output names plain.

    target_xyz:    world grasp point (body-centroid height), from perception.
    target_cloud:  world-frame ring point cloud, from perception.
    target_half_z: perceived half-thickness — locates the top face
                   (``top_z = center_z + half_z``) for the radii slab cut.
    fingertip_axial / finger_half_gap: FK-derived gripper offsets from this
                   subgraph's own derive_gripper_geometry node (the fingertip trails the TCP by
                   ``fingertip_axial`` along the approach axis). finger_half_gap
                   is not consumed by the current wall-midpoint geometry.
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

    # Grasp geometry from the radii + gripper offsets, once per arm. The radii
    # describe the ring and are shared; only the approach mirrors by arm side.
    out: dict = {"hole_radius": hole_radius, "rim_radius": rim_radius}
    for arm_id in (0, 1):
        for leg, value in _arm_legs(ctx, arm_id, x, y, z,
                                    hole_radius, rim_radius, fingertip_axial).items():
            out[f"arm{arm_id}_{leg}"] = value
    return out
