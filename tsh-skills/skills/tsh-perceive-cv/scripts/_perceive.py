"""Shared robust top-face estimator for the CV perceive path.

``_perceive_cv.py`` segments by colour+height and hands its pixel mask to
:func:`top_face_from_mask` here — the SAME back-projection + robust top-face
math the DINO+SAM variant (``tsh-perceive-sam``) uses, so the two front-ends
can't drift apart on the geometry they emit. No ground-truth pose, no object
dimensions, and no scene-specific constants are assumed.
"""

from __future__ import annotations

import numpy as np

from gap_core.types import CameraFrame, pose_to_matrix

from .constants import (
    PERCEIVE_BOTTOM_PCTILE,
    PERCEIVE_FLOOR_PCTILE,
    PERCEIVE_SPRAY_ABOVE_MEDIAN,
    PERCEIVE_TOP_PCTILE,
    PERCEIVE_TOP_SLAB,
)


def top_face_from_mask(cam: CameraFrame, mask, top_slab: float = PERCEIVE_TOP_SLAB, *,
                       label: str = "", raise_if_missing: bool = True,
                       floor_trim: bool = True):
    """Back-project a pixel ``mask`` and finish on the robust top face.

    Returns ``(x, y, top_z, pts)`` (or ``None`` with ``raise_if_missing=False``).

      * x, y  — top-face centre: the mean of the top ``top_slab`` slab. The top
                face of these round objects is symmetric, so it has far less
                single-view self-occlusion bias than the whole-cloud centroid.
      * top_z — robust top face: the 98th percentile of the cloud z. A few mask
                pixels back-project onto a nearby arm/gripper or a grazing edge
                (a <2% but extreme minority), so ``max`` would latch the top ~10-20
                cm high; the 98th percentile ignores them with no hand-tuned band.
      * pts   — the full world-frame cloud (N,3), table/floor removed.

    ``floor_trim`` drops the bottom ``PERCEIVE_FLOOR_PCTILE`` of the cloud —
    needed for masks that leak table pixels, but a mask that is already
    height-filtered (the CV segmenter) passes ``False``: the trim would chop
    real low side points and bias ``half_z`` short.
    """
    # Use the camera info to back-project those pixels into the world frame.
    c2w = pose_to_matrix(cam["pose"])
    K = np.asarray(cam["intrinsics"], dtype=float)
    depth = np.asarray(cam["depth"], dtype=float)
    # Pinhole model: x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    # Converts 2D pixel coordinates (xs, ys) combined with depth readings zc into real-world 3D coordinates.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.where(mask > 0)
    zc = depth[ys, xs]
    ok = zc > 0
    xs, ys, zc = xs[ok], ys[ok], zc[ok]
    cam_pts = np.stack([(xs - cx) * zc / fx, (ys - cy) * zc / fy, zc,
                        np.ones_like(zc)], axis=1)
    pts = (c2w @ cam_pts.T).T[:, :3]
    ## Noisy data filtering, floor removal + outlier rejection
    if floor_trim:
        z_floor = np.percentile(pts[:, 2], PERCEIVE_FLOOR_PCTILE)
        pts = pts[pts[:, 2] > z_floor]
    if len(pts) == 0:
        if not raise_if_missing:
            return None
        raise RuntimeError(f"perceive: empty cloud for '{label}'")
    # Reject the high-z depth-discontinuity spray. At grazing view angles a
    # minority of mask-edge / gripper-adjacent pixels back-project ~10 cm ABOVE
    # the object; the 98th-percentile top face then latches onto them and the
    # top-slab centroid is dragged off (a ~15 cm miss). Points more than
    # PERCEIVE_SPRAY_ABOVE_MEDIAN above the median cloud height cannot be the flat
    # top of a table-resting object, so drop them. A clean, head-on cloud has no
    # such spray — nothing is removed and the estimate is bit-for-bit unchanged.
    z_med = float(np.median(pts[:, 2]))
    pts = pts[pts[:, 2] < z_med + PERCEIVE_SPRAY_ABOVE_MEDIAN]
    ### top center calculation
    top_z = float(np.percentile(pts[:, 2], PERCEIVE_TOP_PCTILE))
    top = pts[pts[:, 2] > top_z - top_slab]
    x, y = top[:, :2].mean(0)
    return float(x), float(y), top_z, pts


def estimate_half_thickness(pts, top_z):
    """Half-height of a flat object from its cloud: ``(top_z − robust bottom) / 2``.

    ``top_z`` is the robust top face from :func:`top_face_from_mask`; the bottom
    is the ``PERCEIVE_BOTTOM_PCTILE`` of the (already floor-trimmed) cloud — for
    an object resting on the table that is its base. Robust to the top-face
    outliers the percentiles already reject. Returns metres (``>= 0``).
    """
    bottom_z = float(np.percentile(pts[:, 2], PERCEIVE_BOTTOM_PCTILE))
    return max((float(top_z) - bottom_z) / 2.0, 0.0)
