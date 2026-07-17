"""Shared RGB-D object-localization core for the perceive_* handover steps.

perceive_tape and perceive_duct run the SAME pipeline — DINO detect -> SAM box
segment -> depth back-projection -> robust top face — and differ only in what they
derive from the result (tape: grasp point at centroid height + the cloud; duct:
the top-face centre). This is that shared core, split out so the two can't drift
apart. No ground-truth pose, no object dimensions, and no scene-specific
constants are assumed.
"""

from __future__ import annotations

import os

import numpy as np

from gap import NodeContext
from gap_core.types import CameraFrame, pose_to_matrix

from .constants import (
    PERCEIVE_BOTTOM_PCTILE,
    PERCEIVE_FLOOR_PCTILE,
    PERCEIVE_SPRAY_ABOVE_MEDIAN,
    PERCEIVE_TOP_PCTILE,
    PERCEIVE_TOP_SLAB,
    RING_HOLE_PCTILE,
    RING_RIM_PCTILE,
)


def perceive_top_face(ctx: NodeContext, cameras: list[CameraFrame], key: str = "",
                      top_slab: float = PERCEIVE_TOP_SLAB, *,
                      query: str | None = None,
                      raise_if_missing: bool = True):
    """Localize an object from RGB-D. Returns ``(x, y, top_z, pts)``.

    The DINO query is either passed directly (``query="yellow tape"`` — the
    generic ``perceive_object`` path, no body key needed) or derived from a
    body key (e.g. ``yellow_tape_1`` -> ``"yellow tape"``, ``duct_tape_1`` ->
    ``"duct tape"`` — the legacy ``perceive_tape``/``perceive_duct`` path) ->
    SAM box segmentation -> depth back-projection to a world cloud (the
    geometry.mask_to_world_points math, inlined so we don't depend on that
    out-of-process bundle).

    ``raise_if_missing=False`` returns ``None`` instead of raising when DINO
    finds no match — the clean-all-items loop exit (perception_any pattern).

      * x, y  — top-face centre: the mean of the top ``top_slab`` slab. The top
                face of these round objects is symmetric, so it has far less
                single-view self-occlusion bias than the whole-cloud centroid.
      * top_z — robust top face: the 98th percentile of the cloud z. A few mask
                pixels back-project onto a nearby arm/gripper or a grazing edge
                (a <2% but extreme minority), so ``max`` would latch the top ~10-20
                cm high; the 98th percentile ignores them with no hand-tuned band.
      * pts   — the full world-frame cloud (N,3), table/floor removed.

    Uses the ``agentview`` external camera; angle is irrelevant to RGB-D
    back-projection. ``cam["pose"]`` is the OpenCV camera-to-world the connector
    emits; depth is metric metres.
    """
    if query is None:
        query = key.rsplit("_", 1)[0].replace("_", " ")
        # The grey/white duct scores low for a bare "duct tape" query, and the
        # bright yellow tape ring outscores it — so DINO's top box lands on the
        # WRONG object and place is sent to the giver's side of the table
        # (unreachable). Anchor the duct query on its colour to lift the real
        # duct above the score noise. (Env-overridable for tuning; default is
        # the validated query.)
        if "duct" in key.lower():
            query = os.environ.get("GAP_DUCT_QUERY", "gray tape")
    # Exterior RGB-D view for perception, selected in-script from the shared
    # observation (the wrist cameras are not wired as a fallback here), so the
    # graph never has to name a camera.
    cam = next((c for c in cameras if c["name"] == "agentview"), None)
    if cam is None:
        raise RuntimeError(
            f"perceive: no 'agentview' camera in {[c.get('name') for c in cameras]}")
    # Query DINO in the RGB frame to get a 2D bounding box around the object.
    dets = ctx.tool("grounding-dino.detect", image=cam["rgb"], query=query)["detections"]
    if not dets:
        if not raise_if_missing:
            return None
        raise RuntimeError(f"perceive: DINO found no '{query}' in agentview")
    box = max(dets, key=lambda d: d["score"])["box"]
    seg = ctx.tool("sam3.segment_box", image=cam["rgb"], box=box)
    # Highest-confidence mask isolates the pixels belonging to the object's face.
    mask = np.asarray(seg["masks"][0])

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
    z_floor = np.percentile(pts[:, 2], PERCEIVE_FLOOR_PCTILE)
    pts = pts[pts[:, 2] > z_floor]
    if len(pts) == 0:
        if not raise_if_missing:
            return None
        raise RuntimeError(f"perceive: empty cloud for '{query}'")
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

    ``top_z`` is the robust top face from :func:`perceive_top_face`; the bottom is
    the ``PERCEIVE_BOTTOM_PCTILE`` of the (already floor-trimmed) cloud — for an
    object resting on the table that is its base. Robust to the top-face outliers
    the percentiles already reject. Returns metres (``>= 0``).
    """
    bottom_z = float(np.percentile(pts[:, 2], PERCEIVE_BOTTOM_PCTILE))
    return max((float(top_z) - bottom_z) / 2.0, 0.0)


def estimate_ring_radii(pts, center_xy, top_z=None, top_slab=PERCEIVE_TOP_SLAB):
    """Estimate hole and rim radii from a ring-shaped point cloud.

    Returns ``(hole_radius, rim_radius)`` in metres. Uses the 2nd and 98th
    percentiles of the radial distance distribution for robustness against
    depth noise (points leaking into the hole) and edge outliers.

    When ``top_z`` is given, the radii are measured on the TOP-FACE SLAB only
    (``z > top_z - top_slab``). This is essential from an angled view: pixels
    inside the hole back-project through it onto the table (a low-z blob near the
    centre), which collapses the full-cloud hole radius to a few mm and trips the
    sanity band into the tuned fallback. The flat top annulus has no such points,
    so both the hole and rim edges read true.
    """
    if top_z is not None:
        pts = pts[pts[:, 2] > top_z - top_slab]
    dists = np.linalg.norm(pts[:, :2] - np.asarray(center_xy), axis=1)
    hole_r = float(np.percentile(dists, RING_HOLE_PCTILE))
    rim_r = float(np.percentile(dists, RING_RIM_PCTILE))
    return hole_r, rim_r
