"""Perceive the yellow tape's grasp point + point cloud from RGB-D (pickup step 1).

A standalone perception step: DINO detect -> SAM box segment -> depth
back-projection on the agentview, returning the tape's grasp point ``(x, y, z)``
and the full world-frame point cloud. No ground-truth pose and no tape
dimensions are assumed (same pipeline as ``perceive_duct``).

The emitted ``tape_cloud`` is the tape geometry the handover needs (the giver's
tape-as-EE collision body); ``tape_xyz`` is the grasp point the ``pickup``
step drives to. Splitting perception from motion lets the handover consume the
cloud without re-running (or knowing about) the grasp.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap_core.types import PointCloud

from ._perceive import estimate_half_thickness, estimate_ring_radii, perceive_top_face
from .constants import (
    PERCEIVE_TOP_SLAB,
    RING_HOLE_MAX,
    RING_HOLE_MIN,
    RING_RIM_MARGIN,
    TAPE_HALF_MAX,
    TAPE_HALF_MIN,
)


class Output(TypedDict):
    tape_xyz: list  # [x, y, z] world-frame grasp point (body-centroid height)
    tape_half_z: float  # perceived tape half-thickness (m); place rests the tape flush
    tape_cloud: PointCloud  # world frame
    hole_radius: float | None  # inner hole radius (m), from point cloud
    rim_radius: float | None   # outer rim radius (m), from point cloud


def perceive_tape_grasp_xyz(ctx: NodeContext, cameras: list, object_key: str,
                            top_slab: float = PERCEIVE_TOP_SLAB):
    """Perceive the tape's grasp point from RGB-D instead of ground truth.

    Thin wrapper over the shared ``perceive_top_face`` core. Returns
    ``(x, y, z, half_z, pts)``: x,y is the top-face-annulus centre; ``half_z`` is
    the tape half-thickness DERIVED from the cloud (``(top − bottom)/2``); a
    degenerate cloud (half_z outside the sanity band) RAISES — no tuned fallback.
    z is the top face minus ``half_z`` — the body-centroid height the grasp
    offsets are calibrated against.
    ``pts`` is the full world-frame cloud (the tape collision geometry the handover
    consumes).
    """
    x, y, top_z, pts = perceive_top_face(ctx, cameras, object_key, top_slab=top_slab)
    half_z = estimate_half_thickness(pts, top_z)
    if not TAPE_HALF_MIN < half_z < TAPE_HALF_MAX:  # degenerate cloud → fail loud
        raise RuntimeError(
            f"perceive_tape: derived tape half-thickness {half_z*1000:.1f}mm "
            f"outside sanity band [{TAPE_HALF_MIN*1000:.0f},{TAPE_HALF_MAX*1000:.0f}]mm "
            "— degenerate cloud, no tuned fallback")
    return float(x), float(y), float(top_z - half_z), float(half_z), pts


def run(ctx: NodeContext, *, object_key: str, cameras: list) -> Output:
    """Localize the tape's grasp point + cloud, emit ``tape_xyz`` and ``tape_cloud``.

    object_key: the tape's MJCF body key (e.g. ``"yellow_tape_1"``); the DINO
                query is derived from it (``"yellow tape"``).
    cameras:    the shared observation's camera list. REQUIRED — must come from
                an explicit ``observe`` (``robot.get_observation``) node inside
                THIS subgraph (``Ref("observe.cameras")``), authored by the
                subgraph_agent. Not a cross-subgraph input — see SKILL.md.
    """
    x, y, z, half_z, pts = perceive_tape_grasp_xyz(ctx, cameras, object_key)
    hole_radius = rim_radius = None
    hr, rr = estimate_ring_radii(pts, (x, y), top_z=z + half_z)
    if RING_HOLE_MIN < hr < RING_HOLE_MAX and rr > hr + RING_RIM_MARGIN:  # sanity band: reject noisy clouds
        hole_radius, rim_radius = hr, rr
    print(f"[perceive_tape] {object_key}: grasp=({x:.4f},{y:.4f},{z:.4f}) "
          f"half_z={half_z*1000:.1f}mm hole_r={hr*1000:.1f}mm rim_r={rr*1000:.1f}mm "
          f"({'ok' if hole_radius else 'out of range — downstream skills will raise'})",
          flush=True)
    return {"tape_xyz": [x, y, z], "tape_half_z": half_z,
            "tape_cloud": {"points": pts.astype(np.float32)},
            "hole_radius": hole_radius, "rim_radius": rim_radius}

