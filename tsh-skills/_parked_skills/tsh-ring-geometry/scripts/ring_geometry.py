"""Derive the tape ring's hole/rim radii from its perceived point cloud.

A pure geometry post-processor on the generic ``perceive_object`` output —
no cameras, no tools, no scene constants. Splitting this out of perception
keeps the perception skill object-agnostic (it works unchanged for the duct,
a colour-sorted stack, or any tabletop object); only ring-grasp consumers
(tsh-pickup, tsh-route) need radii, and only for ring-shaped targets.

Raises when the radii fall outside the sanity band (degenerate cloud) — no
tuned fallback; the failure routes to ``on_error`` where the graph can
re-perceive or abort.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext

from ._ring import radii_ok, ring_radii


class Output(TypedDict):
    hole_radius: float  # inner hole radius (m)
    rim_radius: float   # outer rim radius (m)


def run(ctx: NodeContext, *, cloud, center_xyz: list, half_z: float) -> Output:
    """Measure the ring radii on the cloud's top-face slab.

    cloud:      world-frame point cloud from ``perceive_object``.
    center_xyz: the object's centre at body-centroid height (``center_xyz``
                output of ``perceive_object``); the radial distribution is
                measured about its XY.
    half_z:     perceived half-thickness — locates the top face
                (``top_z = center_z + half_z``) for the slab cut.
    """
    raw = cloud.get("points") if isinstance(cloud, dict) else cloud
    pts = np.asarray(raw, dtype=float)
    if pts.ndim != 2 or len(pts) < 4:
        raise RuntimeError("ring_geometry: cloud missing or degenerate (<4 points)")
    if isinstance(center_xyz, dict):
        center_xyz = [center_xyz["x"], center_xyz["y"], center_xyz["z"]]
    x, y, z = (float(v) for v in center_xyz)
    hole_r, rim_r = ring_radii(pts, (x, y), top_z=z + float(half_z))
    if not radii_ok(hole_r, rim_r):
        raise RuntimeError(
            f"ring_geometry: radii hole={hole_r*1000:.1f}mm rim={rim_r*1000:.1f}mm "
            "outside sanity band — degenerate cloud, no tuned fallback")
    print(f"[ring_geometry] hole_r={hole_r*1000:.1f}mm rim_r={rim_r*1000:.1f}mm",
          flush=True)
    return {"hole_radius": hole_r, "rim_radius": rim_r}
