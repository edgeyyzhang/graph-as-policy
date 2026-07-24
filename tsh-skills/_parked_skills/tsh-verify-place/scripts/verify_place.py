"""Verify the placed object actually rests on the destination — by looking.

Re-perceives the object AFTER the release (same classical colour+height CV
core as the perception subgraphs; the gripper has retracted, so the view is
clean again) and checks, with no ground truth, that:

  * the object's XY centre is within ``xy_tol`` of the destination centre,
  * its top face is within ``z_tol`` of the expected rest height
    (destination top + 2 x the object's half thickness).

The verdict is a ROUTED field: ``placed`` finishes the item, ``retry`` loops
the graph back (re-perceive the object wherever it landed, re-route,
re-pick — the full pipeline is the recovery), ``give_up`` routes to abort.
Bounded by ``max_attempts`` (per-process counter, one run = one episode).
"""

from __future__ import annotations

import itertools
from typing import TypedDict

import numpy as np

from gap import NodeContext

from ._perceive import estimate_half_thickness
from ._perceive_cv import perceive_top_face_cv

_ATTEMPTS = itertools.count(1)


class Output(TypedDict):
    verdict: str        # "placed" | "retry" | "give_up" — the routing field
    placed: bool
    xy_error_m: float   # object centre to destination centre, XY
    z_error_m: float    # object top face to expected rest height


def run(ctx: NodeContext, *, object_query: str, dest_xyz: list, half_z: float,
        cameras: list, xy_tol: float = 0.06, z_tol: float = 0.04,
        max_attempts: int = 2) -> Output:
    """Re-perceive ``object_query`` and check it rests on the destination.

    object_query: literal colour-anchored noun phrase for the placed object.
    dest_xyz:     destination top-face centre (the perception the place used).
    half_z:       the object's perceived half thickness (rest-height model).
    cameras:      fresh observation cameras — MUST come from an ``observe``
                  node inside THIS subgraph, captured after the retract.
    """
    if isinstance(dest_xyz, dict):
        dest_xyz = [dest_xyz["x"], dest_xyz["y"], dest_xyz["z"]]
    dest = np.asarray([float(v) for v in dest_xyz])

    got = perceive_top_face_cv(ctx, cameras, object_query,
                               raise_if_missing=False)
    if got is None:
        xy_err = z_err = float("inf")
        placed = False
    else:
        x, y, top_z, pts = got
        obj_half = estimate_half_thickness(pts, top_z)
        expected_top = dest[2] + 2.0 * float(half_z if half_z else obj_half)
        xy_err = float(np.linalg.norm(np.array([x, y]) - dest[:2]))
        z_err = float(abs(top_z - expected_top))
        placed = xy_err < float(xy_tol) and z_err < float(z_tol)

    attempts = next(_ATTEMPTS)
    if placed:
        verdict = "placed"
    elif attempts < int(max_attempts) + 1:
        verdict = "retry"
    else:
        verdict = "give_up"
    print(f"[verify_place] '{object_query}' xy_err={xy_err*1000:.0f}mm "
          f"z_err={z_err*1000:.0f}mm attempt={attempts} -> {verdict}", flush=True)
    return {"verdict": verdict, "placed": placed,
            "xy_error_m": xy_err, "z_error_m": z_err}
