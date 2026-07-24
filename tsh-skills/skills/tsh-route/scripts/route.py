"""Decide the transport route: which arm picks, and whether a handover is needed.

A handover is not a fixed pipeline stage — it exists only because the place
destination is outside the picking arm's workspace. This node makes that
decision explicit and cheap: it probes reachability with the canonical curobo
bundle (``execute=False`` — plans only, never touches the sim) using the
EXACT grasp poses the pickup will execute (shared ``_ring.ring_grasp_poses``)
and the place hover's yaw sweep with the PREDICTED held-tape offset, then
routes:

  * ``direct``   — one arm can both grasp the tape AND reach the place hover:
                   pick_arm = place_arm, no exchange.
  * ``handover`` — the tape is only graspable by one arm and the destination
                   only reachable by the other: pick, exchange at the meet
                   point, place.
  * (raise)      — no arm can grasp, or nothing can place: ``on_error``.

This is the "reachability probe first" practice promoted into the graph —
kinematic dead zones are found in milliseconds of planning instead of
minutes of sim, and the agent gets a real branch to reason about.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from ._motion import plan_tool_move
from ._ring import ring_grasp_poses
from .constants import (
    DOWN_QUAT,
    PLACE_YAW_SWEEP_DEG,
    PLACE_Z_APPROACH,
    RECV_GRASP_DY_MARGIN,
    RECV_GRASP_DZ,
)


class Output(TypedDict):
    route: str        # "direct" | "handover" — the conditional-edge field
    pick_arm: int     # arm that grasps the tape
    place_arm: int    # arm that places it (== pick_arm on the direct route)
    giver_arm: int    # aliases for the handover skill's input names
    receiver_arm: int


def _as3(v):
    if isinstance(v, dict):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return [float(x) for x in v]


def _place_hover_reachable(ctx: NodeContext, arm_id: int, container_xyz, half_z,
                           offset_local) -> bool:
    """Probe the place hover (the most reach-constrained place waypoint) with
    the same yaw sweep place.py runs: pivot the TCP about the tape centre
    until any yaw plans. Mirrors place.py's sweep — probe only."""
    offset_local = np.asarray(offset_local, dtype=float)
    dest = np.asarray(_as3(container_xyz), dtype=float)
    centre = dest + np.array([0.0, 0.0, float(half_z)])
    hover = centre + np.array([0.0, 0.0, PLACE_Z_APPROACH])
    base_xy = np.asarray(
        ctx.tool("libero-yam.arm_base_pose", arm_id=arm_id)["position"])[:2]
    qw0, qx0, qy0, qz0 = DOWN_QUAT
    R_down = Rotation.from_quat([qx0, qy0, qz0, qw0])
    h = R_down.apply(offset_local)[:2]
    from_base = centre[:2] - base_xy
    yaw0 = np.arctan2(from_base[1], from_base[0]) - np.arctan2(h[1], h[0])
    for dyaw in PLACE_YAW_SWEEP_DEG:
        R_place = Rotation.from_rotvec([0.0, 0.0, yaw0 + np.radians(dyaw)]) * R_down
        qx, qy, qz, qw = R_place.as_quat()
        if plan_tool_move(ctx, arm_id, list(hover), (qw, qx, qy, qz),
                          tool_offset=list(offset_local), execute=False) is not None:
            return True
    return False


def run(ctx: NodeContext, *, target_xyz: list, container_xyz: list, half_z: float,
        hole_radius: float, rim_radius: float,
        fingertip_axial: float, finger_half_gap: float) -> Output:
    """Probe grasp + place reachability per arm and pick the route.

    target_xyz: tape grasp point (body-centroid height), from perception.
    container_xyz:   destination top-face centre, from perception.
    half_z:     perceived tape half-thickness (place rest height).
    hole_radius / rim_radius: perceived ring geometry (tsh-ring-geometry).
    fingertip_axial / finger_half_gap: FK-derived gripper offsets
        (tsh-gripper-geometry) — the probe builds the pickup's exact poses.
    """
    target = _as3(target_xyz)
    arms = (0, 1)

    grasp_ok: dict[int, bool] = {}
    grasp_geom: dict[int, dict] = {}
    for a in arms:
        g = ring_grasp_poses(ctx, a, target,
                             hole_radius=hole_radius, rim_radius=rim_radius,
                             fingertip_axial=fingertip_axial,
                             finger_half_gap=finger_half_gap)
        grasp_geom[a] = g
        gx, gy = g["grasp_xy"]
        ok = plan_tool_move(ctx, a, [gx, gy, g["hover_z"]], g["grasp_quat"],
                            execute=False) is not None
        if ok:
            ok = plan_tool_move(ctx, a, [gx, gy, g["seat_z"]], g["grasp_quat"],
                                execute=False) is not None
        grasp_ok[a] = ok

    if not any(grasp_ok.values()):
        raise RuntimeError(
            f"route: no arm can reach the grasp at "
            f"{[round(v, 3) for v in target]} — kinematic dead zone")

    # Place probe per arm. The held offset differs by how the arm would come
    # to hold the tape: its own ring grasp (direct) or the receiver's rim
    # thread (after a handover). The yaw sweep pivots about the tape centre,
    # so the probe is dominated by destination reach, not offset direction.
    recv_offset = [0.0, rim_radius + RECV_GRASP_DY_MARGIN, -RECV_GRASP_DZ]
    place_ok: dict[int, bool] = {}
    for a in arms:
        off = grasp_geom[a]["held_offset"] if grasp_ok[a] else recv_offset
        place_ok[a] = _place_hover_reachable(ctx, a, container_xyz, half_z, off)

    # Prefer the grasp-capable arm nearer the tape (less reach = more margin).
    def _base_dist(a: int) -> float:
        b = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=a)["position"])[:2]
        return float(np.linalg.norm(np.asarray(target[:2]) - b))

    pickers = sorted((a for a in arms if grasp_ok[a]), key=_base_dist)

    for a in pickers:  # direct if any picker can also place
        if place_ok[a]:
            print(f"[route] direct: arm {a} grasps and places "
                  f"(grasp_ok={grasp_ok}, place_ok={place_ok})", flush=True)
            return {"route": "direct", "pick_arm": a, "place_arm": a,
                    "giver_arm": a, "receiver_arm": 1 - a}
    for a in pickers:  # handover if the other arm can place
        other = 1 - a
        if place_ok[other]:
            print(f"[route] handover: arm {a} picks, arm {other} places "
                  f"(grasp_ok={grasp_ok}, place_ok={place_ok})", flush=True)
            return {"route": "handover", "pick_arm": a, "place_arm": other,
                    "giver_arm": a, "receiver_arm": other}

    raise RuntimeError(
        f"route: grasp reachable (arms {pickers}) but no arm can reach the "
        f"place hover at {[round(v, 3) for v in _as3(container_xyz)]} — "
        f"kinematic dead zone")
