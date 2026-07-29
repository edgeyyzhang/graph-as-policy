"""Decide the transport route: which arm picks, and whether a handover is needed.

A handover is not a fixed pipeline stage — it exists only because the place
destination is outside the picking arm's workspace. This node makes that
decision explicit and cheap: it probes reachability with the canonical curobo
bundle (``execute=False`` — plans only, never touches the sim) using the EXACT
grasp poses ``pickup`` will execute — supplied by one
``calculate-grasp-ring`` instance PER ARM (route consumes the poses as
data; it derives no grasp geometry itself) — and the place hover's yaw sweep
with the predicted held-tape offset, then routes:

  * ``direct``         — one arm can both grasp the tape AND reach the place
                          hover: pick_arm = place_arm, no exchange.
  * ``needs_handover``  — the tape is only graspable by one arm and the
                          destination only reachable by the other: pick,
                          exchange at the meet point, place.
  * (raise)            — no arm can grasp, or nothing can place: ``on_error``.

Because the probe replays the SAME poses the grasp skill produced, the route's
feasibility answer can never disagree with the grasp actually run.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import Quaternion, Vec3

from ._motion import plan_tool_move
from .constants import (
    DOWN_QUAT,
    PLACE_YAW_SWEEP_DEG,
    PLACE_Z_APPROACH,
    RECV_GRASP_DY_MARGIN,
    RECV_GRASP_DZ,
)


class Output(TypedDict):
    route: str        # "direct" | "needs_handover" — the conditional-edge field
    pick_arm: int     # arm that grasps the tape
    place_arm: int    # arm that places it (== pick_arm on the direct route)
    giver_arm: int    # aliases for the handover skill's input names
    receiver_arm: int
    # The CHOSEN arm's grasp legs, relayed so pickup consumes them directly
    # (no separate per-arm grasp instance for the pickup side).
    pick_hover_xyz: Vec3
    pick_seat_xyz: Vec3
    pick_lift_xyz: Vec3
    pick_grasp_quat: Quaternion


def _as3(v) -> list:
    return [float(v["x"]), float(v["y"]), float(v["z"])]


def _as_wxyz(q) -> list:
    return [float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])]


def _vec3(v: list) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(q: list) -> Quaternion:
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


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
        rim_radius: float,
        arm0_hover_xyz: list, arm0_seat_xyz: list, arm0_lift_xyz: list,
        arm0_grasp_quat: list, arm0_held_offset: list,
        arm1_hover_xyz: list, arm1_seat_xyz: list, arm1_lift_xyz: list,
        arm1_grasp_quat: list, arm1_held_offset: list) -> Output:
    """Probe grasp + place reachability per arm and pick the route.

    target_xyz:    tape grasp point (body-centroid height), from perception —
                   used only to prefer the picking arm nearer the tape.
    container_xyz: destination top-face centre, from perception.
    half_z:        perceived tape half-thickness (place rest height).
    rim_radius:    perceived rim radius (from calculate-grasp-ring) — the
                   predicted receiver thread offset for the place probe.
    arm{0,1}_hover_xyz / _seat_xyz / _lift_xyz / _grasp_quat / _held_offset: the
                   grasp legs + predicted held offset for each arm, each from its
                   OWN calculate-grasp-ring instance (arm_id = 0 and = 1). The
                   probe replays these EXACT poses (hover + seat), so it can't
                   disagree with the grasp pickup runs; the chosen arm's legs are
                   relayed to pickup as the pick_* outputs.
    """
    target = _as3(target_xyz)
    arms = (0, 1)
    legs = {
        0: (_as3(arm0_hover_xyz), _as3(arm0_seat_xyz), _as3(arm0_lift_xyz),
            _as_wxyz(arm0_grasp_quat), _as3(arm0_held_offset)),
        1: (_as3(arm1_hover_xyz), _as3(arm1_seat_xyz), _as3(arm1_lift_xyz),
            _as_wxyz(arm1_grasp_quat), _as3(arm1_held_offset)),
    }

    grasp_ok: dict[int, bool] = {}
    held: dict[int, list] = {}
    for a in arms:
        hov, seat, _lift, quat, hoff = legs[a]
        held[a] = hoff
        ok = plan_tool_move(ctx, a, hov, quat, execute=False) is not None
        if ok:
            ok = plan_tool_move(ctx, a, seat, quat, execute=False) is not None
        grasp_ok[a] = ok

    if not any(grasp_ok.values()):
        raise RuntimeError(
            f"route: no arm can reach the grasp at "
            f"{[round(v, 3) for v in target]} — kinematic dead zone")

    # Place probe per arm. The held offset differs by how the arm would come to
    # hold the tape: its own ring grasp (direct) or the receiver's rim thread
    # (after a handover). The yaw sweep pivots about the tape centre, so the
    # probe is dominated by destination reach, not offset direction.
    recv_offset = [0.0, rim_radius + RECV_GRASP_DY_MARGIN, -RECV_GRASP_DZ]
    place_ok: dict[int, bool] = {}
    for a in arms:
        off = held[a] if grasp_ok[a] else recv_offset
        place_ok[a] = _place_hover_reachable(ctx, a, container_xyz, half_z, off)

    # Prefer the grasp-capable arm nearer the tape (less reach = more margin).
    def _base_dist(a: int) -> float:
        b = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=a)["position"])[:2]
        return float(np.linalg.norm(np.asarray(target[:2]) - b))

    pickers = sorted((a for a in arms if grasp_ok[a]), key=_base_dist)

    def _pick_legs(a: int) -> dict:
        """The picking arm's grasp legs, relayed for pickup to execute."""
        hov, seat, lift, quat, _hoff = legs[a]
        return {"pick_hover_xyz": _vec3(hov), "pick_seat_xyz": _vec3(seat),
                "pick_lift_xyz": _vec3(lift), "pick_grasp_quat": _quat(quat)}

    for a in pickers:  # direct if any picker can also place
        if place_ok[a]:
            print(f"[route] direct: arm {a} grasps and places "
                  f"(grasp_ok={grasp_ok}, place_ok={place_ok})", flush=True)
            return {"route": "direct", "pick_arm": a, "place_arm": a,
                    "giver_arm": a, "receiver_arm": 1 - a, **_pick_legs(a)}
    for a in pickers:  # needs_handover if the other arm can place
        other = 1 - a
        if place_ok[other]:
            print(f"[route] needs_handover: arm {a} picks, arm {other} places "
                  f"(grasp_ok={grasp_ok}, place_ok={place_ok})", flush=True)
            return {"route": "needs_handover", "pick_arm": a, "place_arm": other,
                    "giver_arm": a, "receiver_arm": other, **_pick_legs(a)}

    raise RuntimeError(
        f"route: grasp reachable (arms {pickers}) but no arm can reach the "
        f"place hover at {[round(v, 3) for v in _as3(container_xyz)]} — "
        f"kinematic dead zone")
