"""Emit the handover station geometry — meet point + presentation orientations —
derived from the two arm-base poses.

Thin GaP node over :mod:`_station_geometry`: the meet point is the arm-base
midpoint offset forward (+X) and up (+Z) by ~half the base separation, and the
giver/receiver quaternions are the base-geometry presentation frame (approach ⊥
baseline, finger-spread along baseline, up = world-up; receiver at splay 0). This
reproduces the tuned MEET_XYZ / GIVER_QUAT / RECV_QUAT and self-configures for a
restationed pair. tsh-handover consumes ``meet_xyz``, ``giver_quat``, ``recv_quat``.

Pure geometry (arm bases only) — no reachability probe, since the geometric meet
was validated reachable across the grid. A reachability-refined variant lives in
``_station_geometry.derive_station_geometry`` for a station that needs it.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Vec3

from ._station_geometry import derive_meet_xyz, derive_presentation_quats


class Output(TypedDict):
    meet_xyz: Vec3        # world-frame handover point
    giver_quat: Quaternion  # giver present orientation, wxyz
    recv_quat: Quaternion   # receiver thread orientation, wxyz (canonical; exchange mirrors)


def _vec3(v) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(q) -> Quaternion:
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


def run(ctx: NodeContext, *, giver_arm: int = 0, receiver_arm: int = 1) -> Output:
    """Derive the meet point + presentation quats from the arm bases (no GT)."""
    giver_quat, recv_quat = derive_presentation_quats(
        ctx, giver_arm=giver_arm, receiver_arm=receiver_arm)
    meet = derive_meet_xyz(ctx, giver_arm=giver_arm, receiver_arm=receiver_arm)
    print(f"[station_geometry] meet={[round(v, 4) for v in meet]} "
          f"giver_quat={[round(v, 4) for v in giver_quat]}", flush=True)
    return {"meet_xyz": _vec3(meet),
            "giver_quat": _quat(giver_quat), "recv_quat": _quat(recv_quat)}
