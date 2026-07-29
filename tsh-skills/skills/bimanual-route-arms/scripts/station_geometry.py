"""Emit the handover station geometry — meet point + presentation orientations —
derived from the two arm-base poses.

Thin GaP node over :mod:`_station_geometry`: the meet point is the arm-base
midpoint offset forward (+X) and up (+Z) by ~half the base separation, and the
giver/receiver quaternions are the base-geometry presentation frame (approach ⊥
baseline, finger-spread along baseline, up = world-up; receiver at splay 0). This
reproduces the tuned MEET_XYZ / GIVER_QUAT / RECV_QUAT and self-configures for a
restationed pair. bimanual-handover consumes ``meet_xyz``, ``giver_quat``, ``recv_quat``.

Pure geometry (arm bases only) — no reachability probe, since the geometric meet
was validated reachable across the grid. A reachability-refined variant lives in
``_station_geometry.derive_station_geometry`` for a station that needs it.

This node runs UNCONDITIONALLY, on both routes, and relays ``route`` so the
subgraph branches on that field here instead. Gating it behind the
needs_handover exit would save ~0ms (it is arithmetic on two arm-base poses)
while making meet_xyz/giver_quat/recv_quat/hold_target_* bound on only one
path — a mismatch W8 cannot see, because the names have declared producers and
whether they are BOUND is a runtime property. Downstream then dies with
"input 'hold_target_xyz' has no upstream producer" after a clean validation.
Running it always is what makes those outputs safe to consume anywhere.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Vec3

from ._station_geometry import derive_meet_xyz, derive_presentation_quats


class Output(TypedDict):
    route: str            # echo of the routing decision — this node runs on EVERY
                          # path, so the subgraph branches on this field HERE
                          # rather than gating this node behind an exit. Keeping
                          # it unconditional is what makes the outputs below
                          # bound on every path (see module docstring).
    meet_xyz: Vec3        # world-frame handover point
    giver_quat: Quaternion  # giver present orientation, wxyz
    recv_quat: Quaternion   # receiver thread orientation, wxyz (canonical; exchange mirrors)
    hold_target_xyz: Vec3   # alias of meet_xyz — the canonical "where the held
                            # object's centre goes" name transport-with-held-object requires,
                            # so the PRESENT leg auto-wires by exact name like the
                            # place legs do (calculate-place-pose emits the same pair).
                            # Deliberately NOT target_xyz, which would collide with
                            # perception's own target_xyz and silently carry the
                            # tape back to its pickup spot instead of the meet point.
    hold_target_quat: Quaternion  # alias of giver_quat — same contract


def _vec3(v) -> Vec3:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _quat(q) -> Quaternion:
    return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}


def run(ctx: NodeContext, *, route: str, giver_arm: int = 0,
        receiver_arm: int = 1) -> Output:
    """Derive the meet point + presentation quats from the arm bases (no GT).

    route: the routing decision from ``route.py``, echoed straight back out —
           this node sits on every path, so the subgraph's branch reads it from
           here rather than gating this node behind an exit.
    """
    giver_quat, recv_quat = derive_presentation_quats(
        ctx, giver_arm=giver_arm, receiver_arm=receiver_arm)
    meet = derive_meet_xyz(ctx, giver_arm=giver_arm, receiver_arm=receiver_arm)
    print(f"[station_geometry] meet={[round(v, 4) for v in meet]} "
          f"giver_quat={[round(v, 4) for v in giver_quat]}", flush=True)
    return {"route": route,
            "meet_xyz": _vec3(meet),
            "giver_quat": _quat(giver_quat), "recv_quat": _quat(recv_quat),
            "hold_target_xyz": _vec3(meet), "hold_target_quat": _quat(giver_quat)}
