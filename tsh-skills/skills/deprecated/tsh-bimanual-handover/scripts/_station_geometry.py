"""Derive the bimanual handover meet point from the two arm bases.

The tuned ``MEET_XYZ`` is (to a few mm) a fixed function of the arm-base
geometry: the midpoint of the two bases in XY, offset forward (+X) and up (+Z) by
~half the base separation — the natural spot a bimanual pair meets. The Y
*midpoint* (no "own-side" bias) was verified reachable AND collision-free, so no
convention fudge is needed and the meet point is pure base geometry — it
self-configures if the arms are restationed.

``derive_meet_xyz`` returns the geometric nominal, optionally refined by a small
forward/up cuRobo reachability search (the same ``plan_to_pose`` probe the
exchange uses) so a candidate that one arm can't quite reach is nudged to one
both can. The forward/up fractions are an ergonomic presentation prior (~0.5 of
the base separation); the reachability probe, not the fraction, is what pins the
final pose.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

# Presentation-height priors as a fraction of the base separation. A bimanual
# handover sits roughly midway forward of and above the two bases; the exact
# value is refined by the reachability probe, so these only seed the search.
MEET_FORWARD_FRAC = 0.54  # +X in front of the base midplane
MEET_UP_FRAC = 0.45       # +Z above the base plane


def derive_meet_xyz(ctx: NodeContext, *, giver_arm: int = 0, receiver_arm: int = 1,
                    probes=(), forward_frac: float = MEET_FORWARD_FRAC,
                    up_frac: float = MEET_UP_FRAC) -> list:
    """World-frame handover meet point from the two arm bases.

    probes: optional iterable of ``callable(meet_xyz) -> bool`` reachability
            checks (e.g. giver-present + receiver-thread cuRobo probes). When
            given, a small forward/up grid around the geometric nominal is
            searched and the first pose satisfying ALL probes is returned; the
            nominal is returned as-is when no probes are supplied or none pass.
    """
    bg = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=giver_arm)["position"],
                    dtype=float)
    br = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=receiver_arm)["position"],
                    dtype=float)
    mid = (bg + br) / 2.0
    sep = float(np.linalg.norm(bg - br))
    nominal = np.array([mid[0] + forward_frac * sep, mid[1], mid[2] + up_frac * sep])

    probes = tuple(probes)
    if not probes:
        return [float(v) for v in nominal]

    # Refine: nudge forward/up (Y stays at the midpoint) to a both-arms-reachable
    # pose, seeded at the nominal and spiralling outward.
    offsets = [0.0, 0.03, -0.03, 0.06, -0.06]
    for f in offsets:
        for u in offsets:
            cand = nominal + np.array([f, 0.0, u])
            if all(p(cand) for p in probes):
                return [float(v) for v in cand]
    return [float(v) for v in nominal]


def _mat_to_wxyz(R: np.ndarray) -> tuple:
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def derive_presentation_quats(ctx: NodeContext, *, giver_arm: int = 0,
                              receiver_arm: int = 1, recv_splay_deg: float = 0.0):
    """Derive the giver/receiver presentation orientations from the arm bases.

    The presentation frame is pure station geometry: the gripper approach
    (tool-z) points FORWARD (horizontal, perpendicular to the base baseline,
    away from the bases), the finger-spread axis (tool-x) lies ALONG the
    baseline, and tool-y is world-up. This reproduces the tuned ``GIVER_QUAT``
    exactly. The receiver uses the same frame plus a small ``recv_splay_deg``
    rotation about the up axis (the tuned ``RECV_QUAT`` is a ~8.5deg splay) — a
    wrist-collision refinement that can be swept for reachability.

    Both quats are CANONICAL (as if the receiver sits on -Y); the exchange's
    ``_rim_grasp`` mirrors them across Y=0 for a reversed (+Y) receiver, so the
    splay is applied toward -Y here and the mirror handles the other side —
    matching how the tuned RECV_QUAT is defined. Returns
    ``(giver_quat_wxyz, recv_quat_wxyz)``.
    """
    bg = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=giver_arm)["position"],
                    dtype=float)
    br = np.asarray(ctx.tool("libero-yam.arm_base_pose", arm_id=receiver_arm)["position"],
                    dtype=float)
    # Baseline oriented so the giver is on +Y of it (canonical), independent of
    # which physical arm is giver, so the derived quats match the canonical
    # RECV_QUAT convention the mirror expects.
    baseline = np.array([0.0, abs(bg[1] - br[1]), 0.0])
    b_hat = baseline / np.linalg.norm(baseline)
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.cross(b_hat, up)
    fwd = fwd / np.linalg.norm(fwd)
    if fwd[0] < 0.0:  # forward points AWAY from the bases (into the workspace)
        fwd = -fwd
    # gripper frame columns: x = finger-spread (baseline), y = up, z = approach (fwd)
    R = np.column_stack([b_hat, np.cross(fwd, b_hat), fwd])
    giver_quat = _mat_to_wxyz(R)
    # Receiver: canonical splay toward -Y about the up axis (mirror handles +Y).
    Rr = R @ Rotation.from_euler("y", -np.radians(recv_splay_deg)).as_matrix()
    return giver_quat, _mat_to_wxyz(Rr)


def derive_station_geometry(ctx: NodeContext, *, giver_arm: int = 0,
                            receiver_arm: int = 1, giver_present_probe,
                            recv_thread_probe):
    """Derive the whole handover station geometry from the arm bases.

    Combines :func:`derive_presentation_quats` (giver/receiver orientation) and
    :func:`derive_meet_xyz` (meet position). The receiver uses the pure geometric
    frame (splay 0): once the meet point is DERIVED it is the symmetric arm
    midpoint, at which the frame is reachable with no wrist splay — the tuned
    RECV_QUAT's ~8.5deg splay was compensating for the off-centre tuned meet, and
    is unnecessary here. (``derive_presentation_quats`` keeps a ``recv_splay_deg``
    knob for a station that does need one.) The two probe callbacks supply the
    cuRobo reachability checks so this module needn't import the exchange's grasp
    helpers:

      ``giver_present_probe(meet, giver_quat) -> bool``
      ``recv_thread_probe(meet, recv_quat)  -> bool``

    Returns ``(meet_xyz, giver_quat, recv_quat)``.
    """
    giver_quat, recv_quat = derive_presentation_quats(
        ctx, giver_arm=giver_arm, receiver_arm=receiver_arm)
    meet = derive_meet_xyz(
        ctx, giver_arm=giver_arm, receiver_arm=receiver_arm,
        probes=(lambda m: giver_present_probe(m, giver_quat),
                lambda m: recv_thread_probe(m, recv_quat)))
    return meet, giver_quat, recv_quat
