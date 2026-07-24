"""Perceive the duct-tape (place target) top-face centre from RGB-D.

A small, standalone perception step: DINO detect -> SAM box segment -> depth
back-projection on the agentview, returning the duct's TOP-face centre
``(x, y, top_z)`` in world frame. No ground-truth pose and no duct dimensions
are assumed (same pipeline as ``perceive_tape``).

Run this UP FRONT, before the grasp, while the duct view is clean. At place time
the held tape + gripper hover directly over the duct and leak into the DINO box /
SAM mask, which skews the 98th-percentile top face LOW; the descent then drives
the tape into the duct and the tape's stiff contact (solref=0.005) launches the
light duct roll off the table. The duct is a static target, so an early read
stays valid and sidesteps that occlusion entirely.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext

from ._perceive import perceive_top_face
from .constants import PERCEIVE_TOP_SLAB


class Output(TypedDict):
    duct_xyz: list  # [x, y, top_z] world-frame duct top-face centre


def perceive_duct_top_xyz(ctx: NodeContext, cameras: list, target_key: str,
                          top_slab: float = PERCEIVE_TOP_SLAB):
    """Perceive the duct top-face centre from RGB-D (no ground-truth, no duct
    dimensions). Thin wrapper over the shared ``perceive_top_face`` core: the
    placed tape rests just above ``top_z``, so we take the top face directly and
    drop the cloud."""
    x, y, top_z, _ = perceive_top_face(ctx, cameras, target_key, top_slab=top_slab)
    return x, y, top_z


def run(ctx: NodeContext, *, target_key: str, cameras: list) -> Output:
    """Localize ``target_key``'s top-face centre and emit it as ``duct_xyz``.

    target_key: the place target's MJCF body key (e.g. ``"duct_tape_1"``); the
                DINO query is derived from it (``"duct tape"``).
    cameras:    the shared observation's camera list (``Ref("observe.cameras")``).
    """
    x, y, top_z = perceive_duct_top_xyz(ctx, cameras, target_key)
    print(f"[perceive_duct] {target_key} top-centre="
          f"({x:.4f}, {y:.4f}, {top_z:.4f})", flush=True)
    return {"duct_xyz": [x, y, top_z]}
