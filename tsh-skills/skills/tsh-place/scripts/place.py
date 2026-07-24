"""Descend to a pre-approached hover, release, retract.

The holder does NOT hold the tape centred in its gripper: after a ring grasp
or a rim thread the tape centre sits several cm off the TCP. The tape is
therefore treated as the holder's end effector: the upstream step MEASURED
the held-tape offset (``held_offset`` — the pickup's ``tape_in_giver`` on the
direct route, or the exchange's ``receiver_offset`` after a handover; both
via rigid-grip FK, no ground truth), composed once here to get the actual
release TCP pose.

WHERE/WHICH orientation come from ``tsh-place-pose`` (the yaw-reachability
probe); a preceding ``tsh-transport-held`` node has already carried the arm
to the collision-aware hover above ``place_xyz``. This script only does the
straight-down set-down and straight-up retract (``curobo_linear_move``,
orientation locked) — the final legs, never a free transport.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from gap import NodeContext

from ._held import as_vec3, as_wxyz, q_to_R
from ._motion import curobo_linear_move
from .constants import PLACE_DROP_CLEARANCE, PLACE_Z_APPROACH


class Output(TypedDict):
    placed: bool
    place_tcp: list  # world TCP pose whose grip matches the RESTING tape,
                     # [x,y,z,qw,qx,qy,qz] (the release pose minus the drop
                     # clearance). A return leg re-closes exactly here to
                     # re-acquire the tape with the same measured grip.
    place_arm: int   # echo of the placing arm (checkpoint anchor)


def run(ctx: NodeContext, *, held_offset: list, place_xyz: list, place_quat: list,
        arm_id: int = 1) -> Output:
    """Release the held tape at ``place_xyz``/``place_quat``, then retract up.

    held_offset: tape centre in the holder's TCP frame — the pickup's
                 ``tape_in_giver`` (direct route) or the exchange's measured
                 ``receiver_offset`` (handover route); the holder's tool
                 offset here.
    place_xyz:   the tape's final rest centre — from ``tsh-place-pose``.
    place_quat:  the reachable presentation orientation (wxyz) — from
                 ``tsh-place-pose``; the preceding ``tsh-transport-held``
                 node has already carried the arm to the hover above this.
    arm_id:      the holding arm (route-decided).
    """
    offset_local = as_vec3(held_offset)
    place_xyz = as_vec3(place_xyz)
    place_quat = list(as_wxyz(place_quat))
    tcp_place = place_xyz - q_to_R(place_quat).apply(offset_local)

    # Descend most of the way, stop PLACE_DROP_CLEARANCE above the surface so
    # the fingers can open without jamming against the dest, then retract.
    drop_descent = PLACE_Z_APPROACH - PLACE_DROP_CLEARANCE
    tcp_release = tcp_place + np.array([0.0, 0.0, PLACE_DROP_CLEARANCE])
    curobo_linear_move(ctx, arm_id, list(tcp_release), place_quat, allowed_axes=["Z"],
                       distance=drop_descent, direction=(0.0, 0.0, -1.0))
    ctx.tool("robot.open_gripper", arm_id=arm_id)
    curobo_linear_move(ctx, arm_id,
                       list(tcp_place + np.array([0.0, 0.0, PLACE_Z_APPROACH])),
                       place_quat, allowed_axes=["Z"],
                       distance=PLACE_Z_APPROACH - PLACE_DROP_CLEARANCE,
                       direction=(0.0, 0.0, 1.0))

    return {"placed": True,
            "place_tcp": [float(v) for v in (*tcp_place, *place_quat)],
            "place_arm": int(arm_id)}
