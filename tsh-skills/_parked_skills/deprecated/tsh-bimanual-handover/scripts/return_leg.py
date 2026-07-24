"""Return the tape to where it was picked up, by replaying the forward handover
in reverse — same world-frame gripper poses, arm roles swapped.

The forward leg already demonstrated every pose this leg needs: the pickup
grasp, the giver's present pose, the receiver's threaded grab pose (the two
coexisted collision-free at the exchange), and the place release pose. Rather
than deriving new mirrored geometry (the arms are identical translated copies,
not mirror images — a true mirrored exchange is not reachable), this leg drives
each arm back through those RECORDED poses in reverse order:

  1. re-pick   — the original receiver re-perceives the resting ring (once,
                 right after the place — never at the hand-back), then descends
                 to the perception-centred RELEASE pose (never deeper) and
                 re-closes: its grip on the tape is the measured
                 ``receiver_offset`` shifted by the known drop clearance,
                 corrected exactly in step 2.
  2. re-meet   — it carries the tape back to its recorded exchange grab pose;
                 the tape is presented exactly as it was at the forward grab.
  3. re-grab   — the original giver re-approaches its recorded present pose
                 (reversing its own post-exchange retract) and closes on the rim
                 exactly where its forward grip was.
  4. extract   — the original receiver reverses its insert (partial open, pull
                 straight out of the hole, clear to its own side).
  5. re-place  — the original giver reverses its pickup: back to the recorded
                 grasp pose, descend, open, retract.

No new constants and no tuned mirror geometry: every target is a pose the
forward leg reached, so reachability and arm-arm clearance hold by construction
(for the same scene the forward leg just ran in). The only perception is the
single post-place re-perceive that centres the re-pick.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext
from gap_core.types import PointCloud

from ._motion import curobo_linear_move, plan_tool_move
from ._perceive import perceive_top_face
from .place import _approach_with_attached_tape
from .constants import (
    EXCHANGE_RETRACT_D,
    GIVER_RELEASE_FRACTION,
    GRASP_CLOSE_RAMP_STEPS,
    GRASP_CLOSE_SETTLE_STEPS,
    GRASP_HOOK_DZ,
    GRASP_LIFT_CLEARANCE,
    GRASP_PRE_DZ,
    PLACE_DROP_CLEARANCE,
    PLACE_Z_APPROACH,
    RECV_INSERT_OVERSHOOT,
    RECV_OPEN_SETTLE_STEPS,
    RECV_PRE_BACK,
)


class Output(TypedDict):
    returned: bool


# Extra raise of the re-grasp above the recorded release pose. At the release
# pose the pad BOTTOMS (the gripper's lowest points — the tips are raked
# sideways, not downward) sit <1 mm above whatever the ring rests on, so the
# close scrapes/wedges the duct under the ring. The ring wall (~31 mm) is
# taller than the pads (18 mm), so raising by this much keeps the pads FULLY
# on the wall while clearing the duct top by the same amount. Folded into the
# same known-grip-shift correction as the drop clearance (offset_prime).
REGRIP_RAISE = 0.008  # m


def _pose(rec):
    """Split a recorded [x,y,z,qw,qx,qy,qz] pose into (pos ndarray, quat wxyz)."""
    rec = [float(v) for v in rec]
    return np.asarray(rec[:3]), tuple(rec[3:])


def run(ctx: NodeContext, *,
        grasp_tcp: list,
        giver_tcp: list,
        receiver_tcp: list,
        receiver_offset: list,
        place_tcp: list,
        object_key: str,
        giver_arm: int = 0,
        receiver_arm: int = 1,
        tape_cloud: PointCloud | None = None) -> Output:
    """Replay the forward handover in reverse (see module docstring).

    grasp_tcp / giver_tcp / receiver_tcp / place_tcp: the forward leg's recorded
        world TCP poses (pickup close, exchange present, exchange grab, place
        release-minus-drop). ``giver_arm``/``receiver_arm`` are the FORWARD
        roles; this leg swaps them (the forward receiver gives the tape back).
    receiver_offset: tape centre in the (forward) receiver's TCP frame, measured
        at the forward grab — re-acquired (drop-clearance-corrected) by
        re-closing at the recorded release pose.
    object_key: the tape's key (DINO query is derived from it) — the resting
        ring is re-perceived ONCE, after the place and before the re-pick, to
        re-centre the grasp (the fingertips clear the duct walls by only a few
        mm, so the recorded rest pose alone is not accurate enough). There is
        deliberately NO perception at or before the hand-back itself.
    tape_cloud: forward pickup's perceived cloud; attached as the collision body
        for the carry back to the meet (sphere fallback inside the helper).
    """
    offset_local = np.asarray(
        [receiver_offset["x"], receiver_offset["y"], receiver_offset["z"]]
        if isinstance(receiver_offset, dict) else receiver_offset, dtype=float)
    p_grasp, q_grasp = _pose(grasp_tcp)
    p_give, q_give = _pose(giver_tcp)
    p_recv, q_recv = _pose(receiver_tcp)
    p_place, q_place = _pose(place_tcp)

    R_place = Rotation.from_quat([q_place[1], q_place[2], q_place[3], q_place[0]])
    side = 1.0 if float(ctx.tool("libero-yam.arm_base_pose",
                                 arm_id=receiver_arm)["position"][1]) > 0 else -1.0

    # 1. Re-pick. First re-perceive the RESTING ring — after the place, never at
    #    the hand-back: the release drop re-seats the ring a few mm, and at grasp
    #    depth the fingertips clear the duct's core/outer walls by only ~5 mm, so
    #    descending on the recorded rest pose alone risks pinching the duct too
    #    (runs 19/20). The arm parked directly above the ring; step aside to its
    #    own side for a clean camera view, then centre the grasp on the
    #    perceived XY (Z is trusted: the ring rests flush on the duct).
    c_rec = p_place + R_place.apply(offset_local)  # recorded rest centre
    park = p_place + [0.0, 0.0, PLACE_Z_APPROACH]
    for off in ([0.0, side * 0.20, 0.05], [0.0, side * 0.12, 0.12],
                [-0.08, side * 0.10, 0.15], [0.0, 0.0, 0.20]):
        aside = plan_tool_move(ctx, receiver_arm, park + off, q_place,
                               execute=False)
        if aside is not None:
            ctx.tool("libero-yam.execute_trajectory", trajectory=aside,
                     arm_id=receiver_arm)
            break
    else:  # never abort the leg over the view — perceive past the parked arm
        print("[return] WARNING: no reachable step-aside pose; "
              "perceiving in place", flush=True)
    delta = np.zeros(3)
    try:
        cameras = ctx.tool("robot.get_observation")["cameras"]
        px, py, _tz, _pts = perceive_top_face(ctx, cameras, object_key)
        delta[:2] = px - c_rec[0], py - c_rec[1]
        if np.linalg.norm(delta) > 0.06:  # implausible — grasp as recorded
            print(f"[return] WARNING: perceived ring "
                  f"{np.linalg.norm(delta)*1000:.0f}mm off the recorded rest; "
                  f"ignoring the correction", flush=True)
            delta[:] = 0.0
    except Exception as exc:
        print(f"[return] WARNING: post-place re-perceive failed ({exc}); "
              f"using the recorded rest pose", flush=True)
    print(f"[return] re-pick centred by perception: "
          f"delta=({delta[0]*1000:+.1f},{delta[1]*1000:+.1f})mm", flush=True)

    # Descend EXACTLY to the (re-centred) recorded release pose and no deeper —
    # the fingertip must not follow the dropped tape down, or it pokes past the
    # resting ring into whatever the ring rests on (the duct) and the close
    # pinches both. Closing one drop clearance above the wall-centred grab
    # shifts the re-acquired grip by a KNOWN amount, corrected via offset_prime.
    ctx.tool("robot.open_gripper", arm_id=receiver_arm,
             settle_steps=RECV_OPEN_SETTLE_STEPS)
    regrip_dz = PLACE_DROP_CLEARANCE + REGRIP_RAISE
    p_rel = p_place + delta + [0.0, 0.0, regrip_dz]
    hover = p_place + delta + [0.0, 0.0, PLACE_Z_APPROACH]
    drop_descent = PLACE_Z_APPROACH - regrip_dz
    plan_tool_move(ctx, receiver_arm, hover, q_place)
    curobo_linear_move(ctx, receiver_arm, list(p_rel), q_place,
                       allowed_axes=["Z"], distance=drop_descent,
                       direction=(0.0, 0.0, -1.0))
    ctx.tool("robot.close_gripper", arm_id=receiver_arm,
             settle_steps=GRASP_CLOSE_SETTLE_STEPS,
             ramp_steps=GRASP_CLOSE_RAMP_STEPS)
    curobo_linear_move(ctx, receiver_arm, list(hover), q_place,
                       allowed_axes=["Z"], distance=drop_descent,
                       direction=(0.0, 0.0, 1.0))
    print("[return] re-picked the tape at the recorded release pose", flush=True)

    # The re-acquired grip: the resting tape sits one drop clearance plus the
    # regrip raise BELOW where this TCP pose would hold it, so the tape centre
    # in the TCP frame is the measured receiver_offset minus that (world-down)
    # shift, rotated into the TCP frame.
    offset_prime = offset_local - R_place.inv().apply([0.0, 0.0, regrip_dz])

    # 2. Re-meet: carry the tape back to its forward-grab pose (the gripper-down
    #    -> face-on reorient of the forward place, reversed), with the held tape
    #    attached as a collision body; plain plan as fallback. The TAPE-centre
    #    target is exact, so the giver-side re-grab below replays verbatim; the
    #    grip delta is absorbed by this arm's wrist position instead.
    R_recv = Rotation.from_quat([q_recv[1], q_recv[2], q_recv[3], q_recv[0]])
    tape_at_meet = p_recv + R_recv.apply(offset_local)
    tape_now = hover + R_place.apply(offset_prime)  # held at the hover now
    if not _approach_with_attached_tape(ctx, receiver_arm, tape_at_meet, q_recv,
                                        offset_prime, tape_now, tape_cloud):
        plan_tool_move(ctx, receiver_arm, tape_at_meet, q_recv,
                       tool_offset=offset_prime)
    print("[return] presented the tape back at the exchange pose", flush=True)

    # 3. Re-grab: the original giver reverses its own post-exchange retract —
    #    it is still parked EXCHANGE_RETRACT_D behind its present pose — and
    #    closes on the rim exactly where its forward grip was.
    ctx.tool("robot.open_gripper", arm_id=giver_arm)
    pre_give = p_give - [EXCHANGE_RETRACT_D, 0.0, 0.0]
    plan_tool_move(ctx, giver_arm, pre_give, q_give)
    curobo_linear_move(ctx, giver_arm, list(p_give), q_give,
                       allowed_axes=["X"], distance=EXCHANGE_RETRACT_D,
                       direction=(1.0, 0.0, 0.0))
    ctx.tool("robot.close_gripper", arm_id=giver_arm)

    # 4. Extract: the forward receiver reverses its insert — partial open so the
    #    pads clear the rim, pull straight out of the hole along −X, then clear
    #    to its own side (sign from its base) and fully open. Its TCP sits at
    #    the grip-delta-shifted meet pose (see step 2), so back off from there.
    ctx.tool("robot.open_gripper", arm_id=receiver_arm,
             fraction=GIVER_RELEASE_FRACTION)
    tcp_meet = tape_at_meet - R_recv.apply(offset_prime)
    back = tcp_meet + [RECV_PRE_BACK - RECV_INSERT_OVERSHOOT, 0.0, 0.0]
    curobo_linear_move(ctx, receiver_arm, list(back), q_recv,
                       allowed_axes=["X"],
                       distance=RECV_INSERT_OVERSHOOT - RECV_PRE_BACK,
                       direction=(-1.0, 0.0, 0.0))
    ctx.tool("robot.open_gripper", arm_id=receiver_arm)
    plan_tool_move(ctx, receiver_arm,
                   back + [0.0, side * EXCHANGE_RETRACT_D, 0.0], q_recv)
    print("[return] handed back; forward receiver clear", flush=True)

    # 5. Re-place: the original giver reverses its pickup — free plan back to
    #    the lift pose above the recorded grasp, straight descent, release,
    #    straight retract. The tape lands where perception first found it.
    lift_dz = GRASP_LIFT_CLEARANCE - GRASP_HOOK_DZ
    plan_tool_move(ctx, giver_arm, p_grasp + [0.0, 0.0, lift_dz], q_grasp)
    curobo_linear_move(ctx, giver_arm, list(p_grasp), q_grasp,
                       allowed_axes=["Z"], distance=lift_dz,
                       direction=(0.0, 0.0, -1.0))
    ctx.tool("robot.open_gripper", arm_id=giver_arm)
    curobo_linear_move(ctx, giver_arm,
                       p_grasp + [0.0, 0.0, GRASP_PRE_DZ - GRASP_HOOK_DZ],
                       q_grasp, allowed_axes=["Z"],
                       distance=GRASP_PRE_DZ - GRASP_HOOK_DZ,
                       direction=(0.0, 0.0, 1.0))
    print("[return] tape placed back at the original pickup spot", flush=True)

    return {"returned": True}
