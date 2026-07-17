"""Bimanual insertion handover: giver presents the ring face-on, receiver inserts.

Both grippers point +X (gripper-z = +X), so the giver presents the tape "o"
face-on along world X and the receiver threads a finger into the same hole. A
naive straight -X receiver approach drives it THROUGH the giver's arm, so the
receiver pre-positions on its own -Y side (its wrist passes beside the giver's),
aligns while still behind the tape, and inserts purely along +X onto the open
rim (one finger in the hole, one outside -> wall pinch).

The tape is the giver's end effector: the pickup measured the tape centre in the
giver TCP frame (``tape_in_giver``, rigid under the grip), and that offset is
composed into the planner's tcp_offset (``plan_tool_move``), so the present move
literally plans "tape centre to ``meet_xyz``" — no ground truth, no back-solve.

Arm layout (LIBERO-YAM): canonical is arm0 (left, base y~+0.31) giver, arm1
(right, base y~-0.31) receiver. The geometry is authored for that handedness;
passing the swapped pair (giver_arm=1, receiver_arm=0) mirrors the whole
exchange across Y=0 automatically — the handedness is read from the receiver
base, so no per-side constants are duplicated.
"""

from __future__ import annotations

import os
from typing import TypedDict

import numpy as np
from scipy.spatial.transform import Rotation

from gap import NodeContext

from ._motion import curobo_linear_move, plan_tool_move
from .constants import (
    EXCHANGE_RETRACT_D,
    GIVER_RELEASE_FRACTION,
    PLAN_POSITION_THRESHOLD,
    PLAN_ROTATION_THRESHOLD,
    RECV_ANGLE_SWEEP_DEG,
    RECV_CLOSE_SETTLE_STEPS,
    RECV_GRASP_ANGLE_DEG,
    RECV_GRASP_DY_MARGIN,
    RECV_GRASP_DZ,
    RECV_INSERT_OVERSHOOT,
    RECV_OPEN_SETTLE_STEPS,
    RECV_PRE_BACK,
)

## This is used for the return
class Output(TypedDict):
    handed_over: bool
    receiver_offset: list  # tape centre in the receiver TCP frame (m), measured at
                           # the grab instant via the giver's rigid grip + FK (no
                           # GT). Consumed by place to land the tape centre on target.
    giver_tcp: list     # world giver TCP pose at the present, [x,y,z,qw,qx,qy,qz].
                        # A return leg replays these two proven-feasible, proven-
                        # collision-free poses with the arm roles swapped.
    receiver_tcp: list  # world receiver TCP pose at the grab, same layout.


# Reflection across the world Y=0 plane. The exchange geometry below is authored
# for the CANONICAL layout — giver on +Y, receiver inserting from its own -Y
# side. Swapping the arms makes the whole handover the mirror image across Y=0,
# so we reflect every pose: positions y -> -y, orientations R -> M R M (a proper
# rotation — conjugating a rotation by a reflection preserves handedness). The
# handedness is read from the receiver base (see run), never a tuned per-side set.
_M = np.diag([1.0, -1.0, 1.0])



def _rim_grasp(hx, hy, hz, angle_deg, mirror=False, *, quat, recv_grasp_dy):
    """Receiver grasp pose, revolved ``angle_deg`` around the ring's axis.

    The ring is presented face-on (axis ~ world +X), so its circumference lies in
    the world Y-Z plane. The base grasp (angle 0) is the -Y rim point: TCP offset
    ``(0, -recv_grasp_dy, +RECV_GRASP_DZ)`` from the hole centre with orientation
    ``quat`` (the required canonical receiver quat, supplied base-geometry-derived
    by tsh-station-geometry). ``recv_grasp_dy`` is DERIVED per-call from the
    perceived ``rim_radius`` (stable run-to-run, unlike ``hole_radius``) plus a
    small fixed margin — see ``run()``. Revolving applies a RIGID rotation of
    the whole grasp frame about the world-X axis through the hole centre — offset
    AND orientation together — carrying the threading finger to a new clock
    position while its threading geometry stays intact. ``mirror`` reflects the
    (relative) grasp frame across Y=0 for a receiver whose base sits on +Y — the
    hole itself is real, so only the offset and orientation flip. Returns
    ``((tx,ty,tz), quat_wxyz)``.
    """
    a = np.radians(angle_deg)
    ## rotating around x sweeps gripper along circular rim
    Rx = Rotation.from_rotvec([a, 0.0, 0.0])  # about world X = the hole axis
    off = Rx.apply([0.0, -recv_grasp_dy, RECV_GRASP_DZ])
    qw, qx, qy, qz = quat
    ### hand orientation changes as we revolve around the circle
    R = Rx * Rotation.from_quat([qx, qy, qz, qw])
    if mirror:  # receiver on +Y: reflect offset + orientation across Y=0
        off = _M @ off
        R = Rotation.from_matrix(_M @ R.as_matrix() @ _M)
    tx, ty, tz = hx + off[0], hy + off[1], hz + off[2] # detected center of the tape, adds offset + rotation
    rx, ry, rz, rw = R.as_quat()
    return (tx, ty, tz), (rw, rx, ry, rz)


def _ee(ctx: NodeContext, arm_id: int):
    """Current TCP position + rotation for ``arm_id``."""
    ee = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    p = np.array([ee["position"]["x"], ee["position"]["y"], ee["position"]["z"]])
    r = ee["rotation"]
    return p, Rotation.from_quat([r["x"], r["y"], r["z"], r["w"]])


def _as_wxyz(q):
    """Accept a Quaternion dict {w,x,y,z} (subgraph type coercion) or a wxyz
    sequence; return a plain wxyz tuple."""
    return (q["w"], q["x"], q["y"], q["z"]) if isinstance(q, dict) else tuple(q)


def run(ctx: NodeContext, *,
        giver_arm: int = 0,
        receiver_arm: int = 1,
        tape_in_giver: list,
        rim_radius: float,
        meet_xyz: tuple,
        giver_quat: tuple,
        recv_quat: tuple) -> Output:
    """Transfer the tape from giver to receiver at ``meet_xyz``.

    tape_in_giver: tape centre in the giver TCP frame, measured by the pickup —
                   the giver's tool offset for the whole exchange.
    rim_radius:    perceived outer rim radius (m), relayed from tsh-pickup (the
                   same value it used for the giver's grasp). Stable run-to-run
                   (~1mm spread), unlike hole_radius (~17mm spread) — so the
                   receiver's thread offset derives from this, not hole_radius.
    meet_xyz:      world-frame handover location (where the tape centre is
                   presented). REQUIRED — supplied by tsh-station-geometry (no
                   tuned fallback); ``GAP_HANDOVER_XYZ="x,y,z"`` overrides at runtime.
    giver_quat / recv_quat: presentation orientations (wxyz). REQUIRED — supplied
                   base-geometry-derived by tsh-station-geometry (no tuned
                   fallback). ``recv_quat`` is canonical (as if the receiver sits
                   on -Y); ``_rim_grasp`` mirrors it for a +Y receiver.
    """
    recv_grasp_dy = rim_radius + RECV_GRASP_DY_MARGIN
    ### Flexible set of inputs for the agent
    if isinstance(meet_xyz, dict):
        meet_xyz = (meet_xyz["x"], meet_xyz["y"], meet_xyz["z"])
    override = os.environ.get("GAP_HANDOVER_XYZ")
    if override:
        meet_xyz = tuple(float(v) for v in override.split(","))
    tig = np.asarray(
        [tape_in_giver["x"], tape_in_giver["y"], tape_in_giver["z"]]
        if isinstance(tape_in_giver, dict) else tape_in_giver, dtype=float)
    # Presentation quats may arrive as a Quaternion dict {w,x,y,z} (subgraph type
    # coercion) or as a wxyz sequence; normalize to a wxyz tuple.
    giver_quat = _as_wxyz(giver_quat)
    recv_quat = _as_wxyz(recv_quat)

    # Handedness: the two arms are IDENTICAL copies (not mirror images), so both
    # reach the SAME orientation set. The giver presents with ``giver_quat``
    # regardless of side; for a reversed pair (receiver on +Y) the receiver
    # threads from its own +Y side (grasp frame reflected across Y=0). Side is
    # read from the base and drives the ``_rim_grasp`` mirror below. The meet
    # point is taken verbatim — tsh-station-geometry already computes it for the
    # actual base configuration.
    recv_on_plus_y = float(
        ctx.tool("libero-yam.arm_base_pose", arm_id=receiver_arm)["position"][1]) > 0.0

    # The giver presents with ``giver_quat``; the receiver threads with the
    # canonical ``recv_quat`` (``_rim_grasp`` mirrors it for a +Y receiver), both
    # supplied base-geometry-derived by tsh-station-geometry.
    recv_base_quat = recv_quat

    def _hole():
        """Live tape (hole) centre from the giver's FK + the rigid tape-in-giver
        offset — no ground truth."""
        p, R = _ee(ctx, giver_arm)
        w = p + R.apply(tig) ## rotates tape into world orientaiton, add to giver IK
        return float(w[0]), float(w[1]), float(w[2])

    # 1. Giver presents the tape "o" face-on at the meeting point: the TAPE is
    #    the end effector (tool_offset=tig), so cuRobo plans the tape centre to
    #    meet_xyz directly — repeatable regardless of how the ring settled on the
    #    finger.
    plan_tool_move(ctx, giver_arm, meet_xyz, giver_quat, tool_offset=tig,
                   position_threshold=PLAN_POSITION_THRESHOLD,
                   rotation_threshold=PLAN_ROTATION_THRESHOLD)

    # 2. Receiver opens and pre-positions on its OWN Y side, level with the hole
    #    and backed off in -X, so its finger can sweep into the hole face along +X.
    #    Sweep the grasp angle around the ring circumference and take the first
    #    clock position that can BOTH pre-position AND reach the inserted pose —
    #    different angles thread the same hole from different clock positions, so
    #    the ring's symmetry is the reach margin. Probing the insert too (not just
    #    the pre-position) rejects angles whose thread can't finish — essential for
    #    the reversed handover, where the receiver reaches from the far side.
    ctx.tool("robot.open_gripper", arm_id=receiver_arm, settle_steps=RECV_OPEN_SETTLE_STEPS)
    hx, hy, hz = _hole()
    chosen = None
    ### Calculate a clock position sweep around the tape for a reachable IK
    for dangle in RECV_ANGLE_SWEEP_DEG:
        angle = RECV_GRASP_ANGLE_DEG + dangle ## increment the angle
        (tx, ty, tz), rq = _rim_grasp(hx, hy, hz, angle, recv_on_plus_y,
                                      quat=recv_base_quat, recv_grasp_dy=recv_grasp_dy)
        pre_traj = plan_tool_move(ctx, receiver_arm, [tx + RECV_PRE_BACK, ty, tz], rq,
                                  execute=False)
        if pre_traj is None:
            continue
        if plan_tool_move(ctx, receiver_arm, [tx + RECV_INSERT_OVERSHOOT, ty, tz], rq,
                          execute=False) is None:
            continue  # pre-position ok but the thread can't finish — try next clock
        chosen = (angle, pre_traj)
        break
    if chosen is None:
        raise RuntimeError("exchange: no reachable receiver grasp angle found") # failed
    angle, pre_traj = chosen
    ctx.tool("libero-yam.execute_trajectory", trajectory=pre_traj, arm_id=receiver_arm)

    # 3. Re-query the (settled) hole, slide to the standoff while still behind in
    #    X, then insert purely along +X — a jerk-limited planned trajectory. The
    #    receiver wrist stays on the open side the whole time.
    hx, hy, hz = _hole()
    (tx, ty, tz), recv_quat = _rim_grasp(hx, hy, hz, angle, recv_on_plus_y,
                                         quat=recv_base_quat, recv_grasp_dy=recv_grasp_dy)

    #### BOTH OF THE BELOW were heavily tuned
    ### line up squarely with the tape
    curobo_linear_move(ctx, receiver_arm, [tx + RECV_PRE_BACK, ty, tz], recv_quat,
                       allowed_axes=["X", "Y", "Z"])
    ### then insert
    curobo_linear_move(ctx, receiver_arm, [tx + RECV_INSERT_OVERSHOOT, ty, tz],
                       recv_quat, allowed_axes=["X"],
                       distance=RECV_INSERT_OVERSHOOT - RECV_PRE_BACK,
                       direction=(1.0, 0.0, 0.0))

    # 4. Receiver closes on the rim and settles BEFORE the giver moves at all — the
    #    settle steps run the sim to let the receiver's jaw contact stabilize while
    #    the giver's gripper command hasn't fired yet, so the giver can't start
    #    releasing while the receiver's grip is still closing. Then recover the
    #    tape's world pose from the giver's FK and express the tape centre in the
    #    RECEIVER TCP frame — the measured held-tape offset place needs. The
    #    receiver's own grasp error is thereby accounted for; no GT.
    ctx.tool("robot.close_gripper", arm_id=receiver_arm,
             settle_steps=RECV_CLOSE_SETTLE_STEPS)
    tape_world = np.array(_hole())
    p_recv, R_recv = _ee(ctx, receiver_arm)
    # The ring slides ~2cm back into the receiver's designed rim seat the instant
    # the giver lets go (measured ~20mm, mostly along the insert axis). Anchoring
    # the offset to _hole() at the OVER-INSERTED TCP (p_recv) therefore captures the
    # transient dual-pinch pose and is stale by place time. Anchor instead to the
    # grasp TARGET (tx,ty,tz) — the seat the ring actually settles into — so the
    # rigid offset matches the settled ring. GT-free (pure grasp geometry).
    receiver_offset = R_recv.inv().apply(tape_world - np.array([tx, ty, tz])).tolist()
    ctx.tool("robot.open_gripper", arm_id=giver_arm, fraction=GIVER_RELEASE_FRACTION)

    # 5. Giver retracts in −X first (perpendicular to the tape axis, clearing the
    #    receiver's path to place), then the receiver retracts along its own side
    #    in Y (−Y canonically, +Y when mirrored — always away from the centre).
    gp, gR = _ee(ctx, giver_arm)
    gqx, gqy, gqz, gqw = gR.as_quat()
    giver_tcp = [float(v) for v in (*gp, gqw, gqx, gqy, gqz)]
    receiver_tcp = [float(v) for v in
                    (tx + RECV_INSERT_OVERSHOOT, ty, tz, *recv_quat)]
    plan_tool_move(ctx, giver_arm,
                   [gp[0] - EXCHANGE_RETRACT_D, gp[1], gp[2]], giver_quat)
    ctx.tool("robot.open_gripper", arm_id=giver_arm)
    recv_retract = EXCHANGE_RETRACT_D if recv_on_plus_y else -EXCHANGE_RETRACT_D
    plan_tool_move(ctx, receiver_arm,
                   [tx + RECV_PRE_BACK, ty + recv_retract, tz], recv_quat)

    return {"handed_over": True, "receiver_offset": receiver_offset,
            "giver_tcp": giver_tcp, "receiver_tcp": receiver_tcp}

