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
    GIVER_QUAT,
    GIVER_RELEASE_FRACTION,
    MEET_XYZ,
    PLAN_POSITION_THRESHOLD,
    PLAN_ROTATION_THRESHOLD,
    RECV_ANGLE_SWEEP_DEG,
    RECV_GRASP_ANGLE_DEG,
    RECV_GRASP_DY,
    RECV_GRASP_DZ,
    RECV_INSERT_OVERSHOOT,
    RECV_OPEN_SETTLE_STEPS,
    RECV_PRE_BACK,
    RECV_QUAT,
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



def _rim_grasp(hx, hy, hz, angle_deg, mirror=False, quat=RECV_QUAT):
    """Receiver grasp pose, revolved ``angle_deg`` around the ring's axis.

    The ring is presented face-on (axis ~ world +X), so its circumference lies in
    the world Y-Z plane. The base grasp (angle 0) is the -Y rim point: TCP offset
    ``(0, -RECV_GRASP_DY, +RECV_GRASP_DZ)`` from the hole centre with orientation
    ``quat`` (canonical ``RECV_QUAT`` by default; the station-geometry derivation
    passes a base-geometry quat instead). Revolving applies a RIGID rotation of
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
    ### offset from center hole to pinch the rim: this was TUNED (to avoid arm-arm collisions)
    off = Rx.apply([0.0, -RECV_GRASP_DY, RECV_GRASP_DZ])
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


def run(ctx: NodeContext, *,
        giver_arm: int = 0,
        receiver_arm: int = 1,
        tape_in_giver: list,
        meet_xyz: tuple = MEET_XYZ) -> Output:
    """Transfer the tape from giver to receiver at ``meet_xyz``.

    tape_in_giver: tape centre in the giver TCP frame, measured by the pickup —
                   the giver's tool offset for the whole exchange.
    meet_xyz:      world-frame handover location (where the tape centre is
                   presented). Defaults to the validated constants.MEET_XYZ;
                   ``GAP_HANDOVER_XYZ="x,y,z"`` overrides at runtime to sweep
                   handover positions without rebuilding the graph.
    """
    ### Flexible set of inputs for the agent
    default_meet = meet_xyz is MEET_XYZ  # caller didn't name a handover point
    if isinstance(meet_xyz, dict):
        meet_xyz = (meet_xyz["x"], meet_xyz["y"], meet_xyz["z"])
    override = os.environ.get("GAP_HANDOVER_XYZ")
    if override:
        meet_xyz = tuple(float(v) for v in override.split(","))
    tig = np.asarray(
        [tape_in_giver["x"], tape_in_giver["y"], tape_in_giver["z"]]
        if isinstance(tape_in_giver, dict) else tape_in_giver, dtype=float)

    # Handedness: the two arms are IDENTICAL copies (not mirror images), so both
    # reach the SAME orientation set. The giver therefore presents with the
    # canonical GIVER_QUAT regardless of side. What DOES flip for a reversed pair
    # (receiver on +Y => giver on -Y): the receiver threads from its own +Y side
    # (its grasp frame is reflected across Y=0), and the rendezvous mirrors to the
    # giver's own -Y side so the arms present on their own halves instead of
    # crossing. Side read from the base; an explicit/overridden meet is verbatim.
    recv_on_plus_y = float(
        ctx.tool("libero-yam.arm_base_pose", arm_id=receiver_arm)["position"][1]) > 0.0
    if recv_on_plus_y and default_meet and not override:
        mx, my, mz = meet_xyz
        meet_xyz = (mx, -my, mz)

    # Opt-in: derive the meet point AND the giver presentation orientation from
    # the arm-base geometry instead of the tuned constants (self-configures for a
    # restationed pair). Off by default; the tuned MEET_XYZ / GIVER_QUAT are the
    # fallback the probe seeds from.
    giver_quat = GIVER_QUAT
    recv_base_quat = RECV_QUAT
    if os.environ.get("GAP_DERIVE_MEET") and default_meet and not override:
        from ._station_geometry import derive_station_geometry

        def _giver_present(m, q):  # can the giver present the tape at m, orient q?
            return plan_tool_move(ctx, giver_arm, list(m), q, tool_offset=tig,
                                  execute=False) is not None

        def _recv_thread(m, q):  # can the receiver thread the rim at m, orient q?
            (tx, ty, tz), rq = _rim_grasp(m[0], m[1], m[2], RECV_GRASP_ANGLE_DEG,
                                          recv_on_plus_y, quat=q)
            return plan_tool_move(ctx, receiver_arm, [tx + RECV_PRE_BACK, ty, tz],
                                  rq, execute=False) is not None

        meet, giver_quat, recv_base_quat = derive_station_geometry(
            ctx, giver_arm=giver_arm, receiver_arm=receiver_arm,
            giver_present_probe=_giver_present, recv_thread_probe=_recv_thread)
        meet_xyz = tuple(meet)
        print(f"[exchange] derived meet_xyz={[round(v, 4) for v in meet_xyz]} "
              f"giver_quat={[round(v, 4) for v in giver_quat]} (GAP_DERIVE_MEET)",
              flush=True)

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
                                      quat=recv_base_quat)
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
                                         quat=recv_base_quat)

    #### BOTH OF THE BELOW were heavily tuned
    ### line up squarely with the tape
    curobo_linear_move(ctx, receiver_arm, [tx + RECV_PRE_BACK, ty, tz], recv_quat,
                       allowed_axes=["X", "Y", "Z"])
    ### then insert
    curobo_linear_move(ctx, receiver_arm, [tx + RECV_INSERT_OVERSHOOT, ty, tz],
                       recv_quat, allowed_axes=["X"],
                       distance=RECV_INSERT_OVERSHOOT - RECV_PRE_BACK,
                       direction=(1.0, 0.0, 0.0))

    # 4. Receiver closes on the rim. BEFORE the giver releases (grip still rigid),
    #    recover the tape's world pose from the giver's FK and express the tape
    #    centre in the RECEIVER TCP frame — the measured held-tape offset place
    #    needs. The receiver's own grasp error is thereby accounted for; no GT.
    ctx.tool("robot.close_gripper", arm_id=receiver_arm)
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

