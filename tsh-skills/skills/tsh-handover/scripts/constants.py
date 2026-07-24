# ── Coordinated grasp clock-angle preset ───────────────────────────────────────
# The pickup revolve (GRASP_RING_ANGLE_DEG, in tsh-pickup) and the receiver
# revolve (RECV_GRASP_ANGLE_DEG) are a COUPLED pair: rotating the pickup moves
# the giver's finger around the presented ring, so the receiver must shift by
# the matching amount to keep threading the OPEN side (else it threads into
# the giver → drop). Both validated; flip GRASP_PRESET to switch:
#   "rotated"  → pickup 45 CW / receiver 30  : ACTIVE; zero arm contacts
#   "baseline" → pickup 0     / receiver 45  : original; works with a benign 0mm graze
GRASP_PRESET = "rotated"
_GRASP_ANGLE_PRESETS = {"rotated": (45.0, 0.0), "baseline": (0.0, 45.0)}  # (pickup, receiver)
RECV_GRASP_ANGLE_DEG = _GRASP_ANGLE_PRESETS[GRASP_PRESET][1]  # deg — receiver revolve about the
                                # presented ring axis (Y-Z plane) from the pure -Y point, wrist
                                # rolled to match. Set by GRASP_PRESET (coupled to the pickup angle).

# ── Receiver grasp ───────────────────────────────────────────────────────────
# Receiver grasp — grab the OPEN -Y rim from the receiver's OWN -Y side.
# The giver presents the ring vertically and pinches only its +Y rim, so the
# whole -Y half is free. recv_grasp_dy = rim_radius + RECV_GRASP_DY_MARGIN (NOT
# hole_radius-based: hole_radius is too noisy run-to-run, rim_radius is stable):
# the near (threading) finger lands just INSIDE the -Y edge of the hole, the far
# finger on the -Y outer wall (wall pinch).
RECV_GRASP_DY_MARGIN = 0.0105  # m — added to the perceived rim_radius; see above
RECV_GRASP_DZ         = 0.045   # m — raise the TCP so the fingertips reach hole height: the
                                # fingers hang ~44 mm BELOW the wrist for this +X pose, so
                                # TCP must sit that far above the hole centre.
RECV_PRE_BACK         = -0.10   # m — X back-off for the receiver pre-position (enough for the
                                # finger to clear the ring face before the +X insert).
RECV_INSERT_OVERSHOOT = 0.025   # m — push +X to seat the finger in the hole. Under-seating it
                                # shifts the grip geometry → the place IK fails and a real wrist
                                # collision appears. Deep seat = robust.
EXCHANGE_RETRACT_D  = 0.10   # m — retract distance after exchange

# ── Motion-planner tolerances ────────────────────────────────────────────────
# Pose-match thresholds for the giver present (tape-as-EE onto meet_xyz).
PLAN_POSITION_THRESHOLD = 0.01  # m
PLAN_ROTATION_THRESHOLD = 0.05  # rad

# ── Gripper actuation timing ────────────────────────────────────────────────────
RECV_OPEN_SETTLE_STEPS   = 15   # sim steps to settle the receiver's pre-thread open
RECV_CLOSE_SETTLE_STEPS  = 30   # sim steps to settle the receiver's rim grip closed —
                                # BEFORE the giver's open_gripper call fires (below the
                                # base close_gripper default of 60, the giver could start
                                # releasing while the receiver's grip is still settling).
GIVER_RELEASE_FRACTION   = 0.5  # partial giver open at the grab instant, before full release

# ── Reachability search sweep ────────────────────────────────────────────────
# Clock-angle deltas (deg) about the presented ring axis, tried in order until a
# receiver grasp can BOTH pre-position and finish the thread. The ring's rotational
# symmetry is the reach margin, so every angle threads the same hole.
RECV_ANGLE_SWEEP_DEG = (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60)
