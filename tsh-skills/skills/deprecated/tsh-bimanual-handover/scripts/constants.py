# ──────────────────────────────────────────────────────────────────────────────
# World-frame axis convention (LIBERO-YAM bimanual scene)
# ──────────────────────────────────────────────────────────────────────────────
#   X  (+X = forward, away from robot backs)
#      Both arm bases at x≈0.25; tape objects at x≈0.55.
#   Y  (+Y = arm0 / left side;  −Y = arm1 / right side)
#      arm0 base y=+0.31, arm1 base y=−0.31.
#      yellow_tape at y≈+0.375 (arm0 side); duct_tape at y≈−0.375 (arm1 side).
#   Z  (+Z = up; table surface ≈0.75 m; tape centroids ≈0.762 m)
# ──────────────────────────────────────────────────────────────────────────────

# Tape half-thickness (top face above the body centroid). DERIVED at runtime from
# the perceived cloud — (robust top − robust bottom)/2 in perceive_tape — so it
# tracks the true tape, whatever its size. The constant below is only the FALLBACK
# used when the cloud is too degenerate to measure (mirrors the GRASP_RING_* /
# hole-radius fallback pattern). Measured from the scaled yellow_tape mesh.
TAPE_TOP_Z   = 0.016   # m — fallback tape half-thickness
TAPE_HALF_MIN = 0.005  # m — sanity band on the perceived half-thickness; outside it,
TAPE_HALF_MAX = 0.040  # m   fall back to TAPE_TOP_Z rather than trust a bad cloud

# ── Grasp parameters (pickup.py) ─────────────────────────────────────
# Strategy: ring-grasp from above with DOWN_QUAT.
# DOWN:  TCP z-axis → world −Z (top-down approach, fingers spread along world Y).
#   With DOWN_QUAT the finger pair spreads along world Y (half-gap ≈ 39 mm).
#
#   ring_dx = +0.03 m  shifts the TCP forward in X (away from robot backs) to
#             centre the approach over the tape without colliding with the rim.
#   ring_dy = +0.043 m shifts the TCP in +Y (toward arm0 base side) so that:
#             • near finger (TCP − 39 mm in Y) lands ≈4 mm from tape centre
#               — well inside the 38 mm inner hole                           ✓
#             • far  finger (TCP + 39 mm in Y) lands ≈82 mm from centre
#               — well outside the 54 mm outer radius                        ✓
#   hook_dz = 0.0      places the TCP at the tape midplane for maximum
#             outer-wall contact; closing the gripper hooks the inner rim.
# With DOWN_QUAT the finger tips sit ~38 mm BEHIND the TCP in X.
# ring_dx = 0.038 compensates: finger tip lands at tape centre in X.
# ring_dx = 0.050 puts finger tip 12 mm INSIDE the hole in X (more clearance).
DOWN_QUAT     = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)
GRASP_RING_DX = 0.048   # m — rotated ~18° CW on ring; fingertip 10mm from centre in X
# Fingers spread along world X (half-gap ≈ 40 mm). With ring_dx=0.038 the near
# finger lands ≈ tape centre (inside the 47 mm hole) and the far finger ≈ 78 mm
# out (outside the 62 mm rim) → a true wall pinch. ring_dy shifts both in Y;
# 0.038 keeps the near finger clearly inside the hole. Re-tuned for the 1.3×
# tape under realistic friction (1.0): combined with the kp=400 clamp (run.py)
# the grip survives the 90° reorientation to the present pose.
GRASP_RING_DY = 0.024   # m — rotated grip: near finger 17mm inside hole, far 2mm outside rim
# ── Coordinated grasp clock-angle preset ───────────────────────────────────────
# The pickup revolve (GRASP_RING_ANGLE_DEG) and the receiver revolve
# (RECV_GRASP_ANGLE_DEG) are a COUPLED pair: rotating the pickup moves the giver's
# finger around the presented ring, so the receiver must shift by the matching
# amount to keep threading the OPEN side (else it threads into the giver → drop).
# Both validated; flip GRASP_PRESET to switch — both angles read from it:
#   "rotated"  → pickup 45 CW / receiver 30  : ACTIVE; zero arm contacts
#   "baseline" → pickup 0     / receiver 45  : original; works with a benign 0mm graze
GRASP_PRESET = "rotated"
_GRASP_ANGLE_PRESETS = {"rotated": (45.0, 0.0), "baseline": (0.0, 45.0)}  # (pickup, receiver)
GRASP_RING_ANGLE_DEG = _GRASP_ANGLE_PRESETS[GRASP_PRESET][0]  # deg — pickup revolve CW about the
                             # vertical hole axis (set by GRASP_PRESET above).
GRASP_PRE_DZ  = 0.07    # m — hover height above grasp point before descent
GRASP_HOOK_DZ = -0.005 # m — deep seat: finger pads centered on the 32mm wall/inner rim.
# TCP world-z = centroid + hook_dz. The contact pads span TCP-3.8mm..TCP+14.2mm in
# world-z (measured from yam.xml FK: pads at link6-z 0.1205..0.1385, TCP at 0.1347).
# At -0.005 the deepest pad sits ~0.753 (clear of the table/tape-bottom ≈0.746) and
# the shallowest ~0.771, so the pads engage the full wall instead of just the top lip
# (+0.015 only clipped the top ~5mm). Floor: -0.012 puts the deepest pad on the table.
# (Old note "hz<0.015 slips" predates the friction=1.0 / kp=800 / real-inertia fixes.)
GRASP_LIFT_CLEARANCE = 0.085  # m — lift the grasped tape this far ABOVE its perceived
                              # grasp height (grasp_z + clearance). Relative, so it can't
                              # silently break if the table height changes — an absolute
                              # world-Z target would.

# ── Gripper geometry (YAM-specific, empirical FK under DOWN_QUAT) ─────────────
# (each fingertip sits ~39 mm from the TCP along the spread axis — the half-gap
# referenced in the grasp-offset rationale above)
GRIPPER_FINGERTIP_BEHIND = 0.038 # m — fingertip sits this far behind TCP along the approach axis

# ── Handover (bimanual_exchange.py) ────────────────────────────────────────────
# Everything that defines WHERE and in WHAT ORIENTATION the two arms meet.
#
# GOAL: present the tape "o" face-on along the world X axis (ring normal ∥ X) so
# the hole reads as an O from the front. Because this is an insertion grasp, the
# ring normal is locked to the gripper z-axis, so "o on X" ⟺ gripper-z on X.
# The giver CANNOT aim its gripper at -X (tool folded back over its own base —
# 0/528 IK solutions, even at 10° tolerance). It CAN aim +X (tool forward), which
# gives the identical face-on "o on X" view with the gripper tucked behind the
# tape. Both arms therefore meet pointing +X and insert into the same hole from
# the -X side (a two-fingers-in-one-hole transfer), exactly as the OpenPI human
# demo does (openpi/examples/joints.h5). Values below are that demo's giver/
# receiver poses at the grab instant (t=185), converted from link_6 to TCP space
# (TCP = link_6 + 0.1347·gripper_z = YAM_TCP_OFFSET), IK-verified for both arms.
#
# Orientations (wxyz quaternions); both have gripper z-axis ≈ +X (tool forward):
#   GIVER: z-axis [1.00, 0.00, 0.00]  presents the tape o FACE-ON (normal ∥ +X)
#   RECV:  z-axis [1.00, 0.00, 0.00]  also threads the hole straight along +X
# GIVER_QUAT was the demo's tilted pose (z-axis [0.95,-0.32,0.05] → tape o tilted
# ~10° in +Y on camera). Re-solved to the nearest IK-reachable orientation at
# MEET_XYZ with gripper-z on pure +X (19° wrist change, finger-spread axis nearly
# unchanged so the pinch survives) → tape o now presents square to X.
# Demo poses (openpi/examples/joints.h5, grab instant t=185, FK link_6 world quat).
# These are the ACTUAL teleop demo wrist orientations — restored after the
# straight-+X RECV_QUAT was found to swing the receiver wrist across into the
# giver (gripper-body collision). The giver gripper-z splays toward -Y
# ([0.95,-0.32,0.05]); the receiver gripper-z splays toward +Y
# ([0.92,+0.31,-0.23]) — opposite splays keep the two wrists ~0.17 m apart in Y
# while the fingertips converge in the hole. That splay IS the de-confliction.
# SYMMETRIC threading: receiver is the exact Y-MIRROR of the giver (gz mirrored in
# Y: giver [0.95,-0.32,0.05] -> receiver [0.95,+0.32,0.05]). Both thread the hole
# from -X, splayed to opposite Y sides, BOTH FLAT (no down-tilt) — so the receiver
# grip no longer cocks the ring, and the mirror splay separates the wrists
# (collision-free by geometry, no Z-stagger needed). The old demo RECV_QUAT
# (0.4023,0.4028,0.6723,0.4732) tilted down ~23deg, which cocked the held ring.
GIVER_QUAT = (0.5, 0.5, 0.5, 0.5)  # gz=[1,0,0] — tool on pure +X so the ring plane is parallel
                                   # to the Y axis (hole axis exactly along X, no tilt)
# Receiver wrist: splayed toward -Y (gz=[0.99,-0.15,0]), i.e. the SAME splay as
# the giver. Two reasons: (1) a -Y splay throws the receiver hand toward -Y, away
# from the giver (a +Y splay or a pure +X pose threw it toward the giver / sat on
# an arm-1 IK branch boundary that made the arm thrash); (2) being ~8.5° off pure
# +X keeps arm 1 off that singularity so consecutive IK solves stay on one branch.
RECV_QUAT  = (0.5358, 0.5358, 0.4614, 0.4614)  # gz=[0.99,-0.15,0] — splay toward -Y
#
# meet_xyz: world-frame TCP point where arm0 (giver) presents the tape.
#   = demo giver TCP @ t=185. y=0.050 (NOT 0.22) keeps the giver on its own side.
MEET_XYZ            = (0.586, 0.050, 1.032)  # demo giver TCP @ t=185
# Receiver grasp — grab the OPEN -Y rim from the receiver's OWN -Y side.
# The giver presents the ring vertically and pinches only its +Y rim, so the
# whole -Y half is free. The receiver therefore targets the -Y rim (NOT the hole
# centre, which sits in the giver's corridor): TCP at hole - RECV_GRASP_DY in Y,
# so the near finger threads the hole edge and the far finger lands on the -Y
# outer wall (wall pinch). It approaches from -Y and -X, slides +Y to the rim,
# then inserts +X — its wrist never enters the central corridor where the giver's
# forearm (link_5) sits. The receiver re-queries the LIVE tape pose (tape_key),
# so these are offsets from the true hole centre and generalise across positions.
RECV_GRASP_DY         = 0.075   # m — TCP offset in -Y from the hole centre. Sized so the
                                # threading (near) finger lands just INSIDE the -Y edge of the
                                # hole (≈1 cm in), NOT on the rim wall: the giver's inner finger
                                # holds the +Y edge, so the receiver threads the opposite (-Y)
                                # edge ~7 cm away across the 9.4 cm hole. The far finger lands
                                # outside the -Y rim (free space). (Too large → finger hits the
                                # -Y rim and shoves the tape; too small → finger meets the
                                # giver's finger at the centre.) Kept a tuned constant rather than
                                # derived from the perceived hole radius: that estimate is noisy
                                # run-to-run (~31–48 mm) and would swing the receiver grab ~17 mm.
RECV_GRASP_DZ         = 0.045   # m — raise the TCP so the fingertips reach hole height: the
                                # fingers hang ~44 mm BELOW the wrist for this +X pose, so
                                # TCP must sit that far above the hole centre.
RECV_GRASP_ANGLE_DEG = _GRASP_ANGLE_PRESETS[GRASP_PRESET][1]  # deg — receiver revolve about the
                                # presented ring axis (Y-Z plane) from the pure -Y point, wrist
                                # rolled to match. Set by GRASP_PRESET (coupled to the pickup angle).
RECV_PRE_BACK         = -0.10   # m — X back-off for the receiver pre-position (enough for the
                                # finger to clear the ring face before the +X insert).
RECV_INSERT_OVERSHOOT = 0.025   # m — push +X to seat the finger in the hole. 0.025 fully threads
                                # the hole for a secure grip; under-seating it (tried 0.018) shifts
                                # the grip geometry → the place IK fails and a real wrist collision
                                # appears. Deep seat = robust.
EXCHANGE_RETRACT_D  = 0.10   # m — retract distance after exchange

# ── Place parameters (place.py) ────────────────────────────────────────────────
# place target: the placed TAPE centre = perceived duct TOP-face centre + PLACE_OFFSET,
#   PLUS the perceived tape half-thickness in Z so the tape rests FLUSH on the duct
#   surface (tape bottom on the duct top). The Z rest height is therefore DERIVED, not
#   tuned — no tape or duct dimensions assumed. PLACE_OFFSET is now just an optional
#   XY/Z nudge on top of that flush pose (default: centred, no nudge).
# retract_offset: (dx, dy, dz) from the duct top-face centre to the retract TCP.
PLACE_OFFSET         = (0.0,  0.00,  0.00)  # centred on the duct (stacking on top, no rim to dodge)
# curobo plans the descent in joint space; hover = place + z_approach.
PLACE_Z_APPROACH     = 0.08
PLACE_DROP_CLEARANCE = 0.05  # release this far above the rest pose and let the tape DROP the
                             # last stretch: the receiver's finger (threaded in the hole) slips
                             # out during the fall and the gripper is clear before it retracts,
                             # so the straight-up retract can't hook and lift the ring. 0.02 was
                             # too low — the finger stayed in the hole and carried the tape up.
PLACE_RETRACT_OFFSET = (0.0, -0.05,  0.12)  # up/away from the duct after release
# Fallback collision body for the tape-attached approach swing, used only when no
# perceived cloud is available (a convex hull of the cloud is preferred).
PLACE_TAPE_SPHERE_RADIUS = 0.08   # m — bounding-sphere radius
PLACE_TAPE_SPHERE_SUBDIV = 2      # icosphere tessellation level

# ── Motion-planner tolerances (bimanual_exchange.py / place.py) ─────────────────
# Pose-match thresholds for the two tolerance-sensitive plans: the giver present
# (tape-as-EE onto MEET_XYZ) and the place hover with the tape attached. Everywhere
# else the bundle's own defaults are fine.
PLAN_POSITION_THRESHOLD = 0.01  # m
PLAN_ROTATION_THRESHOLD = 0.05  # rad

# ── Gripper actuation timing ────────────────────────────────────────────────────
# ramp then settle so the light ring is not kicked as the jaws seat on close.
GRASP_CLOSE_SETTLE_STEPS = 150  # sim steps to settle the giver grip after the ramp
GRASP_CLOSE_RAMP_STEPS   = 60   # sim steps to ramp the jaws gently closed
RECV_OPEN_SETTLE_STEPS   = 200  # sim steps to settle the receiver's pre-thread open
GIVER_RELEASE_FRACTION   = 0.5  # partial giver open at the grab instant, before full release

# ── Reachability search sweeps ──────────────────────────────────────────────────
# Clock-angle deltas (deg) about the presented ring axis, tried in order until a
# receiver grasp can BOTH pre-position and finish the thread. The ring's rotational
# symmetry is the reach margin, so every angle threads the same hole.
RECV_ANGLE_SWEEP_DEG = (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60)
# Yaw deltas (deg) about the tape's vertical axis, tried in order until the place
# hover (the most reach-constrained waypoint) is plannable. The round tape lays flat
# at any yaw, so the sweep only trades reach.
PLACE_YAW_SWEEP_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                       110, -110, 130, -130, 150, -150, 180)

# ── Perception (_perceive.py, perceive_tape.py, perceive_duct.py) ───────────────
# No scene-specific constants here — these parametrise the ROBUST estimators over
# the back-projected cloud, chosen to reject the minority of bad pixels/points.
PERCEIVE_TOP_SLAB     = 0.006  # m — top-face slab whose mean gives the (x, y) centre
PERCEIVE_TOP_PCTILE   = 98     # robust top face: drops the <2% arm/gripper/grazing-edge
                               # outliers that a plain max() would latch onto
PERCEIVE_FLOOR_PCTILE = 10     # z-floor: drops the bottom slab (table/floor pixels)
PERCEIVE_SPRAY_ABOVE_MEDIAN = 0.03  # m — drop cloud points more than this ABOVE the median
                               # height: the depth-discontinuity spray (mask-edge / gripper
                               # pixels that back-project ~10cm high at grazing views) that
                               # otherwise poisons the top face. Head-on clouds have none, so
                               # the estimate is unchanged there — validated on the y=0.375 row.
PERCEIVE_BOTTOM_PCTILE = 2     # robust bottom of the (already floor-trimmed) cloud, used
                               # with the top face to derive the tape half-thickness
# Ring radii from the radial-distance distribution about the top-face centre.
RING_HOLE_PCTILE = 2   # inner-hole radius (robust to points leaking into the hole)
RING_RIM_PCTILE  = 98  # outer-rim radius (robust to edge outliers)
# Sanity band on the perceived radii; outside it the pickup falls back to the tuned
# GRASP_RING_* defaults rather than trusting a noisy cloud.
RING_HOLE_MIN   = 0.015  # m — reject implausibly small holes
RING_HOLE_MAX   = 0.080  # m — reject implausibly large holes
RING_RIM_MARGIN = 0.005  # m — rim must exceed the hole by at least this

