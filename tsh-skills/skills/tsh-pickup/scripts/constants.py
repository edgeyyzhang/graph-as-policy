# ── Grasp parameters (pickup.py / _ring.py) ────────────────────────────────────
# Strategy: ring-grasp from above with DOWN_QUAT.
# DOWN: TCP z-axis → world −Z (top-down approach, fingers spread along world Y).
DOWN_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# The pickup revolve (GRASP_RING_ANGLE_DEG) and the receiver revolve
# (RECV_GRASP_ANGLE_DEG, in tsh-handover) are a COUPLED pair: rotating the
# pickup moves the giver's finger around the presented ring, so the receiver
# must shift by the matching amount to keep threading the OPEN side (else it
# threads into the giver → drop). Both validated; flip GRASP_PRESET to switch:
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
GRASP_LIFT_CLEARANCE = 0.085  # m — lift the grasped tape this far ABOVE its perceived
                              # grasp height (grasp_z + clearance). Relative, so it can't
                              # silently break if the table height changes — an absolute
                              # world-Z target would.

# ── Gripper actuation timing (pickup.py) ───────────────────────────────────────
# ramp then settle so the light ring is not kicked as the jaws seat on close.
GRASP_CLOSE_SETTLE_STEPS = 18   # sim steps to settle the giver grip after the ramp
GRASP_CLOSE_RAMP_STEPS   = 12   # sim steps to ramp the jaws gently closed

# ── Perception passthrough (_ring.py reads the perceived cloud's top slab) ────
PERCEIVE_TOP_SLAB = 0.006  # m — top-face slab whose mean gives the (x, y) centre

# ── Ring radii sanity band (_ring.py) ──────────────────────────────────────────
# Radii come from tsh-ring-geometry; outside this band _ring.py falls back to
# the tuned GRASP_RING_* defaults rather than trusting a noisy cloud.
RING_HOLE_PCTILE = 2   # inner-hole radius (robust to points leaking into the hole)
RING_RIM_PCTILE  = 98  # outer-rim radius (robust to edge outliers)
RING_HOLE_MIN   = 0.015  # m — reject implausibly small holes
RING_HOLE_MAX   = 0.080  # m — reject implausibly large holes
RING_RIM_MARGIN = 0.005  # m — rim must exceed the hole by at least this
