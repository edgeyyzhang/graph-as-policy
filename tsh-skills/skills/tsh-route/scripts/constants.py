# ── Grasp parameters (_ring.py) ─────────────────────────────────────────────────
# Strategy: ring-grasp from above with DOWN_QUAT.
# DOWN: TCP z-axis → world −Z (top-down approach, fingers spread along world Y).
DOWN_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# The pickup revolve (GRASP_RING_ANGLE_DEG) and the receiver revolve
# (RECV_GRASP_ANGLE_DEG, in tsh-handover) are a COUPLED pair — flip GRASP_PRESET
# to switch both angles together (see tsh-pickup for the full rationale).
GRASP_PRESET = "rotated"
_GRASP_ANGLE_PRESETS = {"rotated": (45.0, 0.0), "baseline": (0.0, 45.0)}  # (pickup, receiver)
GRASP_RING_ANGLE_DEG = _GRASP_ANGLE_PRESETS[GRASP_PRESET][0]  # deg — pickup revolve CW about the
                             # vertical hole axis (set by GRASP_PRESET above).
GRASP_PRE_DZ  = 0.07    # m — hover height above grasp point before descent
GRASP_HOOK_DZ = -0.005 # m — deep seat: finger pads centered on the 32mm wall/inner rim.
GRASP_LIFT_CLEARANCE = 0.085  # m — lift the grasped tape this far ABOVE its perceived
                              # grasp height (grasp_z + clearance).

# ── Perception passthrough (_ring.py reads the perceived cloud's top slab) ────
PERCEIVE_TOP_SLAB = 0.006  # m — top-face slab whose mean gives the (x, y) centre

# ── Ring radii sanity band (_ring.py) ──────────────────────────────────────────
RING_HOLE_PCTILE = 2   # inner-hole radius (robust to points leaking into the hole)
RING_RIM_PCTILE  = 98  # outer-rim radius (robust to edge outliers)
RING_HOLE_MIN   = 0.015  # m — reject implausibly small holes
RING_HOLE_MAX   = 0.080  # m — reject implausibly large holes
RING_RIM_MARGIN = 0.005  # m — rim must exceed the hole by at least this

# ── Receiver grasp reach probe (route.py) ──────────────────────────────────────
# recv_grasp_dy = rim_radius + RECV_GRASP_DY_MARGIN — see tsh-handover for the
# full rationale; route probes reachability with the same offset.
RECV_GRASP_DY_MARGIN = 0.0105  # m — added to the perceived rim_radius
RECV_GRASP_DZ        = 0.045   # m — raise the TCP so the fingertips reach hole height

# ── Place reach probe (route.py) ───────────────────────────────────────────────
PLACE_Z_APPROACH = 0.08  # m — curobo plans the descent in joint space; hover = place + this
# Yaw deltas (deg) about the tape's vertical axis, tried in order until the place
# hover (the most reach-constrained waypoint) is plannable.
PLACE_YAW_SWEEP_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                       110, -110, 130, -130, 150, -150, 180)
