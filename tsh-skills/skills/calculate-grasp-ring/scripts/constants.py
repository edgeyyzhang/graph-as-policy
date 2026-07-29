# Ring-grasp geometry constants (calculate_grasp_ring.py).
# Strategy: ring-grasp from above with DOWN_QUAT.
# DOWN: TCP z-axis → world −Z (top-down approach, fingers spread along world Y).
DOWN_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# The pickup revolve (GRASP_RING_ANGLE_DEG) and the receiver revolve
# (RECV_GRASP_ANGLE_DEG, in bimanual-handover) are a COUPLED pair: rotating the
# pickup moves the giver's finger around the presented ring, so the receiver
# must shift by the matching amount to keep threading the OPEN side. Flip
# GRASP_PRESET to switch both together (see bimanual-handover for the full rationale).
GRASP_PRESET = "rotated"
_GRASP_ANGLE_PRESETS = {"rotated": (45.0, 0.0), "baseline": (0.0, 45.0)}  # (pickup, receiver)
GRASP_RING_ANGLE_DEG = _GRASP_ANGLE_PRESETS[GRASP_PRESET][0]  # deg — pickup revolve CW about the
                             # vertical hole axis (set by GRASP_PRESET above).
GRASP_PRE_DZ  = 0.07    # m — hover height above grasp point before descent
GRASP_HOOK_DZ = -0.005  # m — deep seat: finger pads centered on the 32mm wall/inner rim.
GRASP_LIFT_CLEARANCE = 0.085  # m — lift the grasped tape this far ABOVE its perceived
                              # grasp height (grasp_z + clearance). Relative, so it can't
                              # silently break if the table height changes.

# ── Ring radii estimation (folded in from tsh-ring-geometry) ───────────────────
# Radii are measured on the cloud's TOP-FACE SLAB (2nd pctile = hole edge,
# 98th = rim edge; both robust to depth noise). Outside the sanity band the
# node RAISES (degenerate cloud, no tuned fallback) — the failure routes to
# on_error where the graph can re-perceive or abort.
PERCEIVE_TOP_SLAB = 0.006  # m — top-face slab whose radial spread gives the radii
RING_HOLE_PCTILE = 2   # inner-hole radius (robust to points leaking into the hole)
RING_RIM_PCTILE  = 98  # outer-rim radius (robust to edge outliers)
RING_HOLE_MIN   = 0.015  # m — reject implausibly small holes
RING_HOLE_MAX   = 0.080  # m — reject implausibly large holes
RING_RIM_MARGIN = 0.005  # m — rim must exceed the hole by at least this
