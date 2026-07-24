# Tape half-thickness (top face above the body centroid). DERIVED at runtime from
# the perceived cloud — (robust top − robust bottom)/2 in perceive_tape. Outside the
# sanity band below, perceive_tape RAISES (degenerate cloud, no tuned fallback).
TAPE_HALF_MIN = 0.005  # m — sanity band on the perceived half-thickness; outside it,
TAPE_HALF_MAX = 0.040  # m   perceive_tape RAISES (degenerate cloud, no tuned fallback)

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
                               # otherwise poisons the top face.
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
