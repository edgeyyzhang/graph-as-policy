# ── Perception (_perceive.py, _perceive_cv.py) ──────────────────────────────
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
                               # with the top face to derive the object half-thickness

# ── Classical-CV segmentation (_perceive_cv.py) ─────────────────────────────────
# Parameters of the colour+height segmenter — like the percentiles above, these
# tune ROBUST estimators, not scene positions. The table plane itself is
# re-estimated from depth on every call.
CV_TABLE_BIN        = 0.005  # m — height-histogram bin; the modal bin is the table plane
CV_MIN_ABOVE_TABLE  = 0.004  # m — below this counts as table/shadow, not object
CV_MAX_ABOVE_TABLE  = 0.20   # m — above this is arms/gate, never a resting object
CV_MIN_BLOB_PX      = 300    # px — connected components smaller than this are speckle
CV_ACHROMATIC_MAX_SAT = 60   # HSV S (0-255) ceiling for "gray"/"white" queries
CV_ACHROMATIC_MIN_VAL = 90   # HSV V (0-255) floor for "gray"/"white" queries
                             # (keeps dark shadows/cavities out of the grey mask)

# Table footprint (world XY), from the scene's own table_plane geom
# (pos=(0.6,0,0.75), size=(0.2975,0.65,0.01) in
# libero_yam_tabletop_base_style.xml) -- a physical fact of the scene, not a
# tuned position. A resting object is always within this box; the robot arms
# reach through it but their own links extend well outside it (confirmed: a
# stray achromatic arm-link blob measured at world x=0.123, well outside
# [0.3025,0.8975]). Filters candidate blobs before the largest-blob tie-break
# so an arm segment can't out-count a real (thin-ring) object.
CV_TABLE_X = (0.3025, 0.8975)  # m
CV_TABLE_Y = (-0.65, 0.65)     # m
