# Strategy: ring-grasp from above with DOWN_QUAT.
# DOWN: TCP z-axis → world −Z (top-down approach, fingers spread along world Y).
DOWN_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# place target: the placed TAPE centre = perceived duct TOP-face centre + PLACE_OFFSET
# (an optional XY/Z nudge on top of the flush pose; default: centred, no nudge).
PLACE_OFFSET = (0.0, 0.00, 0.00)  # centred on the duct (stacking on top, no rim to dodge)
# curobo plans the descent in joint space; hover = place + z_approach.
PLACE_Z_APPROACH = 0.08

# Yaw deltas (deg) about the tape's vertical axis, tried in order until the place
# hover (the most reach-constrained waypoint) is plannable. The round tape lays flat
# at any yaw, so the sweep only trades reach.
PLACE_YAW_SWEEP_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                       110, -110, 130, -130, 150, -150, 180)
