# Route probe constants. Grasp-pose geometry now lives in
# tsh-calculate-grasp-ring (route consumes its poses); only the place-reach
# probe + the predicted receiver offset are route's own.

# DOWN: TCP z-axis → world −Z (top-down); the place-hover yaw sweep pivots this.
DOWN_QUAT = (0.0, 0.7071067811865476, 0.7071067811865476, 0.0)

# ── Receiver grasp reach probe (predicted, radii-derived) ──────────────────────
# recv_grasp_dy = rim_radius + RECV_GRASP_DY_MARGIN — see tsh-handover for the
# full rationale; route probes the non-picking arm's place reach with this.
RECV_GRASP_DY_MARGIN = 0.0105  # m — added to the perceived rim_radius
RECV_GRASP_DZ        = 0.045   # m — raise the TCP so the fingertips reach hole height

# ── Place reach probe ──────────────────────────────────────────────────────────
PLACE_Z_APPROACH = 0.08  # m — curobo plans the descent in joint space; hover = place + this
# Yaw deltas (deg) about the tape's vertical axis, tried in order until the place
# hover (the most reach-constrained waypoint) is plannable.
PLACE_YAW_SWEEP_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
                       110, -110, 130, -130, 150, -150, 180)
