# ── Gripper actuation timing (pickup.py) ───────────────────────────────────────
# ramp then settle so the light ring is not kicked as the jaws seat on close.
# (Grasp-pose geometry now lives in calculate-grasp-ring, not here.)
GRASP_CLOSE_SETTLE_STEPS = 18   # sim steps to settle the giver grip after the ramp
GRASP_CLOSE_RAMP_STEPS   = 12   # sim steps to ramp the jaws gently closed
