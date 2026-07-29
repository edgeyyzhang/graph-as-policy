# curobo plans the descent in joint space; hover = place + z_approach (the
# preceding transport-held-with-object node's approach target).
PLACE_Z_APPROACH = 0.08
PLACE_DROP_CLEARANCE = 0.02  # release this far above the rest pose and let the tape DROP the
                             # last stretch: the receiver's finger (threaded in the hole) slips
                             # out during the fall and the gripper is clear before it retracts,
                             # so the straight-up retract can't hook and lift the ring.
