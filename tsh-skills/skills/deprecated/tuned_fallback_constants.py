"""Archived tuned fallback constants — NO LONGER USED by any active skill.

These are the hand-tuned "safety net" values that used to sit behind the
runtime-derived quantities in the tape-handover skills, in the pattern
``x = derived if derived is not None else CONST``. The skills now REQUIRE the
derived value and FAIL LOUD (raise) when a derivation returns None/degenerate,
rather than silently substituting one of these tuned numbers. They are kept here
only as a historical record of the values that were validated for the demo
LIBERO-YAM scene, and are imported by nobody.

If a derivation regresses and you need to compare against the old tuned value,
read it here — do not re-wire these into the active path.

Provenance: extracted from tsh-skills/skills/*/scripts/constants.py.
"""

# ── Tape half-thickness (was: tsh-perceive / tsh-place fallback) ──────────────
# Formerly the fallback for the cloud-DERIVED half-thickness ((top-bottom)/2),
# used when the cloud was too degenerate to measure. Measured from the scaled
# yellow_tape mesh.
TAPE_TOP_Z = 0.016  # m — fallback tape half-thickness

# ── Ring-grasp offsets (was: tsh-pickup fallback for missing gripper-geometry /
#    perceived hole+rim radii) ─────────────────────────────────────────────────
GRASP_RING_DX = 0.048  # m — rotated ~18° CW on ring; fingertip 10mm from centre in X
GRASP_RING_DY = 0.024  # m — rotated grip: near finger 17mm inside hole, far 2mm outside rim
GRIPPER_FINGERTIP_BEHIND = 0.038  # m — fingertip this far behind TCP along the approach axis

# ── Handover station geometry (was: tsh-handover fallback for missing
#    tsh-station-geometry) ─────────────────────────────────────────────────────
# Present the tape "o" face-on along world +X (ring normal ∥ X), both wrists
# splayed to opposite Y sides, both flat (no down-tilt).
GIVER_QUAT = (0.5, 0.5, 0.5, 0.5)  # gz=[1,0,0] — tool on pure +X, ring plane ∥ Y, no tilt
RECV_QUAT = (0.5358, 0.5358, 0.4614, 0.4614)  # gz=[0.99,-0.15,0] — ~8.5° splay toward -Y
MEET_XYZ = (0.586, 0.050, 1.032)  # world giver-TCP present point (demo giver TCP @ t=185)

# ── Place collision-body fallback (was: tsh-place fallback for missing/degenerate
#    tape_cloud) ────────────────────────────────────────────────────────────────
# Formerly used to approximate the held tape as a bounding sphere for the
# collision-aware approach when no perceived cloud (or <4 points) was available;
# the convex hull of the actual cloud is preferred and is now required.
PLACE_TAPE_SPHERE_RADIUS = 0.08  # m — bounding-sphere radius
PLACE_TAPE_SPHERE_SUBDIV = 2  # icosphere tessellation level
