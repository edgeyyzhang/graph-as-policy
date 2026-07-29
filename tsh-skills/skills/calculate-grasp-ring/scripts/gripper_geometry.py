"""Emit the gripper's grasp offsets, derived from the robot model via FK.

Thin GaP node over :func:`_gripper_geometry.grip_geometry`: it loads the arm
model, sets the gripper aperture, and reads the fingertip contact geometry in the
``grasp_site`` (TCP) frame. The output ``gripper_offsets`` is what the ring-grasp
(pickup) consumes to place its fingers on the ring — self-configuring from the
URDF, so a different gripper needs no re-tuning.
"""

from __future__ import annotations

from typing import TypedDict

from gap import NodeContext

from ._gripper_geometry import grip_geometry


class Output(TypedDict):
    fingertip_axial: float   # fingertip offset ahead of the TCP along the approach (m)
    finger_half_gap: float   # half the open-aperture finger-pair spread (m)


def run(ctx: NodeContext) -> Output:
    """Derive the gripper grasp offsets from the arm model (no ground truth).

    Emits the two offsets the ring grasp consumes as individual scalars (the full
    ``grip_geometry`` dict — pad z-span, aperture — is available in the helper for
    a future seat-depth derivation).
    """
    g = grip_geometry()
    print(f"[gripper_geometry] fingertip_axial={g['fingertip_axial']*1000:.1f}mm "
          f"finger_half_gap={g['finger_half_gap']*1000:.1f}mm "
          f"pad_z=[{g['pad_z_min']*1000:.1f},{g['pad_z_max']*1000:.1f}]mm", flush=True)
    return {"fingertip_axial": g["fingertip_axial"],
            "finger_half_gap": g["finger_half_gap"]}
