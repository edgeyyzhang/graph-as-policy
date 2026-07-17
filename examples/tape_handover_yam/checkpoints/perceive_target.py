"""Checkpoints for `perceive_target` — perception vs privileged pose.

The predicate anchors on privileged state (the sim body pose) and compares
the subgraph's OUTPUT against it — never on camera-derived features (that
would be circular: perception validating itself).
"""

import numpy as np

from gap.runtime.verify import Checkpoint

TAPE = "yellow_tape_1"


def _xy(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"])])
    return np.array([float(v[0]), float(v[1])])


def _perceived_within_5cm(w, outputs) -> bool:
    got = outputs.get("target_xyz")
    if got is None:
        return False
    return float(np.linalg.norm(_xy(got) - w.body(TAPE).position[:2])) < 0.05


def _diag(w, outputs) -> dict:
    got = outputs.get("target_xyz")
    gt = w.body(TAPE).position
    return {
        "perceived_xyz": got if not isinstance(got, dict) else [got["x"], got["y"], got["z"]],
        "gt_xyz": [float(v) for v in gt],
        "xy_error_m": (float(np.linalg.norm(_xy(got) - gt[:2]))
                       if got is not None else None),
    }


def _half_z_plausible(w, outputs) -> bool:
    hz = outputs.get("target_half_z")
    return hz is not None and 0.005 < float(hz) < 0.040


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="target_perceived_within_5cm",
        subgraph="perceive_target",
        predicate=_perceived_within_5cm,
        diagnostics_fn=_diag,
        rationale="the perceived tape grasp point matches the privileged "
                  "body pose to 5 cm in XY — the accuracy the ring grasp "
                  "needs to land a finger in the hole",
        validate=True,
    ),
    Checkpoint(
        name="target_half_z_plausible",
        subgraph="perceive_target",
        predicate=_half_z_plausible,
        rationale="derived half-thickness inside the tape sanity band "
                  "(diagnostic probe; the scripts enforce their own band)",
        validate=False,
    ),
]
