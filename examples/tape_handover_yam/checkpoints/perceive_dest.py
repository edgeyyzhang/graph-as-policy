"""Checkpoints for `perceive_dest` — destination perception vs privileged pose."""

import numpy as np

from gap.runtime.verify import Checkpoint

DUCT = "duct_tape_1"


def _xy(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"])])
    return np.array([float(v[0]), float(v[1])])


def _perceived_within_5cm(w, outputs) -> bool:
    got = outputs.get("dest_xyz")
    if got is None:
        return False
    # dest_xyz is the TOP-face centre; compare XY only (Z differs from the
    # body centroid by the duct's half height).
    return float(np.linalg.norm(_xy(got) - w.body(DUCT).position[:2])) < 0.05


def _diag(w, outputs) -> dict:
    got = outputs.get("dest_xyz")
    gt = w.body(DUCT).position
    return {
        "perceived_xyz": got if not isinstance(got, dict) else [got["x"], got["y"], got["z"]],
        "gt_xyz": [float(v) for v in gt],
        "xy_error_m": (float(np.linalg.norm(_xy(got) - gt[:2]))
                       if got is not None else None),
    }


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="dest_perceived_within_5cm",
        subgraph="perceive_dest",
        predicate=_perceived_within_5cm,
        diagnostics_fn=_diag,
        rationale="the perceived destination top-face centre matches the "
                  "privileged duct pose to 5 cm in XY — the place lands "
                  "inside the duct footprint at that accuracy",
        validate=True,
    ),
]
