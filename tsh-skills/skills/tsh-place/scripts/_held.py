"""Shared held-object ("tool-as-EE") pose helpers for tsh-place.

A held object is planned AS the end effector: the measured rigid offset of
the object centre in the holder's TCP frame (``held_offset`` — from the
pickup's pre-close FK anchor or the exchange's grab-instant measurement) is
composed relative to a target pose to get the actual TCP target
(``place.py`` does this once, for the final rest pose).

The collision-aware approach itself (attaching the object's perceived cloud
as a cuRobo collision body) now lives in ``tsh-transport-held``, composed as
a node ahead of this one — see ``tsh-place-pose`` for the reachable-pose probe
that feeds both.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def q_to_R(quat_wxyz) -> Rotation:
    qw, qx, qy, qz = quat_wxyz
    return Rotation.from_quat([qx, qy, qz, qw])


def as_vec3(v) -> np.ndarray:
    """Accept a Vec3 dict {x,y,z} (subgraph type coercion) or a sequence."""
    if isinstance(v, dict):
        return np.asarray([v["x"], v["y"], v["z"]], dtype=float)
    return np.asarray(list(v), dtype=float)


def as_wxyz(q) -> tuple:
    """Accept a Quaternion dict {w,x,y,z} or a wxyz sequence; return wxyz tuple."""
    return (q["w"], q["x"], q["y"], q["z"]) if isinstance(q, dict) else tuple(q)
