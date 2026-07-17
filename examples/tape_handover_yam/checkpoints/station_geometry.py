"""Checkpoints for `station_geometry` — derived meet point is plausible."""

from gap.runtime.verify import Checkpoint


def _as3(v):
    if isinstance(v, dict):
        return [float(v["x"]), float(v["y"]), float(v["z"])]
    return [float(x) for x in v]


def _meet_plausible(w, outputs) -> bool:
    meet = outputs.get("meet_xyz")
    if meet is None:
        return False
    x, y, z = _as3(meet)
    # Above the table, below max reach, near the arm-baseline midline.
    return 0.80 < z < 1.40 and abs(y) < 0.30 and 0.2 < x < 1.0


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        name="meet_point_plausible",
        subgraph="station_geometry",
        predicate=_meet_plausible,
        rationale="the derived rendezvous sits above the table between the "
                  "two arm bases — a bad base read lands it outside this box",
        validate=True,
    ),
]
