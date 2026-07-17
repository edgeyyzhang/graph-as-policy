#!/usr/bin/env python
"""Train/test split over the benchmark.py anchor-pair grid, for policy
training data collection.

Split is an INTERPOLATION holdout at the anchor-PAIR level: a random subset
of pairs is reserved as test-only and never appears in training data
collection. This is orthogonal to (and composes with) the per-trial jitter
(_JITTER_M) already applied in benchmark.py/parallel_benchmark.py when
collecting demos at each anchor — jitter tests robustness to small
perturbations around a trained position; this split tests generalization to
an anchor pair never trained on at all. Collect training demos only for the
"train" pairs below; hold "test" pairs out until evaluation.

The original 3x3 grid (72 pairs, 100% success in benchmark_runs/
20260716_215711_parallel) is marked "golden" and always placed in train —
it's the pre-verified core; only the newly-densified pairs are eligible for
the test split.

Usage:
    python examples/tape_handover_yam/make_position_splits.py
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Mirrors benchmark.py's _ANCHOR_GRID/_YELLOW_ANCHORS/_DUCT_ANCHORS (keep in
# sync with that file). Duplicated rather than imported so this stays a
# lightweight, no-GPU script — importing benchmark.py pulls in mujoco/gap.
_ANCHOR_GRID = [(x, y) for x in (0.40, 0.45, 0.50, 0.55, 0.60)
                for y in (0.25, 0.375, 0.50)]
_YELLOW_ANCHORS = [a for a in _ANCHOR_GRID if a != (0.60, 0.50)]
_DUCT_ANCHORS = [(x, -y) for x, y in _ANCHOR_GRID]

_GOLDEN_X = (0.40, 0.50, 0.60)
_GOLDEN_Y = (0.25, 0.375, 0.50)

_TEST_FRAC = 0.20
_SEED = 0


def _is_golden(yellow_xy: tuple, duct_xy: tuple) -> bool:
    yx, yy = yellow_xy
    dx, dy = duct_xy
    return yx in _GOLDEN_X and yy in _GOLDEN_Y and dx in _GOLDEN_X and -dy in _GOLDEN_Y


def main() -> int:
    pairs = list(itertools.product(_YELLOW_ANCHORS, _DUCT_ANCHORS))
    golden = [p for p in pairs if _is_golden(*p)]
    new = [p for p in pairs if not _is_golden(*p)]

    rng = random.Random(_SEED)
    shuffled = new[:]
    rng.shuffle(shuffled)
    n_test = round(len(shuffled) * _TEST_FRAC)
    test, train_new = shuffled[:n_test], shuffled[n_test:]

    train = golden + train_new

    def _row(yellow_xy, duct_xy, split):
        return {"yellow_xy": list(yellow_xy), "duct_xy": list(duct_xy),
                "golden": _is_golden(yellow_xy, duct_xy), "split": split}

    rows = ([_row(y, d, "train") for y, d in train] +
            [_row(y, d, "test") for y, d in test])

    out_path = HERE / "position_splits.json"
    out_path.write_text(json.dumps(rows, indent=2))

    print(f"positions: {len(pairs)} total ({len(golden)} golden + {len(new)} new)")
    print(f"train: {len(train)} ({len(golden)} golden + {len(train_new)} new)")
    print(f"test:  {len(test)} (interpolation holdout, new pairs only)")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
