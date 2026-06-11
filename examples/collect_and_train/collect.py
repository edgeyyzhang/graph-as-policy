#!/usr/bin/env python
"""Collect demonstrations by running a gap graph as a scripted policy.

Each rollout executes the quickstart pick-and-place graph on a fresh
variation seed while a :class:`gap.connector.collector.DataCollector`
records synchronized observation/action/reward triples at every control
step into HDF5 (LeRobot-convertible layout — see the README).

    python collect.py --episodes 25 --out demos.hdf5 \
        --graph ../libero_quickstart/graph
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gap
from gap.connector.collector import DataCollector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--out", default="demos.hdf5")
    ap.add_argument(
        "--graph",
        default=str(Path(__file__).parent.parent / "libero_quickstart" / "graph"),
    )
    ap.add_argument("--skills", default=None,
                    help="open-robot-skills checkout path (default: auto-discovered)")
    ap.add_argument("--suite", default="libero_object_all_variance")
    ap.add_argument("--task", type=int, default=0)
    args = ap.parse_args()

    kept = 0
    with gap.connector.sim("libero", task=f"{args.suite}/{args.task}") as conn:
        collector = DataCollector(conn, args.out)
        try:
            for episode in range(args.episodes):
                seed = episode + 1  # seeds index the suite's baked variations
                conn.reset(seed=seed)
                collector.start_episode()
                result = gap.execute(
                    args.graph, conn, skills=args.skills, checkpoints="warn"
                )
                success, reward = conn.check_success()
                collector.end_episode(success=success)
                kept += int(success)
                print(
                    f"[collect] episode {episode + 1}/{args.episodes} "
                    f"seed={seed} success={success} reward={reward:.2f} "
                    f"graph_exit={result.exit_status} kept={kept}"
                )
        finally:
            collector.close()
    print(f"[collect] wrote {kept} successful / {args.episodes} total episodes → {args.out}")
    print("[collect] filter to successes during conversion (see README).")


if __name__ == "__main__":
    main()
