#!/usr/bin/env python
"""Parallel launcher for benchmark.py: shards the yellow-anchor list across
multiple GPUs, running one benchmark.py subprocess per shard concurrently.

Each shard sweeps its yellow-anchor slice against the FULL duct-anchor list
(benchmark.py's own cross product), so the shards partition the full
yellow x duct grid with no overlap and no gaps. All shards append to one
shared results.jsonl — benchmark.py already appends line-buffered, so
concurrent writers are safe (see its module docstring).

Usage:
    uv run --no-sync python examples/tape_handover_yam/parallel_benchmark.py \
        examples/tape_handover_yam/generated/task_00 --gpus 1,3,4,5 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Mirrors benchmark.py's _ANCHOR_GRID/_YELLOW_ANCHORS/_DUCT_ANCHORS (keep in
# sync with that file, including the (0.60, 0.50) pickup dead-zone exclusion).
# Duplicated rather than imported so this launcher stays a lightweight
# subprocess-orchestrator — it never touches mujoco/gap/curobo itself, only
# the benchmark.py children do.
_ANCHOR_GRID = [(x, y) for x in (0.40, 0.45, 0.50, 0.55, 0.60)
                for y in (0.25, 0.375, 0.50)]
_YELLOW_ANCHORS = [a for a in _ANCHOR_GRID if a != (0.60, 0.50)]
_DUCT_ANCHORS = [(x, -y) for x, y in _ANCHOR_GRID]


def _fmt_anchors(anchors: list[tuple[float, float]]) -> str:
    return ",".join(f"{x:.4f}:{y:.4f}" for x, y in anchors)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph_dir",
                    help="agent-authored graph dir (e.g. generated/task_00; see GENERATE.md)")
    ap.add_argument("--gpus", required=True,
                    help="comma-separated CUDA device ids to shard across, e.g. 1,3,4,5")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--record-extra", action="store_true")
    ap.add_argument("--trial-timeout-s", type=float, default=None)
    ap.add_argument("--pairs-file", default=None,
                    help="JSON list of {yellow_xy, duct_xy[, golden]} anchor "
                         "pairs (e.g. position_splits.json) to shard across "
                         "GPUs instead of the full yellow x duct grid")
    ap.add_argument("--skip-golden", action="store_true",
                    help="with --pairs-file, drop entries with golden=true "
                         "(already-collected/verified pairs)")
    args = ap.parse_args(argv)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("[parallel] --gpus must list at least one device", file=sys.stderr)
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "benchmark_runs" / f"{ts}_parallel"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.touch()

    pairs_file = None
    if args.pairs_file:
        raw_pairs = json.loads(Path(args.pairs_file).read_text())
        if args.skip_golden:
            raw_pairs = [p for p in raw_pairs if not p.get("golden")]
        # Round-robin the pairs themselves across GPUs — an arbitrary pair
        # subset (e.g. a densified grid's new pairs) isn't a yellow x duct
        # rectangle, so it can't be sharded via YELLOW_ANCHORS alone.
        shards = [raw_pairs[i::len(gpus)] for i in range(len(gpus))]
        n_trials = len(raw_pairs) * args.seeds
    else:
        # Round-robin the yellow anchors across GPUs; each shard crosses its
        # slice against the full duct list, partitioning the yellow x duct grid.
        shards = [_YELLOW_ANCHORS[i::len(gpus)] for i in range(len(gpus))]
        n_trials = len(_YELLOW_ANCHORS) * len(_DUCT_ANCHORS) * args.seeds

    procs, log_files = [], []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        log_path = out_dir / f"gpu{gpu}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["RESULTS_PATH"] = str(results_path)
        if args.trial_timeout_s is not None:
            env["TRIAL_TIMEOUT_S"] = str(args.trial_timeout_s)
        # graph_dir is relative to the repo root (how benchmark.py is normally
        # invoked: `cd graph-as-policy && python examples/.../benchmark.py
        # examples/.../generated/task_00`) — so the child must run with that
        # same cwd, not HERE, or its own relative graph_dir resolves wrong.
        cmd = ["uv", "run", "--no-sync", "python", str(HERE / "benchmark.py"), args.graph_dir,
               "--seeds", str(args.seeds), "--out", str(out_dir)]
        if args.record_extra:
            cmd.append("--record-extra")
        if args.pairs_file:
            shard_path = out_dir / f"pairs_gpu{gpu}.json"
            shard_path.write_text(json.dumps(shard))
            cmd += ["--pairs-file", str(shard_path)]
            print(f"[parallel] gpu={gpu} pairs={len(shard)} -> {log_path}", flush=True)
        else:
            env["YELLOW_ANCHORS"] = _fmt_anchors(shard)
            env["DUCT_ANCHORS"] = _fmt_anchors(_DUCT_ANCHORS)
            print(f"[parallel] gpu={gpu} yellow_anchors={shard} -> {log_path}", flush=True)
        log_fh = open(log_path, "w")
        log_files.append(log_fh)
        procs.append(subprocess.Popen(cmd, cwd=HERE.parent.parent, env=env, stdout=log_fh, stderr=subprocess.STDOUT))

    exit_codes = [p.wait() for p in procs]
    for fh in log_files:
        fh.close()

    print(f"[parallel] all {len(procs)} shards finished (exit codes: {exit_codes})", flush=True)
    print(f"[parallel] results -> {results_path} ({n_trials} trials planned)", flush=True)

    config = {
        "graph_dir": args.graph_dir,
        "pairs_file": args.pairs_file,
        "yellow_anchors": _YELLOW_ANCHORS if not args.pairs_file else None,
        "duct_anchors": _DUCT_ANCHORS if not args.pairs_file else None,
        "seeds_per_pair": args.seeds,
        "gpus": gpus,
        "n_trials_planned": n_trials,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return 0 if all(c == 0 for c in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
