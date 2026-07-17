#!/usr/bin/env python
"""Summarize tape-handover benchmark results into a summary JSON + TSV.

Reads the per-trial jsonl that ``benchmark.py`` appends and produces:
  * ``summary.json`` — machine-readable rollup with per-position breakdown
  * ``summary.tsv``  — spreadsheet-friendly table
  * a human-readable block printed to stdout

Can be run at any time (even mid-benchmark) to get the current success rate.

Usage:
    python summarize_results.py [RESULTS_PATH]
    python summarize_results.py benchmark_runs/20260709_120000/results.jsonl
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS = _HERE / "benchmark_runs"


def _find_latest_results() -> Path | None:
    if not _DEFAULT_RESULTS.is_dir():
        return None
    runs = sorted(_DEFAULT_RESULTS.iterdir(), reverse=True)
    for d in runs:
        p = d / "results.jsonl"
        if p.exists():
            return p
    return None


def _parse_args(argv: list[str]) -> tuple[Path, Path | None]:
    results_path: Path | None = None
    out_path: Path | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            i += 1
            out_path = Path(argv[i]) if i < len(argv) else out_path
        elif a.startswith("--out="):
            out_path = Path(a.split("=", 1)[1])
        elif not a.startswith("-") and results_path is None:
            results_path = Path(a)
        i += 1
    if results_path is None:
        results_path = _find_latest_results()
    return results_path, out_path


def _load_rows(results_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(results_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _mean(xs: list[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    n_succ = sum(1 for r in rows if bool(r.get("success")))
    n_graph = sum(1 for r in rows if bool(r.get("graph_success")))
    success_rate = (n_succ / total) if total else 0.0
    graph_rate = (n_graph / total) if total else 0.0

    cells: dict[str, list[dict]] = {}
    for r in rows:
        ya = tuple(r.get("yellow_anchor", [0, 0]))
        da = tuple(r.get("duct_anchor", [0, 0]))
        key = f"{ya[0]:.2f},{ya[1]:.2f}|{da[0]:.2f},{da[1]:.2f}"
        cells.setdefault(key, []).append(r)

    per_position = []
    for key in sorted(cells.keys()):
        group = cells[key]
        ya = group[0].get("yellow_anchor", [0, 0])
        da = group[0].get("duct_anchor", [0, 0])
        s = sum(1 for r in group if bool(r.get("success")))
        per_position.append({
            "yellow_anchor": ya,
            "duct_anchor": da,
            "n_trials": len(group),
            "n_success": s,
            "success_rate": s / len(group) if group else 0.0,
            "mean_wall_s": round(_mean([r.get("wall_s", 0) for r in group]), 1),
        })

    return {
        "overall": {
            "n_trials": total,
            "n_success": n_succ,
            "success_rate": success_rate,
            "graph_success_rate": graph_rate,
            "mean_wall_s": round(_mean([r.get("wall_s", 0) for r in rows]), 1),
        },
        "per_position": per_position,
    }


def _write_tsv(summary: dict, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["yellow_x", "yellow_y", "duct_x", "duct_y",
                     "n_trials", "n_success", "success_rate", "mean_wall_s"])
        for p in summary["per_position"]:
            w.writerow([p["yellow_anchor"][0], p["yellow_anchor"][1],
                        p["duct_anchor"][0], p["duct_anchor"][1],
                        p["n_trials"], p["n_success"],
                        f"{p['success_rate']:.4f}", p["mean_wall_s"]])
        w.writerow([])
        o = summary["overall"]
        w.writerow(["OVERALL", "", "", "", o["n_trials"], o["n_success"],
                     f"{o['success_rate']:.4f}", o["mean_wall_s"]])


def _print_block(summary: dict, results_path: Path, out_path: Path) -> None:
    o = summary["overall"]
    print("=" * 60)
    print("tape-handover benchmark SUMMARY")
    print(f"  source: {results_path}")
    print("-" * 60)
    print(f"  total trials:    {o['n_trials']}")
    print(f"  success:         {o['n_success']}/{o['n_trials']}"
          + (f"  =  {o['success_rate']:.1%}" if o['n_trials'] else "  (no trials)"))
    print(f"  graph success:   {o['graph_success_rate']:.1%}")
    print(f"  mean wall/trial: {o['mean_wall_s']:.1f}s")
    print("-" * 60)
    print("  per-position (yellow -> duct):  success_rate  n  mean_wall")
    for p in summary["per_position"]:
        ya, da = p["yellow_anchor"], p["duct_anchor"]
        print(f"    ({ya[0]:.2f},{ya[1]:.2f})->({da[0]:.2f},{da[1]:.2f}):  "
              f"{p['success_rate']:6.1%}  n={p['n_trials']}  "
              f"{p['mean_wall_s']:.1f}s")
    print("-" * 60)
    print(f"  summary -> {out_path}")
    print("=" * 60)


def main(argv: list[str]) -> int:
    results_path, out_path = _parse_args(argv)
    if results_path is None or not results_path.exists():
        target = results_path or "benchmark_runs/*/results.jsonl"
        print(f"[summarize] no results file at {target} — run benchmark.py first.",
              file=sys.stderr)
        return 1

    out_dir = out_path or results_path.parent
    out_json = out_dir / "summary.json" if out_dir.is_dir() else out_dir
    out_tsv = out_json.with_suffix(".tsv")

    rows = _load_rows(results_path)
    summary = summarize(rows)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    _write_tsv(summary, out_tsv)
    _print_block(summary, results_path, out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
