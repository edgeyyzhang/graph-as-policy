"""Run the whole zero-shot prompt matrix, one subprocess per case.

    MUJOCO_GL=egl uv run python \
        examples/zero_shot_prompts/run_all.py \
        --out outputs/zero_shot_prompts/run1

Each case runs in a fresh subprocess (own CUDA/EGL context) under a
wall-clock timeout. On timeout the whole process GROUP is killed —
the tester matrix includes a prompt whose baseline is a consistent
mid-grasp hang, so hangs are an expected, first-class outcome: a
``timeout_marker.json`` is dropped and the case is re-scored from its
flushed partial trace (``--re-eval``), which recovers the hang site
(last in-flight tool call) from ``events.jsonl``.

Re-running with the same ``--out`` skips cases that already have a
``result.json`` (resume); ``--force`` re-runs everything.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cases import CASES, Case  # noqa: E402

REPO_ROOT = _HERE.parents[1]


def _spawn(case: Case, args: argparse.Namespace, out_root: Path,
           extra: list[str]) -> int:
    cmd = [
        sys.executable, str(_HERE / "run_prompt.py"),
        "--case", case.id, "--out", str(out_root),
        "--provider", args.provider, "--model", args.model,
    ] + extra
    if args.no_video:
        cmd.append("--no-video")
    log_path = out_root / case.id / "console.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n===== {' '.join(cmd)} @ {time.strftime('%F %T')}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(REPO_ROOT),
        )
        timeout_s = 60 * (args.timeout_min or case.timeout_min)
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            (out_root / case.id / "timeout_marker.json").write_text(
                json.dumps({
                    "timed_out": True,
                    "timeout_min": timeout_s // 60,
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }) + "\n")
            return -9


def run_one(case: Case, args: argparse.Namespace, out_root: Path) -> str:
    case_dir = out_root / case.id
    result_path = case_dir / "result.json"
    if result_path.exists() and not args.force:
        return "cached"
    marker = case_dir / "timeout_marker.json"
    if marker.exists():
        marker.unlink()

    extra = ["--force-generate"] if args.force else []
    code = _spawn(case, args, out_root, extra)
    if code == -9 or not result_path.exists():
        # killed (hang) or died before writing the aggregate: score the
        # partial traces post-mortem in a fresh process.
        _spawn(case, args, out_root, ["--re-eval"])
        return "timeout" if code == -9 else f"exit={code} (re-evaled)"
    return f"exit={code}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _load(out_root: Path) -> list[dict]:
    rows = []
    for case in CASES.values():
        p = out_root / case.id / "result.json"
        if not p.exists():
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            rows.append({"case": {"id": case.id},
                         "verdict": "UNREADABLE", "score": 0.0, "seeds": []})
    return rows


def write_report(out_root: Path) -> Path:
    rows = _load(out_root)
    lines = [
        "# Zero-shot prompt matrix — measured results",
        "",
        f"- output root: `{out_root}`",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- cases with results: {len(rows)}/{len(CASES)}",
        "",
        "| case | verdict | score | per-seed | summary |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        cid = r.get("case", {}).get("id", "?")
        seeds = r.get("seeds", [])
        per_seed = " ".join(
            f"s{p.get('seed')}:{p.get('evaluation', {}).get('verdict', '?')}"
            for p in seeds
        ) or "—"
        summary = (seeds and str(
            seeds[-1].get("evaluation", {}).get("summary", ""))
            or str(r.get("generation", {}).get("error", ""))[:120])
        lines.append(
            f"| {cid} | {r.get('verdict')} | "
            f"{float(r.get('score', 0.0)):.2f} | {per_seed} | "
            f"{summary.replace('|', '/')[:140]} |"
        )
    n_pass = sum(r.get("verdict") == "PASS" for r in rows)
    lines += ["", f"**{n_pass}/{len(rows)} passed**", ""]
    for r in rows:
        c = r.get("case", {})
        lines += [
            f"## {c.get('id', '?')}",
            "",
            f"- prompt: “{c.get('prompt', '')}”",
            f"- tester baseline: {c.get('baseline', '')}",
            f"- criterion: {c.get('criterion', '')}",
            f"- verdict: **{r.get('verdict')}** "
            f"(score {float(r.get('score', 0.0)):.2f})",
        ]
        gen = r.get("generation", {})
        if gen:
            lines.append(
                f"- generation: {gen.get('duration_s', '?')} s"
                + (", cached" if gen.get("cached") else "")
                + (f", ERROR: {str(gen.get('error'))[:200]}"
                   if gen.get("error") else "")
            )
        st = r.get("generated_graph_structure", {})
        if st.get("available"):
            lines.append(
                f"- generated graph: top-level cycle="
                f"{st.get('top_level_has_cycle')}, subgraph cycles="
                f"{st.get('subgraph_has_cycle')}"
            )
        for p in r.get("seeds", []):
            ev = p.get("evaluation", {})
            lines.append(
                f"- seed {p.get('seed')}: **{ev.get('verdict')}** "
                f"({float(ev.get('score', 0.0)):.2f}) — "
                f"{ev.get('summary', '')}"
            )
            for chk in ev.get("checks", []):
                mark = "✔" if chk.get("passed") else "✘"
                lines.append(
                    f"  - {mark} `{chk.get('name')}`: {chk.get('detail')}")
            diag = ev.get("diagnostics", {})
            if diag.get("hang"):
                lines.append(f"  - hang site: `{diag['hang']}`")
            if diag.get("planner_errors"):
                lines.append(
                    f"  - planner errors: {len(diag['planner_errors'])} "
                    f"(first: {diag['planner_errors'][0]})")
            if diag.get("max_node_iterations"):
                lines.append(
                    f"  - max node revisits observed at runtime: "
                    f"{diag['max_node_iterations']}")
        lines.append("")
    report = out_root / "report.md"
    report.write_text("\n".join(lines))
    (out_root / "report.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", default=None,
                    help=f"comma-separated ids (default all): "
                         f"{','.join(CASES)}")
    ap.add_argument("--provider",
                    default=os.environ.get("GAP_LLM_PROVIDER", "vertex"))
    ap.add_argument("--model",
                    default=os.environ.get("GAP_LLM_MODEL",
                                           "gemini-3.1-pro-preview"))
    ap.add_argument("--timeout-min", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out or
                    f"outputs/zero_shot_prompts/{time.strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        print(f"report: {write_report(out_root)}")
        return 0

    if args.cases:
        ids = args.cases.split(",")
        unknown = [c for c in ids if c not in CASES]
        if unknown:
            print(f"unknown case(s): {unknown}; have {sorted(CASES)}")
            return 2
    else:
        ids = list(CASES)

    print(f"matrix: {len(ids)} case(s) -> {out_root}")
    for i, cid in enumerate(ids, 1):
        case = CASES[cid]
        t0 = time.time()
        print(f"[{i}/{len(ids)}] {cid} "
              f"(timeout {args.timeout_min or case.timeout_min} min) ... ",
              end="", flush=True)
        status = run_one(case, args, out_root)
        result = {}
        rp = out_root / cid / "result.json"
        if rp.exists():
            try:
                result = json.loads(rp.read_text())
            except Exception:
                pass
        print(f"{status} {result.get('verdict', '?')} "
              f"score={float(result.get('score', 0.0)):.2f} "
              f"({(time.time() - t0) / 60:.1f} min)")

    report = write_report(out_root)
    print(f"report: {report}")
    rows = _load(out_root)
    print(f"passed {sum(r.get('verdict') == 'PASS' for r in rows)}"
          f"/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
