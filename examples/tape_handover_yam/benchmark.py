#!/usr/bin/env python
"""Position-generalization benchmark for the bimanual tape-handover graph.

Sweeps the yellow (pickup) tape and duct (place) tape over their reachable
grids with small random jitter, running the given graph at each sampled
position. Results are appended one JSON line per trial to RESULTS_PATH so
concurrent / resumed runs can share the same file.

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
      uv run --no-sync python examples/tape_handover_yam/benchmark.py GRAPH_DIR

GRAPH_DIR is REQUIRED — point it at an agent-authored graph (e.g.
generated/task_00; see GENERATE.md).

Env knobs:
    SEEDS              jittered samples per anchor pair (default 5)
    TRIAL_TIMEOUT_S    soft per-trial wall watchdog via SIGALRM (default 300)
    RESULTS_PATH       jsonl append target (default benchmark_runs/<ts>/results.jsonl)
    RECORD_VIDEO       '1' -> record ALL trials (default), '0' -> record NONE
    VIDEO_DIR          mp4 dir (default benchmark_runs/<ts>/video)
    YELLOW_ANCHORS     override yellow anchors (comma-sep x:y pairs, e.g. "0.40:0.25,0.50:0.50")
    DUCT_ANCHORS       override duct anchors (same format; every yellow anchor is tried
                       against every duct anchor, so counts need not match)
"""

from __future__ import annotations

import itertools
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

import gap

from gap.connector.libero_yam import libero_yam as _libero_yam
from run import BDDL, OPEN_ROBOT_SKILLS, TSH_SKILLS, ArmCollisionMonitor, _free_camera

HERE = Path(__file__).resolve().parent

# ── skill registries (precedence: tsh-skills > open-robot-skills) ───────────
_SKILLS = [str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)]

# ── position grid ───────────────────────────────────────────────────────────
# 5x3: the original 3x3 (0.40/0.50/0.60 x 0.25/0.375/0.50) densified with
# midpoint x columns (0.45, 0.55). The midpoints were spot-checked reachable
# (6/6, both pickup and place roles, all 3 y rows) via a probe run before
# being folded in here — see git history for the probe results — and no
# interior dead zones are expected: reachability over this box is smooth
# except at the one known extreme corner below.
_ANCHOR_GRID = [(x, y) for x in (0.40, 0.45, 0.50, 0.55, 0.60)
                for y in (0.25, 0.375, 0.50)]
# (0.60, 0.50) is a confirmed pickup-side IK dead zone: the approach/hover
# waypoint above the tape lands at world (0.68, 0.51, 0.85), past the
# documented x=0.70 reach ceiling once combined with that much lateral
# offset — 0/6 across every duct pairing in the 81-trial sweep
# (`plan_tool_move: no plan for arm 0 to [0.68, 0.514, 0.853]`). Excluded
# from pickup only; the mirrored place position (0.60, -0.50) is unaffected
# (6/6) and stays in the duct grid.
_YELLOW_ANCHORS = [a for a in _ANCHOR_GRID if a != (0.60, 0.50)]
_DUCT_ANCHORS = [(x, -y) for x, y in _ANCHOR_GRID]
_JITTER_M = 0.02
_DEFAULT_SEEDS = 5

# ── env knobs ───────────────────────────────────────────────────────────────
_TRIAL_TIMEOUT_S = float(os.environ.get("TRIAL_TIMEOUT_S", "300"))
_RECORD_VIDEO = os.environ.get("RECORD_VIDEO", "1") != "0"


def _parse_anchors(env_key: str, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    pairs = []
    for tok in raw.split(","):
        x, y = tok.strip().split(":")
        pairs.append((float(x), float(y)))
    return pairs


class _TrialTimeout(BaseException):
    """Not an ``Exception`` subclass on purpose — mirrors gap_core's
    ``GuardLimitExceeded``: a bare ``except Exception`` inside a node/subgraph
    (which is exactly what ``on_error`` routing does) must not be able to
    silently absorb the watchdog and report a misleading graceful failure
    instead of a timeout.
    """


def _run_trial_guarded(graph_dir: str, yellow_xy: tuple[float, float],
                       duct_xy: tuple[float, float], video_dir: Path | None,
                       trial_id: str, timeout_s: float,
                       extra_dir: Path | None = None) -> dict:
    """Run one trial with a SIGALRM wall timeout. Returns a result dict."""

    def _on_alarm(signum, frame):
        raise _TrialTimeout()

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)

    os.environ["GAP_YELLOW_XY"] = f"{yellow_xy[0]:.4f},{yellow_xy[1]:.4f}"
    os.environ["GAP_DUCT_XY"] = f"{duct_xy[0]:.4f},{duct_xy[1]:.4f}"

    conn = None
    recorder = None
    ok = False
    graph_ok = False
    note = ""
    collision_report = ""
    video_path = None
    t0 = time.perf_counter()

    try:
        conn = _libero_yam(bddl=str(BDDL), cameras=["agentview", "left", "right"])
        conn.reset()
        env = conn.env

        frames: list = []
        frames_opp: list = []
        # --record-extra covers front(agentview)+wrist already, so skip these
        # two more free-camera renders every 8 steps under it — halves the
        # per-trial render cost on top of TrialRecorder's own overhead.
        record = _RECORD_VIDEO and video_dir is not None and extra_dir is None

        if record:
            renderer, cam = _free_camera(env.model, azimuth=10.0)
            renderer_opp, cam_opp = _free_camera(env.model, azimuth=190.0)
            back_wall = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "back_wall")

        collision = ArmCollisionMonitor(env.model)
        step_count = [0]

        if extra_dir is not None:
            from trial_recorder import TrialRecorder
            recorder = TrialRecorder(env, task_prompt=BDDL.stem.replace("_", " "))

        def on_step(e):
            collision(e)
            if recorder is not None:
                recorder.on_step(e)
            if not record:
                return
            step_count[0] += 1
            if step_count[0] % 8 == 0:
                renderer.update_scene(e.data, camera=cam)
                frames.append(renderer.render().copy())
                alpha = env.model.geom_rgba[back_wall, 3]
                env.model.geom_rgba[back_wall, 3] = 0.0
                renderer_opp.update_scene(e.data, camera=cam_opp)
                frames_opp.append(renderer_opp.render().copy())
                env.model.geom_rgba[back_wall, 3] = alpha

        env.on_step = on_step

        result = gap.execute(graph_dir, conn, skills=_SKILLS)
        graph_ok = bool(result.success)
        ok = bool(env.task_completed())
        if result.error:
            note = str(result.error)
        collision_report = collision.report().splitlines()[0]

        if record:
            video_dir.mkdir(parents=True, exist_ok=True)
            for name, buf in (("handover.mp4", frames), ("handover_opposite.mp4", frames_opp)):
                if buf:
                    path = video_dir / name
                    with imageio.get_writer(str(path), fps=30, codec="h264", quality=8) as w:
                        for f in buf:
                            w.append_data(f)
                    video_path = video_path or str(path)
            renderer.close()
            renderer_opp.close()

        if recorder is not None:
            recorder.save(extra_dir)

    except _TrialTimeout:
        note = "timeout"
        print(f"    [watchdog] trial exceeded {timeout_s:.0f}s -> record failed(timeout)",
              flush=True)
    except Exception as e:
        note = f"error:{e!r}"
        print(f"    [error] trial raised: {e!r}", flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        # Close the recorder's cached renderers BEFORE conn.close() tears down
        # the shared EGL display, and unconditionally (not just on the happy
        # path) — a timeout/exception mid-trial used to leave them open,
        # orphaning them against a display conn.close() was about to kill and
        # poisoning EGL for whatever the process runs next.
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    dt = time.perf_counter() - t0
    return {
        "trial_id": trial_id,
        "yellow_xy": [round(yellow_xy[0], 4), round(yellow_xy[1], 4)],
        "duct_xy": [round(duct_xy[0], 4), round(duct_xy[1], 4)],
        "success": ok,
        "graph_success": graph_ok,
        "wall_s": round(dt, 2),
        "note": note,
        "collision": collision_report,
        "video": video_path,
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph_dir",
                    help="agent-authored graph dir (e.g. generated/task_00; see GENERATE.md)")
    ap.add_argument("--seeds", type=int,
                    default=int(os.environ.get("SEEDS", str(_DEFAULT_SEEDS))))
    ap.add_argument("--out", default=None)
    ap.add_argument("--record-extra", action="store_true",
                    help="also save per-step joint/cartesian trajectories plus "
                         "front (agentview) and wrist (left/right) camera feeds "
                         "per trial, via trial_recorder.TrialRecorder — off by "
                         "default, independent of RECORD_VIDEO")
    ap.add_argument("--pairs-file", default=None,
                    help="JSON list of {yellow_xy, duct_xy[, golden]} anchor "
                         "pairs (e.g. position_splits.json) to run instead of "
                         "the full YELLOW_ANCHORS x DUCT_ANCHORS cross product "
                         "— use this to target an arbitrary subset (a "
                         "densified grid's new pairs are not a rectangle, so "
                         "the anchor-list override can't select them)")
    ap.add_argument("--skip-golden", action="store_true",
                    help="with --pairs-file, drop entries with golden=true "
                         "(already-collected/verified pairs)")
    args = ap.parse_args(argv)

    graph_dir = args.graph_dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else HERE / "benchmark_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs_file:
        raw_pairs = json.loads(Path(args.pairs_file).read_text())
        if args.skip_golden:
            raw_pairs = [p for p in raw_pairs if not p.get("golden")]
        anchor_pairs = [(tuple(p["yellow_xy"]), tuple(p["duct_xy"])) for p in raw_pairs]
    else:
        yellow_anchors = _parse_anchors("YELLOW_ANCHORS", _YELLOW_ANCHORS)
        duct_anchors = _parse_anchors("DUCT_ANCHORS", _DUCT_ANCHORS)
        anchor_pairs = list(itertools.product(yellow_anchors, duct_anchors))

    results_path = Path(os.environ.get(
        "RESULTS_PATH", str(out_dir / "results.jsonl")))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir = Path(os.environ.get("VIDEO_DIR", str(out_dir / "video")))
    extra_dir = out_dir / "extra" if args.record_extra else None

    rng = random.Random(0)
    plan: list[tuple[tuple, tuple, tuple, tuple]] = []
    for y_anchor, d_anchor in anchor_pairs:
        for _ in range(args.seeds):
            jx = y_anchor[0] + rng.uniform(-_JITTER_M, _JITTER_M)
            jy = y_anchor[1] + rng.uniform(-_JITTER_M, _JITTER_M)
            djx = d_anchor[0] + rng.uniform(-_JITTER_M, _JITTER_M)
            djy = d_anchor[1] + rng.uniform(-_JITTER_M, _JITTER_M)
            plan.append((y_anchor, (jx, jy), d_anchor, (djx, djy)))

    n_trials = len(plan)
    print(f"=== tape-handover benchmark: {len(anchor_pairs)} anchor pairs x "
          f"{args.seeds} seeds = {n_trials} trials ===",
          flush=True)
    print(f"    graph: {graph_dir}", flush=True)
    print(f"    skills: {_SKILLS}", flush=True)
    print(f"    results -> {results_path}   "
          f"trial_timeout={_TRIAL_TIMEOUT_S:.0f}s   "
          f"video={'ON' if _RECORD_VIDEO else 'OFF'}   "
          f"record_extra={'ON -> ' + str(extra_dir) if extra_dir else 'OFF'}", flush=True)

    results_fh = open(results_path, "a", buffering=1)

    n_succ = 0
    walls: list[float] = []
    t_all = time.perf_counter()

    try:
        for i, (y_anchor, yellow_xy, d_anchor, duct_xy) in enumerate(plan):
            tid = (f"trial_{i:03d}_yx{yellow_xy[0]:.2f}_yy{yellow_xy[1]:.2f}"
                   f"_dx{duct_xy[0]:.2f}_dy{duct_xy[1]:.2f}")
            print(f"\n--- trial {i + 1}/{n_trials} {tid} ---", flush=True)

            trial_video_dir = video_dir / tid if _RECORD_VIDEO else None
            trial_extra_dir = extra_dir / tid if extra_dir else None
            row = _run_trial_guarded(
                graph_dir, yellow_xy, duct_xy, trial_video_dir, tid,
                _TRIAL_TIMEOUT_S, extra_dir=trial_extra_dir)
            row["yellow_anchor"] = list(y_anchor)
            row["duct_anchor"] = list(d_anchor)

            results_fh.write(json.dumps(row) + "\n")
            results_fh.flush()

            walls.append(row["wall_s"])
            if row["success"]:
                n_succ += 1

            print(f"    -> success={row['success']} graph={row['graph_success']} "
                  f"wall={row['wall_s']:.1f}s | {row['collision']} "
                  f"note={row['note'] or '-'}", flush=True)
    finally:
        try:
            results_fh.close()
        except Exception:
            pass

    total_s = time.perf_counter() - t_all
    n_done = len(walls)

    print(f"\n=== RESULT ===", flush=True)
    if n_done:
        print(f"  trials: {n_done}   success: {n_succ}/{n_done} "
              f"({n_succ / n_done:.1%})", flush=True)
        print(f"  wall (trials only): {total_s:.1f}s   "
              f"mean/trial: {sum(walls) / n_done:.1f}s", flush=True)
    print(f"  results -> {results_path}", flush=True)

    config = {
        "graph_dir": graph_dir,
        "pairs_file": args.pairs_file,
        "anchor_pairs": [[list(y), list(d)] for y, d in anchor_pairs],
        "seeds_per_anchor": args.seeds,
        "jitter_m": _JITTER_M,
        "trial_timeout_s": _TRIAL_TIMEOUT_S,
        "record_video": _RECORD_VIDEO,
        "wall_clock_s": round(total_s, 1),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"  config -> {out_dir / 'config.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
