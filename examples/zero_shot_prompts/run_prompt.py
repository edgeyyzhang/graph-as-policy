"""Run one zero-shot prompt end-to-end: generate -> execute -> score.

    MUJOCO_GL=egl uv run python \
        examples/zero_shot_prompts/run_prompt.py \
        --case soup_basket --out outputs/zero_shot_prompts/dev

Pipeline per case:

1. ``gap.agent.generate_sync(prompt)`` — the prompt verbatim, once per
   case (cached in ``<case>/gen/task_00``; ``--force-generate`` refreshes).
2. For each seed: a fresh sim connector, ``StateRecorder`` around the
   tool dispatch, ``gap.execute(..., checkpoints="warn")``, video saved.
3. ``criteria.evaluate_seed`` over the recorded state trace ->
   ``seed<k>/result_seed.json``; the case aggregate (PASS only if every
   seed passes) -> ``<case>/result.json``.

``--re-eval`` re-scores existing traces from disk without sim or LLM —
this is also how a timed-out (hung) run gets its post-mortem verdict:
``run_all.py`` kills the process group, drops ``timeout_marker.json``,
and re-invokes this script with ``--re-eval``.

Exit codes: 0 pass, 1 fail, 2 infra crash, 3 generation error/refusal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cases import CASES, Case  # noqa: E402
import criteria  # noqa: E402
from state_log import StateRecorder, load_jsonl  # noqa: E402

logger = logging.getLogger("zero_shot_prompts")


def _default_gcp_project() -> str | None:
    """Project for the vertex provider: the ADC quota project on this
    machine (gap.agent.llm reads $GOOGLE_CLOUD_PROJECT; google-genai
    otherwise needs an explicit project for vertexai=True)."""
    adc = Path.home() / ".config/gcloud/application_default_credentials.json"
    try:
        return json.loads(adc.read_text()).get("quota_project_id") or None
    except Exception:
        return None


def _pick_freest_gpu() -> None:
    """Pin this case to the GPU with the most free memory (shared box:
    device 0 has been observed at 80/82 GB from other users' jobs, which
    makes CuRobo's planner die with CUDA OOM mid-case). Respects an
    explicit CUDA_VISIBLE_DEVICES; MUJOCO_EGL_DEVICE_ID is set to the
    same physical index so EGL rendering follows."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    import subprocess
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        idx, free = max(
            (r.split(",") for r in rows), key=lambda r: int(r[1]))
        os.environ["CUDA_VISIBLE_DEVICES"] = idx.strip()
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", idx.strip())
        logger.info("pinned to GPU %s (%s MiB free)", idx.strip(),
                    free.strip())
    except Exception as exc:
        logger.warning("GPU auto-pick failed (%s); using defaults", exc)


def env_hygiene() -> None:
    _pick_freest_gpu()
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Keep frame buffers off /tmp (measured near-full); home volume instead.
    tmp = Path.home() / ".cache" / "gap_zero_shot_prompts_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp))
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        proj = _default_gcp_project()
        if proj:
            os.environ["GOOGLE_CLOUD_PROJECT"] = proj
    # perceiving-next-item's decide script treats the env's own
    # check_success as authoritative ONLY when this opt-in is set; the
    # env's task is not the prompt's task, so it must stay off here.
    os.environ.pop("GAP_DECIDE_TRUST_ENV", None)


DEFAULT_PROVIDER = os.environ.get("GAP_LLM_PROVIDER", "vertex")
DEFAULT_MODEL = os.environ.get("GAP_LLM_MODEL", "gemini-3.1-pro-preview")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(case: Case, case_dir: Path, provider: str, model: str,
             force: bool) -> tuple[Path | None, dict]:
    info: dict = {"provider": provider, "model": model, "refused": False,
                  "error": None, "duration_s": 0.0, "cached": False}
    wf_dir = case_dir / "gen" / "task_00"
    if (wf_dir / "workflow.json").exists() and not force:
        info["cached"] = True
        return wf_dir, info
    from gap.agent import generate_sync

    t0 = time.time()
    try:
        graph = generate_sync(
            case.prompt, provider=provider, model=model,
            out_dir=case_dir / "gen",
        )
    except BaseException as exc:
        info["duration_s"] = round(time.time() - t0, 1)
        info["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        info["refused"] = "missing capabilit" in str(exc).lower()
        return None, info
    info["duration_s"] = round(time.time() - t0, 1)
    try:
        (case_dir / "graph.txt").write_text(str(graph))
    except Exception:
        pass
    return Path(graph.path), info


def _load_workflow(wf_dir: Path | None) -> dict | None:
    if wf_dir is None:
        return None
    try:
        return json.loads((wf_dir / "workflow.json").read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Execution (one seed)
# ---------------------------------------------------------------------------


def run_seed(case: Case, wf_dir: Path, seed_dir: Path, seed: int,
             video: bool) -> dict:
    """Execute the generated graph on one seed; returns execution meta.

    events.jsonl / state.jsonl are streamed to disk as the run goes, so
    a kill at any point leaves a scoreable partial trace.
    """
    import gap.connector
    from gap.runtime.execute import execute

    seed_dir.mkdir(parents=True, exist_ok=True)
    # Generated decide-loops keep a no-progress counter in
    # $TMPDIR/.gap_next_item_progress_<sha1(trace_dir)> (see
    # open-robot-skills perceiving-next-item/scripts/decide_next_item.py).
    # Our TMPDIR persists and trace paths repeat across re-runs of the
    # same --out, so a previous run's counter would leak into this one's
    # first pass — clear the scratch before every fresh episode.
    import glob
    import tempfile
    for stale in glob.glob(os.path.join(
            tempfile.gettempdir(), ".gap_next_item_progress_*")):
        try:
            os.unlink(stale)
        except OSError:
            pass
    execution: dict = {
        "seed": seed, "ran": False, "success": False, "exit_status": None,
        "error": None, "duration_s": 0.0, "harness_crash": None,
    }
    conn = gap.connector.sim(
        "libero", task=case.sim_task, record_video=video, seed=seed,
    )
    rec = StateRecorder(conn, seed_dir)
    try:
        conn.reset()
        rec.install()
        rec.snapshot("__initial__")
        result = execute(
            str(wf_dir), conn,
            trace_dir=seed_dir / "trace", checkpoints="warn",
        )
        execution.update(
            ran=True,
            success=bool(result.success),
            exit_status=result.exit_status,
            error=str(result.error)[:800] if result.error else None,
            duration_s=round(result.duration_s, 1),
            checkpoints=[
                {"subgraph": c.subgraph, "name": c.name,
                 "passed": bool(c.passed)}
                for c in (result.checkpoint_results or [])
            ],
        )
        rec.snapshot("__final__")
        try:
            done, reward = conn.check_success()
            execution["env_task_completed"] = bool(done)   # advisory only:
            execution["env_reward"] = float(reward)         # env's own task
        except Exception:
            pass
        if video:
            try:
                saved = conn.save_video(str(seed_dir / "run_video.mp4"))
                execution["video"] = saved.get("file_path")
            except Exception as exc:
                execution["video"] = f"save failed: {exc}"
    except BaseException as exc:
        execution["harness_crash"] = f"{type(exc).__name__}: {exc}"[:800]
        execution["traceback"] = traceback.format_exc(limit=15)
        try:
            rec.snapshot("__final__")
        except Exception:
            pass
    finally:
        rec.uninstall()
        try:
            conn.close()
        except Exception:
            pass
    (seed_dir / "execution.json").write_text(
        json.dumps(execution, indent=2, default=str) + "\n")
    return execution


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def evaluate_from_disk(case: Case, case_dir: Path,
                       timed_out_case: bool) -> dict:
    """Score every seed dir on disk; aggregate to the case verdict."""
    wf_dir = case_dir / "gen" / "task_00"
    if not wf_dir.exists():
        gen_path = case_dir / "generation.json"
        try:
            override = json.loads(gen_path.read_text()).get("graph_override")
            wf_dir = Path(override) if override else wf_dir
        except Exception:
            pass
    workflow = _load_workflow(wf_dir if wf_dir.exists() else None)
    structure = criteria.workflow_structure(workflow)

    seed_payloads = []
    for seed in case.seeds:
        seed_dir = case_dir / f"seed{seed}"
        if not seed_dir.exists():
            if timed_out_case:
                seed_payloads.append({
                    "seed": seed,
                    "evaluation": {
                        "verdict": "NOT_RUN", "score": 0.0,
                        "summary": "case timed out before this seed started",
                        "checks": [], "diagnostics": {}},
                })
            continue
        states = load_jsonl(seed_dir / "state.jsonl")
        events = load_jsonl(seed_dir / "events.jsonl")
        exec_path = seed_dir / "execution.json"
        if exec_path.exists():
            execution = json.loads(exec_path.read_text())
        else:
            # run_seed never finished for this seed: the case-level kill
            # (or a hard crash) hit while it was in flight.
            execution = {"seed": seed, "ran": True,
                         "timed_out": timed_out_case,
                         "harness_crash": None if timed_out_case
                         else "no execution.json (process died)"}
        extras = {
            "generated_graph_structure": structure,
            "max_node_iterations":
                criteria.max_node_iterations(seed_dir / "trace"),
        }
        ev = criteria.evaluate_seed(case, states, events, execution,
                                    extras)
        payload = {"seed": seed, "execution": execution,
                   "evaluation": ev.to_dict()}
        (seed_dir / "result_seed.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n")
        seed_payloads.append(payload)
    if not seed_payloads:
        return {}
    return _aggregate(case, case_dir, seed_payloads, structure)


def _aggregate(case: Case, case_dir: Path, seed_payloads: list[dict],
               structure: dict) -> dict:
    verdicts = [p["evaluation"]["verdict"] for p in seed_payloads]
    scores = [float(p["evaluation"]["score"]) for p in seed_payloads]
    if all(v == "PASS" for v in verdicts) and verdicts:
        verdict = "PASS"
    elif "TIMEOUT" in verdicts:
        verdict = "TIMEOUT"
    else:
        verdict = "FAIL"
    gen_path = case_dir / "generation.json"
    generation = (json.loads(gen_path.read_text())
                  if gen_path.exists() else {})
    payload = {
        "case": {
            "id": case.id, "prompt": case.prompt,
            "baseline": case.baseline, "criterion": case.criterion,
            "seeds": list(case.seeds), "sim_task": case.sim_task,
        },
        "generation": generation,
        "generated_graph_structure": structure,
        "verdict": verdict,
        "score": round(sum(scores) / max(1, len(scores)), 4),
        "seeds": seed_payloads,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (case_dir / "result.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    per_seed = ", ".join(
        f"seed{p['seed']}={p['evaluation']['verdict']}"
        f"({float(p['evaluation']['score']):.2f})"
        for p in seed_payloads
    )
    print(f"CASE {case.id}: {verdict} score={payload['score']:.2f} "
          f"[{per_seed}]")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_case(case: Case, out_root: Path, *, provider: str, model: str,
             video: bool, force_generate: bool,
             graph_override: Path | None, only_seed: int | None) -> dict:
    case_dir = out_root / case.id
    case_dir.mkdir(parents=True, exist_ok=True)

    if graph_override is not None:
        wf_dir: Path | None = graph_override
        gen_info = {"graph_override": str(graph_override), "refused": False}
    else:
        wf_dir, gen_info = generate(case, case_dir, provider, model,
                                    force_generate)
    (case_dir / "generation.json").write_text(
        json.dumps(gen_info, indent=2) + "\n")

    if wf_dir is None:
        verdict = "GEN_REFUSED" if gen_info.get("refused") else "GEN_ERROR"
        payload = {
            "case": {"id": case.id, "prompt": case.prompt,
                     "baseline": case.baseline},
            "generation": gen_info,
            "verdict": verdict, "score": 0.0, "seeds": [],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (case_dir / "result.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n")
        print(f"CASE {case.id}: {verdict} — {gen_info.get('error', '')[:200]}")
        return payload

    seeds = [only_seed] if only_seed is not None else list(case.seeds)
    for seed in seeds:
        logger.info("case %s: executing seed %d", case.id, seed)
        run_seed(case, wf_dir, case_dir / f"seed{seed}", seed, video)
    return evaluate_from_disk(case, case_dir, timed_out_case=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=None,
                    help="run only this seed (default: the case's list)")
    ap.add_argument("--graph", default=None,
                    help="skip generation; execute this workflow dir")
    ap.add_argument("--force-generate", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--re-eval", action="store_true",
                    help="re-score existing traces (no sim, no LLM); "
                         "honors timeout_marker.json for hang verdicts")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    env_hygiene()
    # Out-of-process tool bundles (perception VLM, ...) resolve their
    # provider from the environment (GAP_VLM_PROVIDER > GAP_LLM_PROVIDER
    # > "openrouter" — open-robot-skills/tools/vlm/tools.py); export the
    # chosen provider/model so the runtime bundles match generation.
    os.environ["GAP_LLM_PROVIDER"] = args.provider
    os.environ["GAP_LLM_MODEL"] = args.model
    case = CASES[args.case]
    out_root = Path(args.out)
    case_dir = out_root / case.id

    if args.re_eval:
        timed_out = (case_dir / "timeout_marker.json").exists()
        payload = evaluate_from_disk(case, case_dir, timed_out_case=timed_out)
        if not payload:
            gen_path = case_dir / "generation.json"
            gen_info = (json.loads(gen_path.read_text())
                        if gen_path.exists() else {})
            verdict = ("GEN_REFUSED" if gen_info.get("refused")
                       else "TIMEOUT" if timed_out else "GEN_ERROR")
            payload = {
                "case": {"id": case.id, "prompt": case.prompt,
                         "baseline": case.baseline},
                "generation": gen_info, "verdict": verdict,
                "score": 0.0, "seeds": [],
            }
            (case_dir / "result.json").write_text(
                json.dumps(payload, indent=2, default=str) + "\n")
            print(f"CASE {case.id}: {verdict}")
    else:
        payload = run_case(
            case, out_root, provider=args.provider, model=args.model,
            video=not args.no_video, force_generate=args.force_generate,
            graph_override=Path(args.graph) if args.graph else None,
            only_seed=args.seed,
        )

    verdict = payload.get("verdict")
    if verdict == "PASS":
        return 0
    if verdict in ("GEN_ERROR", "GEN_REFUSED"):
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
