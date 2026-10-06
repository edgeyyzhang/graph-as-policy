"""Turn rehearsal records into the author-facing feedback files.

``feedback.json`` holds everything; ``feedback.md`` is the short view an
author reads first. The report is descriptive: it places the graph's own
verdicts next to the privileged world state and counts where they disagree,
but it does not rank what to fix or prescribe an edit order.

Sections:

1. per-case success this round and last (same case ids), with the changes;
2. per-case unit path: each subgraph exit (or top-level node for flat graphs)
   with held body, gripper, and object displacement since the episode start;
3. per-unit counts over cases: visits, verdict distribution, checkpoint
   pass/fail, held-at-exit rate, errors;
4. verdict-versus-world disagreements, from a small claims table mapping exit
   values to the world fact they assert;
5. errors by unit and node;
6. diff from the previous round's graph;
7. trace directory per case.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .collect import checkpoint_rows, has_subgraphs, trace_node_errors, unit_rows
from .diff import is_empty, manifest_diff, workflow_diff

#: exit value -> world fact it asserts. ``held``: some body is in the gripper;
#: ``not_held``: nothing is. Exit values not listed assert nothing.
DEFAULT_CLAIMS: dict[str, str] = {
    "grasped": "held", "held": "held", "carried": "held", "lifted": "held",
    "released": "not_held", "placed": "not_held", "dropped": "not_held",
    "missed": "not_held", "slipped": "not_held",
}


def claim_holds(claim: str, world: dict[str, Any] | None) -> bool | None:
    if world is None:
        return None
    held = world.get("held")
    if claim == "held":
        return held is not None
    if claim == "not_held":
        return held is None
    return None


def _load_previous(previous: Path | None) -> dict[str, Any] | None:
    if previous is None:
        return None
    path = Path(previous) / "feedback.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        return None


def _compact_world(row: dict[str, Any]) -> dict[str, Any]:
    world = row.get("world") or {}
    since = row.get("since_start") or {}
    moved = {k: v for k, v in (since.get("displacement_m") or {}).items() if v >= 0.01}
    return {
        "held": world.get("held"),
        "gripper": world.get("gripper_open_fraction"),
        "moved_since_start_m": moved,
        "robot_contacts": world.get("robot_contacts", []),
    }


def build_feedback(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    previous: Path | None = None,
    workflow_dir: Path | None = None,
    claims: dict[str, str] | None = None,
) -> dict[str, Any]:
    claims_table = dict(DEFAULT_CLAIMS)
    if claims:
        claims_table.update(claims)
    prev = _load_previous(previous)
    prev_cases = {int(c["case"]): c for c in (prev or {}).get("cases", [])} if prev else {}

    # 1. per-case success, this round and last
    cases_out = []
    for r in records:
        case = int(r["case"])
        before = prev_cases.get(case, {}).get("success") if prev_cases else None
        now = r.get("success")
        change = None
        if before is not None and now is not None and before != now:
            change = "fixed" if now else "broken"
        cases_out.append({
            "case": case, "success": now, "previous_success": before, "change": change,
            "graph_exit": r.get("exit_status"), "error": r.get("error"),
            "duration_s": r.get("duration_s"), "trace_dir": r.get("trace_dir"),
        })
    n = len(records)
    successes = sum(1 for c in cases_out if c["success"])
    prev_successes = sum(1 for c in cases_out if c["previous_success"]) if prev_cases else None

    # 2. per-case unit path
    per_case = []
    unit_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "visits": 0, "verdicts": Counter(), "errors": 0, "held_at_exit": Counter(), "world_seen": 0,
        "checkpoints": defaultdict(lambda: {"passed": 0, "failed": 0}),
    })
    disagreements: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in records:
        rows = unit_rows(r)
        path = []
        for row in rows:
            unit, verdict = row.get("unit"), row.get("verdict")
            world = row.get("world")
            path.append({
                "unit": unit, "visit": row.get("visit"), "verdict": verdict,
                "error_path": row.get("error_path"), **_compact_world(row),
            })
            if unit is None:
                continue
            st = unit_stats[unit]
            st["visits"] += 1
            st["verdicts"][str(verdict)] += 1
            if row.get("error_path"):
                st["errors"] += 1
            if world is not None:
                st["world_seen"] += 1
                st["held_at_exit"][str(world.get("held"))] += 1
            claim = claims_table.get(str(verdict))
            if claim is not None:
                ok = claim_holds(claim, world)
                if ok is False:
                    key = (unit, str(verdict), claim)
                    entry = disagreements.setdefault(key, {"unit": unit, "verdict": verdict, "claims": claim, "count": 0, "cases": []})
                    entry["count"] += 1
                    if r["case"] not in entry["cases"]:
                        entry["cases"].append(r["case"])
        for cp in checkpoint_rows(r):
            st = unit_stats[cp["unit"]]["checkpoints"][cp["name"]]
            st["passed" if cp["passed"] else "failed"] += 1
        per_case.append({"case": r["case"], "success": r.get("success"), "path": path,
                         "final": _compact_world({"world": r.get("final_world"),
                                                  "since_start": None}) if r.get("final_world") else None})

    # 3. per-unit counts
    units_out = {}
    for unit, st in unit_stats.items():
        units_out[unit] = {
            "visits": st["visits"],
            "verdicts": dict(st["verdicts"]),
            "errors": st["errors"],
            "held_at_exit": dict(st["held_at_exit"]),  # body name (or 'None') -> count over visits
            "checkpoints": {k: dict(v) for k, v in st["checkpoints"].items()},
        }

    # 5. errors
    errors = []
    for r in records:
        if r.get("error"):
            errors.append({"case": r["case"], "node": None, "message": r["error"]})
        for e in trace_node_errors(r):
            errors.append({"case": r["case"], "node": e["node"], "message": e["message"]})
        for v in r.get("visits", []):
            if v.get("success") is False:
                errors.append({"case": r["case"], "node": v.get("node"), "message": None, "kind": "node_failed"})
    error_digest = Counter((e.get("node"), (e.get("message") or "")[:120]) for e in errors)

    # 6. diff from previous
    diff_out: dict[str, Any] | None = None
    if previous is not None:
        prev_wf = _load_json(Path(previous) / "workflow.snapshot.json")
        cur_wf = _load_json(Path(workflow_dir) / "workflow.json") if workflow_dir else None
        prev_run = _load_json(Path(previous) / "run.json") or {}
        diff_out = {
            "workflow": workflow_diff(prev_wf, cur_wf) if (prev_wf is not None and cur_wf is not None) else None,
            "files": manifest_diff((prev_run.get("manifest") or {}).get("files"), (meta.get("manifest") or {}).get("files")),
        }

    return {
        "meta": {k: v for k, v in meta.items() if k != "manifest"},
        "summary": {
            "cases": n, "successes": successes,
            "previous_successes": prev_successes,
            "fixed": [c["case"] for c in cases_out if c["change"] == "fixed"],
            "broken": [c["case"] for c in cases_out if c["change"] == "broken"],
            "graph_has_subgraphs": any(has_subgraphs(r) for r in records),
        },
        "cases": cases_out,
        "per_case": per_case,
        "units": units_out,
        "disagreements": sorted(disagreements.values(), key=lambda d: -d["count"]),
        "errors": [{"node": k[0], "message": k[1], "count": v} for k, v in error_digest.most_common()],
        "diff": diff_out,
        "claims": claims_table,
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(cell(x) for x in r) + " |")
    return "\n".join(lines)


def render_markdown(fb: dict[str, Any]) -> str:
    s = fb["summary"]
    out = [f"# Rehearsal feedback", ""]
    out.append(f"Graph: `{fb['meta'].get('workflow_dir')}` · sim `{fb['meta'].get('sim')}` · cases {fb['meta'].get('cases')}")
    line = f"Native success: **{s['successes']} / {s['cases']}**"
    if s.get("previous_successes") is not None:
        line += f" (previous round {s['previous_successes']} / {s['cases']}; fixed {s['fixed']}, broken {s['broken']})"
    out += ["", line, ""]

    out += ["## Per-case success", ""]
    out.append(_md_table(
        ["case", "success", "previous", "change", "graph exit", "error"],
        [[c["case"], c["success"], c["previous_success"], c["change"], c["graph_exit"], (c["error"] or "")[:80]] for c in fb["cases"]],
    ))

    out += ["", "## Per-case path (graph verdict beside world state)", ""]
    for pc in fb["per_case"]:
        out.append(f"### case {pc['case']} — {'success' if pc['success'] else 'FAIL' if pc['success'] is False else 'unknown'}")
        rows = []
        for step in pc["path"]:
            moved = ", ".join(f"{k} {v:.2f}m" for k, v in (step.get("moved_since_start_m") or {}).items())
            rows.append([step["unit"], step.get("visit"), step["verdict"], step.get("held"), step.get("gripper"), moved,
                         ", ".join(step.get("robot_contacts") or [])])
        out.append(_md_table(["unit", "visit", "verdict", "held", "gripper", "moved since start", "robot contacts"], rows))
        out.append("")

    out += ["## Per-unit counts", ""]
    rows = []
    for unit, u in fb["units"].items():
        cps = "; ".join(f"{k}: {v['passed']}✓/{v['failed']}✗" for k, v in u["checkpoints"].items())
        held = ", ".join(f"{k} ×{v}" for k, v in u["held_at_exit"].items())
        rows.append([unit, u["visits"], json.dumps(u["verdicts"]), u["errors"], held, cps])
    out.append(_md_table(["unit", "visits", "verdicts", "errors", "held at exit", "checkpoints"], rows))

    out += ["", "## Verdict vs world disagreements", ""]
    if fb["disagreements"]:
        out.append(_md_table(["unit", "verdict", "claims", "count", "cases"],
                             [[d["unit"], d["verdict"], d["claims"], d["count"], d["cases"]] for d in fb["disagreements"]]))
    else:
        out.append("None: every exit value that asserts a world fact was consistent with the simulator.")

    out += ["", "## Errors", ""]
    if fb["errors"]:
        out.append(_md_table(["node", "message", "count"], [[e["node"], e["message"], e["count"]] for e in fb["errors"]]))
    else:
        out.append("None.")

    if fb.get("diff"):
        out += ["", "## Changes since previous round", ""]
        wd = fb["diff"].get("workflow")
        if wd is None:
            out.append("Previous workflow snapshot unavailable.")
        elif is_empty(wd):
            out.append("No node, parameter, edge, or routing changes.")
        else:
            out.append("```json")
            out.append(json.dumps(wd, indent=2))
            out.append("```")
        files = fb["diff"].get("files") or {}
        if any(files.values()):
            out.append(f"Files: added {files.get('added')}, removed {files.get('removed')}, modified {files.get('modified')}")

    out += ["", "## Traces", ""]
    out += [f"- case {c['case']}: `{c['trace_dir']}`" for c in fb["cases"]]
    out.append("")
    return "\n".join(out)


def write_feedback(out_dir: Path, fb: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feedback.json").write_text(json.dumps(fb, indent=2, default=str))
    (out_dir / "feedback.md").write_text(render_markdown(fb))


__all__ = ["build_feedback", "render_markdown", "write_feedback", "DEFAULT_CLAIMS", "claim_holds"]
