#!/usr/bin/env python3
"""Policy card: one self-contained HTML report about a generated policy graph.

    policy_card.py GRAPH_DIR EVAL_DIR [--out DIR] [--optimization JSON] [--max-mb 15]
                   [--max-failure-videos 4] [--height 360] [--speed 3] [--keep-media]

GRAPH_DIR holds the ``workflow.json`` and scripts of the policy. EVAL_DIR is one
``gap rehearse`` run of that graph, made with ``--frames --video``, over the cases
to report: either every layout of the task or, as policy_report.sh does by
default, a random sample of them recorded in ``EVAL_DIR/sample.json``. The
report has four sections:

  0. the task: suite, instruction, goal predicate, objects, cases;
  1. the policy: the graph drawing, its subgraphs, and a two-panel video of one
     successful case (simulator left, graph with the active node right);
  2. the evaluation: success per case, steps and time, and the trajectories:
     with ``--all-videos`` a two-panel video of every evaluated case, otherwise
     one video per failed case, each with the node it ended in;
  3. the parameters of every node: literal inputs, the constants its script
     defines, and the learnable offsets when an optimization file is given.

Everything is embedded in ``report.html`` (images and videos as data URIs) so the
file can be published as a page or sent on its own. ``report.json`` beside it
holds the numbers. Videos are encoded small and the total is kept under
``--max-mb`` so the page fits the artifact limit; failure videos beyond
``--max-failure-videos`` are listed without a video.

This file belongs to the ``policy-report`` agent skill (``agent/skills/policy-report``);
``policy_report.sh`` beside it runs the evaluation and then this generator with the
checkout's ``.venv`` Python. Run alone with ``.venv/bin/python`` and ``PYTHONPATH``
covering the gap checkout and ``gap-core/src``.
"""
from __future__ import annotations

import argparse
import ast
import base64
import html
import json
import re
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent            # <gap>/agent/skills/policy-report/scripts
GAP_ROOT = HERE.parents[3]
sys.path[:0] = [str(GAP_ROOT / "examples" / "rehearse_loop"),   # compose_video
                str(GAP_ROOT), str(GAP_ROOT / "gap-core" / "src")]

MB = 1024 * 1024
ENCODINGS = [  # tried in order until the page fits the budget
    {"height": 480, "quality": 6},
    {"height": 360, "quality": 6},
    {"height": 300, "quality": 5},
    {"height": 240, "quality": 4},
]


# --------------------------------------------------------------------------- inputs

def load_json(path: Path):
    return json.loads(path.read_text())


def case_dirs(eval_dir: Path) -> list[Path]:
    return sorted(p for p in (eval_dir / "cases").glob("case_*") if (p / "case.json").exists())


def load_cases(eval_dir: Path) -> list[dict]:
    rows = []
    for d in case_dirs(eval_dir):
        c = load_json(d / "case.json")
        visits = c.get("visits") or []
        last = visits[-1] if visits else {}
        traj = c.get("trajectory") or {}
        latency = c.get("latency") or {}
        rows.append({
            "case": c["case"],
            "dir": d,
            "success": bool(c.get("success")),
            "graph_exit": c.get("exit_status"),
            "error": c.get("error"),
            "last_node": last.get("node"),
            "visits": len(visits),
            "steps": traj.get("steps") or latency.get("control_steps"),
            "duration_s": c.get("duration_s"),
            "has_video": (d / "video.mp4").exists() and (d / "video.mp4.frames.json").exists(),
        })
    return rows


def task_description(sim: str) -> dict:
    """Suite, index, name, instruction and goal predicate of a LIBERO task."""
    suite, _, idx = sim.partition("/")
    out = {"suite": suite, "index": int(idx) if idx.isdigit() else idx, "name": None,
           "instruction": None, "goal": None}
    root = GAP_ROOT / "third_party/LIBERO-PRO/libero/libero"
    task_map = root / "benchmark/libero_suite_task_map.py"
    if not task_map.exists() or not idx.isdigit():
        return out
    tree = ast.parse(task_map.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            tasks = ast.literal_eval(node.value).get(suite) or []
            if int(idx) < len(tasks):
                out["name"] = tasks[int(idx)]
            break
    if out["name"]:
        bddl = root / "bddl_files" / suite / (out["name"] + ".bddl")
        if bddl.exists():
            src = bddl.read_text()
            m = re.search(r"\(:language\s+([^)]*)\)", src)
            if m:
                out["instruction"] = " ".join(m.group(1).split())
            m = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n", src, re.S)
            if m:
                out["goal"] = " ".join(m.group(1).split())
    return out


# --------------------------------------------------------------------------- graph

def qualified_nodes(wf: dict) -> list[dict]:
    """Every executable node with its owner, in graph order."""
    rows = []
    top = wf.get("nodes") or {}
    subgraphs = wf.get("subgraphs") or {}
    for name, node in top.items():
        if node.get("type") == "subgraph":
            sg = subgraphs.get(node.get("ref"), {})
            for inner, spec in (sg.get("nodes") or {}).items():
                rows.append({"owner": name, "skill": sg.get("skill"), "node": inner,
                             "id": f"{name}.{inner}", "spec": spec})
        else:
            rows.append({"owner": None, "skill": None, "node": name, "id": name, "spec": node})
    return rows


def subgraph_summary(wf: dict) -> list[dict]:
    rows = []
    top = wf.get("nodes") or {}
    cond = wf.get("conditional_edges") or {}
    for name, node in top.items():
        if node.get("type") != "subgraph":
            rows.append({"name": name, "skill": None, "nodes": 1, "exits": {},
                         "type": node.get("type")})
            continue
        sg = (wf.get("subgraphs") or {}).get(node.get("ref"), {})
        routes = (cond.get(name) or {}).get("mapping") or {}
        rows.append({"name": name, "skill": sg.get("skill"), "nodes": len(sg.get("nodes") or {}),
                     "exits": routes, "type": "subgraph",
                     "on_error": sg.get("on_error")})
    return rows


def literal_inputs(spec: dict) -> dict:
    """Inputs given as values in the graph (not wired from another node)."""
    out = {}
    for key, value in (spec.get("inputs") or {}).items():
        if isinstance(value, dict) and "$ref" in value:
            out[key] = {"ref": value["$ref"]}
        else:
            out[key] = {"value": value}
    return out


def script_constants(path: Path) -> list[dict]:
    """Module-level names bound to numbers (or containers of numbers), with the
    comment on the same line as their description."""
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return []
    lines = src.splitlines()
    rows = []

    def numeric(v) -> bool:
        if isinstance(v, bool):
            return True
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, (list, tuple)):
            return bool(v) and all(numeric(x) for x in v)
        if isinstance(v, dict):
            return bool(v) and all(numeric(x) for x in v.values())
        return False

    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        try:
            v = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        if not numeric(v):
            continue
        # palettes, matrices and display settings are not behavior parameters
        if isinstance(v, (list, tuple)) and any(isinstance(x, (list, tuple, dict)) for x in v):
            continue
        for t in targets:
            if t.id.startswith("__") or re.search(r"COLOU?R|FONT|PALETTE|DEBUG|VERBOSE|LOG", t.id, re.I):
                continue
            comment = ""
            line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
            if "#" in line:
                comment = line.split("#", 1)[1].strip()
            rows.append({"name": t.id, "value": v, "note": comment, "line": node.lineno})
    return rows


def learnable(opt: dict | None, node_id: str) -> list[dict]:
    if not opt:
        return []
    entry = (opt.get("nodes") or {}).get(node_id) or {}
    return entry.get("parameters") or []


# --------------------------------------------------------------------------- media

def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def make_videos(graph_dir: Path, picks: list[dict], media: Path, budget_bytes: int,
                height: int, speed: float, title: str | None) -> dict[int, Path]:
    """Compose one two-panel video per picked case, shrinking the encoding until
    the base64 total fits the budget. Drops the last picks when it cannot."""
    from compose_video import compose

    encodings = [e for e in ENCODINGS if e["height"] <= height] or ENCODINGS[-1:]
    if encodings[0]["height"] != height:
        encodings.insert(0, {"height": height, "quality": 6})
    chosen: dict[int, Path] = {}
    for enc in encodings:
        chosen = {}
        total = 0
        for pick in picks:
            out = media / f"{pick['label']}_case_{pick['case']:02d}.mp4"
            compose(pick["dir"], graph_dir, out, height=enc["height"], speed=speed,
                    title=title, quality=enc["quality"])
            total += out.stat().st_size * 4 // 3
            chosen[pick["case"]] = out
        if total <= budget_bytes:
            return chosen
    # Still too large at the smallest encoding: keep the policy video and as many
    # failure videos as fit, in order.
    kept: dict[int, Path] = {}
    total = 0
    for pick in picks:
        p = chosen[pick["case"]]
        size = p.stat().st_size * 4 // 3
        if total + size <= budget_bytes or pick["label"] == "policy":
            kept[pick["case"]] = p
            total += size
        else:
            p.unlink(missing_ok=True)
    return kept


# --------------------------------------------------------------------------- html

def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


STYLE = """
:root {
  --bg: #f7f6f2; --panel: #ffffff; --ink: #1d1c1a; --muted: #6b675f; --line: #ddd8ce;
  --accent: #b5541c; --accent-ink: #ffffff; --ok: #2f7d4a; --fail: #b3261e; --tint: #f1ece2;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #171614; --panel: #201f1c; --ink: #ece8df; --muted: #a39d92; --line: #3a3733;
    --accent: #e0803f; --accent-ink: #1a120a; --ok: #6fc48a; --fail: #f08a82; --tint: #27251f;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #171614; --panel: #201f1c; --ink: #ece8df; --muted: #a39d92; --line: #3a3733;
  --accent: #e0803f; --accent-ink: #1a120a; --ok: #6fc48a; --fail: #f08a82; --tint: #27251f;
  color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); margin: 0; padding-block: 24px 48px; padding-inline: 16px;
       font: 15px/1.5 "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; display: grid; gap: 28px; }
h1, h2, h3 { line-height: 1.2; text-wrap: balance; margin: 0; }
h1 { font-size: 1.7rem; } h2 { font-size: 1.25rem; margin-top: 8px; } h3 { font-size: 1rem; }
p { margin: 0; max-width: 70ch; }
.lede { color: var(--muted); }
.score { display: flex; flex-wrap: wrap; gap: 12px; }
.score div { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; min-width: 120px; }
.score b { display: block; font-size: 1.5rem; font-variant-numeric: tabular-nums; }
.score span { color: var(--muted); font-size: .85rem; text-transform: uppercase; letter-spacing: .04em; }
section { display: grid; gap: 14px; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }
.kv dt { color: var(--muted); } .kv dd { margin: 0; }
figure { margin: 0; display: grid; gap: 8px; }
figure img, figure video { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
figcaption { color: var(--muted); font-size: .9rem; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .92rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
td.num, th.num { text-align: right; }
.ok { color: var(--ok); font-weight: 600; } .fail { color: var(--fail); font-weight: 600; }
code { font: .88em/1.4 "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--tint); padding: 1px 5px; border-radius: 4px; }
.node { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 8px; }
.node h3 code { background: none; padding: 0; font-size: 1em; }
.tag { display: inline-block; background: var(--accent); color: var(--accent-ink); border-radius: 999px; padding: 0 8px; font-size: .75rem; margin-left: 6px; vertical-align: middle; }
.goal { font: .9em/1.5 "IBM Plex Mono", ui-monospace, monospace; background: var(--tint); padding: 8px 10px; border-radius: 6px; overflow-x: auto; }
.footer { color: var(--muted); font-size: .85rem; }
"""


def render_html(ctx: dict) -> str:
    task, cases, sg = ctx["task"], ctx["cases"], ctx["subgraphs"]
    n_ok = sum(c["success"] for c in cases)
    title = ctx["title"]
    parts = [f"<title>{esc(title)}</title>",
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">',
             f"<style>{STYLE}</style>", "<main>"]

    # header
    ok_steps = [c["steps"] for c in cases if c["success"] and c["steps"]]
    ok_time = [c["duration_s"] for c in cases if c["success"] and c["duration_s"]]
    parts.append(f"<header><h1>{esc(title)}</h1><p class='lede'>{esc(task.get('instruction') or ctx['sim'])}</p></header>")
    parts.append("<div class='score'>"
                 f"<div><b>{n_ok} / {len(cases)}</b><span>layouts solved</span></div>"
                 f"<div><b>{fmt(statistics.median(ok_steps)) if ok_steps else '–'}</b><span>median steps, successes</span></div>"
                 f"<div><b>{fmt(statistics.median(ok_time)) if ok_time else '–'} s</b><span>median time, successes</span></div>"
                 f"<div><b>{len(ctx['nodes'])}</b><span>nodes in {len([s for s in sg if s['type']=='subgraph'])} subgraphs</span></div>"
                 "</div>")

    # 0 task
    parts.append("<section><h2>0. Task</h2><dl class='kv'>")
    parts.append(f"<dt>Simulator</dt><dd>LIBERO, <code>{esc(ctx['sim'])}</code>" + (f" ({esc(task['name'])})" if task.get("name") else "") + "</dd>")
    parts.append(f"<dt>Instruction</dt><dd>{esc(task.get('instruction') or 'not recorded')}</dd>")
    if task.get("goal"):
        parts.append(f"<dt>Success predicate</dt><dd><div class='goal'>{esc(task['goal'])}</div></dd>")
    parts.append(f"<dt>Objects in the scene</dt><dd>{esc(', '.join(ctx['objects']))}</dd>")
    smp = ctx.get("sample")
    if smp:
        parts.append(f"<dt>Layouts</dt><dd>{len(cases)} of the task's layouts, sampled at random from cases {esc(smp.get('pool'))} with seed {esc(smp.get('seed'))}: "
                     f"cases {esc(', '.join(str(c['case']) for c in cases))}. Each case resets the simulator to one of the task's stored initial states, so the same case is the same layout every time.</dd>")
    else:
        parts.append(f"<dt>Layouts</dt><dd>cases {esc(ctx['case_spec'])}: each case resets the simulator to one of the task's stored initial states, so the same case is the same layout every time</dd>")
    parts.append(f"<dt>Evaluated</dt><dd>{esc(ctx['started'])}, graph <code>{esc(ctx['graph_hash'])}</code></dd>")
    parts.append("</dl></section>")

    # 1 policy
    parts.append("<section><h2>1. Policy</h2>")
    parts.append(f"<figure><img src='data:image/png;base64,{ctx['graph_png']}' alt='policy graph'>"
                 "<figcaption>The policy graph. Each lane is a subgraph; arrows between lanes are routed on the subgraph's exit value.</figcaption></figure>")
    parts.append("<div class='tbl'><table><tr><th>Subgraph</th><th>Skill</th><th class='num'>Nodes</th><th>Exit → next</th></tr>")
    for s in sg:
        exits = ", ".join(f"{esc(k)} → {esc(v)}" for k, v in (s.get("exits") or {}).items()) or "—"
        parts.append(f"<tr><td><code>{esc(s['name'])}</code></td><td>{esc(s.get('skill') or s.get('type'))}</td>"
                     f"<td class='num'>{s['nodes']}</td><td>{exits}</td></tr>")
    parts.append("</table></div>")
    pv = ctx.get("policy_video")
    if pv:
        parts.append(f"<figure><video controls preload='metadata' src='data:video/mp4;base64,{pv['b64']}'></video>"
                     f"<figcaption>Case {pv['case']}, a representative success: simulator on the left, the graph with the active node on the right. "
                     f"{esc(pv['note'])}</figcaption></figure>")
    else:
        parts.append("<p class='lede'>No successful case with a video was available for the policy video.</p>")
    parts.append("</section>")

    # 2 evaluation
    parts.append("<section><h2>2. Evaluation</h2>")
    parts.append(f"<p>{n_ok} of {len(cases)} layouts end with the task predicate satisfied. "
                 "Steps are simulator control steps; time is wall-clock for the whole case including perception calls.</p>")
    parts.append("<div class='tbl'><table><tr><th>Case</th><th>Result</th><th>Graph exit</th><th class='num'>Steps</th><th class='num'>Time (s)</th><th class='num'>Node visits</th><th>Ended in</th></tr>")
    for c in cases:
        res = "<span class='ok'>success</span>" if c["success"] else "<span class='fail'>failure</span>"
        parts.append(f"<tr><td>{c['case']}</td><td>{res}</td><td>{esc(c['graph_exit'])}</td>"
                     f"<td class='num'>{c['steps'] if c['steps'] is not None else '–'}</td>"
                     f"<td class='num'>{fmt(c['duration_s']) if c['duration_s'] is not None else '–'}</td>"
                     f"<td class='num'>{c['visits']}</td><td><code>{esc(c['last_node'])}</code></td></tr>")
    parts.append("</table></div>")
    def failure_text(c: dict) -> str:
        what = (f"The graph exited with <code>{esc(c['graph_exit'])}</code>"
                + (f" after the error <code>{esc(c['error'])}</code>" if c.get("error") else "")
                + f"; the last node to run was <code>{esc(c['last_node'])}</code>.")
        if c["graph_exit"] == "success":
            what += " The policy believed it had finished, but the simulator's predicate was not met."
        return what

    def video_or_note(c: dict) -> str:
        v = ctx["case_videos"].get(c["case"])
        if v:
            return f"<figure><video controls preload='metadata' src='data:video/mp4;base64,{v}'></video></figure>"
        if pv and pv["case"] == c["case"]:
            return "<p class='lede'>This is the representative case; its video is in section 1.</p>"
        if c["has_video"]:
            return "<p class='lede'>Video not embedded to keep the page within its size limit; it is in the evaluation directory.</p>"
        return "<p class='lede'>The evaluation was run without video.</p>"

    failures = [c for c in cases if not c["success"]]
    if ctx.get("all_videos"):
        parts.append(f"<h3>Trajectories</h3><p>One video per evaluated layout, played at {ctx['speed']:g}x: simulator on the left, the graph with the active node on the right.</p>")
        for c in cases:
            res = "<span class='ok'>success</span>" if c["success"] else "<span class='fail'>failure</span>"
            stats = f"{c['steps']} steps, {fmt(c['duration_s'])} s" if c["steps"] is not None else ""
            parts.append(f"<div class='node'><h3>Case {c['case']} <span class='lede'>{res} {esc(stats)}</span></h3>")
            if not c["success"]:
                parts.append(f"<p>{failure_text(c)}</p>")
            parts.append(video_or_note(c) + "</div>")
    elif failures:
        parts.append("<h3>Failure modes</h3>")
        for c in failures:
            parts.append(f"<div class='node'><h3>Case {c['case']}</h3><p>{failure_text(c)}</p>" + video_or_note(c) + "</div>")
    else:
        parts.append("<p>No failures on the evaluated layouts.</p>")
    parts.append("</section>")

    # 3 parameters
    parts.append("<section><h2>3. Parameters per node</h2>"
                 "<p>For each node: inputs set as values in the graph (wired inputs are named by their source), "
                 "the constants its script defines, and the offsets exposed for learning when an optimization file was given.</p>")
    for owner, rows in ctx["nodes_by_owner"].items():
        parts.append(f"<h3>{esc(owner or 'top level')}</h3>")
        for n in rows:
            spec = n["spec"]
            kind = spec.get("type")
            src = spec.get("script") or spec.get("tool") or spec.get("skill") or ""
            learn = n["learnable"]
            parts.append(f"<div class='node'><h3><code>{esc(n['id'])}</code> <span class='lede'>{esc(kind)} {esc(src)}</span>"
                         + (f"<span class='tag'>{len(learn)} learnable</span>" if learn else "") + "</h3>")
            inputs = n["inputs"]
            lit = {k: v for k, v in inputs.items() if "value" in v}
            wired = {k: v["ref"] for k, v in inputs.items() if "ref" in v}
            if lit or wired:
                parts.append("<div class='tbl'><table><tr><th>Input</th><th>Value or source</th></tr>")
                for k, v in lit.items():
                    parts.append(f"<tr><td><code>{esc(k)}</code></td><td>{esc(fmt(v['value']))}</td></tr>")
                for k, ref in wired.items():
                    parts.append(f"<tr><td><code>{esc(k)}</code></td><td class='lede'>from <code>{esc(ref)}</code></td></tr>")
                parts.append("</table></div>")
            if n["constants"]:
                parts.append("<div class='tbl'><table><tr><th>Constant</th><th>Value</th><th>Note</th></tr>")
                for c in n["constants"]:
                    parts.append(f"<tr><td><code>{esc(c['name'])}</code></td><td>{esc(fmt(c['value']))}</td><td class='lede'>{esc(c['note'])}</td></tr>")
                parts.append("</table></div>")
            if learn:
                parts.append("<div class='tbl'><table><tr><th>Learnable</th><th>Applied to</th><th class='num'>Low</th><th class='num'>Default</th><th class='num'>High</th></tr>")
                for p in learn:
                    parts.append(f"<tr><td><code>{esc(p.get('input'))}</code></td><td>{esc(p.get('destination') or 'input offset')}</td>"
                                 f"<td class='num'>{fmt(p.get('low'))}</td><td class='num'>{fmt(p.get('default'))}</td><td class='num'>{fmt(p.get('high'))}</td></tr>")
                parts.append("</table></div>")
            if not (lit or wired or n["constants"] or learn):
                parts.append("<p class='lede'>No parameters: this node only wires its inputs through.</p>")
            parts.append("</div>")
    parts.append("</section>")

    parts.append(f"<p class='footer'>Generated {esc(ctx['generated'])} by policy_card.py from {esc(ctx['eval_dir'])}.</p>")
    parts.append("</main>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- main

def graph_hash(graph_dir: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(graph_dir.rglob("*")):
        if p.is_file() and "agent_traces" not in p.parts and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(graph_dir)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph_dir", type=Path)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="Report directory (default: EVAL_DIR/../report)")
    ap.add_argument("--optimization", type=Path, default=None, help="optimization.json with learnable parameters")
    ap.add_argument("--title", default=None)
    ap.add_argument("--max-mb", type=float, default=15.0, help="Upper bound for report.html (default 15)")
    ap.add_argument("--max-failure-videos", type=int, default=4,
                    help="Without --all-videos: how many failed cases get an embedded video")
    ap.add_argument("--all-videos", action="store_true",
                    help="Embed a two-panel video of every evaluated case, not only the failures")
    ap.add_argument("--height", type=int, default=480, help="Video panel height (default 480)")
    ap.add_argument("--speed", type=float, default=3.0, help="Video playback speed (default 3x)")
    ap.add_argument("--keep-media", action="store_true", help="Keep the PNG and MP4 files beside the report")
    args = ap.parse_args()

    graph_dir, eval_dir = args.graph_dir.resolve(), args.eval_dir.resolve()
    out = (args.out or eval_dir.parent / "report").resolve()
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)

    run = load_json(eval_dir / "run.json") if (eval_dir / "run.json").exists() else {}
    wf = load_json(graph_dir / "workflow.json")
    opt = load_json(args.optimization) if args.optimization else (
        load_json(graph_dir / "optimization.json") if (graph_dir / "optimization.json").exists() else None)
    cases = load_cases(eval_dir)
    if not cases:
        sys.exit(f"no cases under {eval_dir / 'cases'}")
    sim = run.get("sim") or "unknown"
    task = task_description(sim)
    first = load_json(cases[0]["dir"] / "case.json")
    objects = sorted((first.get("initial_world") or {}).get("objects", {}).keys())
    title = args.title or (task.get("instruction") or sim).capitalize()

    # graph drawing
    from gap.viz import render
    render(graph_dir / "workflow.json", media / "graph.pdf", legend=True, also_png=True)
    graph_png = media / "graph.png"

    # videos: one representative success (median steps) and the failures
    budget = int(args.max_mb * MB) - graph_png.stat().st_size * 4 // 3 - 300 * 1024
    picks = []
    successes = [c for c in cases if c["success"] and c["has_video"] and c["steps"]]
    rep_case = None
    if successes:
        successes.sort(key=lambda c: c["steps"])
        rep_case = successes[len(successes) // 2]["case"]
        picks.append({"label": "policy", "case": rep_case, "dir": successes[len(successes) // 2]["dir"]})
    failures = [c for c in cases if not c["success"] and c["has_video"]]
    if not args.all_videos:
        failures = failures[: args.max_failure_videos]
    for c in failures:
        picks.append({"label": "failure", "case": c["case"], "dir": c["dir"]})
    if args.all_videos:  # the remaining successes, dropped first if the budget is short
        for c in cases:
            if c["success"] and c["has_video"] and c["case"] != rep_case:
                picks.append({"label": "traj", "case": c["case"], "dir": c["dir"]})
    videos = make_videos(graph_dir, picks, media, budget, args.height, args.speed, args.title) if picks else {}

    policy_video = None
    if rep_case in videos:
        rep = next(c for c in cases if c["case"] == rep_case)
        policy_video = {"case": rep["case"], "b64": b64(videos[rep["case"]]),
                        "note": f"{rep['steps']} steps, {fmt(rep['duration_s'])} s, played at {args.speed:g}x."}
    case_videos = {c["case"]: b64(videos[c["case"]]) for c in cases
                   if c["case"] in videos and c["case"] != rep_case}
    failure_videos = {k: v for k, v in case_videos.items()
                      if not next(c for c in cases if c["case"] == k)["success"]}
    sample = load_json(eval_dir / "sample.json") if (eval_dir / "sample.json").exists() else None

    # parameters
    nodes = qualified_nodes(wf)
    by_owner: dict[str | None, list[dict]] = {}
    for n in nodes:
        spec = n["spec"]
        n["inputs"] = literal_inputs(spec)
        n["constants"] = script_constants(graph_dir / spec["script"]) if spec.get("script") else []
        n["learnable"] = learnable(opt, n["id"])
        by_owner.setdefault(n["owner"], []).append(n)

    ctx = {
        "title": title, "sim": sim, "task": task, "objects": objects, "cases": cases,
        "case_spec": ",".join(str(c["case"]) for c in cases) if len(cases) < 6 else f"{cases[0]['case']}-{cases[-1]['case']}",
        "started": run.get("started") or "unknown date", "graph_hash": graph_hash(graph_dir),
        "subgraphs": subgraph_summary(wf), "nodes": nodes, "nodes_by_owner": by_owner,
        "graph_png": b64(graph_png), "policy_video": policy_video, "failure_videos": failure_videos,
        "case_videos": case_videos, "all_videos": args.all_videos, "sample": sample, "speed": args.speed,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval_dir": str(eval_dir),
    }
    page = render_html(ctx)
    (out / "report.html").write_text(page)

    summary = {
        "sim": sim, "task": task, "graph_dir": str(graph_dir), "graph_hash": ctx["graph_hash"],
        "evaluated": ctx["started"], "cases": len(cases), "successes": sum(c["success"] for c in cases),
        "per_case": [{k: v for k, v in c.items() if k != "dir"} for c in cases],
        "subgraphs": ctx["subgraphs"],
        "nodes": [{"id": n["id"], "owner": n["owner"], "type": n["spec"].get("type"),
                   "source": n["spec"].get("script") or n["spec"].get("tool"),
                   "inputs": n["inputs"], "constants": n["constants"], "learnable": n["learnable"]} for n in nodes],
        "videos": {str(k): str(v.relative_to(out)) for k, v in videos.items()},
        "sample": sample, "all_videos": args.all_videos,
        "report_bytes": (out / "report.html").stat().st_size,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.keep_media:
        shutil.rmtree(media, ignore_errors=True)
        summary["videos"] = {}
    size_mb = summary["report_bytes"] / MB
    print(f"{out / 'report.html'}  {size_mb:.1f} MB  "
          f"{summary['successes']}/{summary['cases']} successes  "
          f"{len(videos)} video(s)" + ("" if size_mb <= args.max_mb else "  OVER BUDGET"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
