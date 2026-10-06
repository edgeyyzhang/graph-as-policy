"""One feedback file per graph, in one format.

A subgraph is a graph whose nodes are tools and scripts; the main graph is a
graph whose nodes are subgraphs (or tools and scripts, when it is flat). Both
get the same file with the same sections::

    feedback/main.md                 the main graph
    feedback/subgraphs/<name>.md     one per subgraph

Sections: graph definition, task success, cases, counts over cases,
checkpoints, changes since the previous round.

For every node visit a file records the three elements of the decision
process the graph runs in: the action (the node's inputs), the state (the
privileged world before and after) and the result (the node's outputs and
its exit). Task success is the simulator's verdict for the case. The files
describe; they do not rank causes or suggest edits.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .diff import _literal_inputs, manifest_diff, workflow_diff
from .trajectory_view import view_path
from .values import compact, mapping_text, text
from .world_state import joints_changed, pose_changed

SECTIONS = (
    "Graph definition",
    "Task success",
    "Cases",
    "Counts over cases",
    "Checkpoints",
    "Changes since the previous round",
)


# ---------------------------------------------------------------------------
# Graph definitions
# ---------------------------------------------------------------------------


def _node_definition(node: dict[str, Any]) -> dict[str, Any]:
    wiring = {k: v["$ref"] for k, v in (node.get("inputs") or {}).items()
              if isinstance(v, dict) and "$ref" in v}
    return {
        "type": node.get("type"),
        "target": node.get("ref") or node.get("tool") or node.get("script") or node.get("status"),
        "parameters": compact(_literal_inputs(node)),
        "wiring": wiring,
    }


def _definition(graph: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "nodes": {name: _node_definition(node) for name, node in (graph.get("nodes") or {}).items()},
        "edges": [list(e) for e in (graph.get("edges") or [])],
        "routing": graph.get("conditional_edges") or {},
    }
    for key in ("skill", "inputs", "outputs", "exit", "on_error"):
        if key in graph:
            out[key] = graph[key]
    return out


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def state_change(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any] | None:
    """Components of the state that differ between two summaries."""
    if not before or not after:
        return None
    out: dict[str, Any] = {}
    eb, ea = before.get("ee") or {}, after.get("ee") or {}
    if pose_changed({"position": eb.get("position")}, {"position": ea.get("position")}):
        out["hand position"] = [eb.get("position"), ea.get("position")]
    if pose_changed({"quaternion_wxyz": eb.get("quaternion_wxyz")}, {"quaternion_wxyz": ea.get("quaternion_wxyz")}):
        out["hand orientation"] = [eb.get("quaternion_wxyz"), ea.get("quaternion_wxyz")]
    gb, ga = before.get("gripper_open_fraction"), after.get("gripper_open_fraction")
    if (gb is None) != (ga is None) or (gb is not None and abs(gb - ga) > 1e-3):
        out["gripper opening"] = [gb, ga]
    ob, oa = before.get("objects") or {}, after.get("objects") or {}
    for name in sorted(set(ob) | set(oa)):
        b, a = ob.get(name) or {}, oa.get(name) or {}
        if pose_changed({"position": b.get("position")}, {"position": a.get("position")}):
            out[f"{name} position"] = [b.get("position"), a.get("position")]
        if pose_changed({"quaternion_wxyz": b.get("quaternion_wxyz")}, {"quaternion_wxyz": a.get("quaternion_wxyz")}):
            out[f"{name} orientation"] = [b.get("quaternion_wxyz"), a.get("quaternion_wxyz")]
        if (b.get("contacts") or []) != (a.get("contacts") or []):
            out[f"{name} contacts"] = [b.get("contacts") or [], a.get("contacts") or []]
        if joints_changed(b, a):
            out[f"{name} joints"] = [b.get("joints") or {}, a.get("joints") or {}]
    return out


def _full_state(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Every component of a state, for the start of a case."""
    if not state:
        return None
    out: dict[str, Any] = {}
    ee = state.get("ee") or {}
    if ee.get("position") is not None:
        out["hand position"] = ee.get("position")
    if state.get("gripper_open_fraction") is not None:
        out["gripper opening"] = state.get("gripper_open_fraction")
    for name, body in (state.get("objects") or {}).items():
        out[f"{name} position"] = body.get("position")
        if body.get("quaternion_wxyz") is not None:
            out[f"{name} orientation"] = body.get("quaternion_wxyz")
        if body.get("joints"):
            out[f"{name} joints"] = body.get("joints")
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _node_row(visit: dict[str, Any], case: int, graph: str, trajectories: bool) -> dict[str, Any]:
    success = visit.get("success")
    name = visit.get("node") or ""
    return {
        "node": name.split(".", 1)[1] if "." in name else name,
        "visit": visit.get("visit"),
        "result": "ok" if success else ("error" if success is False else None),
        "error": success is False,
        "action": visit.get("action"),
        "outputs": visit.get("outputs"),
        "state_change": state_change(visit.get("world_before"), visit.get("world_after")),
        "steps": visit.get("steps"),
        "trajectory": view_path(case, graph) if trajectories and visit.get("steps") is not None else None,
    }


def _subgraph_visits(record: dict[str, Any], workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Every subgraph visit of one case, in order, with its bound inputs."""
    subgraphs = workflow.get("subgraphs") or {}
    visits = record.get("visits") or []
    pool: dict[str, Any] = dict(record.get("inputs") or {})
    out = []
    for ex in record.get("exits") or []:
        name, index = ex.get("subgraph"), ex.get("visit")
        inner = [v for v in visits if v.get("unit") == name and v.get("unit_visit") == index
                 and v.get("node") != name]
        declared = (subgraphs.get(name) or {}).get("inputs") or {}
        # The executor binds a subgraph input to the most recent upstream
        # output of the same name, falling back to the workflow inputs.
        inputs = {k: pool.get(k) for k in declared}
        before = inner[0].get("world_before") if inner else None
        out.append({
            "subgraph": name, "visit": index, "seq": ex.get("seq", 0),
            "exit": ex.get("exit"), "error_path": bool(ex.get("error_path")),
            "inputs": inputs, "outputs": ex.get("outputs") or {},
            "state_before": before, "state_after": ex.get("world"),
            "steps": sum(v.get("steps") or 0 for v in inner) if any(v.get("steps") is not None for v in inner) else None,
            "inner": inner,
        })
        pool.update(ex.get("outputs") or {})
    return out


def _main_rows(record: dict[str, Any], workflow: dict[str, Any], sub_visits: list[dict[str, Any]],
               trajectories: bool) -> list[dict[str, Any]]:
    case = int(record["case"])
    names_by_ref: dict[str, list[str]] = {}
    for name, node in (workflow.get("nodes") or {}).items():
        if node.get("type") == "subgraph" and node.get("ref"):
            names_by_ref.setdefault(node["ref"], []).append(name)
    keyed: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for sv in sub_visits:
        nodes = names_by_ref.get(sv["subgraph"]) or []
        keyed.append(((sv["seq"], 0), {
            "node": nodes[0] if len(nodes) == 1 else sv["subgraph"],
            "subgraph": sv["subgraph"],
            "visit": sv["visit"],
            "result": sv["exit"],
            "error": sv["error_path"],
            "action": sv["inputs"],
            "outputs": sv["outputs"],
            "state_change": state_change(sv["state_before"], sv["state_after"]),
            "steps": sv["steps"],
            "trajectory": view_path(case, "main") if trajectories and sv["steps"] is not None else None,
            "detail": f"feedback/subgraphs/{sv['subgraph']}.md",
        }))
    for v in record.get("visits") or []:
        if "." in (v.get("node") or ""):
            continue
        keyed.append(((v.get("seq", 0), 1), _node_row(v, case, "main", trajectories)))
    return [row for _, row in sorted(keyed, key=lambda kv: kv[0])]


# ---------------------------------------------------------------------------
# Shared sections
# ---------------------------------------------------------------------------


def _success_rows(records: list[dict[str, Any]], previous: dict[int, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        case = int(r["case"])
        before = (previous.get(case) or {}).get("success") if previous else None
        now = r.get("success")
        change = None
        if before is not None and now is not None and before != now:
            change = "fixed" if now else "broken"
        rows.append({"case": case, "success": now, "previous_success": before, "change": change,
                     "graph_exit": r.get("exit_status"), "error": r.get("error")})
    return rows


def _counts(cases: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for c in cases:
        for v in c["visits"]:
            for row in v.get("rows") or []:
                st = stats.setdefault(row["node"], {"visits": 0, "results": Counter(), "errors": 0})
                st["visits"] += 1
                st["results"][str(row.get("result"))] += 1
                st["errors"] += 1 if row.get("error") else 0
    return {k: {"visits": v["visits"], "results": dict(v["results"]), "errors": v["errors"]}
            for k, v in stats.items()}


def _checkpoints(records: list[dict[str, Any]], subgraph: str | None) -> dict[str, Any]:
    """Checkpoint results; ``subgraph`` None keeps every subgraph's."""
    out: dict[str, dict[str, Any]] = {}
    for r in records:
        for cp in r.get("checkpoints") or []:
            if subgraph is not None and cp.get("subgraph") != subgraph:
                continue
            key = cp.get("name") if subgraph is not None else f"{cp.get('subgraph')}.{cp.get('name')}"
            entry = out.setdefault(key, {"passed": 0, "failed": 0, "cases": []})
            entry["passed" if cp.get("passed") else "failed"] += 1
            if subgraph is not None:
                entry["cases"].append({
                    "case": r.get("case"), "passed": bool(cp.get("passed")),
                    "eval_error": cp.get("eval_error"), "diagnostics": compact(cp.get("diagnostics")),
                })
    return out


def _split_changes(wf_diff: dict[str, Any] | None, files: dict[str, list[str]] | None,
                   workflow: dict[str, Any]) -> dict[str, Any]:
    """Assign each change to the graph it belongs to."""
    names = list((workflow.get("subgraphs") or {}))
    empty = lambda: {"nodes_added": [], "nodes_removed": [], "nodes_changed": {}, "edges_added": [],  # noqa: E731
                     "edges_removed": [], "routing_changed": {}, "files": {"added": [], "removed": [], "modified": []}}
    out: dict[str, Any] = {"main": empty()}

    def owner(key: str) -> str:
        head = key.split("#", 1)[0].split(".", 1)[0]
        return head if ("." in key or "#" in key) and head else "main"

    def bucket(name: str) -> dict[str, Any]:
        return out.setdefault(name, empty())

    for key in ("nodes_added", "nodes_removed"):
        for item in (wf_diff or {}).get(key) or []:
            bucket(owner(item))[key].append(item)
    for key in ("nodes_changed", "routing_changed"):
        for item, value in ((wf_diff or {}).get(key) or {}).items():
            bucket(owner(item))[key][item] = value
    for key in ("edges_added", "edges_removed"):
        for edge in (wf_diff or {}).get(key) or []:
            bucket(owner(str(edge[0])) if "." in str(edge[0]) else owner(str(edge[1])))[key].append(edge)

    script_owner: dict[str, str] = {}
    for sg_name, sg in (workflow.get("subgraphs") or {}).items():
        for node in (sg.get("nodes") or {}).values():
            if node.get("script"):
                script_owner.setdefault(node["script"], sg_name)
    for kind, paths in (files or {}).items():
        for path in paths or []:
            if path == "workflow.json":
                continue  # holds every graph; its changes are the entries above
            stem = Path(path).stem
            if path.startswith("checkpoints/") and stem in names:
                target = stem
            else:
                target = script_owner.get(path, "main")
            bucket(target)["files"][kind].append(path)
    return out


def _has_changes(change: dict[str, Any] | None) -> bool:
    if not change:
        return False
    return any(change.get(k) for k in change if k != "files") or any((change.get("files") or {}).values())


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_graph_feedback(
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    workflow: dict[str, Any],
    *,
    previous: Path | None = None,
    trajectories: bool = True,
) -> dict[str, Any]:
    """``{"main": data, "subgraphs": {name: data}}``, every ``data`` in one format."""
    prev_cases: dict[int, Any] = {}
    changes: dict[str, Any] | None = None
    if previous is not None:
        prev_fb = _load(Path(previous) / "feedback.json") or {}
        prev_cases = {int(c["case"]): c for c in prev_fb.get("cases", [])}
        prev_wf = _load(Path(previous) / "workflow.snapshot.json")
        prev_run = _load(Path(previous) / "run.json") or {}
        changes = _split_changes(
            workflow_diff(prev_wf, workflow) if prev_wf is not None else None,
            manifest_diff((prev_run.get("manifest") or {}).get("files"), (meta.get("manifest") or {}).get("files")),
            workflow,
        )
    success = _success_rows(records, prev_cases)
    header = {k: meta.get(k) for k in ("workflow_dir", "sim", "cases", "started")}
    sub_visits = {int(r["case"]): _subgraph_visits(r, workflow) for r in records}

    main_cases = []
    for r in records:
        case = int(r["case"])
        main_cases.append({
            "case": case, "success": r.get("success"), "error": r.get("error"),
            "visits": [{
                "visit": 0, "reached": True,
                "inputs": r.get("inputs") or {},
                "initial_state": _full_state(r.get("initial_world")),
                "rows": _main_rows(r, workflow, sub_visits[case], trajectories),
                "outputs": {}, "exit": r.get("exit_status"), "error_path": bool(r.get("error")),
                "state_change": state_change(r.get("initial_world"), r.get("final_world")),
            }],
        })
    main = {
        "graph": "main", "kind": "main graph", "meta": header,
        "definition": _definition(workflow),
        "success": success, "cases": main_cases, "counts": _counts(main_cases),
        "checkpoints": _checkpoints(records, None),
        "changes": None if changes is None else changes.get("main"),
        "compared": previous is not None,
    }

    subgraphs: dict[str, Any] = {}
    for name, sg in (workflow.get("subgraphs") or {}).items():
        cases = []
        for r in records:
            case = int(r["case"])
            visits = []
            for sv in sub_visits[case]:
                if sv["subgraph"] != name:
                    continue
                visits.append({
                    "visit": sv["visit"], "reached": True, "inputs": sv["inputs"], "initial_state": None,
                    "rows": [_node_row(v, case, name, trajectories) for v in sv["inner"]],
                    "outputs": sv["outputs"], "exit": sv["exit"], "error_path": sv["error_path"],
                    "state_change": state_change(sv["state_before"], sv["state_after"]),
                })
            if not visits:
                visits = [{"visit": None, "reached": False, "rows": []}]
            cases.append({"case": case, "success": r.get("success"), "error": None, "visits": visits})
        subgraphs[name] = {
            "graph": name, "kind": "subgraph", "meta": header,
            "definition": _definition(sg),
            "success": success, "cases": cases, "counts": _counts(cases),
            "checkpoints": _checkpoints(records, name),
            "changes": None if changes is None else changes.get(name),
            "compared": previous is not None,
        }
    return {"main": main, "subgraphs": subgraphs}


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:
        return "" if x is None else str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def _change_lines(change: dict[str, Any] | None, indent: str = "  ") -> list[str]:
    if change is None:
        return [f"{indent}not recorded"]
    if not change:
        return [f"{indent}none"]
    return [f"{indent}{name}: {text(pair[0])} -> {text(pair[1])}" for name, pair in change.items()]


def _outcome(success: Any) -> str:
    return "task succeeded" if success else ("task FAILED" if success is False else "task result unknown")


def render_markdown(data: dict[str, Any]) -> str:
    """The same section list for the main graph and for every subgraph."""
    out = [f"# Feedback: {data['graph']} ({data['kind']})", ""]
    meta = data.get("meta") or {}
    out.append(f"Graph `{meta.get('workflow_dir')}`, sim `{meta.get('sim')}`, cases {meta.get('cases')}.")
    out.append("State is the privileged simulator state: positions in metres and orientations as "
               "quaternions (w, x, y, z), in the robot base frame. A state change lists only the "
               "components that differ between the start and the end of the node.")
    out.append("")

    # 1
    definition = data["definition"]
    out += [f"## 1. {SECTIONS[0]}", ""]
    for key, label in (("skill", "Skill"), ("inputs", "Declared inputs"), ("outputs", "Declared outputs"),
                       ("exit", "Exit"), ("on_error", "On error")):
        if key in definition:
            out.append(f"{label}: {text(compact(definition[key]))}")
    if any(k in definition for k in ("skill", "inputs", "outputs", "exit", "on_error")):
        out.append("")
    out.append(_table(
        ["node", "type", "refers to", "parameters", "wiring"],
        [[n, d["type"], d["target"], mapping_text(d["parameters"]) if d["parameters"] else "",
          mapping_text(d["wiring"]) if d["wiring"] else ""] for n, d in definition["nodes"].items()],
    ))
    out += ["", "Edges: " + (", ".join(f"{a} -> {b}" for a, b in definition["edges"]) or "none")]
    for src, spec in (definition.get("routing") or {}).items():
        mapping = (spec or {}).get("mapping") or {}
        out.append(f"Routing at {src} on `{(spec or {}).get('router_field')}`: "
                   + ", ".join(f"{k} -> {v}" for k, v in mapping.items()))
    out.append("")

    # 2
    out += [f"## 2. {SECTIONS[1]}", ""]
    rows = data["success"]
    n = len(rows)
    wins = sum(1 for r in rows if r["success"])
    line = f"Task success, judged by the simulator: {wins} of {n} cases."
    if data.get("compared"):
        prev = sum(1 for r in rows if r["previous_success"])
        fixed = [r["case"] for r in rows if r["change"] == "fixed"]
        broken = [r["case"] for r in rows if r["change"] == "broken"]
        line += f" Previous round: {prev} of {n}. Fixed: {fixed or 'none'}. Broken: {broken or 'none'}."
    out += [line, ""]
    out.append(_table(
        ["case", "task success", "previous round", "change", "graph exit", "error"],
        [[r["case"], r["success"], r["previous_success"], r["change"], r["graph_exit"], (r["error"] or "")[:120]]
         for r in rows],
    ))
    out.append("")

    # 3
    out += [f"## 3. {SECTIONS[2]}", ""]
    for c in data["cases"]:
        out += [f"### Case {c['case']}: {_outcome(c['success'])}", ""]
        for v in c["visits"]:
            if not v.get("reached"):
                out += ["Not reached in this case.", ""]
                continue
            if len(c["visits"]) > 1:
                out += [f"Visit {v['visit']}.", ""]
            out.append(f"Inputs: {mapping_text(v.get('inputs'))}")
            if v.get("initial_state"):
                out.append("State at the start of the case:")
                out += [f"  {k}: {text(val)}" for k, val in v["initial_state"].items()]
            out.append("")
            for row in v["rows"]:
                label = row["node"] + (f" ({row['subgraph']})" if row.get("subgraph") and row["subgraph"] != row["node"] else "")
                out.append(f"#### {label}, visit {row['visit']}: {row['result']}"
                           + (" (left by the error path)" if row.get("error") and data["kind"] == "main graph"
                              and row.get("subgraph") else ""))
                out.append(f"- action: {mapping_text(row.get('action'))}")
                out.append(f"- outputs: {mapping_text(row.get('outputs'))}")
                out.append("- state change:")
                out += _change_lines(row.get("state_change"), indent="    ")
                if row.get("steps") is not None:
                    out.append(f"- simulator steps: {row['steps']}")
                if row.get("trajectory"):
                    out.append(f"- trajectory: `{row['trajectory']}`")
                if row.get("detail"):
                    out.append(f"- nodes of this subgraph: `{row['detail']}`")
                out.append("")
            out.append(f"Outputs: {mapping_text(v.get('outputs'))}")
            out.append(f"Exit: {v.get('exit')}" + (" (by the error path)" if v.get("error_path") else ""))
            out.append("State change over the whole graph:")
            out += _change_lines(v.get("state_change"))
            out.append("")
        if c.get("error"):
            out += [f"Error: {c['error']}", ""]

    # 4
    out += [f"## 4. {SECTIONS[3]}", ""]
    if data["counts"]:
        out.append(_table(
            ["node", "visits", "results", "errors"],
            [[k, v["visits"], ", ".join(f"{a} {b}" for a, b in v["results"].items()), v["errors"]]
             for k, v in data["counts"].items()],
        ))
    else:
        out.append("No node of this graph ran.")
    out.append("")

    # 5
    out += [f"## 5. {SECTIONS[4]}", ""]
    if data["checkpoints"]:
        out.append(_table(["checkpoint", "passed", "failed"],
                          [[k, v["passed"], v["failed"]] for k, v in data["checkpoints"].items()]))
        for name, cp in data["checkpoints"].items():
            for row in cp.get("cases") or []:
                detail = mapping_text(row.get("diagnostics")) if row.get("diagnostics") else "no diagnostics"
                note = f"; evaluation error: {row['eval_error']}" if row.get("eval_error") else ""
                out.append(f"- {name}, case {row['case']}: {'passed' if row['passed'] else 'failed'}; {detail}{note}")
    else:
        out.append("None declared for this graph.")
    out.append("")

    # 6
    out += [f"## 6. {SECTIONS[5]}", ""]
    if not data.get("compared"):
        out.append("No previous round was given.")
    elif not _has_changes(data.get("changes")):
        out.append("No change to this graph.")
    else:
        ch = data["changes"]
        body = {k: v for k, v in ch.items() if k != "files" and v}
        if body:
            out += ["```json", json.dumps(body, indent=2), "```"]
        files = ch.get("files") or {}
        if any(files.values()):
            out.append(f"Files: added {files.get('added')}, removed {files.get('removed')}, "
                       f"modified {files.get('modified')}")
    out.append("")
    return "\n".join(out)


def write_graph_feedback(
    out: Path,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    workflow_dir: Path,
    previous: Path | None = None,
    trajectories: bool = True,
) -> dict[str, Any]:
    """Write ``feedback/main.{md,json}`` and ``feedback/subgraphs/<name>.{md,json}``."""
    workflow = _load(Path(workflow_dir) / "workflow.json") or {}
    data = build_graph_feedback(records, meta, workflow, previous=previous, trajectories=trajectories)
    root = Path(out) / "feedback"
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.json").write_text(json.dumps(data["main"], indent=2, default=str))
    (root / "main.md").write_text(render_markdown(data["main"]), encoding="utf-8")
    if data["subgraphs"]:
        (root / "subgraphs").mkdir(exist_ok=True)
    for name, sub in data["subgraphs"].items():
        (root / "subgraphs" / f"{name}.json").write_text(json.dumps(sub, indent=2, default=str))
        (root / "subgraphs" / f"{name}.md").write_text(render_markdown(sub), encoding="utf-8")
    return data


__all__ = ["build_graph_feedback", "write_graph_feedback", "render_markdown", "state_change", "SECTIONS"]
