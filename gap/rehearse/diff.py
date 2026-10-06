"""Node-level diff between two workflow definitions.

Used by the feedback report to show an author what changed between rounds:
nodes added or removed, literal inputs (parameters) that changed, edges and
routing that changed, and which script files changed. Subgraph nodes are
flattened to ``<subgraph>.<node>`` ids so the diff reads like the trace.
"""

from __future__ import annotations

from typing import Any


def _flatten_nodes(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, node in (workflow.get("nodes") or {}).items():
        out[name] = node
    for sg_name, sg in (workflow.get("subgraphs") or {}).items():
        for name, node in (sg.get("nodes") or {}).items():
            out[f"{sg_name}.{name}"] = node
    return out


def _edges(workflow: dict[str, Any]) -> set[tuple[str, str]]:
    out = {tuple(e) for e in (workflow.get("edges") or [])}
    for sg_name, sg in (workflow.get("subgraphs") or {}).items():
        for a, b in sg.get("edges") or []:
            out.add((f"{sg_name}.{a}", f"{sg_name}.{b}"))
    return out


def _routing(workflow: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(workflow.get("conditional_edges") or {})
    for sg_name, sg in (workflow.get("subgraphs") or {}).items():
        for src, spec in (sg.get("conditional_edges") or {}).items():
            out[f"{sg_name}.{src}"] = spec
        out[f"{sg_name}#exit"] = {"exit": sg.get("exit"), "on_error": sg.get("on_error")}
    return out


def _literal_inputs(node: dict[str, Any]) -> dict[str, Any]:
    """Inputs bound to literals are parameters; ``$ref`` bindings are wiring."""
    lits: dict[str, Any] = {}
    for key, value in (node.get("inputs") or {}).items():
        if isinstance(value, dict) and "$ref" in value:
            continue
        lits[key] = value
    return lits


def workflow_diff(previous: dict[str, Any] | None, current: dict[str, Any] | None) -> dict[str, Any]:
    """Structured diff of two ``workflow.json`` dictionaries."""
    previous = previous or {}
    current = current or {}
    prev_nodes, cur_nodes = _flatten_nodes(previous), _flatten_nodes(current)
    added = sorted(set(cur_nodes) - set(prev_nodes))
    removed = sorted(set(prev_nodes) - set(cur_nodes))
    changed: dict[str, Any] = {}
    for name in sorted(set(cur_nodes) & set(prev_nodes)):
        a, b = prev_nodes[name], cur_nodes[name]
        entry: dict[str, Any] = {}
        for key in ("type", "tool", "script", "ref"):
            if a.get(key) != b.get(key):
                entry[key] = {"from": a.get(key), "to": b.get(key)}
        la, lb = _literal_inputs(a), _literal_inputs(b)
        params = {k: {"from": la.get(k), "to": lb.get(k)} for k in sorted(set(la) | set(lb)) if la.get(k) != lb.get(k)}
        if params:
            entry["parameters"] = params
        refs_a = {k: v for k, v in (a.get("inputs") or {}).items() if isinstance(v, dict) and "$ref" in v}
        refs_b = {k: v for k, v in (b.get("inputs") or {}).items() if isinstance(v, dict) and "$ref" in v}
        if refs_a != refs_b:
            entry["wiring"] = {"from": refs_a, "to": refs_b}
        if entry:
            changed[name] = entry
    ea, eb = _edges(previous), _edges(current)
    ra, rb = _routing(previous), _routing(current)
    routing_changed = {k: {"from": ra.get(k), "to": rb.get(k)} for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)}
    return {
        "nodes_added": added,
        "nodes_removed": removed,
        "nodes_changed": changed,
        "edges_added": sorted(list(e) for e in eb - ea),
        "edges_removed": sorted(list(e) for e in ea - eb),
        "routing_changed": routing_changed,
    }


def manifest_diff(previous: dict[str, str] | None, current: dict[str, str] | None) -> dict[str, list[str]]:
    """Which files were added, removed, or modified between two manifests."""
    previous = previous or {}
    current = current or {}
    return {
        "added": sorted(set(current) - set(previous)),
        "removed": sorted(set(previous) - set(current)),
        "modified": sorted(k for k in set(current) & set(previous) if current[k] != previous[k]),
    }


def is_empty(diff: dict[str, Any]) -> bool:
    return not any(diff.get(k) for k in diff)


__all__ = ["workflow_diff", "manifest_diff", "is_empty"]
