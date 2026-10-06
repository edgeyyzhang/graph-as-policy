"""Load rehearsal records and flatten them into per-unit rows.

A *unit* is the thing the author reasons about: a subgraph visit when the
graph has subgraphs, otherwise a top-level node visit (flat graphs). Each
unit row carries the graph's own verdict (exit value or ok/error) beside the
privileged world at that moment, which is what the report compares.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .world_state import diff as world_diff


def load_records(out_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((Path(out_dir) / "cases").glob("case_*/case.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return records


def trace_node_errors(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-node error messages from the case's ``dag_trace.json`` if present."""
    trace_dir = record.get("trace_dir")
    if not trace_dir:
        return []
    path = Path(trace_dir) / "dag_trace.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    out = []
    for node in data.get("nodes", []):
        if node.get("status") == "error" or node.get("error_message"):
            out.append({"node": node.get("name"), "message": node.get("error_message"), "status": node.get("status")})
    return out


def has_subgraphs(record: dict[str, Any]) -> bool:
    return bool(record.get("exits"))


def unit_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Ordered units for one case with the graph verdict and the world at exit."""
    initial = record.get("initial_world")
    rows: list[dict[str, Any]] = []
    if has_subgraphs(record):
        for ex in record.get("exits", []):
            world = ex.get("world")
            rows.append({
                "case": record.get("case"),
                "unit": ex.get("subgraph"),
                "visit": ex.get("visit"),
                "verdict": ex.get("exit"),
                "error_path": bool(ex.get("error_path")),
                "elapsed_s": ex.get("elapsed_s"),
                "world": world,
                "since_start": world_diff(initial, world) if world is not None else None,
            })
        return rows
    for visit in record.get("visits", []):
        name = visit.get("node") or ""
        if "." in name:
            continue  # inner node of a subgraph; the subgraph exit is the unit
        world = visit.get("world_after")
        success = visit.get("success")
        rows.append({
            "case": record.get("case"),
            "unit": name,
            "visit": visit.get("visit"),
            "verdict": "ok" if success else ("error" if success is False else None),
            "error_path": success is False,
            "elapsed_s": None if not visit.get("t_start") or not visit.get("t_end") else round(visit["t_end"] - visit["t_start"], 3),
            "world": world,
            "since_start": world_diff(initial, world) if world is not None else None,
            "effect": world_diff(visit.get("world_before"), world) if world is not None else None,
        })
    return rows


def checkpoint_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cp in record.get("checkpoints", []) or []:
        rows.append({
            "case": record.get("case"),
            "unit": cp.get("subgraph"),
            "name": cp.get("name"),
            "passed": bool(cp.get("passed")),
            "eval_error": cp.get("eval_error"),
            "diagnostics": cp.get("diagnostics"),
        })
    return rows


__all__ = ["load_records", "unit_rows", "checkpoint_rows", "trace_node_errors", "has_subgraphs"]
