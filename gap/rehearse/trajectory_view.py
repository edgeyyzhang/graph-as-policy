"""Readable views of the per-step trajectories.

``cases/<case>/trajectory.jsonl`` holds every sample and every body. These
views are what a reader opens: per case, ``main.md`` has one section per
unit of the main graph (a subgraph visit, or a node in a flat graph) and
``<subgraph>.md`` has one section per node of that subgraph.

Two display rules, the same for every task:

- one row every ``interval`` simulator steps, plus every boundary sample;
- a column is shown only when its value changes within the section, beyond
  1 mm, 1 degree, or a change of contacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import trajectory
from .world_state import ANGLE_TOL_DEG, POSITION_TOL_M, changed_bodies, joints_changed, pose_changed

LEGEND = (
    "Privileged simulator state. Positions in metres, orientations as quaternions (w, x, y, z), "
    "both in the robot base frame; joints of articulated objects (drawers, doors, knobs) in metres "
    "for slides and radians for hinges. `i` is the sample index in the case's `trajectory.jsonl`. "
    "A column is shown only when its value changes within the section "
    f"(beyond {POSITION_TOL_M * 1000:.0f} mm, {ANGLE_TOL_DEG:.0f} degree, a change of contacts, "
    "or a joint moving)."
)


def _joints_text(entry: dict[str, Any] | None, owner: str) -> str:
    """``bottom_level=0.120, top_level=0.000``; the owner's name prefix is dropped."""
    joints = (entry or {}).get("joints") or {}
    parts = []
    for name in sorted(joints):
        short = name[len(owner) + 1:] if name.startswith(owner + "_") else name
        parts.append(f"{short}={float(joints[name]):.3f}")
    return ", ".join(parts)


def case_dir(out: Path, case: int) -> Path:
    return Path(out) / "trajectories" / f"case_{int(case):04d}"


def view_path(case: int, graph: str) -> str:
    """Path of a view relative to the rehearsal output directory."""
    return f"trajectories/case_{int(case):04d}/{graph}.md"


def _vec(v: list[float] | None) -> str:
    return "" if v is None else "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _select(samples: list[dict[str, Any]], interval: int) -> list[dict[str, Any]]:
    """Boundary samples, every ``interval``-th step, and the last step."""
    interval = max(1, int(interval))
    last_step = max((i for i, s in enumerate(samples) if s.get("kind") == "step"), default=-1)
    rows, steps = [], 0
    for idx, s in enumerate(samples):
        if s.get("kind") != "step":
            rows.append(s)
            continue
        steps += 1
        if steps % interval == 0 or idx == last_step:
            rows.append(s)
    return rows


def _component_changes(states: list[dict[str, Any]], getter) -> tuple[bool, bool]:
    """(position changed, orientation changed) for one entity over the states."""
    entries = [getter(s) for s in states]
    entries = [e for e in entries if e]
    if len(entries) < 2:
        return False, False
    ref = entries[0]
    position = any(
        pose_changed({"position": ref.get("position")}, {"position": e.get("position")}) for e in entries[1:]
    )
    orientation = any(
        pose_changed(
            {"quaternion_wxyz": ref.get("quaternion_wxyz")}, {"quaternion_wxyz": e.get("quaternion_wxyz")}
        )
        for e in entries[1:]
    )
    return position, orientation


def table(samples: list[dict[str, Any]], interval: int) -> str:
    """Markdown table for one section, or a one-line note when nothing changed."""
    states = [s.get("state") or {} for s in samples]
    hand_pos, hand_rot = _component_changes(states, lambda s: s.get("ee"))
    grips = [s.get("gripper_open_fraction") for s in states if s.get("gripper_open_fraction") is not None]
    gripper = bool(grips) and (max(grips) - min(grips) > 1e-3)
    bodies = changed_bodies(states)

    columns: list[tuple[str, Any]] = []
    if hand_pos:
        columns.append(("hand position", lambda s: _vec((s.get("ee") or {}).get("position"))))
    if hand_rot:
        columns.append(("hand orientation", lambda s: _vec((s.get("ee") or {}).get("quaternion_wxyz"))))
    if gripper:
        columns.append(("gripper opening", lambda s: "" if s.get("gripper_open_fraction") is None
                        else f"{s['gripper_open_fraction']:.3f}"))
    for name in bodies:
        pos, rot = _component_changes(states, lambda s, n=name: (s.get("objects") or {}).get(n))
        contact_sets = {tuple((s.get("objects") or {}).get(name, {}).get("contacts") or []) for s in states}
        if pos:
            columns.append((f"{name} position",
                            lambda s, n=name: _vec(((s.get("objects") or {}).get(n) or {}).get("position"))))
        if rot:
            columns.append((f"{name} orientation",
                            lambda s, n=name: _vec(((s.get("objects") or {}).get(n) or {}).get("quaternion_wxyz"))))
        if len(contact_sets) > 1:
            columns.append((f"{name} contacts",
                            lambda s, n=name: ", ".join(((s.get("objects") or {}).get(n) or {}).get("contacts") or [])
                            or "none"))
        entries = [(s.get("objects") or {}).get(name) for s in states]
        entries = [e for e in entries if e]
        if entries and any(joints_changed(entries[0], e) for e in entries[1:]):
            columns.append((f"{name} joints",
                            lambda s, n=name: _joints_text(((s.get("objects") or {}).get(n)), n)))
    if not columns:
        return "No state change."

    head = ["i", "sample", "time s"] + [c[0] for c in columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in _select(samples, interval):
        state = s.get("state") or {}
        t = state.get("time_s")
        cells = [str(s.get("i")), str(s.get("kind")), "" if t is None else f"{t:.2f}"]
        cells += [fn(state) for _, fn in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _span(samples: list[dict[str, Any]]) -> str:
    steps = sum(1 for s in samples if s.get("kind") == "step")
    times = [(s.get("state") or {}).get("time_s") for s in samples]
    times = [t for t in times if t is not None]
    span = f", simulator time {times[0]:.2f} to {times[-1]:.2f} s" if times else ""
    return f"{steps} simulator steps{span}"


def group(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one case into units (each with its nodes) and samples outside any node."""
    units: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    unit_index: dict[tuple[Any, Any], dict[str, Any]] = {}
    for s in samples:
        if s.get("node") is None:
            outside.append(s)
            continue
        ukey = (s.get("unit"), s.get("unit_visit"))
        unit = unit_index.get(ukey)
        if unit is None:
            unit = {"unit": ukey[0], "visit": ukey[1], "samples": [], "nodes": [], "_nodes": {}}
            unit_index[ukey] = unit
            units.append(unit)
        unit["samples"].append(s)
        nkey = (s.get("node"), s.get("visit"))
        node = unit["_nodes"].get(nkey)
        if node is None:
            node = {"node": nkey[0], "visit": nkey[1], "samples": []}
            unit["_nodes"][nkey] = node
            unit["nodes"].append(node)
        node["samples"].append(s)
    return units, outside


def render_main(case: int, units: list[dict[str, Any]], outside: list[dict[str, Any]], interval: int) -> str:
    out = [f"# Trajectory: case {case}, main graph", "", LEGEND, f"One row every {interval} simulator steps.", ""]
    for u in units:
        nested = any(n["node"] != u["unit"] for n in u["nodes"])
        out.append(f"## {u['unit']} (visit {u['visit']})")
        out.append("")
        out.append(_span(u["samples"]) + ".")
        if nested:
            out.append(f"Per-node tables: `{u['unit']}.md`.")
        out += ["", table(u["samples"], interval), ""]
    if outside:
        out += ["## Outside any node", "",
                "Episode start and end, and steps taken by end-node recovery actions.", "",
                _span(outside) + ".", "", table(outside, interval), ""]
    return "\n".join(out)


def render_unit(case: int, name: str, visits: list[dict[str, Any]], interval: int) -> str:
    out = [f"# Trajectory: case {case}, {name}", "", LEGEND, f"One row every {interval} simulator steps.", ""]
    for u in visits:
        if len(visits) > 1:
            out += [f"## visit {u['visit']}", ""]
        for n in u["nodes"]:
            short = n["node"].split(".", 1)[1] if "." in n["node"] else n["node"]
            out += [f"### {short} (visit {n['visit']})", "", _span(n["samples"]) + ".", "",
                    table(n["samples"], interval), ""]
    return "\n".join(out)


def write_views(out: Path, records: list[dict[str, Any]], *, interval: int = 5) -> list[Path]:
    """Render the views of every case that has a trajectory file."""
    written: list[Path] = []
    for record in records:
        info = record.get("trajectory") or {}
        path = info.get("path")
        if not path or not Path(path).exists():
            continue
        samples = list(trajectory.read(Path(path)))
        if not samples:
            continue
        case = int(record["case"])
        units, outside = group(samples)
        directory = case_dir(out, case)
        directory.mkdir(parents=True, exist_ok=True)
        main = directory / "main.md"
        main.write_text(render_main(case, units, outside, interval), encoding="utf-8")
        written.append(main)
        by_name: dict[str, list[dict[str, Any]]] = {}
        for u in units:
            if any(n["node"] != u["unit"] for n in u["nodes"]):
                by_name.setdefault(u["unit"], []).append(u)
        for name, visits in by_name.items():
            target = directory / f"{name}.md"
            target.write_text(render_unit(case, name, visits, interval), encoding="utf-8")
            written.append(target)
    return written


__all__ = ["write_views", "view_path", "case_dir", "table", "group", "render_main", "render_unit", "LEGEND"]
