"""Align two ``dag_trace.json`` files by node name and report agreement.

The two paths (rehearsal sandbox vs. real LIBERO) execute the same
``workflow.json`` so node names line up 1:1 — alignment is a name-keyed
join, not a fuzzy match. Agreement is reported at three granularities:

- *verdict* — did both paths reach the same exit status (``ok`` vs
  not-``ok``)? This is the load-bearing signal: if the workflow exited
  identically on both paths, trial-level pass/fail will agree too.
- *status string* — exact ``status`` match (``ok`` / ``error`` /
  ``skipped`` / ``running``). Strictly stronger than verdict; surfaces
  cases where one path errored and the other was skipped.
- *condition.met* — for nodes that gate downstream branches via a
  condition, did both paths' conditions evaluate the same way? A
  divergence here typically points at perception or sensor drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_dag_trace(trace_dir: Path) -> dict[str, Any]:
    """Load ``dag_trace.json`` from a trace directory.

    ``trace_dir`` may be either the directory containing ``dag_trace.json``
    or the file path itself; we accept both for caller convenience.
    """
    p = Path(trace_dir)
    if p.is_dir():
        p = p / "dag_trace.json"
    if not p.exists():
        raise FileNotFoundError(f"dag_trace.json not found at {p}")
    return json.loads(p.read_text())


def _index_by_name(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return ``{node_name: node_dict}`` from a parsed ``dag_trace.json``."""
    return {n["name"]: n for n in trace.get("nodes", [])}


# ---------------------------------------------------------------------------
# Per-node agreement
# ---------------------------------------------------------------------------


def _is_ok(status: str | None) -> bool:
    """Verdict-level success: exact match against ``ok``."""
    return status == "ok"


def _condition_met(node: dict[str, Any]) -> bool | None:
    cr = node.get("condition_result")
    if not isinstance(cr, dict):
        return None
    met = cr.get("met")
    if met is None:
        return None
    return bool(met)


@dataclass(frozen=True)
class NodeAgreement:
    name: str
    rehearsal_status: str | None
    real_status: str | None
    verdict_agree: bool
    """Both paths either ``ok`` or both not-``ok``."""
    status_agree: bool
    """Status strings match exactly."""
    rehearsal_duration_ms: float | None
    real_duration_ms: float | None
    rehearsal_error: str | None
    real_error: str | None
    rehearsal_condition_met: bool | None
    real_condition_met: bool | None
    condition_agree: bool | None
    """``None`` when neither side has a condition; else exact bool match."""

    @property
    def in_both(self) -> bool:
        return self.rehearsal_status is not None and self.real_status is not None


def _build_node_agreement(
    name: str,
    reh: dict[str, Any] | None,
    real: dict[str, Any] | None,
) -> NodeAgreement:
    reh_status = reh["status"] if reh else None
    real_status = real["status"] if real else None
    verdict_agree = (
        reh is not None and real is not None
        and _is_ok(reh_status) == _is_ok(real_status)
    )
    status_agree = (
        reh is not None and real is not None
        and reh_status == real_status
    )
    reh_cond = _condition_met(reh) if reh else None
    real_cond = _condition_met(real) if real else None
    if reh_cond is None and real_cond is None:
        condition_agree: bool | None = None
    else:
        condition_agree = (reh_cond == real_cond)
    return NodeAgreement(
        name=name,
        rehearsal_status=reh_status,
        real_status=real_status,
        verdict_agree=verdict_agree,
        status_agree=status_agree,
        rehearsal_duration_ms=(reh.get("duration_ms") if reh else None),
        real_duration_ms=(real.get("duration_ms") if real else None),
        rehearsal_error=(reh.get("error_message") if reh else None),
        real_error=(real.get("error_message") if real else None),
        rehearsal_condition_met=reh_cond,
        real_condition_met=real_cond,
        condition_agree=condition_agree,
    )


# ---------------------------------------------------------------------------
# Aggregate diff
# ---------------------------------------------------------------------------


@dataclass
class TraceDiff:
    rehearsal_dir: str
    real_dir: str
    nodes: list[NodeAgreement] = field(default_factory=list)
    rehearsal_only: list[str] = field(default_factory=list)
    """Node names present only on the rehearsal side."""
    real_only: list[str] = field(default_factory=list)

    @property
    def matched(self) -> list[NodeAgreement]:
        return [n for n in self.nodes if n.in_both]

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def verdict_agreement_rate(self) -> float:
        if not self.matched:
            return 0.0
        return sum(1 for n in self.matched if n.verdict_agree) / len(self.matched)

    @property
    def status_agreement_rate(self) -> float:
        if not self.matched:
            return 0.0
        return sum(1 for n in self.matched if n.status_agree) / len(self.matched)

    @property
    def first_divergence(self) -> NodeAgreement | None:
        """First matched node where verdict_agree is False (DAG-order)."""
        for n in self.matched:
            if not n.verdict_agree:
                return n
        return None


def diff_trace_dirs(rehearsal_dir: Path, real_dir: Path) -> TraceDiff:
    """Compute the alignment + agreement between two trace directories.

    The trace directories must each contain a ``dag_trace.json`` written by
    :class:`gap.runtime.tracing.DagTrace`. Pass either the directory or the
    file path itself.
    """
    reh_trace = _load_dag_trace(rehearsal_dir)
    real_trace = _load_dag_trace(real_dir)
    reh = _index_by_name(reh_trace)
    real = _index_by_name(real_trace)

    # Preserve DAG order from the rehearsal trace, then append real-only
    # nodes in their own order. This makes ``first_divergence`` walk the
    # workflow in the order the rehearsal author understood it.
    seen: set[str] = set()
    ordered_names: list[str] = []
    for node in reh_trace.get("nodes", []):
        n = node["name"]
        if n not in seen:
            ordered_names.append(n)
            seen.add(n)
    for node in real_trace.get("nodes", []):
        n = node["name"]
        if n not in seen:
            ordered_names.append(n)
            seen.add(n)

    nodes = [
        _build_node_agreement(n, reh.get(n), real.get(n))
        for n in ordered_names
    ]
    rehearsal_only = sorted(set(reh.keys()) - set(real.keys()))
    real_only = sorted(set(real.keys()) - set(reh.keys()))
    return TraceDiff(
        rehearsal_dir=str(Path(rehearsal_dir).resolve()),
        real_dir=str(Path(real_dir).resolve()),
        nodes=nodes,
        rehearsal_only=rehearsal_only,
        real_only=real_only,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def to_json_dict(diff: TraceDiff) -> dict[str, Any]:
    return {
        "rehearsal_dir": diff.rehearsal_dir,
        "real_dir": diff.real_dir,
        "matched_count": diff.matched_count,
        "verdict_agreement_rate": round(diff.verdict_agreement_rate, 4),
        "status_agreement_rate": round(diff.status_agreement_rate, 4),
        "first_divergence": (
            asdict(diff.first_divergence) if diff.first_divergence else None
        ),
        "rehearsal_only": diff.rehearsal_only,
        "real_only": diff.real_only,
        "nodes": [asdict(n) for n in diff.nodes],
    }


def format_summary(diff: TraceDiff) -> str:
    """Render a tight text summary suitable for printing to stdout."""
    lines: list[str] = []
    lines.append(f"rehearsal: {diff.rehearsal_dir}")
    lines.append(f"real:      {diff.real_dir}")
    lines.append(
        f"matched: {diff.matched_count}  "
        f"verdict-agreement: {diff.verdict_agreement_rate:.2%}  "
        f"status-agreement: {diff.status_agreement_rate:.2%}"
    )
    if diff.rehearsal_only:
        lines.append(f"rehearsal-only nodes: {', '.join(diff.rehearsal_only)}")
    if diff.real_only:
        lines.append(f"real-only nodes:      {', '.join(diff.real_only)}")
    fd = diff.first_divergence
    if fd is None:
        lines.append("first divergence: <none — all matched nodes agree>")
    else:
        lines.append(
            f"first divergence: {fd.name}  "
            f"(rehearsal={fd.rehearsal_status!r}, real={fd.real_status!r})"
        )
    lines.append("")
    lines.append(
        f"{'node':<40}  {'rehearsal':<10}  {'real':<10}  "
        f"{'verdict':<8}  {'status':<7}"
    )
    for n in diff.nodes:
        v = "agree" if n.verdict_agree else "DIFFER"
        s = "agree" if n.status_agree else "DIFFER"
        if not n.in_both:
            v = "missing"
            s = "missing"
        lines.append(
            f"{n.name:<40}  "
            f"{(n.rehearsal_status or '-'):<10}  "
            f"{(n.real_status or '-'):<10}  "
            f"{v:<8}  {s:<7}"
        )
    return "\n".join(lines)
