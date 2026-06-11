"""DAG trace alignment between rehearsal and real runs.

Both paths emit ``dag_trace.json`` files via :mod:`gap.runtime.tracing` with
the same node-name keys (workflow.json node ids are stable across paths).
This package aligns the two by name, compares per-node status / condition /
duration, and reports an aggregate agreement rate.

Use :func:`diff_trace_dirs` to compare two trace directories; use
:func:`format_summary` to render a one-screen text summary, or
:func:`to_json_dict` to ship the structured diff to disk.
"""

from __future__ import annotations

from gap.runtime.trace_diff.differ import (
    NodeAgreement,
    TraceDiff,
    diff_trace_dirs,
    format_summary,
    to_json_dict,
)

__all__ = [
    "NodeAgreement",
    "TraceDiff",
    "diff_trace_dirs",
    "format_summary",
    "to_json_dict",
]
