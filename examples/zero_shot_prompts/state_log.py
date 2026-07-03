"""Ground-truth state recording for one graph execution.

``StateRecorder`` shadows ``conn.tool_registry.invoke`` (an instance
attribute assignment — every script/tool node call in the runtime goes
through this single dispatch point, see ``ExecutionContext.tool`` in
``gap/runtime/context.py``) and, around every tool call, appends:

- ``events.jsonl`` — one line per call/return/error, flushed
  immediately. Because the "call" line is written BEFORE dispatch, a
  hung tool leaves an unmatched "call" as the last line: the hang site
  survives a process kill and is recovered post-mortem.
- ``state.jsonl`` — one compact ground-truth snapshot per completed
  call (plus ``__initial__`` / ``__final__`` bookends), built from
  ``conn.world_snapshot()`` (a ``gap.runtime.verify.World`` in the
  robot-base frame). Criteria evaluate these dicts, so a killed run is
  still scoreable from whatever was flushed.

Snapshots are taken on the calling thread, immediately after the tool
returns — never mid-physics-step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _f(v: Any) -> list[float]:
    return [round(float(x), 5) for x in np.asarray(v).ravel().tolist()]


def _kw_summary(kw: dict[str, Any]) -> dict[str, Any]:
    """Small, JSON-safe view of tool kwargs (no arrays, no blobs)."""
    out: dict[str, Any] = {}
    for k, v in kw.items():
        if k == "ctx":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if not isinstance(v, str) else v[:120]
        elif isinstance(v, np.ndarray):
            out[k] = f"<array {v.shape}>"
        elif isinstance(v, (list, tuple)):
            if len(v) <= 8 and all(isinstance(x, (int, float)) for x in v):
                out[k] = [round(float(x), 4) for x in v]
            else:
                out[k] = f"<list len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = f"<dict keys={sorted(v)[:8]}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def _result_summary(res: Any, depth: int = 0) -> Any:
    """Numeric-leaf summary of a tool result (capped size).

    Keeps scalars, short numeric lists, and status-ish strings so
    diagnostics like 'what bounding box did geometry return' can be
    answered from the event log without re-running anything.
    """
    if depth > 3:
        return "<depth>"
    if isinstance(res, (int, float, bool)) or res is None:
        return res
    if isinstance(res, str):
        return res[:160]
    if isinstance(res, np.ndarray):
        if res.size <= 12:
            return _f(res)
        return f"<array {res.shape}>"
    if isinstance(res, (list, tuple)):
        if len(res) <= 12 and all(
            isinstance(x, (int, float, bool)) for x in res
        ):
            return [round(float(x), 5) for x in res]
        return f"<list len={len(res)}>"
    if isinstance(res, dict):
        out = {}
        for i, (k, v) in enumerate(res.items()):
            if i >= 24:
                out["..."] = f"<{len(res) - 24} more>"
                break
            out[str(k)] = _result_summary(v, depth + 1)
        return out
    return f"<{type(res).__name__}>"


class StateRecorder:
    def __init__(self, conn: Any, out_dir: str | Path) -> None:
        self.conn = conn
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # "w", not "a": a seed dir records exactly one run attempt — a
        # resume after a mid-run kill must not prepend the stale trace.
        self._events = open(self.out_dir / "events.jsonl", "w")
        self._states = open(self.out_dir / "state.jsonl", "w")
        self._t0 = time.monotonic()
        self._seq = 0
        self._installed = False
        self._orig: Any = None

    # -- event / state emission ----------------------------------------

    def _emit(self, fh, payload: dict[str, Any]) -> None:
        fh.write(json.dumps(payload, default=str) + "\n")
        fh.flush()

    def _now(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def snapshot(self, tag: str, node: str | None = None) -> None:
        """Append one ground-truth state line (best-effort)."""
        try:
            w = self.conn.world_snapshot()
        except Exception as exc:
            self._emit(self._states, {
                "seq": self._seq, "t": self._now(), "tool": tag,
                "node": node, "snapshot_error": str(exc)[:300],
            })
            return
        try:
            held = w.held_body()
            held_name = held.name if held is not None else None
        except Exception:
            held_name = None
        rv = w.robot_view
        bodies = {}
        for name in w.body_names():
            b = w.body(name)
            bodies[name] = {
                "p": _f(b.position),
                "q": _f(b.quaternion_wxyz),
                "lo": _f(b.aabb_lower),
                "hi": _f(b.aabb_upper),
                "v": round(float(np.linalg.norm(b.linear_velocity)), 4),
                "g": bool(b.is_grasped()),
            }
        self._emit(self._states, {
            "seq": self._seq,
            "t": self._now(),
            "tool": tag,
            "node": node,
            "held": held_name,
            "grip": None if rv is None else round(
                float(rv.gripper_open_fraction), 4),
            "ee": None if rv is None else _f(rv.ee_position),
            "bodies": bodies,
        })

    # -- invoke wrapping -------------------------------------------------

    def install(self) -> None:
        reg = self.conn.tool_registry
        self._orig = reg.invoke
        recorder = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            name = str(args[0]) if args else str(kw.get("name", "?"))
            ctx = kw.get("ctx")
            node = getattr(ctx, "_node_id", None)
            recorder._seq += 1
            seq = recorder._seq
            recorder._emit(recorder._events, {
                "ev": "call", "seq": seq, "t": recorder._now(),
                "tool": name, "node": node, "kw": _kw_summary(kw),
            })
            try:
                res = recorder._orig(*args, **kw)
            except BaseException as exc:
                recorder._emit(recorder._events, {
                    "ev": "error", "seq": seq, "t": recorder._now(),
                    "tool": name, "node": node,
                    "err": f"{type(exc).__name__}: {exc}"[:400],
                })
                recorder.snapshot(name, node)
                raise
            recorder._emit(recorder._events, {
                "ev": "done", "seq": seq, "t": recorder._now(),
                "tool": name, "node": node,
                "summary": _result_summary(res),
            })
            recorder.snapshot(name, node)
            return res

        reg.invoke = wrapped
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            try:
                del self.conn.tool_registry.invoke  # restore class method
            except AttributeError:
                pass
            self._installed = False
        for fh in (self._events, self._states):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


# -- loading -------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn final line from a killed process
    return out


def hang_site(events: list[dict]) -> dict | None:
    """The last 'call' event with no matching 'done'/'error' — where a
    killed run was stuck."""
    open_calls: dict[int, dict] = {}
    for ev in events:
        if ev.get("ev") == "call":
            open_calls[int(ev.get("seq", -1))] = ev
        elif ev.get("ev") in ("done", "error"):
            open_calls.pop(int(ev.get("seq", -1)), None)
    if not open_calls:
        return None
    return open_calls[max(open_calls)]
