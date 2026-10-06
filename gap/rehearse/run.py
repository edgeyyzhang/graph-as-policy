"""Rehearse a fixed graph over fixed simulator cases and record what happened.

One :class:`~gap.connector.sim.SimConnector` is built per call and reset to
each case's initial state (``reset(seed=case)``; the LIBERO loaders map seed
``k`` to init ``(k-1) % n``). Every case runs through
:func:`gap.runtime.execute.execute`; the harness attaches itself through
``on_executor`` and records, without touching the executor:

- the world (privileged snapshot, summarized) before and after every node
  visit, via the tracer's ``on_node_start`` / ``on_node_end`` hooks;
- every subgraph exit (value, error path, elapsed) via ``subgraph_exit_hook``;
- checkpoint results, the graph's exit status, the native task predicate, and
  the error, from the :class:`ExecutionResult` and the connector.

Per case a ``case_<id>.json`` lands in ``out/cases/``, next to the GaP trace
directory for that case. :mod:`gap.rehearse.report` turns the set into the
author-facing feedback files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from .trajectory import StepSampler
from .values import compact
from .world_state import summarize

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def load_tools_plugin(spec: str | None) -> Callable[[Any], None] | None:
    """Resolve ``module:function`` to a callable taking the connector."""
    if not spec:
        return None
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"--tools expects module:function, got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def graph_manifest(workflow_dir: Path) -> dict[str, Any]:
    """Content hashes of every workflow file, for round-to-round diffs."""
    files: dict[str, str] = {}
    for path in sorted(workflow_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in (".json", ".py", ".md", ".yaml", ".yml", ".txt"):
            continue
        files[str(path.relative_to(workflow_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files}


def _save_frame(connector: Any, path: Path) -> str | None:
    try:
        obs = connector.get_observation()
        cameras = obs.get("cameras") if isinstance(obs, dict) else getattr(obs, "cameras", None)
        if not cameras:
            return None
        cam = cameras[0]
        rgb = cam.get("rgb") if isinstance(cam, dict) else getattr(cam, "rgb", None)
        if rgb is None:
            return None
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path)
        return str(path)
    except Exception:
        logger.debug("frame capture failed", exc_info=True)
        return None


class _CaseRecorder:
    """Collects node visits, subgraph exits and frames for one case."""

    def __init__(self, connector: Any, case_dir: Path, *, objects: list[str] | None, frames: bool,
                 sampler: StepSampler | None = None):
        self.connector = connector
        self.case_dir = case_dir
        self.objects = objects
        self.frames = frames
        self.sampler = sampler
        self._unit_exits: dict[str, int] = {}
        self.visits: list[dict[str, Any]] = []
        self.exits: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}
        self._counts: dict[str, int] = {}
        self.snapshot_errors = 0

    def _world(self) -> dict[str, Any] | None:
        fn = getattr(self.connector, "world_snapshot", None)
        if fn is None:
            return None
        try:
            return summarize(fn(), self.objects)
        except Exception:
            self.snapshot_errors += 1
            logger.debug("world snapshot failed", exc_info=True)
            return None

    def on_node_start(self, name: str, _: Any) -> None:
        index = self._counts.get(name, 0)
        self._counts[name] = index + 1
        # The unit is the graph the node belongs to: its subgraph, or the node
        # itself in a flat graph. A subgraph visit ends at its exit.
        unit = name.split(".", 1)[0] if "." in name else name
        unit_visit = self._unit_exits.get(unit, 0) if "." in name else index
        row = {
            "node": name,
            "visit": index,
            "unit": unit,
            "unit_visit": unit_visit,
            "seq": len(self.visits),
            "t_start": time.time(),
            "world_before": self._world(),
        }
        self._open[name] = row
        self.visits.append(row)
        if self.sampler is not None:
            self.sampler.set_node(name, index, unit, unit_visit)
            self.sampler.mark("start")

    def on_node_end(self, name: str, success: Any) -> None:
        row = self._open.pop(name, None)
        if row is None:
            row = {"node": name, "visit": self._counts.get(name, 0), "seq": len(self.visits),
                   "t_start": None, "world_before": None}
            self.visits.append(row)
        row["t_end"] = time.time()
        row["success"] = None if success is None else bool(success)
        row["world_after"] = self._world()
        if self.sampler is not None:
            self.sampler.mark("end")
            row["steps"] = self.sampler.node_steps
            self.sampler.clear_node()
        if self.frames:
            safe = name.replace("/", "_")
            row["frame"] = _save_frame(
                self.connector, self.case_dir / "frames" / f"{row['seq']:03d}_{safe}_v{row['visit']}.png"
            )

    def on_subgraph_exit(self, event: Any) -> None:
        name = getattr(event, "sg_name", None)
        if name is not None:
            self._unit_exits[name] = self._unit_exits.get(name, 0) + 1
        self.exits.append({
            "subgraph": name,
            "visit": getattr(event, "visit_index", None),
            "exit": getattr(event, "exit_value", None),
            "outputs": compact(getattr(event, "bound_outputs", None) or {}),
            "error_path": bool(getattr(event, "error_path", False)),
            "elapsed_s": getattr(event, "elapsed_s", None),
            "seq": len(self.visits),
            "world": self._world(),
        })

    def attach(self, executor: Any) -> None:
        executor.subgraph_exit_hook = self.on_subgraph_exit
        executor.trace.on_node_start = self.on_node_start
        executor.trace.on_node_end = self.on_node_end


def _trace_value(trace_dir: Path, node: str, visit: int, name: str) -> Any:
    """A node visit's resolved inputs or output as the tracer wrote them."""
    path = trace_dir / "node_data" / node / "iters" / f"{visit:03d}" / name
    if not path.exists():
        return None
    try:
        return compact(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def run_case(
    workflow_dir: Path,
    connector: Any,
    case: int,
    case_dir: Path,
    *,
    skills: Any = None,
    checkpoints: str = "warn",
    max_node_workers: int = 1,
    objects: list[str] | None = None,
    frames: bool = False,
    inputs: dict[str, Any] | None = None,
    sampler: StepSampler | None = None,
    video: bool = False,
    video_interval: int = 5,
) -> dict[str, Any]:
    """Reset the simulator to ``case`` and run the graph once.

    With ``video`` (and a sampler that has a step handle) the exterior camera
    image of every ``video_interval``-th step is written to ``video.mp4``.
    """
    from gap.runtime.execute import execute

    case_dir.mkdir(parents=True, exist_ok=True)
    recorder = _CaseRecorder(connector, case_dir, objects=objects, frames=frames, sampler=sampler)
    started = time.time()
    connector.reset(seed=case)
    initial_world = recorder._world()
    if sampler is not None:
        # Opened after the reset so the simulator's own reset steps stay out.
        sampler.open(case_dir / "trajectory.jsonl",
                     video=(case_dir / "video.mp4") if video else None, video_every=video_interval)
        sampler.mark("initial")
    result = execute(
        workflow_dir,
        connector,
        skills=skills,
        inputs=inputs,
        trace_dir=case_dir / "trace",
        checkpoints=checkpoints,
        max_node_workers=max_node_workers,
        on_executor=recorder.attach,
    )
    native_success, reward = None, None
    check = getattr(connector, "check_success", None)
    if check is not None:
        try:
            native_success, reward = check()
            native_success = bool(native_success)
            reward = float(reward)
        except Exception:
            logger.debug("check_success failed", exc_info=True)
    final_world = recorder._world()
    trajectory = None
    if sampler is not None:
        sampler.clear_node()
        sampler.mark("final")
        sampler.close()
        trajectory = {
            "path": str(case_dir / "trajectory.jsonl"),
            "video": str(sampler.video_written) if sampler.video_written else None,
            "steps": sampler.steps,
            "per_step": sampler.steps_available,
            "errors": sampler.errors,
        }
    # The action of a node is its resolved inputs; both come from the trace.
    for row in recorder.visits:
        row["action"] = _trace_value(case_dir / "trace", row["node"], row["visit"], "resolved_inputs.json")
        row["outputs"] = _trace_value(case_dir / "trace", row["node"], row["visit"], "output.json")
    checkpoint_rows = []
    for cp in getattr(result, "checkpoint_results", None) or []:
        checkpoint_rows.append(cp.to_dict() if hasattr(cp, "to_dict") else _json_safe(cp))
    record = {
        "case": case,
        "success": native_success,
        "reward": reward,
        "graph_success": bool(result.success),
        "exit_status": result.exit_status,
        "error": None if result.error is None else f"{type(result.error).__name__}: {result.error}",
        "duration_s": round(float(result.duration_s), 3),
        "latency": _json_safe(result.latency),
        "wall_s": round(time.time() - started, 3),
        "trace_dir": str(case_dir / "trace"),
        "trajectory": trajectory,
        "inputs": compact(inputs or {}),
        "initial_world": initial_world,
        "final_world": final_world,
        "visits": recorder.visits,
        "exits": recorder.exits,
        "checkpoints": checkpoint_rows,
        "snapshot_errors": recorder.snapshot_errors,
    }
    (case_dir / "case.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def select_ik(connector: Any, ik: str | None) -> None:
    """Choose the connector's inverse-kinematics backend before it is first used.

    ``pyroki`` selects the in-process PyRoKi backend, for machines without
    cuRobo. ``None`` or ``curobo`` keeps the connector's default. A backend
    the caller already set is left alone.
    """
    if ik in (None, "curobo"):
        return
    if ik != "pyroki":
        raise ValueError(f"unknown ik backend {ik!r}; expected 'curobo' or 'pyroki'")
    if getattr(connector, "_ik", None) is not None:
        return
    from gap.connector.ik import PyRokiBackend

    config = connector.config
    connector._ik = PyRokiBackend(tcp_offset=config.tcp_offset, home_joints=config.home_joints)


def boot_bundles(workflow_dir: Path, connector: Any, skills: Any = None) -> Any | None:
    """Boot the workflow's RPC tool bundles once, for every case of a rehearsal.

    :func:`gap.runtime.execute.execute` boots and shuts down bundle servers
    per call, but their registrations stay on the connector's registry, so a
    second call on the same connector would dispatch to closed servers.
    Booting here first means each ``execute`` finds the tools already
    registered and neither boots nor shuts anything down. Returns the manager
    (the caller shuts it down) or ``None`` when nothing needs booting.
    """
    tool_registry = getattr(connector, "tool_registry", None)
    if tool_registry is None:
        return None  # execute() builds a fresh registry per call
    from gap.runtime.execute import _BUNDLE_TOOL_CATALOG
    from gap.runtime.tool_bundle_boot import boot_tool_bundles
    from gap.skills import load_registry_set, resolve_registries

    registry_set = resolve_registries(skills)
    if not registry_set:
        return None
    skill_registry = load_registry_set(registry_set)
    if hasattr(tool_registry, "discover_pending"):
        tool_registry.discover_pending(catalog=_BUNDLE_TOOL_CATALOG)
    return boot_tool_bundles(workflow_dir, skill_registry, tool_registry)


def parse_cases(spec: str | list[int]) -> list[int]:
    """``"1-4,7"`` → ``[1, 2, 3, 4, 7]``."""
    if isinstance(spec, list):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    if not out or len(set(out)) != len(out):
        raise ValueError(f"cases must be a non-empty list of distinct integers, got {spec!r}")
    return out


def rehearse(
    workflow_dir: str | Path,
    *,
    sim: str,
    cases: str | list[int],
    out: str | Path,
    env: str = "libero",
    skills: Any = None,
    tools: str | Callable[[Any], None] | None = None,
    checkpoints: str = "warn",
    max_node_workers: int = 1,
    objects: list[str] | None = None,
    frames: bool = False,
    previous: str | Path | None = None,
    claims: dict[str, str] | None = None,
    inputs: dict[str, Any] | None = None,
    connector: Any = None,
    trajectory: bool = True,
    trajectory_interval: int = 5,
    ik: str | None = None,
    video: bool = False,
    video_interval: int = 5,
) -> dict[str, Any]:
    """Run every case, write per-case records, then the feedback files.

    Returns the feedback dictionary (also written to ``out/feedback.json``).
    ``connector`` lets tests inject a prebuilt connector; otherwise one is
    built with :func:`gap.connector.sim.sim` for ``env`` and ``sim``.

    With ``trajectory`` the privileged state is sampled at every simulator
    step into ``cases/<case>/trajectory.jsonl`` and rendered, one row every
    ``trajectory_interval`` steps, under ``trajectories/``. The per-graph
    feedback files under ``feedback/`` are written either way.
    """
    from .graph_feedback import write_graph_feedback
    from .report import build_feedback, write_feedback
    from .trajectory_view import write_views

    # Absolute paths: simulator construction may change the working directory.
    workflow_dir = Path(workflow_dir).resolve()
    out = Path(out).resolve()
    if previous is not None:
        previous = Path(previous).resolve()
    (out / "cases").mkdir(parents=True, exist_ok=True)
    case_ids = parse_cases(cases)
    plugin = load_tools_plugin(tools) if isinstance(tools, str) else tools

    own_connector = connector is None
    if own_connector:
        from gap.connector.sim import sim as build_sim

        connector = build_sim(env, task=sim, headless=not frames)
    select_ik(connector, ik)
    if plugin is not None:
        plugin(connector)

    meta = {
        "workflow_dir": str(workflow_dir),
        "sim": sim,
        "env": env,
        "cases": case_ids,
        "checkpoints": checkpoints,
        "objects": objects,
        "frames": frames,
        "trajectory": trajectory,
        "trajectory_interval": trajectory_interval,
        "video": video,
        "video_interval": video_interval,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": graph_manifest(workflow_dir),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    # Keep the executed graph next to its results so the next round can diff
    # against it without access to the author's working copy.
    wf = workflow_dir / "workflow.json"
    if wf.exists():
        (out / "workflow.snapshot.json").write_text(wf.read_text())

    records: list[dict[str, Any]] = []
    bundles = None
    sampler = StepSampler(connector, objects=objects) if trajectory else None
    try:
        if sampler is not None and not sampler.attach():
            logger.info("connector has no step handle; trajectory holds node boundaries only")
        bundles = boot_bundles(workflow_dir, connector, skills)
        for case in case_ids:
            case_dir = out / "cases" / f"case_{case:04d}"
            if bundles is not None and hasattr(bundles, "revive_dead"):
                for name in bundles.revive_dead():
                    logger.warning("case %d: tool bundle %r had exited and was restarted", case, name)
            try:
                record = run_case(
                    workflow_dir, connector, case, case_dir,
                    skills=skills, checkpoints=checkpoints, max_node_workers=max_node_workers,
                    objects=objects, frames=frames, inputs=inputs, sampler=sampler,
                    video=video, video_interval=video_interval,
                )
            except Exception as exc:  # infrastructure failure, not a graph failure
                logger.exception("case %d aborted", case)
                record = {
                    "case": case, "success": None, "graph_success": False, "exit_status": None,
                    "error": f"harness: {type(exc).__name__}: {exc}", "visits": [], "exits": [],
                    "checkpoints": [], "trace_dir": str(case_dir / "trace"),
                }
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "case.json").write_text(json.dumps(record, indent=2, default=str))
            records.append(record)
            logger.info(
                "case %d: native_success=%s exit=%s error=%s",
                case, record.get("success"), record.get("exit_status"), record.get("error"),
            )
    finally:
        if sampler is not None:
            sampler.close()
            sampler.detach()
        if bundles is not None:
            try:
                bundles.shutdown_all()
            except Exception:
                logger.debug("tool bundle shutdown failed", exc_info=True)
        if own_connector:
            try:
                connector.close()
            except Exception:
                logger.debug("connector close failed", exc_info=True)

    feedback = build_feedback(
        records, meta,
        previous=Path(previous) if previous else None,
        workflow_dir=workflow_dir,
        claims=claims,
    )
    write_feedback(out, feedback)
    if trajectory:
        write_views(out, records, interval=trajectory_interval)
    write_graph_feedback(
        out, records, meta,
        workflow_dir=workflow_dir,
        previous=Path(previous) if previous else None,
        trajectories=trajectory,
    )
    return feedback


__all__ = ["rehearse", "run_case", "parse_cases", "graph_manifest", "load_tools_plugin", "boot_bundles",
           "select_ik"]
