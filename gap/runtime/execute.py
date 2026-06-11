"""gap.execute — one-call workflow execution facade.

Wraps :class:`gap.runtime.executor.WorkflowExecutor` behind a single
function::

    import gap

    conn = gap.connector.sim("libero", task="libero_object/0")
    result = gap.execute(graph, conn)   # open-robot-skills checkout auto-discovered

``connector`` is duck-typed (the Connector class itself lands in P3):

- ``.tool_registry`` — a :class:`gap.tools.ToolRegistry` with the
  connector's ``robot.*`` / ``sim.*`` tools registered;
- ``.get_observation`` — zero-arg observation poll fn (drives the
  executor's :class:`~gap.runtime.observation_stream.ObservationStream`);
- ``.world_snapshot`` (optional) — zero-arg ground-truth
  :class:`gap.runtime.verify.World` factory, enables checkpoint
  enforcement;
- ``.capabilities`` (optional) — capability metadata; unused here.

``connector=None`` runs tools-only against the default ``@tool`` registry
(tests, dry-runs).
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gap.runtime.executor import WorkflowExecutor

logger = logging.getLogger(__name__)

# Persistent catalog of skill-bundle @tool registrations. Bundle modules
# import (and push to the decorator's pending queue) only once per process,
# but each execute() call may bring a fresh connector tool registry —
# without this catalog, the second call's registry would miss every bundle
# tool (the first registry drained the queue).
_BUNDLE_TOOL_CATALOG: list = []


@dataclass
class ExecutionResult:
    """Outcome of one :func:`execute` call."""

    success: bool
    exit_status: str | None          # "success" | "failure" | None
    outputs: dict[str, Any]          # {subgraph_name: bound outputs}
    trace_path: Path | None
    checkpoint_results: list[Any]    # gap.runtime.verify.CheckpointResult
    error: Exception | None
    duration_s: float


def execute(
    graph: Any,
    connector: Any = None,
    *,
    skills: str | Path | None = None,
    inputs: dict[str, Any] | None = None,
    trace_dir: str | Path | None = None,
    checkpoints: str = "warn",
    max_node_workers: int = 8,
) -> ExecutionResult:
    """Execute a workflow graph and return an :class:`ExecutionResult`.

    Args:
        graph: A workflow directory or workflow.json path (str | Path), a
            materialized workflow dict, or an object exposing
            ``to_dict()`` (a :class:`gap.builder.Workflow`). Dicts and
            builders are materialized to a temp directory.
        connector: Duck-typed connector (see module docstring); ``None``
            runs tools-only.
        skills: Optional open-robot-skills checkout path. When omitted, the
            checkout is auto-discovered via
            :func:`gap.skills.find_skills_path` (``$GAP_SKILLS_PATH`` or
            a ``open-robot-skills`` directory next to the graph-as-policy checkout); when
            none is found, execution proceeds without a skill registry.
            Loaded via :func:`gap.skills.load_skills`; the registry is
            passed to the executor as ``skill_registry`` and the bundles'
            ``@tool`` registrations are drained into the active tool
            registry.
        inputs: Optional initial inputs, addressable at the top level as
            ``{"$ref": "in.<name>"}`` and as base producers for subgraph
            input binding.
        trace_dir: Trace output directory (default: ``GAP_TRACE_DIR`` or
            the workflow directory).
        checkpoints: ``"off"`` | ``"warn"`` | ``"raise"`` — enforcement
            mode for ``validate=True`` postcondition checkpoints (only
            active when the connector exposes ``world_snapshot``).
        max_node_workers: Thread budget per parallel super-step.

    Never raises on workflow failure — the exception lands in
    ``result.error`` with ``result.success == False``.
    """
    if not isinstance(graph, str | Path) and hasattr(graph, "to_dict"):
        graph = graph.to_dict()
    if isinstance(graph, dict):
        target = Path(tempfile.mkdtemp(prefix="gap_workflow_"))
        (target / "workflow.json").write_text(json.dumps(graph, indent=2))
    else:
        target = Path(graph)

    tool_registry = getattr(connector, "tool_registry", None)
    if tool_registry is None:
        from gap.tools import default_tool_registry
        tool_registry = default_tool_registry()
    observation_poll_fn = getattr(connector, "get_observation", None)
    world_snapshot_fn = getattr(connector, "world_snapshot", None)

    if skills is None:
        # Default: discover the side-by-side checkout / $GAP_SKILLS_PATH.
        # Discovery failure is fine here — graphs that use no bundle
        # scripts or tools execute against the connector registry alone.
        from gap.skills import find_skills_path

        skills = find_skills_path()
        if skills is not None:
            logger.debug("auto-discovered open-robot-skills checkout: %s", skills)

    skill_registry = None
    if skills is not None:
        try:
            from gap.skills import load_skills
        except ImportError as e:  # pragma: no cover - skills lands in parallel
            raise ImportError(
                f"skills={skills!r} was given but gap.skills is "
                f"unavailable: {e}"
            ) from e
        skill_registry = load_skills(skills)
        # Bundle tools.py imports pushed @tool registrations onto the
        # pending list; drain them into the active registry. The module
        # catalog keeps them available for later execute() calls whose
        # connectors bring fresh registries.
        if hasattr(tool_registry, "discover_pending"):
            tool_registry.discover_pending(catalog=_BUNDLE_TOOL_CATALOG)

    executor = WorkflowExecutor(
        target,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        observation_poll_fn=observation_poll_fn,
        trace_dir=trace_dir,
        checkpoints=checkpoints,
        world_snapshot_fn=world_snapshot_fn,
        max_node_workers=max_node_workers,
    )
    if inputs:
        executor.initial_inputs.update(inputs)

    error: Exception | None = None
    t0 = time.perf_counter()
    try:
        executor.execute()
    except Exception as exc:
        error = exc
        logger.warning("workflow execution failed: %s", exc)
    duration_s = time.perf_counter() - t0

    trace_path = getattr(executor.trace, "_output_dir", None)
    return ExecutionResult(
        success=error is None and executor.exit_status == "success",
        exit_status=executor.exit_status,
        outputs=dict(executor.cross_subgraph_outputs),
        trace_path=Path(trace_path) if trace_path is not None else None,
        checkpoint_results=list(executor.checkpoint_results),
        error=error,
        duration_s=duration_s,
    )


