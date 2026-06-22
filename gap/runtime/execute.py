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
from collections.abc import Sequence
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
    latency: dict[str, Any] | None = None
    """Connector-reported episode latency snapshot (``{}``-or-``None``
    when the connector doesn't expose ``get_latency_info``). Today only
    :class:`gap.connector.sim.SimConnector` (LIBERO/VAB) populates this.
    Keys: ``control_steps`` (int), ``control_freq`` (Hz),
    ``sim_physics_wall_s`` (s). ``duration_s`` above is the total wall
    time; ``compute_overhead_s = duration_s − sim_physics_wall_s``."""


def execute(
    graph: Any,
    connector: Any = None,
    *,
    skills: str | Path | Sequence[str | Path] | None = None,
    inputs: dict[str, Any] | None = None,
    trace_dir: str | Path | None = None,
    checkpoints: str = "warn",
    max_node_workers: int = 8,
    policies: dict[str, dict[str, Any]] | None = None,
) -> ExecutionResult:
    """Execute a workflow graph and return an :class:`ExecutionResult`.

    Args:
        graph: A workflow directory or workflow.json path (str | Path), a
            materialized workflow dict, or an object exposing
            ``to_dict()`` (a :class:`gap.builder.Workflow`). Dicts and
            builders are materialized to a temp directory.
        connector: Duck-typed connector (see module docstring); ``None``
            runs tools-only.
        skills: Optional skill registry root(s) — one path or a
            precedence-ordered sequence. When omitted, the active
            registries are resolved via
            :func:`gap.skills.resolve_registries` (``$GAP_SKILLS_PATH``
            list > project ``[tool.gap]`` > user config > the
            open-robot-skills checkout next to the gap checkout); when
            none are found, execution proceeds without a skill registry.
            Loaded via :func:`gap.skills.load_registry_set`; the merged
            registry is passed to the executor as ``skill_registry`` and
            the bundles' ``@tool`` registrations are drained into the
            active tool registry.
        inputs: Optional initial inputs, addressable at the top level as
            ``{"$ref": "in.<name>"}`` and as base producers for subgraph
            input binding.
        trace_dir: Trace output directory (default: ``GAP_TRACE_DIR`` or
            the workflow directory).
        checkpoints: ``"off"`` | ``"warn"`` | ``"raise"`` — enforcement
            mode for ``validate=True`` postcondition checkpoints (only
            active when the connector exposes ``world_snapshot``).
        max_node_workers: Thread budget per parallel super-step.
        policies: Optional ``{policy_id: entry}`` overrides for learned-policy
            servers (e.g. ``{"pi05-libero": {"url": "ws://host:port"}}``).
            Policy skills otherwise auto-resolve to their shipped preset, so
            this is only needed to point a policy at an external/custom
            server.

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
        from gap_core.tools import default_tool_registry
        tool_registry = default_tool_registry()
    observation_poll_fn = getattr(connector, "get_observation", None)
    world_snapshot_fn = getattr(connector, "world_snapshot", None)

    # Resolve the active registry set (explicit arg > $GAP_SKILLS_PATH >
    # project [tool.gap] > user config > auto-discovered sibling).
    # Resolution failure is fine here — graphs that use no bundle scripts
    # or tools execute against the connector registry alone.
    from gap.skills import load_registry_set, resolve_registries

    registry_set = resolve_registries(skills)
    skill_registry = None
    if registry_set:
        logger.debug(
            "active skill registries: %s",
            ", ".join(f"{s.name} ({s.path})" for s in registry_set),
        )
        skill_registry = load_registry_set(registry_set)
        # Bundle tools.py imports pushed @tool registrations onto the
        # pending list; drain them into the active registry. The module
        # catalog keeps them available for later execute() calls whose
        # connectors bring fresh registries.
        if hasattr(tool_registry, "discover_pending"):
            tool_registry.discover_pending(catalog=_BUNDLE_TOOL_CATALOG)

    # Boot any learned-policy servers the workflow references. A policy skill
    # owns its preset, so this "just works" without a `policies:` block; pass
    # `policies` to override (e.g. an external `url:`). Booted before the
    # executor so the PolicyExecutor can be threaded in, and torn down in the
    # finally regardless of outcome. RPC tool bundles (those with
    # `gap.serving.protocol: stdio-msgpack`) boot the same way — their
    # ToolBundleManager registers each catalog entry with the runtime
    # ToolRegistry so the executor sees the RPC-routed tools alongside the
    # in-process @tool catalog.
    executor: WorkflowExecutor | None = None
    policy_manager = None
    policy_executor = None
    tool_bundle_manager = None
    error: Exception | None = None
    t0 = time.perf_counter()
    try:
        from gap.runtime.policy_boot import boot_policies
        from gap.runtime.tool_bundle_boot import boot_tool_bundles

        policy_manager, policy_executor = boot_policies(
            target, skill_registry, config_policies=policies,
        )
        tool_bundle_manager = boot_tool_bundles(
            target, skill_registry, tool_registry,
        )
        executor = WorkflowExecutor(
            target,
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            policy_executor=policy_executor,
            observation_poll_fn=observation_poll_fn,
            trace_dir=trace_dir,
            checkpoints=checkpoints,
            world_snapshot_fn=world_snapshot_fn,
            max_node_workers=max_node_workers,
        )
        if inputs:
            executor.initial_inputs.update(inputs)
        executor.execute()
    except Exception as exc:
        error = exc
        logger.warning("workflow execution failed: %s", exc)
    finally:
        if tool_bundle_manager is not None:
            tool_bundle_manager.shutdown_all()
        if policy_manager is not None:
            policy_manager.shutdown_all()
    duration_s = time.perf_counter() - t0

    trace_path = (
        getattr(executor.trace, "_output_dir", None)
        if executor is not None else None
    )

    latency: dict[str, Any] | None = None
    if connector is not None:
        latency_fn = getattr(connector, "get_latency_info", None)
        if latency_fn is not None:
            try:
                latency = dict(latency_fn())
            except Exception:
                logger.debug("connector.get_latency_info failed", exc_info=True)

    return ExecutionResult(
        success=(
            error is None
            and executor is not None
            and executor.exit_status == "success"
        ),
        exit_status=executor.exit_status if executor is not None else None,
        outputs=(
            dict(executor.cross_subgraph_outputs) if executor is not None else {}
        ),
        trace_path=Path(trace_path) if trace_path is not None else None,
        checkpoint_results=(
            list(executor.checkpoint_results) if executor is not None else []
        ),
        error=error,
        duration_s=duration_s,
        latency=latency,
    )


