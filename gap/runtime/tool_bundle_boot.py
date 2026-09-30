"""Discover which RPC-protocol tool bundles a workflow needs and boot them.

Parallel to :mod:`gap.runtime.policy_boot`. We discover the bundle set
from two sources:

1. Direct ``tool:`` nodes in the workflow (e.g. a node with
   ``tool: "sam3.segment"`` → the ``sam3`` bundle).
2. Literal ``ctx.tool("bundle.tool", ...)`` calls in local script nodes.
3. The ``allowed_tools`` list of every skill subgraph the workflow uses.
   Script nodes call tools via ``ctx.tool_call(...)``, not as workflow
   nodes — so a workflow's `target_sg` subgraph referencing a
   ``perceiving-objects`` skill must pre-boot every bundle that
   skill's ``allowed_tools`` declares (sam3, grounding-dino, vlm, …).

A bundle whose SKILL.md declares ``gap.serving.protocol == "stdio-msgpack"``
is booted out-of-process via :class:`ToolBundleManager`. Bundles without a
``gap.serving:`` block (or with ``in-process`` protocol) are unaffected:
their @tool registrations land in the registry via the in-process path
exactly as today.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def required_rpc_tool_bundles(
    workflow_dir: str | Path, skill_registry: Any, tool_registry: Any | None = None,
) -> set[str]:
    """Bundle names a workflow references whose serving.protocol is
    ``stdio-msgpack``."""
    out: set[str] = set()
    if skill_registry is None:
        return out
    from gap.runtime.workflow import load_workflow

    try:
        wf = load_workflow(Path(workflow_dir) / "workflow.json")
    except Exception:
        return out

    candidate_tool_names: set[str] = set()
    candidate_skills: set[str] = set()
    script_paths: set[Path] = set()
    # The root workflow owns nodes directly; nested subgraphs are additional
    # scopes, not the only scopes. Scan both.
    for sg in (wf, *wf.subgraphs.values()):
        # Skill that owns this subgraph contributes its allowed_tools — script
        # nodes call those via ctx.tool_call(...) and aren't workflow nodes.
        skill_name = getattr(sg, "skill", None)
        if skill_name:
            candidate_skills.add(skill_name)
        for node in sg.nodes.values():
            if node.type == "tool" and node.tool:
                candidate_tool_names.add(node.tool)
            if node.script:
                script_paths.add(Path(workflow_dir) / node.script)

    # Local script nodes are first-class workflow nodes too. Their tool calls
    # are not represented as explicit `tool` nodes, so discover literal calls
    # before the bundle manager starts. This mirrors `gap skills install
    # --workflow` and keeps model-backed tools available to saved graphs.
    tool_call_re = re.compile(r"""ctx\.tool\(\s*["']([\w-]+\.[\w.-]+)["']""")
    for path in sorted(script_paths):
        try:
            candidate_tool_names.update(tool_call_re.findall(path.read_text()))
        except OSError:
            logger.warning("cannot inspect workflow script %s for tool dependencies", path)

    for skill_name in candidate_skills:
        try:
            info = skill_registry.get(skill_name)
        except Exception:
            continue
        for tool_name in getattr(info.meta, "allowed_tools", []) or []:
            candidate_tool_names.add(tool_name)

    for tool_name in candidate_tool_names:
        # A connector may provide the same public contract in-process (for
        # example a simulator-native pure geometry implementation). In that
        # case the workflow's dependency is already satisfied; booting the RPC
        # bundle only to collide on the name is both wasteful and incorrect.
        if tool_registry is not None:
            try:
                if tool_name in tool_registry:
                    continue
            except TypeError:
                # Older duck-typed registries may expose ``get`` but no
                # membership protocol. Fall through and preserve old behavior.
                pass
        bundle = tool_name.split(".", 1)[0]
        if bundle in out:
            continue
        try:
            info = skill_registry.get(bundle)
        except Exception:
            continue
        serving = getattr(info.meta, "serving", None)
        if serving is None:
            continue
        if getattr(serving, "protocol", None) == "stdio-msgpack":
            out.add(bundle)
    return out


def boot_tool_bundles(
    workflow_dir: str | Path,
    skill_registry: Any,
    tool_registry: Any,
    *,
    startup_timeout_s: float = 60.0,
    evict_grace_s: float = 5.0,
) -> Any | None:
    """Boot every RPC tool bundle the workflow needs.

    Returns the :class:`ToolBundleManager` or ``None`` when no RPC bundle
    is referenced (the common case today — all current tool bundles run
    in-process). The caller MUST call ``manager.shutdown_all()`` in a
    ``finally`` to terminate the subprocesses.
    """
    required = required_rpc_tool_bundles(workflow_dir, skill_registry, tool_registry)
    if not required:
        return None

    from gap.runtime.tool_bundle_manager import ToolBundleManager

    manager = ToolBundleManager(
        skill_registry=skill_registry,
        tool_registry=tool_registry,
        startup_timeout_s=startup_timeout_s,
        evict_grace_s=evict_grace_s,
    )
    manager.boot_all(required)
    return manager


class ToolBundleSession:
    """Run-scoped owner for RPC tool bundles across workflow executions."""

    def __init__(self) -> None:
        self._manager: Any | None = None
        self._lock = threading.Lock()

    def prepare(
        self, workflow_dir: str | Path, skill_registry: Any, tool_registry: Any
    ) -> Any | None:
        """Attach live servers to a new episode registry, booting only once."""
        if skill_registry is None:
            return None
        with self._lock:
            if self._manager is not None:
                self._manager.attach_registry(tool_registry)
            required = required_rpc_tool_bundles(
                workflow_dir, skill_registry, tool_registry
            )
            if self._manager is None and required:
                from gap.runtime.tool_bundle_manager import ToolBundleManager

                self._manager = ToolBundleManager(skill_registry, tool_registry)
            if self._manager is not None:
                self._manager.boot_all(required)
            return self._manager

    def close(self) -> None:
        """Close all model servers at the end of the owning run."""
        with self._lock:
            manager, self._manager = self._manager, None
        if manager is not None:
            manager.shutdown_all()


__all__ = ["ToolBundleSession", "boot_tool_bundles", "required_rpc_tool_bundles"]
