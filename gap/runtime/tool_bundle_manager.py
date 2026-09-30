"""Lifecycle manager for out-of-process tool bundle servers.

Parallel to :mod:`gap.runtime.policy_manager` — the policy manager owns
long-running learned-policy websocket servers; this one owns long-running
``stdio-msgpack`` tool servers (one subprocess per bundle that declares
``gap.serving.protocol: stdio-msgpack`` in its SKILL.md). The launcher
calls :meth:`boot_all` after policies are up; on workflow exit the caller
MUST invoke :meth:`shutdown_all` from a ``finally``.

A booted bundle's tools register with the gap-runtime ToolRegistry via
:meth:`gap_core.tools.ToolRegistry.register_rpc` so the executor sees the
RPC-routed tools side-by-side with the in-process @tool catalog. The LLM
catalog and the workflow validator are unaware of the transport — they
just see another ``runtime`` tool.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class ToolBundleStartupError(RuntimeError):
    """A bundle's tool server failed to come up (or its catalog handshake
    failed). Surfaces the bundle name + the underlying error."""


@dataclass
class _Managed:
    bundle: str
    client: Any  # ToolClient
    tool_names: list[str]


class ToolBundleManager:
    """Boot every RPC-protocol tool bundle the workflow references; shut down on exit.

    Typical use (from the launcher)::

        manager = ToolBundleManager(skill_registry=skill_registry,
                                    tool_registry=tool_registry)
        manager.boot_all(["sam3", "curobo", "geometry"])  # noqa
        try:
            executor.execute()
        finally:
            manager.shutdown_all()
    """

    def __init__(self, skill_registry: Any, tool_registry: Any,
                 *, startup_timeout_s: float = 60.0,
                 evict_grace_s: float = 5.0,
                 call_timeout_s: float | None = None) -> None:
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._timeout = float(startup_timeout_s)
        self._grace = float(evict_grace_s)
        # Per-call reply timeout passed to each ToolClient. None lets the
        # client resolve GAP_TOOL_CALL_TIMEOUT_S / its built-in default.
        self._call_timeout_s = call_timeout_s
        self._lock = threading.Lock()
        self._managed: dict[str, _Managed] = {}

    def _register_client(self, registry: Any, bundle: str, client: Any) -> list[str]:
        """Register one live client's catalog into ``registry``."""
        tool_names: list[str] = []
        for entry in client.catalog:
            if entry.name in registry:
                existing = registry.get(entry.name)
                if existing.transport == "python" and bundle in existing.tags:
                    logger.info(
                        "[tool-bundle:%s] retaining in-process %s",
                        bundle, entry.name,
                    )
                    continue
                raise ValueError(f"Tool name collision: {entry.name!r} already registered")
            registry.register_rpc(
                entry.name,
                client,
                summary=entry.summary,
                tags=tuple(entry.tags),
            )
            tool_names.append(entry.name)
        return tool_names

    def attach_registry(self, tool_registry: Any) -> None:
        """Expose the already-running servers through a fresh registry.

        Connectors are intentionally episode-scoped, while expensive model
        servers can be run-scoped.  Rebinding only installs RPC descriptors;
        no model process is restarted and no episode-local connector state is
        retained.
        """
        with self._lock:
            managed = list(self._managed.values())
            self._tool_registry = tool_registry
        for item in managed:
            try:
                self._register_client(tool_registry, item.bundle, item.client)
            except ValueError as exc:
                raise ToolBundleStartupError(
                    f"tool bundle {item.bundle!r}: cannot attach to fresh registry — {exc}"
                ) from exc

    def boot_all(self, bundle_names: Iterable[str]) -> None:
        """Spawn one server per bundle, register every catalog entry.

        Bundles whose SKILL.md does NOT declare ``serving.protocol ==
        "stdio-msgpack"`` are silently skipped — they run in-process.
        """
        from gap_core.rpc.client import ToolClient

        names = sorted(set(bundle_names))
        if not names:
            return
        started: list[_Managed] = []
        try:
            for name in names:
                with self._lock:
                    if name in self._managed:
                        continue
                info = self._skill_registry.get(name)
                serving = info.meta.serving
                if serving is None or serving.protocol != "stdio-msgpack":
                    continue
                logger.info("[tool-bundle:%s] booting", name)
                try:
                    client = ToolClient(
                        bundle_name=name,
                        bundle_dir=info.meta.bundle_dir,
                        env=serving.env,
                        evict_grace_s=self._grace,
                        call_timeout_s=self._call_timeout_s,
                    )
                except BaseException as exc:
                    raise ToolBundleStartupError(
                        f"tool bundle {name!r}: failed to start its server "
                        f"({type(exc).__name__}: {exc})"
                    ) from exc

                try:
                    tool_names = self._register_client(self._tool_registry, name, client)
                except ValueError as exc:
                    client.close()
                    raise ToolBundleStartupError(
                        f"tool bundle {name!r}: cannot register its catalog — {exc}"
                    ) from exc
                started.append(_Managed(bundle=name, client=client,
                                         tool_names=tool_names))
                logger.info("[tool-bundle:%s] %d tools registered",
                            name, len(tool_names))

            with self._lock:
                for m in started:
                    self._managed[m.bundle] = m
        except BaseException:
            # Tear down anything we started before re-raising.
            for m in started:
                try:
                    m.client.close()
                except BaseException:
                    logger.warning("[tool-bundle:%s] close during boot rollback failed",
                                   m.bundle, exc_info=True)
            raise

    def shutdown_all(self) -> None:
        """Terminate every tool bundle subprocess we started. Idempotent."""
        with self._lock:
            managed = list(self._managed.values())
            self._managed.clear()

        if not managed:
            return
        logger.info("ToolBundleManager: shutting down %d tool bundle%s",
                    len(managed), "" if len(managed) == 1 else "s")
        for m in managed:
            try:
                m.client.close()
            except BaseException:
                logger.warning("[tool-bundle:%s] close failed",
                               m.bundle, exc_info=True)

    def loaded_bundles(self) -> list[str]:
        return sorted(self._managed.keys())
__all__ = ["ToolBundleManager", "ToolBundleStartupError"]
