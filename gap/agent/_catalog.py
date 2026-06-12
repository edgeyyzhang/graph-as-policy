"""Codegen-time registry assembly.

The codegen pipeline needs two registries per run:

- a :class:`gap.skills.SkillsRegistry` built from the configured
  open-robot-skills checkout, and
- a :class:`gap.tools.ToolRegistry` carrying everything the generated
  graphs may dispatch on: the connector-owned ``robot.*`` / ``sim.*``
  tools, the skill bundles' ``@tool`` functions, and the codegen-scope
  meta-tools.

Connector tools are registered by a live connector at *execution* time —
none exists at codegen time. Rather than hand-maintaining a static table
of their signatures, we build a throwaway :class:`SimConnector` around a
null env (its ``__init__`` and tool registration never touch the env) and
copy the resulting descriptors. That keeps the codegen catalog identical
to the real connector's by construction; a parity test pins the
assumption that construction stays env-free.

Bundle ``@tool`` registrations are drained from the decorator's pending
list exactly once per process (modules are cached in ``sys.modules``), so
we archive every drained entry and re-apply the archive to each fresh
registry — re-registration of the same function is idempotent.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gap.skills import SkillsRegistry
from gap.tools import ToolDescriptor, ToolRegistry
from gap.tools import _registry as _tools_registry_module

from ._meta_tools import register_codegen_meta_tools

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

#: Archive of every bundle @tool registration ever drained in this
#: process, keyed by tool name. Re-applied to each fresh codegen registry.
_TOOL_ARCHIVE: dict[str, dict[str, Any]] = {}

_CONNECTOR_DESCRIPTORS: dict[str, ToolDescriptor] | None = None


def connector_tool_descriptors() -> dict[str, ToolDescriptor]:
    """Descriptors (name → schema/summary/tags) for the connector-owned
    ``robot.*`` / ``sim.*`` tools, derived from the real connector code.

    The returned descriptors are schema-only for codegen purposes — they
    are NOT dispatchable (the throwaway connector has no env).
    """
    global _CONNECTOR_DESCRIPTORS
    with _LOCK:
        if _CONNECTOR_DESCRIPTORS is None:
            from gap.connector.sim import SimConnector

            conn = SimConnector(None, SimpleNamespace())
            _CONNECTOR_DESCRIPTORS = dict(conn.tool_registry._tools)
        return dict(_CONNECTOR_DESCRIPTORS)


def _drain_pending_with_archive(registry: ToolRegistry) -> None:
    """Drain pending ``@tool`` registrations into *registry*, archiving
    them so later registries in the same process see them too."""
    with _LOCK:
        for entry in _tools_registry_module._PENDING_TOOLS:
            _TOOL_ARCHIVE[entry["name"]] = dict(entry)
        registry.discover_pending()
        for entry in _TOOL_ARCHIVE.values():
            try:
                registry._register_python(
                    entry["name"], entry["summary"],
                    entry["scope"], entry["tags"], entry["fn"],
                )
            except ValueError:
                # A different implementation already owns this name in
                # this registry — first registration wins.
                logger.debug("tool %r already registered; keeping existing", entry["name"])


def build_codegen_tool_registry(
    skills_registry: SkillsRegistry | None = None,
) -> ToolRegistry:
    """Build the flat tool catalog the codegen prompts render from.

    Contains: connector tool descriptors (schema-only), every bundle
    ``@tool`` (imported by *skills_registry* discovery), and the
    codegen-scope meta-tools.
    """
    registry = ToolRegistry()
    for name, descriptor in connector_tool_descriptors().items():
        registry._tools.setdefault(name, descriptor)
    _drain_pending_with_archive(registry)
    register_codegen_meta_tools(registry)
    return registry


def load_codegen_registries(
    skills_path: str | Path | Sequence[str | Path],
    *,
    only: list[str] | None = None,
    disable: list[str] | None = None,
) -> tuple[SkillsRegistry, ToolRegistry]:
    """Load the skill registry root(s) and build the matching tool catalog."""
    from gap.skills import load_registry_set

    skills = load_registry_set(skills_path, only=only, disable=disable)
    tools = build_codegen_tool_registry(skills)
    return skills, tools
