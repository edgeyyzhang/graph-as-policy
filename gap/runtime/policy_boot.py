"""Boot the policy servers a workflow needs, then tear them down.

A learned-policy skill is a ``kind='policy'`` bundle (under
``<registry>/policies/<name>/``); the bundle name is the registered preset
/ ``policy_id`` it drives, and its SKILL.md ``gap.serving:`` block carries
the launch recipe. Before executing a workflow, the launcher scans it for
such skills, boots one server per referenced bundle (running the bundle's
own venv via ``uv run --project <bundle_dir>``), and hands the executor a
:class:`gap.runtime.policy.PolicyExecutor`. The manager is the caller's to
shut down (in a ``finally``).

This lives apart from ``gap.runtime.policy`` because discovery needs the
skills registry; keeping it here avoids importing ``gap.skills`` from the
hot ``gap.runtime`` import path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def required_policies(workflow_dir: str | Path, skill_registry: Any) -> set[str]:
    """Policy-skill bundle names a workflow references.

    A policy skill is a ``kind='policy'`` bundle; the bundle name is the
    preset / ``policy_id`` the skill drives
    (:attr:`gap.runtime.policy_skill.PolicyLoopSkill.preset`). Both the
    bundle-name callable (``pi05-libero``) and its ``<bundle>.run`` tool
    form are recognised. Returns an empty set when there is no registry,
    the workflow can't be loaded, or it uses no policy skills.
    """
    out: set[str] = set()
    if skill_registry is None:
        return out
    from gap.runtime.workflow import load_workflow

    try:
        wf = load_workflow(Path(workflow_dir) / "workflow.json")
    except Exception:
        return out
    for sg in wf.subgraphs.values():
        for node in sg.nodes.values():
            if node.type != "tool" or not node.tool:
                continue
            bundle = node.tool.split(".", 1)[0]
            try:
                info = skill_registry.get(bundle)
            except Exception:
                continue
            if info.kind == "policy":
                out.add(bundle)
    return out


def boot_policies(
    workflow_dir: str | Path,
    skill_registry: Any,
    *,
    config_policies: dict[str, dict[str, Any]] | None = None,
    startup_timeout_s: float = 120.0,
    evict_grace_s: float = 10.0,
) -> tuple[Any | None, Any | None]:
    """Boot every policy server the workflow needs.

    Returns ``(policy_manager, policy_executor)`` — both ``None`` when the
    workflow references no policy skill (the common case, so non-policy runs
    pay nothing). A referenced policy is served by its ``config_policies``
    entry when present (e.g. an external ``url:`` or a hand-authored
    ``command:`` override); otherwise it is auto-resolved from the policy
    bundle's SKILL.md ``gap.serving:`` block so a policy skill "just works"
    without any ``policies:`` config. A policy whose bundle declares no
    ``gap.serving:`` block surfaces a clear
    :class:`PolicyConfigError`.

    The caller MUST call ``policy_manager.shutdown_all()`` in a ``finally``.
    """
    required = required_policies(workflow_dir, skill_registry)
    if not required:
        return None, None

    from gap.runtime.policy import PolicyExecutor
    from gap.runtime.policy_manager import PolicyConfigError, PolicyManager

    entries: dict[str, dict[str, Any]] = dict(config_policies or {})
    for name in sorted(required):
        if name in entries:
            continue
        info = skill_registry.get(name)
        serving = info.meta.serving
        if serving is None:
            raise PolicyConfigError(
                f"policy bundle {name!r}: SKILL.md declares no `gap.serving:` "
                f"block — every kind='policy' bundle must declare its server "
                f"launch recipe there (command + protocol). Either add the "
                f"block to the bundle or override it with an explicit "
                f"`policies: {{{name}: {{...}}}}` entry."
            )
        entries[name] = {
            "command": list(serving.command),
            "bundle_dir": info.meta.bundle_dir,
            "env": dict(serving.env or {}),
        }

    manager = PolicyManager(
        entries=entries,
        startup_timeout_s=startup_timeout_s,
        evict_grace_s=evict_grace_s,
    )
    manager.boot_all(required)
    return manager, PolicyExecutor(manager)


__all__ = ["required_policies", "boot_policies"]
