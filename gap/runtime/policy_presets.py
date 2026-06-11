"""Named serving presets for the ``policies:`` config block.

A preset bundles the checkpoint URI and start_cmd template for one
known-good policy-server recipe, so a task.yaml entry can say
``preset: pi05-libero`` instead of spelling out the start command.
:func:`resolve_policies` expands preset entries into the full managed
entries :class:`gap.runtime.policy_manager.PolicyManager` consumes
(``start_cmd`` with a ``{port}`` placeholder + ``env`` overrides).

Both templates reference ``$GAP_OPENPI_DIR`` — the user's openpi (or
MolmoAct) checkout. The manager spawns ``start_cmd`` through the shell,
so the variable is expanded at spawn time from the environment.
"""

from __future__ import annotations

from typing import Any

from .policy_manager import PolicyConfigError

PRESETS: dict[str, dict[str, Any]] = {
    "pi05-libero": {
        "checkpoint_uri": "s3://openpi-assets/checkpoints/pi05_libero",
        "start_cmd": (
            "cd $GAP_OPENPI_DIR && uv run scripts/serve_policy.py "
            "policy:checkpoint --policy.config=pi05_libero "
            "--policy.dir=s3://openpi-assets/checkpoints/pi05_libero "
            "--port {port}"
        ),
        "env": {},
        "notes": (
            "openpi reference recipe for the LIBERO π0.5 checkpoint; "
            "serve_policy.py downloads the checkpoint from openpi-assets "
            "on first run. Set GAP_OPENPI_DIR to your openpi checkout."
        ),
    },
    "molmoact-libero": {
        "checkpoint_uri": "hf://allenai/MolmoAct-7B-D-LIBERO-0812",
        "start_cmd": (
            "cd $GAP_OPENPI_DIR && python scripts/serve_policy_vllm.py "
            "--checkpoint allenai/MolmoAct-7B-D-LIBERO-0812 "
            "--port {port}"
        ),
        "env": {},
        "notes": (
            "MolmoAct LIBERO checkpoint behind a vLLM-style serve script "
            "that speaks the openpi websocket protocol. The user supplies "
            "the openpi/MolmoAct checkout path via GAP_OPENPI_DIR, which "
            "the shell expands in the start_cmd template at spawn time."
        ),
    },
}


def resolve_policies(
    entries: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Expand ``preset: <name>`` entries into full managed entries.

    Non-preset entries (external ``url:`` / explicit ``start_cmd:``) pass
    through unchanged. A preset entry may carry its own ``env`` mapping,
    which overrides the preset's ``env`` key-by-key.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for pid, entry in dict(entries or {}).items():
        if not (isinstance(entry, dict) and "preset" in entry):
            resolved[pid] = entry
            continue
        name = str(entry["preset"])
        preset = PRESETS.get(name)
        if preset is None:
            raise PolicyConfigError(
                f"policy {pid!r}: unknown preset {name!r} "
                f"(available: {', '.join(sorted(PRESETS))})"
            )
        if "url" in entry or "start_cmd" in entry:
            raise PolicyConfigError(
                f"policy {pid!r}: specify either 'preset' or an explicit "
                f"'url'/'start_cmd', not both"
            )
        expanded: dict[str, Any] = {
            "start_cmd": preset["start_cmd"],
            "env": dict(preset["env"]),
        }
        extra_env = entry.get("env")
        if extra_env:
            expanded["env"].update(extra_env)
        resolved[pid] = expanded
    return resolved


__all__ = [
    "PRESETS",
    "resolve_policies",
]
