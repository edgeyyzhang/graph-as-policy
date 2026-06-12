"""The committed CLI reference must match the live parser.

Touch the CLI (new command, flag, help text), regenerate:

    uv run python agent/scripts/gen_cli_reference.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI_MD = REPO / "agent" / "skills" / "gap" / "references" / "cli.md"


def _render_live() -> str:
    spec = importlib.util.spec_from_file_location(
        "gap_agent_gen_cli_reference",
        REPO / "agent" / "scripts" / "gen_cli_reference.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.render()


def test_cli_reference_is_fresh():
    committed = CLI_MD.read_text(encoding="utf-8")
    live = _render_live()
    assert committed == live, (
        "agent/skills/gap/references/cli.md is stale — regenerate it:\n"
        "    uv run python agent/scripts/gen_cli_reference.py"
    )
