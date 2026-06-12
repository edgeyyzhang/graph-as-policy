"""Every `gap <sub> [<subsub>]` command the skill mentions must exist.

Catches renames/removals the moment they land: the skill content and the
hand-written references may only cite commands the live parser actually
registers (the generated cli.md is exempt — it IS the parser output).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "agent" / "skills" / "gap"

#: `gap <word> [<word>]` on one line (no newline-spanning: `import gap\nconn`
#: in python snippets must not read as a command).
_CMD_RE = re.compile(r"\bgap[ \t]+([a-z][a-z-]+)(?:[ \t]+([a-z][a-z-]+))?")

#: Words after `gap <sub>` that are arguments, not sub-subcommands.
_NON_SUBCOMMAND_WORDS = {
    "my-skill", "my-tool", "waving-gripper", "graph", "outputs",
}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _extract_commands(text: str) -> set[tuple[str, str | None]]:
    commands: set[tuple[str, str | None]] = set()
    # Shell fences + inline code spans only: python fences legitimately
    # contain `gap` the module, and prose says things like "gap extensions".
    chunks = re.findall(r"```(?:bash|sh|console)\n(.*?)```", text, flags=re.DOTALL)
    chunks += re.findall(r"`([^`\n]+)`", text)
    for chunk in chunks:
        for match in _CMD_RE.finditer(chunk):
            sub, subsub = match.group(1), match.group(2)
            if subsub in _NON_SUBCOMMAND_WORDS:
                subsub = None
            commands.add((sub, subsub))
    return commands


def test_every_cited_command_exists():
    from gap.cli import build_parser

    parser = build_parser()
    top = _subparsers(parser)

    sources = [SKILL_DIR / "SKILL.md"] + sorted(
        p for p in (SKILL_DIR / "references").glob("*.md") if p.name != "cli.md"
    )
    unknown: list[str] = []
    seen_any = False
    for path in sources:
        for sub, subsub in _extract_commands(path.read_text(encoding="utf-8")):
            seen_any = True
            if sub not in top:
                unknown.append(f"{path.name}: gap {sub}")
                continue
            if subsub is not None:
                nested = _subparsers(top[sub])
                if nested and subsub not in nested:
                    unknown.append(f"{path.name}: gap {sub} {subsub}")
    assert seen_any, "no gap commands found in the skill content?"
    assert not unknown, (
        "skill content cites commands the CLI does not register: "
        + ", ".join(sorted(set(unknown)))
    )
