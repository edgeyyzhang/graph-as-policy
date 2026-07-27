#!/usr/bin/env python3
"""Check the skill registry is self-consistent BEFORE spending a `gap generate` run.

Every cross-subgraph coupling in GaP is an exact name match: a consumer's
``required_inputs`` name binds to whichever upstream subgraph declares an
output of the SAME name. So any name a consumer needs that no producer
declares is not a wiring detail the agent can infer — it is a guess, and the
agent gets it right some runs and wrong others.

This lints for that class directly: for every ``required_inputs`` entry, is
there some skill whose ``produces_outputs`` declares that exact name? Names
carrying a ``<a|b>`` template (the tsh-perceive-cv / tsh-calculate-grasp-ring
pattern for one skill instantiated per role) are expanded to their
alternatives first.

Run it after ANY SKILL.md edit::

    python tsh-skills/lint_couplings.py

Exit 0 = every declared input has a declared producer. Exit 1 = at least one
coupling exists only in prose, which is the failure mode this exists to catch.

Deliberately standalone: it parses the frontmatter itself rather than importing
gap, so it stays a skills-side tool with no engine dependency.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Names a script node supplies directly (literals baked into the node's inputs,
# or values produced INSIDE the subgraph), so they need no upstream producer.
# Keep this list short and justified — every entry is a coupling the lint
# stops checking.
_LOCALLY_SUPPLIED = {
    "arm_id",       # a literal (0/1) baked per instance, never auto-wired
    "object_query", # a literal DINO noun phrase
}


def _frontmatter(path: Path) -> str:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    return m.group(1) if m else ""


def _block(fm: str, key: str) -> dict[str, str]:
    """Pull a `key:` mapping out of the `gap:` frontmatter, without a YAML dep."""
    out: dict[str, str] = {}
    lines = fm.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != f"{key}:":
            continue
        indent = len(line) - len(line.lstrip())
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip())
            if nxt_indent <= indent:
                break
            if ":" not in nxt:
                continue
            name, _, typ = nxt.strip().partition(":")
            out[name.strip().strip('"\'')] = typ.strip()
        break
    return out


def _expand(name: str) -> list[str]:
    """`<a|b>_foo` -> [`a_foo`, `b_foo`]; plain names pass through."""
    m = re.search(r"<([^>]+)>", name)
    if not m:
        return [name]
    return [name.replace(m.group(0), alt) for alt in m.group(1).split("|")]


def main() -> int:
    root = Path(__file__).resolve().parent / "skills"
    skills = sorted(p for p in root.glob("*/SKILL.md"))
    if not skills:
        print(f"no SKILL.md found under {root}", file=sys.stderr)
        return 1

    produced: dict[str, list[str]] = {}
    required: list[tuple[str, str, str]] = []
    for path in skills:
        fm = _frontmatter(path)
        skill = path.parent.name
        for raw in _block(fm, "produces_outputs"):
            for name in _expand(raw):
                produced.setdefault(name, []).append(skill)
        for raw, typ in _block(fm, "required_inputs").items():
            for name in _expand(raw):
                required.append((skill, name, typ))

    errors = []
    for skill, name, typ in required:
        if name in _LOCALLY_SUPPLIED or name in produced:
            continue
        errors.append(f"  {skill}: required input '{name}' ({typ}) "
                      f"has NO skill declaring it as an output")

    print(f"checked {len(skills)} skills: "
          f"{len(produced)} declared output names, {len(required)} declared inputs")

    # Produced-but-never-consumed is not an error (checkpoints and debugging
    # read these), but an output that exists ONLY to feed a consumer is worth
    # surfacing — an orphan there usually means a rename landed on one side.
    consumed = {n for _, n, _ in required}
    orphans = sorted(n for n, who in produced.items()
                     if n not in consumed and not n.endswith(("_tcp", "_arm"))
                     and n not in {"route"})
    if orphans:
        print("\nnote — declared outputs nothing declares as an input "
              "(fine for checkpoint/debug values, suspicious after a rename):")
        for n in orphans:
            print(f"  {n:22s} from {', '.join(produced[n])}")

    if errors:
        print(f"\nFAIL: {len(errors)} coupling(s) exist only in prose:")
        print("\n".join(errors))
        return 1
    print("\nOK: every declared input has at least one declaring producer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
