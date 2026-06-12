"""The agent skill's own conformance: SKILL.md + plugin manifests.

The engine's ``parse_skill_md`` enforces the Agent Skills spec (name ==
dirname ≤64 chars, description ≤1024, gap-extension keys nested), so the
gap repo's Claude Code skill is linted by the same parser that validates
registry bundles — the skill can never drift out of the format the
engine itself preaches.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "agent" / "skills" / "gap"


def test_skill_md_passes_the_engine_parser():
    from gap.skills import parse_skill_md

    meta = parse_skill_md(SKILL_DIR / "SKILL.md")
    assert meta.name == "gap"
    assert "Use when" in meta.description
    assert len(meta.description) <= 1024


def test_referenced_reference_files_exist():
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    refs_dir = SKILL_DIR / "references"
    assert refs_dir.is_dir()
    import re

    referenced = set(re.findall(r"references/([a-z-]+\.md)", body))
    assert referenced, "SKILL.md should point at its references/"
    for name in referenced:
        assert (refs_dir / name).is_file(), f"references/{name} is missing"


def test_manifests_parse_and_paths_exist():
    marketplace = json.loads(
        (REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert marketplace["name"] == "gap"
    names = [p["name"] for p in marketplace["plugins"]]
    assert names == ["gap", "open-robot-skills"]
    for plugin in marketplace["plugins"]:
        source = plugin["source"]
        if isinstance(source, str):
            assert source.startswith("./")
            assert (REPO / source).is_dir(), f"plugin source {source} missing"
        else:
            assert source["source"] == "github"
            assert "/" in source["repo"]

    plugin = json.loads(
        (REPO / "agent" / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    assert plugin["name"] == "gap"
    # No version field on purpose: omitted version = git-SHA versioning,
    # so every push is an update (recommended while iterating).
    assert "version" not in plugin


def test_install_doc_uses_the_real_marketplace_commands():
    install = (REPO / "agent" / "INSTALL.md").read_text(encoding="utf-8")
    assert "claude plugin marketplace add graph-robots/graph-as-policy" in install
    assert "claude plugin install gap@gap" in install
