"""Tests for the gap.skills bundle loader against the fixture checkout.

``fixtures/`` is a mini open-robot-skills checkout with the two bundle roots:

- ``tools/fixture-tool/``   — SKILL.md (``gap.tools`` list) + ``tools.py``
  exposing one ``@tool`` function.
- ``skills/fixture-skill/`` — SKILL.md (``gap:`` extensions) +
  ``scripts/hello.py`` + ``prompts/greet.md``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from gap.skills import (
    SkillsRegistry,
    load_prompt,
    load_skills,
)
from gap_core.skills import CanonicalScript, Skill

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_bundle(root: Path, folder: str, name: str, frontmatter: str, body: str = "# body") -> Path:
    bundle_dir = root / folder / name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return bundle_dir


# ---------------------------------------------------------------------------
# Discovery: both bundle roots, kind from the folder
# ---------------------------------------------------------------------------


def test_load_skills_discovers_both_kinds() -> None:
    reg = load_skills(FIXTURES)
    assert isinstance(reg, SkillsRegistry)
    assert len(reg) == 2
    assert "fixture-tool" in reg
    assert "fixture-skill" in reg
    assert reg.get("fixture-tool").kind == "tool"
    assert reg.get("fixture-skill").kind == "skill"
    assert reg.get("fixture-tool").namespace == "tools"
    assert reg.get("fixture-skill").namespace == "skills"


def test_list_skills_kind_filter() -> None:
    reg = load_skills(FIXTURES)
    assert [i.name for i in reg.list_skills(kind="skill")] == ["fixture-skill"]
    assert [i.name for i in reg.list_skills(kind="tool")] == ["fixture-tool"]
    assert {i.name for i in reg.list_skills(category="testing")} == {
        "fixture-skill", "fixture-tool",
    }


def test_name_collision_across_roots_raises(tmp_path: Path) -> None:
    fm = "name: twin-bundle\ndescription: A twin.\n"
    _write_bundle(tmp_path, "tools", "twin-bundle", fm)
    _write_bundle(tmp_path, "skills", "twin-bundle", fm)
    with pytest.raises(ValueError, match="collision"):
        load_skills(tmp_path)


# ---------------------------------------------------------------------------
# Frontmatter → SkillMeta mapping
# ---------------------------------------------------------------------------


def test_frontmatter_maps_into_skillmeta() -> None:
    reg = load_skills(FIXTURES)
    meta = reg.get("fixture-skill").meta

    # Agent Skills spec core
    assert meta.name == "fixture-skill"
    assert meta.description.startswith("Produce a greeting")
    assert meta.compatibility == "requires gap>=0.1"
    assert meta.metadata["category"] == "testing"
    assert meta.category == "testing"
    assert meta.tags == ["fixture", "greeting"]

    # gap: extensions, same attribute names the prompt assembler reads
    assert meta.allowed_tools == ["robot.get_observation", "fixture-tool.echo"]
    assert set(meta.exit_conditions) == {"greeted", "failed"}
    assert meta.produces_outputs == {"greeting": "str"}
    assert meta.required_inputs == {"who": "str"}
    assert meta.canonical_scripts == [CanonicalScript(name="hello", path="scripts/hello.py")]
    assert meta.prompts == {"greet": "prompts/greet.md"}
    assert meta.hard_rules == ["Always greet politely."]
    assert meta.streaming is False
    assert meta.kind == "skill"

    # body captured verbatim for the prompt assembler
    assert meta.body.startswith("# fixture-skill")
    assert meta.bundle_dir == FIXTURES / "skills" / "fixture-skill"


def test_tool_bundle_meta_maps_gap_tools() -> None:
    reg = load_skills(FIXTURES)
    meta = reg.get("fixture-tool").meta
    assert meta.kind == "tool"
    assert meta.license == "MIT"
    assert meta.tools == {"fixture-tool.echo": "Echo a string back, uppercased."}
    assert meta.canonical_scripts == []
    assert meta.exit_conditions == {}


def test_canonical_script_schema_extracted() -> None:
    reg = load_skills(FIXTURES)
    script = reg.get("fixture-skill").canonical_scripts["hello"]
    assert script.bundle_relative == "scripts/hello.py"
    assert script.schema.inputs["who"].required is False
    assert "greeting" in script.schema.outputs


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_name_dirname_mismatch_raises(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path, "skills", "actual-dirname",
        "name: declared-name\ndescription: Mismatched bundle.\n",
    )
    with pytest.raises(ValueError, match="must equal its bundle directory"):
        load_skills(tmp_path)


def test_missing_name_raises(tmp_path: Path) -> None:
    _write_bundle(tmp_path, "skills", "anonymous-bundle", "description: No name here.\n")
    with pytest.raises(ValueError, match="no `name` field"):
        load_skills(tmp_path)


def test_description_over_1024_chars_raises(tmp_path: Path) -> None:
    long_description = "x" * 1025
    _write_bundle(
        tmp_path, "skills", "wordy-bundle",
        f"name: wordy-bundle\ndescription: {long_description}\n",
    )
    with pytest.raises(ValueError, match="1024"):
        load_skills(tmp_path)


def test_legacy_frontmatter_keys_raise(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path, "skills", "legacy-bundle",
        "name: legacy-bundle\ndescription: Old shape.\nruntime:\n  shape: composite\n",
    )
    with pytest.raises(ValueError, match="removed legacy"):
        load_skills(tmp_path)


def test_gap_extension_at_top_level_raises(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path, "skills", "flat-bundle",
        "name: flat-bundle\ndescription: Extensions outside gap.\n"
        "exit_conditions: {done: ok}\n",
    )
    with pytest.raises(ValueError, match="nest them under the `gap:` key"):
        load_skills(tmp_path)


# ---------------------------------------------------------------------------
# compatibility: parsing
# ---------------------------------------------------------------------------


def test_compatibility_warning_on_absurd_requirement(tmp_path: Path, caplog) -> None:
    _write_bundle(
        tmp_path, "skills", "futuristic-bundle",
        "name: futuristic-bundle\ndescription: Needs a future gap.\n"
        "compatibility: requires gap>=999.0\n",
    )
    with caplog.at_level(logging.WARNING, logger="gap.skills._registry"):
        reg = load_skills(tmp_path)
    # Warns but still registers — compatibility is advisory.
    assert "futuristic-bundle" in reg
    assert any("requires gap>=999.0" in r.getMessage() for r in caplog.records)


def test_satisfied_compatibility_is_silent(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="gap.skills._registry"):
        load_skills(FIXTURES)  # both fixtures require gap>=0.1
    assert not [r for r in caplog.records if "requires gap" in r.getMessage()]


def test_unparseable_compatibility_warns(tmp_path: Path, caplog) -> None:
    _write_bundle(
        tmp_path, "skills", "cryptic-bundle",
        "name: cryptic-bundle\ndescription: Strange constraint.\n"
        "compatibility: needs a quantum computer\n",
    )
    with caplog.at_level(logging.WARNING, logger="gap.skills._registry"):
        load_skills(tmp_path)
    assert any("unparseable compatibility" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Synthetic packages + load_prompt
# ---------------------------------------------------------------------------


def test_synthetic_packages_installed() -> None:
    load_skills(FIXTURES)
    pkg = sys.modules["gap_skills.skills.fixture-skill"]
    assert pkg.__path__ == [str(FIXTURES / "skills" / "fixture-skill")]
    scripts_pkg = sys.modules["gap_skills.skills.fixture-skill.scripts"]
    assert scripts_pkg.__path__ == [str(FIXTURES / "skills" / "fixture-skill" / "scripts")]


def test_load_prompt_renders_from_script_via_synthetic_package() -> None:
    reg = load_skills(FIXTURES)
    script = reg.get("fixture-skill").canonical_scripts["hello"]
    # The script's own load_prompt(__package__, ...) call resolves the
    # bundle root through the synthetic package.
    out = script.module.run(None, who="gap")
    assert out == {"greeting": "Hello, gap!"}


def test_load_prompt_conditional_block() -> None:
    load_skills(FIXTURES)
    text = load_prompt(
        "gap_skills.skills.fixture-skill.scripts", "greet", who="gap", excited=True,
    )
    assert text == "Hello, gap! So glad you are here!"


def test_load_prompt_missing_prompt_or_variable_raises() -> None:
    load_skills(FIXTURES)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt("gap_skills.skills.fixture-skill", "nonexistent")
    with pytest.raises(KeyError, match="undefined variable"):
        load_prompt("gap_skills.skills.fixture-skill", "greet")  # no who=


# ---------------------------------------------------------------------------
# tools.py → gap.tools._PENDING_TOOLS integration
# ---------------------------------------------------------------------------


def test_tools_py_registers_pending_tools() -> None:
    from gap_core.tools import _registry as tools_registry

    reg = load_skills(FIXTURES)
    info = reg.get("fixture-tool")
    assert info.tools_module is not None
    # The @tool decorator ran at import time and stashed the registration
    # for ToolRegistry.discover_pending() to drain. Another test in the
    # same process may already have drained the queue (any execute() /
    # registry build does), in which case the entry lives on in the
    # drained-replay list — membership in the union is the invariant.
    entries = [
        e
        for e in (*tools_registry._PENDING_TOOLS, *tools_registry._DRAINED_TOOLS)
        if e["name"] == "fixture-tool.echo"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["summary"] == "Echo a string back, uppercased."
    assert entry["fn"]("hi") == "HI"


# ---------------------------------------------------------------------------
# Class-based Skill + callable import paths
# ---------------------------------------------------------------------------


def test_class_based_skill_in_skill_py(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path, "skills", "counting-things",
        "name: counting-things\ndescription: Counts invocations across calls.\n",
    )
    (bundle / "skill.py").write_text(
        "from typing import TypedDict\n"
        "from gap_core.skills import Skill\n"
        "\n"
        "class Output(TypedDict):\n"
        "    count: int\n"
        "\n"
        "class Counter(Skill):\n"
        "    def __init__(self):\n"
        "        self.count = 0\n"
        "\n"
        "    def run(self, ctx) -> Output:\n"
        "        self.count += 1\n"
        "        return {'count': self.count}\n",
        encoding="utf-8",
    )
    reg = load_skills(tmp_path)
    info = reg.get("counting-things")
    assert info.skill_class is not None
    assert issubclass(info.skill_class, Skill)
    # SKILL.md-derived meta installed onto the class (ClassVar contract).
    assert info.skill_class.meta is info.meta
    assert "count" in info.schema.outputs

    # One instance per workflow execution: state persists across calls.
    instances: dict = {}
    assert reg.call("counting-things", None, skill_instances=instances) == {"count": 1}
    assert reg.call("counting-things", None, skill_instances=instances) == {"count": 2}


def test_function_style_run_in_skill_py(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path, "skills", "doubling-numbers",
        "name: doubling-numbers\ndescription: Doubles a number.\n",
    )
    (bundle / "skill.py").write_text(
        "from typing import TypedDict\n"
        "\n"
        "class Output(TypedDict):\n"
        "    doubled: int\n"
        "\n"
        "def run(ctx, value: int = 1, **_ignored) -> Output:\n"
        "    return {'doubled': value * 2}\n",
        encoding="utf-8",
    )
    reg = load_skills(tmp_path)
    info = reg.get("doubling-numbers")
    assert info.skill_class is None
    assert info.module is not None
    assert info.schema.inputs["value"].required is False
    # kwargs not in the signature are filtered out before dispatch.
    assert reg.call("doubling-numbers", None, value=4, extraneous=True) == {"doubled": 8}


def test_calling_non_callable_bundle_raises() -> None:
    reg = load_skills(FIXTURES)
    with pytest.raises(ValueError, match="no callable run"):
        reg.call("fixture-skill", None)


# ---------------------------------------------------------------------------
# Docs rendering
# ---------------------------------------------------------------------------


def test_generate_docs_renders_scripts_and_tools() -> None:
    reg = load_skills(FIXTURES)
    docs = reg.generate_docs()
    assert "### fixture-skill" in docs
    assert "`scripts/hello.py`" in docs
    assert "### fixture-tool" in docs
    assert "`fixture-tool.echo` — Echo a string back, uppercased." in docs
