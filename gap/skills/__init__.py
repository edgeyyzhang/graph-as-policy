"""gap.skills — the skill-bundle loader and the skill-authoring surface.

A bundle is an Agent Skills-format directory (``SKILL.md`` + resources)
living in an open-robot-skills checkout under one of two roots — the folder conveys
the kind:

- ``tools/<bundle>/``  — model-backed callables; ``tools.py`` exposes typed
  functions via :func:`tool`.
- ``skills/<bundle>/`` — manipulation strategies; ``scripts/`` holds the
  canonical Python files the subgraph_agent emits as ``type: script``
  states, ``prompts/`` holds Jinja-style templates loaded by scripts at
  runtime, ``references/``/``examples/`` hold lazy-loaded long-form docs.

Build a registry from a checkout (omit the path to auto-discover it via
``$GAP_SKILLS_PATH`` or the side-by-side checkout layout)::

    from gap.skills import find_skills_path, load_skills

    registry = load_skills(find_skills_path(required=True))

Skill authors import from here (the stable surface)::

    from gap.skills import tool, Skill, SkillMeta, load_prompt
"""

from __future__ import annotations

from ._meta_from_skill_md import parse_skill_md
from ._prompt_loader import load_prompt
from ._registry import ScriptInfo, SkillInfo, SkillsRegistry, load_skills
from .discovery import find_skills_path, looks_like_skills_checkout
from .meta import CanonicalScript, Param, Skill, SkillMeta, SkillRequires
from .registries import (
    RegistrySet,
    RegistrySpec,
    as_registry_paths,
    load_registry_set,
    resolve_registries,
)

try:
    # gap.tools is ported in parallel; guard so the loader surface works
    # even when the tool registry isn't importable.
    from gap.tools import tool
except ImportError:  # pragma: no cover
    tool = None  # type: ignore[assignment]

__all__ = [
    "CanonicalScript",
    "Param",
    "RegistrySet",
    "RegistrySpec",
    "ScriptInfo",
    "Skill",
    "SkillInfo",
    "SkillMeta",
    "SkillRequires",
    "SkillsRegistry",
    "as_registry_paths",
    "find_skills_path",
    "load_prompt",
    "load_registry_set",
    "load_skills",
    "looks_like_skills_checkout",
    "parse_skill_md",
    "resolve_registries",
    "tool",
]
