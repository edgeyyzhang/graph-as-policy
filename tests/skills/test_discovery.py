"""Unit tests for gap.skills.find_skills_path (checkout discovery).

Resolution order under test: explicit argument > $GAP_SKILLS_PATH >
sibling-walk from the given roots > None / FileNotFoundError. The walk
roots are injected via ``search_from`` so the tests are hermetic — the
developer machine's real side-by-side checkout never leaks in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gap.skills import find_skills_path, looks_like_skills_checkout
from gap.skills.discovery import GAP_SKILLS_PATH_ENV


def _make_checkout(root: Path) -> Path:
    """A minimal directory tree that passes looks_like_skills_checkout."""
    for kind, bundle in (("tools", "fixture-tool"), ("skills", "fixture-skill")):
        d = root / kind / bundle
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n")
    return root


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(GAP_SKILLS_PATH_ENV, raising=False)


# ---------------------------------------------------------------------------
# looks_like_skills_checkout
# ---------------------------------------------------------------------------


def test_checkout_shape_requires_both_roots_with_bundles(tmp_path: Path):
    assert not looks_like_skills_checkout(tmp_path)

    co = _make_checkout(tmp_path / "open-robot-skills")
    assert looks_like_skills_checkout(co)

    # tools/ without any SKILL.md bundle does not count.
    empty = tmp_path / "empty"
    (empty / "tools" / "thing").mkdir(parents=True)
    (empty / "skills" / "thing").mkdir(parents=True)
    assert not looks_like_skills_checkout(empty)

    # Underscore-prefixed dirs are skipped (mirrors the loader).
    underscored = tmp_path / "underscored"
    for kind in ("tools", "skills"):
        d = underscored / kind / "_private"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("x")
    assert not looks_like_skills_checkout(underscored)


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_explicit_argument_wins(tmp_path: Path, monkeypatch):
    co = _make_checkout(tmp_path / "explicit")
    other = _make_checkout(tmp_path / "env-checkout")
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, str(other))
    # Explicit beats env and is returned as-given (resolved), even without
    # checkout-shape validation (load_skills surfaces problems loudly).
    assert find_skills_path(co) == co.resolve()
    assert find_skills_path(tmp_path / "nonexistent") == (tmp_path / "nonexistent").resolve()


def test_env_var_resolves(tmp_path: Path, monkeypatch):
    co = _make_checkout(tmp_path / "from-env")
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, str(co))
    assert find_skills_path(search_from=[tmp_path / "elsewhere"]) == co.resolve()


def test_env_var_invalid_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, str(tmp_path / "not-a-checkout"))
    with pytest.raises(ValueError, match="GAP_SKILLS_PATH"):
        find_skills_path(search_from=[tmp_path])


def test_sibling_walk_finds_checkout_upwards(tmp_path: Path):
    co = _make_checkout(tmp_path / "open-robot-skills")
    deep = tmp_path / "gap" / "examples" / "libero_quickstart"
    deep.mkdir(parents=True)
    assert find_skills_path(search_from=[deep]) == co.resolve()


def test_walk_accepts_being_inside_the_checkout(tmp_path: Path):
    co = _make_checkout(tmp_path / "open-robot-skills")
    inner = co / "skills"
    assert find_skills_path(search_from=[inner]) == co.resolve()


def test_miss_returns_none_and_required_raises(tmp_path: Path):
    lonely = tmp_path / "gap" / "src"
    lonely.mkdir(parents=True)
    assert find_skills_path(search_from=[lonely]) is None
    with pytest.raises(FileNotFoundError) as exc_info:
        find_skills_path(search_from=[lonely], required=True)
    message = str(exc_info.value)
    # The error lists everything that was tried, in order.
    assert "explicit path argument (not given)" in message
    assert GAP_SKILLS_PATH_ENV in message
    assert str(lonely) in message
    assert "open-robot-skills" in message
