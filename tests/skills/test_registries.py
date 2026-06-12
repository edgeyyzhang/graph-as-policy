"""Unit tests for gap.skills.registries (multi-registry resolution + merge).

Resolution order under test: explicit argument > $GAP_SKILLS_PATH
(os.pathsep list) > project pyproject [tool.gap].registries > user
~/.config/gap/registries.toml > sibling auto-discovery. All sources are
injected (tmp_path cwd, monkeypatched XDG_CONFIG_HOME, explicit
search_from) so the developer machine's real checkouts and config never
leak in.

Bundle names are prefixed ``rgt-`` (registry test) and unique per test:
the ``gap_skills.*`` synthetic namespace is process-global, so reusing a
name across tests would alias previously imported modules.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from gap.skills import (
    as_registry_paths,
    find_skills_path,
    load_registry_set,
    resolve_registries,
)
from gap.skills.discovery import GAP_SKILLS_PATH_ENV
from gap.skills.registries import (
    find_project_config,
    read_user_registries,
    registry_dist_name,
    user_config_path,
    write_user_registries,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No env var, and user config redirected into the test's tmp dir."""
    monkeypatch.delenv(GAP_SKILLS_PATH_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def make_registry(
    root: Path,
    *,
    tools: tuple[str, ...] = (),
    skills: tuple[str, ...] = (),
    dist_name: str | None = None,
    marker: str | None = None,
) -> Path:
    """Fabricate a minimal valid registry checkout at *root*."""
    for kind, names in (("tools", tools), ("skills", skills)):
        for name in names:
            d = root / kind / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: A fixture bundle. "
                f"Use when testing registries.\n---\n"
            )
            if marker is not None:
                (d / "tools.py").write_text(f"MARKER = {marker!r}\n")
    if dist_name is not None:
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "{dist_name}"\nversion = "0.0.0"\n'
        )
    return root


# ---------------------------------------------------------------------------
# as_registry_paths
# ---------------------------------------------------------------------------


def test_as_registry_paths_normalization(tmp_path: Path):
    assert as_registry_paths(None) is None
    assert as_registry_paths([]) is None
    single = as_registry_paths(str(tmp_path))
    assert single == [tmp_path.resolve()]
    assert as_registry_paths(tmp_path) == [tmp_path.resolve()]
    many = as_registry_paths([tmp_path / "a", str(tmp_path / "b")])
    assert many == [(tmp_path / "a").resolve(), (tmp_path / "b").resolve()]


# ---------------------------------------------------------------------------
# Resolution precedence
# ---------------------------------------------------------------------------


def test_explicit_is_a_full_override(tmp_path: Path, monkeypatch):
    env_reg = make_registry(tmp_path / "env-reg", tools=("rgt-env",))
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, str(env_reg))
    a, b = tmp_path / "a", tmp_path / "b"
    rs = resolve_registries([a, b], cwd=tmp_path, search_from=[tmp_path])
    # Exactly the explicit entries, in order, unvalidated (may not exist),
    # nothing merged in from env/config/auto.
    assert rs.paths() == [a.resolve(), b.resolve()]
    assert [s.source for s in rs] == ["flag", "flag"]


def test_env_single_path_back_compat(tmp_path: Path, monkeypatch):
    reg = make_registry(tmp_path / "from-env", tools=("rgt-envtool",))
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, str(reg))
    rs = resolve_registries(search_from=[tmp_path / "elsewhere"])
    assert rs.paths() == [reg.resolve()]
    spec = rs.primary()
    assert spec is not None and spec.source == "env"
    assert spec.origin == f"${GAP_SKILLS_PATH_ENV}"


def test_env_pathsep_list_in_order(tmp_path: Path, monkeypatch):
    first = make_registry(tmp_path / "first", tools=("rgt-f",))
    second = make_registry(tmp_path / "second", skills=("rgt-s",))
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, os.pathsep.join([str(first), str(second)]))
    rs = resolve_registries(search_from=[tmp_path / "elsewhere"])
    assert rs.paths() == [first.resolve(), second.resolve()]
    assert [s.origin for s in rs] == [
        f"${GAP_SKILLS_PATH_ENV}[0]", f"${GAP_SKILLS_PATH_ENV}[1]",
    ]


def test_env_invalid_entry_raises_naming_it(tmp_path: Path, monkeypatch):
    good = make_registry(tmp_path / "good", tools=("rgt-g",))
    bad = tmp_path / "not-a-registry"
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, os.pathsep.join([str(good), str(bad)]))
    with pytest.raises(ValueError, match="not-a-registry"):
        resolve_registries(search_from=[tmp_path])


def test_project_config_walkup_and_relative_paths(tmp_path: Path):
    reg = make_registry(tmp_path / "project-skills", skills=("rgt-proj",))
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "robot-project"\n'
        '[tool.gap]\nregistries = ["./project-skills"]\n'
    )
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    rs = resolve_registries(cwd=nested, search_from=[tmp_path / "nowhere"])
    assert rs.paths() == [reg.resolve()]
    spec = rs.primary()
    assert spec is not None and spec.source == "project"
    assert "[tool.gap]" in spec.origin


def test_user_config_round_trip_and_resolution(tmp_path: Path):
    reg = make_registry(tmp_path / "lab-skills", tools=("rgt-lab",))
    write_user_registries([("lab", reg)])
    assert user_config_path().is_file()
    assert read_user_registries() == [("lab", reg)]

    rs = resolve_registries(cwd=tmp_path / "nowhere", search_from=[tmp_path / "nowhere"])
    assert rs.paths() == [reg.resolve()]
    spec = rs.primary()
    assert spec is not None and spec.source == "user" and spec.name == "lab"


def test_union_order_project_user_auto_with_dedup(tmp_path: Path):
    shared = make_registry(tmp_path / "shared", tools=("rgt-shared",))
    user_only = make_registry(tmp_path / "user-only", skills=("rgt-uo",))
    sibling = make_registry(tmp_path / "open-robot-skills", tools=("rgt-sib",))

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[tool.gap]\nregistries = ["../shared"]\n'
    )
    # `shared` also in user config -> deduped, project provenance wins.
    write_user_registries([("user-only", user_only), ("shared-again", shared)])

    rs = resolve_registries(cwd=project_dir, search_from=[tmp_path / "anchor"])
    assert rs.paths() == [shared.resolve(), user_only.resolve(), sibling.resolve()]
    assert [s.source for s in rs] == ["project", "user", "auto"]


def test_broken_user_entry_skipped_with_warning(tmp_path: Path, caplog):
    reg = make_registry(tmp_path / "ok-reg", tools=("rgt-ok",))
    write_user_registries([("broken", tmp_path / "gone"), ("ok", reg)])
    with caplog.at_level(logging.WARNING, logger="gap.skills.registries"):
        rs = resolve_registries(cwd=tmp_path / "nowhere", search_from=[tmp_path / "nowhere"])
    assert rs.paths() == [reg.resolve()]
    assert any("broken" in r.message for r in caplog.records)


def test_required_raises_listing_everything_tried(tmp_path: Path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_registries(cwd=lonely, search_from=[lonely], required=True)
    message = str(exc_info.value)
    assert "explicit path argument (not given)" in message
    assert GAP_SKILLS_PATH_ENV in message
    assert "[tool.gap]" in message
    assert "gap registry add" in message
    assert str(lonely) in message


def test_include_config_false_skips_project_and_user(tmp_path: Path):
    configured = make_registry(tmp_path / "configured", tools=("rgt-cfg",))
    write_user_registries([("configured", configured)])
    (tmp_path / "pyproject.toml").write_text(
        '[tool.gap]\nregistries = ["./configured"]\n'
    )
    rs = resolve_registries(
        cwd=tmp_path, search_from=[tmp_path / "nowhere"], include_config=False,
    )
    assert not rs


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_names_prefer_dist_name_then_dirname(tmp_path: Path):
    named = make_registry(
        tmp_path / "dir-name", tools=("rgt-n1",), dist_name="published-skills",
    )
    unnamed = make_registry(tmp_path / "plain-dir", tools=("rgt-n2",))
    assert registry_dist_name(named) == "published-skills"
    assert registry_dist_name(unnamed) is None

    rs = resolve_registries([named, unnamed])
    assert rs.names() == ["published-skills", "plain-dir"]


def test_duplicate_names_get_suffixed(tmp_path: Path):
    a = make_registry(tmp_path / "a" / "same", tools=("rgt-d1",))
    b = make_registry(tmp_path / "b" / "same", tools=("rgt-d2",))
    rs = resolve_registries([a, b])
    assert rs.names() == ["same", "same-2"]
    assert rs.get("same").path == a.resolve()
    assert rs.get("same-2").path == b.resolve()
    with pytest.raises(KeyError, match="same-2"):
        rs.get("missing")


# ---------------------------------------------------------------------------
# Merged loading
# ---------------------------------------------------------------------------


def test_load_registry_set_merges_with_attribution(tmp_path: Path):
    lab = make_registry(
        tmp_path / "lab", skills=("rgt-merge-lab",), dist_name="lab-skills",
    )
    public = make_registry(
        tmp_path / "public", tools=("rgt-merge-pub",), dist_name="public-skills",
    )
    reg = load_registry_set(resolve_registries([lab, public]))
    assert len(reg) == 2
    assert reg.get("rgt-merge-lab").registry == "lab-skills"
    assert reg.get("rgt-merge-pub").registry == "public-skills"
    assert reg.get("rgt-merge-pub").kind == "tool"


def test_cross_registry_shadowing_first_wins_without_import(tmp_path: Path, caplog):
    winner = make_registry(
        tmp_path / "winner", tools=("rgt-shadowed",), marker="winner",
        dist_name="winner-skills",
    )
    loser = make_registry(
        tmp_path / "loser", tools=("rgt-shadowed",), marker="loser",
        dist_name="loser-skills",
    )
    with caplog.at_level(logging.WARNING, logger="gap.skills._registry"):
        reg = load_registry_set(resolve_registries([winner, loser]))

    assert len(reg) == 1
    info = reg.get("rgt-shadowed")
    assert info.registry == "winner-skills"
    assert info.bundle_dir == winner / "tools" / "rgt-shadowed"
    assert any("shadowed" in r.message for r in caplog.records)

    # The shadowed bundle's modules were never imported: the synthetic
    # module belongs to the winner.
    mod = sys.modules["gap_skills.tools.rgt-shadowed.tools"]
    assert mod.MARKER == "winner"


def test_intra_registry_collision_still_raises(tmp_path: Path):
    reg_dir = make_registry(
        tmp_path / "selfclash", tools=("rgt-clash",), skills=("rgt-clash",),
    )
    with pytest.raises(ValueError, match="name collision"):
        load_registry_set(resolve_registries([reg_dir]))


# ---------------------------------------------------------------------------
# find_skills_path shim
# ---------------------------------------------------------------------------


def test_find_skills_path_returns_primary_of_env_list(tmp_path: Path, monkeypatch):
    first = make_registry(tmp_path / "one", tools=("rgt-shim1",))
    second = make_registry(tmp_path / "two", tools=("rgt-shim2",))
    monkeypatch.setenv(GAP_SKILLS_PATH_ENV, os.pathsep.join([str(first), str(second)]))
    assert find_skills_path(search_from=[tmp_path / "elsewhere"]) == first.resolve()


def test_find_skills_path_ignores_registry_config(tmp_path: Path):
    configured = make_registry(tmp_path / "cfg-reg", tools=("rgt-shim3",))
    write_user_registries([("cfg-reg", configured)])
    # Legacy shim semantics: user/project config is not consulted.
    assert find_skills_path(search_from=[tmp_path / "elsewhere"]) is None


def test_single_root_registry_counts(tmp_path: Path):
    skills_only = make_registry(tmp_path / "skills-only", skills=("rgt-solo",))
    rs = resolve_registries([skills_only])
    reg = load_registry_set(rs)
    assert reg.get("rgt-solo").kind == "skill"
    assert find_project_config(tmp_path) is None
