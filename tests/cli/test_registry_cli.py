"""CLI plumbing for ``gap registry`` (init / list / add / remove)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gap.cli import main as cli_main
from gap.skills import resolve_registries
from gap.skills.registries import read_user_registries, user_config_path
from gap.skills.validate import validate_checkout


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["gap", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    return int(exc.value.code or 0)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_SKILLS_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    lonely = tmp_path / "lonely-anchor"
    lonely.mkdir()
    monkeypatch.chdir(lonely)
    monkeypatch.setattr(
        "gap.skills.registries._default_search_from", lambda: (lonely,),
    )


def _make_registry(root: Path, bundle: str = "rgt-reg-fixture") -> Path:
    d = root / "tools" / bundle
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {bundle}\ndescription: Fixture. Use when testing.\n"
        f"gap:\n  tools:\n    - {bundle}.run: A fixture tool.\n---\n"
    )
    return root


# ---------------------------------------------------------------------------
# add / remove / list round-trip
# ---------------------------------------------------------------------------


def test_add_list_remove_round_trip(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "lab-skills")

    assert _run_cli(monkeypatch, ["registry", "add", "lab", str(reg)]) == 0
    out = capsys.readouterr().out
    assert "added 'lab'" in out and "highest precedence" in out
    assert read_user_registries() == [("lab", reg)]

    assert _run_cli(monkeypatch, ["registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "lab" in out and "user" in out and str(reg) in out

    # Duplicate name -> error suggesting remove.
    assert _run_cli(monkeypatch, ["registry", "add", "lab", str(reg)]) == 1
    assert "already configured" in capsys.readouterr().out

    assert _run_cli(monkeypatch, ["registry", "remove", "lab"]) == 0
    assert read_user_registries() == []

    assert _run_cli(monkeypatch, ["registry", "remove", "lab"]) == 1
    assert "not in" in capsys.readouterr().out


def test_add_prepends_for_precedence(monkeypatch, capsys, tmp_path):
    first = _make_registry(tmp_path / "first", "rgt-reg-a")
    second = _make_registry(tmp_path / "second", "rgt-reg-b")
    _run_cli(monkeypatch, ["registry", "add", "first", str(first)])
    _run_cli(monkeypatch, ["registry", "add", "second", str(second)])
    # Most recently added shadows -> it is first in the file.
    assert [name for name, _ in read_user_registries()] == ["second", "first"]
    rs = resolve_registries()
    assert rs.names()[:2] == ["second", "first"]


def test_add_rejects_remote_urls(monkeypatch, capsys):
    for url in (
        "git@github.com:lab/skills.git",
        "https://github.com/lab/skills.git",
        "https://example.com/skills",
    ):
        assert _run_cli(monkeypatch, ["registry", "add", "lab", url]) == 2
        out = capsys.readouterr().out
        assert "not supported yet" in out
        assert "git clone" in out
    assert not user_config_path().is_file()


def test_add_rejects_bad_name_and_non_registry(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "ok-reg")
    assert _run_cli(monkeypatch, ["registry", "add", "Bad_Name", str(reg)]) == 2
    assert "invalid registry name" in capsys.readouterr().out

    not_reg = tmp_path / "plain"
    not_reg.mkdir()
    assert _run_cli(monkeypatch, ["registry", "add", "plain", str(not_reg)]) == 2
    assert "gap registry init" in capsys.readouterr().out


def test_list_shows_broken_user_entries(monkeypatch, capsys, tmp_path):
    reg = _make_registry(tmp_path / "good-reg")
    _run_cli(monkeypatch, ["registry", "add", "good", str(reg)])
    # Break a configured entry after the fact.
    from gap.skills.registries import write_user_registries

    write_user_registries([("gone", tmp_path / "gone"), ("good", reg)])
    assert _run_cli(monkeypatch, ["registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "broken (not a registry checkout)" in out
    assert "gone" in out


def test_list_json_and_empty_guidance(monkeypatch, capsys, tmp_path):
    assert _run_cli(monkeypatch, ["registry", "list"]) == 0
    assert "gap registry init" in capsys.readouterr().out

    reg = _make_registry(tmp_path / "json-reg")
    _run_cli(monkeypatch, ["registry", "add", "json-reg", str(reg)])
    capsys.readouterr()
    assert _run_cli(monkeypatch, ["registry", "list", "--format", "json"]) == 0
    import json

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "json-reg"
    assert rows[0]["source"] == "user"
    assert rows[0]["tools"] == 1


# ---------------------------------------------------------------------------
# --project scope
# ---------------------------------------------------------------------------


def test_project_add_creates_and_prepends(monkeypatch, capsys, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "robot-proj"\n')
    monkeypatch.chdir(project)

    reg_a = _make_registry(tmp_path / "reg-a", "rgt-reg-pa")
    assert _run_cli(
        monkeypatch, ["registry", "add", "reg-a", str(reg_a), "--project"],
    ) == 0
    text = (project / "pyproject.toml").read_text()
    assert "[tool.gap]" in text and "reg-a" in text

    reg_b = _make_registry(tmp_path / "reg-b", "rgt-reg-pb")
    assert _run_cli(
        monkeypatch, ["registry", "add", "reg-b", str(reg_b), "--project"],
    ) == 0
    rs = resolve_registries(cwd=project)
    # Most recently added first within the project layer.
    assert rs.paths() == [reg_b.resolve(), reg_a.resolve()]
    assert {s.source for s in rs} == {"project"}

    # remove by derived name (dirname)
    assert _run_cli(monkeypatch, ["registry", "remove", "reg-a", "--project"]) == 0
    rs = resolve_registries(cwd=project)
    assert rs.paths() == [reg_b.resolve()]


def test_project_add_refuses_multiline_array(monkeypatch, capsys, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.gap]\nregistries = [\n  "./a",\n  "./b",\n]\n'
    )
    monkeypatch.chdir(project)
    reg = _make_registry(tmp_path / "reg-c", "rgt-reg-pc")
    assert _run_cli(
        monkeypatch, ["registry", "add", "reg-c", str(reg), "--project"],
    ) == 1
    assert "manually" in capsys.readouterr().out
    # File untouched.
    assert '"./a"' in (project / "pyproject.toml").read_text()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_scaffolds_a_valid_registry(monkeypatch, capsys, tmp_path):
    target = tmp_path / "my-lab-skills"
    assert _run_cli(monkeypatch, ["registry", "init", str(target)]) == 0
    out = capsys.readouterr().out
    assert "scaffolded registry 'my-lab-skills'" in out
    assert "gap registry add" in out

    assert (target / "pyproject.toml").is_file()
    assert (target / "tools").is_dir() and (target / "skills").is_dir()
    assert (target / "tests" / "conftest.py").is_file()
    assert (target / "README.md").is_file()
    from gap.skills.registries import is_registry_dir, registry_dist_name

    assert is_registry_dir(target)
    assert registry_dist_name(target) == "my-lab-skills"
    assert validate_checkout(target) == []  # no bundles yet, nothing invalid

    # Empty-but-initialized registries are addable and resolvable.
    assert _run_cli(monkeypatch, ["registry", "add", "my-lab", str(target)]) == 0
    rs = resolve_registries()
    assert target.resolve() in rs.paths()


def test_init_add_flag_and_refusals(monkeypatch, capsys, tmp_path):
    target = tmp_path / "lab2"
    assert _run_cli(
        monkeypatch, ["registry", "init", str(target), "--name", "lab-two", "--add"],
    ) == 0
    assert read_user_registries()[0][0] == "lab-two"

    # Refuse to scaffold over an existing registry.
    assert _run_cli(monkeypatch, ["registry", "init", str(target)]) == 1
    assert "refusing" in capsys.readouterr().out

    # Bad implied name -> ask for --name.
    bad = tmp_path / "Bad_Dir"
    bad.mkdir()
    assert _run_cli(monkeypatch, ["registry", "init", str(bad)]) == 2
    assert "--name" in capsys.readouterr().out
