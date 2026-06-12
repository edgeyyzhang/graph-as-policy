"""CLI plumbing for ``gap check`` (exit policy, JSON schema, filters)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.cli import main as cli_main
from gap.skills import capability as cap
from gap.skills.capability import ProbeResult


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["gap", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    return int(exc.value.code or 0)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_SKILLS_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(cap, "probe_gpu", lambda **kw: ProbeResult("ok", "FAKE GPU"))


@pytest.fixture()
def fixture_registry(tmp_path: Path) -> Path:
    reg = tmp_path / "reg"
    for name, gap_block in (
        ("rgt-cli-ready", ""),
        ("rgt-cli-needskey", "  requires: {env: [RGT_CLI_UNSET_KEY]}\n"),
    ):
        d = reg / "tools" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Fixture. Use when testing CLI.\n"
            f"gap:\n  tools:\n    - {name}.run: A fixture tool.\n{gap_block}---\n"
        )
        (d / "tools.py").write_text("")
    return reg


def test_check_json_schema_and_exit_zero(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch, ["check", "--skills", str(fixture_registry), "--format", "json"],
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == 1
    by_name = {b["name"]: b for b in data["bundles"]}
    assert by_name["rgt-cli-ready"]["status"] == "ready"
    needs = by_name["rgt-cli-needskey"]
    assert needs["status"] == "not-ready"
    (req,) = needs["requirements"].values()
    assert "RGT_CLI_UNSET_KEY" in req["detail"]
    assert "export RGT_CLI_UNSET_KEY" in req["fix_hint"]


def test_check_pretty_mentions_fix_hints(monkeypatch, capsys, fixture_registry):
    code = _run_cli(monkeypatch, ["check", "--skills", str(fixture_registry)])
    assert code == 0
    out = capsys.readouterr().out
    assert "rgt-cli-needskey: NOT READY" in out
    assert "export RGT_CLI_UNSET_KEY" in out
    assert "1 ready, 1 not ready" in out


def test_check_strict_flips_exit(monkeypatch, capsys, fixture_registry):
    assert _run_cli(
        monkeypatch, ["check", "--skills", str(fixture_registry), "--strict"],
    ) == 1
    monkeypatch.setenv("RGT_CLI_UNSET_KEY", "x")
    assert _run_cli(
        monkeypatch, ["check", "--skills", str(fixture_registry), "--strict"],
    ) == 0


def test_check_zero_registries_is_informative(monkeypatch, capsys, tmp_path):
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    monkeypatch.chdir(lonely)
    monkeypatch.setattr(
        "gap.skills.registries._default_search_from", lambda: (lonely,),
    )
    code = _run_cli(monkeypatch, ["check"])
    assert code == 0
    out = capsys.readouterr().out
    assert "no skill registries active" in out
    assert "gap registry add" in out


def test_check_registry_filter_unknown_name(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["check", "--skills", str(fixture_registry), "--registry", "nope"],
    )
    assert code == 2
    assert "not found" in capsys.readouterr().out
