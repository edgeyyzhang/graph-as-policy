"""CLI plumbing for ``gap tools`` (live vs static rows, show, suggestions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.cli import main as cli_main


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["gap", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    return int(exc.value.code or 0)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv("GAP_SKILLS_PATH", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture()
def fixture_registry(tmp_path: Path) -> Path:
    reg = tmp_path / "reg"

    good = reg / "tools" / "rgt-tools-good"
    good.mkdir(parents=True)
    (good / "SKILL.md").write_text(
        "---\nname: rgt-tools-good\n"
        "description: Echo bundle. Use when testing tools list.\n"
        "metadata: {category: test, tags: [echoing]}\n"
        "gap:\n  tools:\n    - rgt-tools-good.echo: Echo a string.\n---\n"
    )
    (good / "tools.py").write_text(
        "from typing import TypedDict\n"
        "from gap.tools import tool\n\n\n"
        "class Out(TypedDict):\n"
        "    text: str\n\n\n"
        '@tool(name="rgt-tools-good.echo", summary="Echo a string.", tags=("echoing",))\n'
        "def echo(text: str, times: int = 1) -> Out:\n"
        '    """Echo text a number of times."""\n'
        '    return {"text": text * times}\n'
    )

    broken = reg / "tools" / "rgt-tools-broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text(
        "---\nname: rgt-tools-broken\n"
        "description: Broken bundle. Use when testing static fallback.\n"
        "gap:\n  tools:\n    - rgt-tools-broken.run: Declared but uninstallable.\n---\n"
    )
    (broken / "tools.py").write_text("import rgt_tools_no_such_dep\n")
    return reg


def test_list_live_and_static_rows(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "list", "--skills", str(fixture_registry), "--format", "json"],
    )
    assert code == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}

    live = rows["rgt-tools-good.echo"]
    assert live["status"] == "ok"
    assert live["inputs"] == {"text": "str", "times": "int"}
    assert live["outputs"] == {"text": "str"}
    assert live["registry"] == "reg"

    static = rows["rgt-tools-broken.run"]
    assert "deps not installed" in static["status"]
    assert static["inputs"] is None

    # Connector rows are always present.
    assert rows["robot.get_observation"]["bundle"] == "connector"


def test_list_static_flag_skips_imports(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "list", "--skills", str(fixture_registry), "--static",
         "--format", "json"],
    )
    assert code == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    # Even the importable bundle is reported statically.
    assert rows["rgt-tools-good.echo"]["status"] == "static (--static)"
    assert rows["rgt-tools-good.echo"]["inputs"] is None


def test_list_tag_filter_and_markdown(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "list", "--skills", str(fixture_registry), "--tag", "echoing"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "rgt-tools-good.echo" in out
    assert "robot.get_observation" not in out

    code = _run_cli(
        monkeypatch,
        ["tools", "list", "--skills", str(fixture_registry),
         "--format", "markdown", "--tag", "echoing"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.startswith("| NAME |")
    assert "`rgt-tools-good.echo`" in out

    code = _run_cli(
        monkeypatch,
        ["tools", "list", "--skills", str(fixture_registry), "--tag", "nope"],
    )
    assert code == 1


def test_show_live_schema_and_runnability(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "show", "rgt-tools-good.echo", "--skills", str(fixture_registry)],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "text: str  (required)" in out
    assert "times: int  (default=1)" in out
    assert "runnability: ready" in out


def test_show_static_fallback_and_json(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "show", "rgt-tools-broken.run", "--skills", str(fixture_registry)],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "schemas unavailable" in out
    assert "not-ready" in out

    code = _run_cli(
        monkeypatch,
        ["tools", "show", "rgt-tools-good.echo", "--skills",
         str(fixture_registry), "--format", "json"],
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["inputs"]["times"] == {
        "type": "int", "required": False, "default": 1, "description": "",
    }
    assert data["runnability"] == "ready"


def test_show_unknown_suggests_close_matches(monkeypatch, capsys, fixture_registry):
    code = _run_cli(
        monkeypatch,
        ["tools", "show", "rgt-tools-good.eco", "--skills", str(fixture_registry)],
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "unknown tool" in out
    assert "rgt-tools-good.echo" in out
