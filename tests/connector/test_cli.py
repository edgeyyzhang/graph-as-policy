"""CLI argument plumbing: gap run --validate-only, gap skills list/check/new."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.cli import main as cli_main
from gap.cli.run import _parse_inputs

_FIXTURES = Path(__file__).resolve().parents[1] / "skills" / "fixtures"


def _valid_workflow() -> dict:
    return {
        "version": 3,
        "meta": {"name": "cli_fixture"},
        "nodes": {
            "sg": {"type": "subgraph", "ref": "sg_def"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "sg"]],
        "conditional_edges": {
            "sg": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "sg_def": {
                "skill": "generic",
                "inputs": {},
                "outputs": {},
                "nodes": {"ok": {"type": "noop"}},
                "edges": [["START", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["gap", *argv])
    with pytest.raises(SystemExit) as exc:
        cli_main()
    return int(exc.value.code or 0)


# ---------------------------------------------------------------------------
# gap run
# ---------------------------------------------------------------------------


def test_validate_only_ok(tmp_path, monkeypatch, capsys):
    (tmp_path / "workflow.json").write_text(json.dumps(_valid_workflow()))
    code = _run_cli(monkeypatch, ["run", str(tmp_path), "--validate-only"])
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_validate_only_bad_graph(tmp_path, monkeypatch, capsys):
    graph = _valid_workflow()
    graph["edges"] = []  # W2: no edge from START
    (tmp_path / "workflow.json").write_text(json.dumps(graph))
    code = _run_cli(monkeypatch, ["run", str(tmp_path), "--validate-only"])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_run_executes_graph(tmp_path, monkeypatch, capsys):
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(_valid_workflow()))
    trace_dir = tmp_path / "trace"
    code = _run_cli(
        monkeypatch,
        ["run", str(wf_dir), "--trace-dir", str(trace_dir), "--checkpoints", "off"],
    )
    assert code == 0
    assert "SUCCESS" in capsys.readouterr().out


def test_run_no_trace(tmp_path, monkeypatch):
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(_valid_workflow()))
    monkeypatch.chdir(tmp_path)
    code = _run_cli(monkeypatch, ["run", str(wf_dir), "--no-trace"])
    assert code == 0
    assert not (tmp_path / "outputs").exists()


def test_video_skipped_gracefully_without_sim(tmp_path, monkeypatch, capsys):
    """Video is ON by default, but a tools-only run (no --sim) has nothing to
    render, so it's silently skipped rather than erroring. The `--no-video`
    opt-out and the deprecated `--record-video` no-op are both accepted."""
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(_valid_workflow()))
    monkeypatch.chdir(tmp_path)

    code = _run_cli(monkeypatch, ["run", str(wf_dir)])
    assert code == 0
    assert "video:" not in capsys.readouterr().out  # no sim → nothing recorded

    # Both flags are accepted on a tools-only run and don't change the outcome.
    assert _run_cli(monkeypatch, ["run", str(wf_dir), "--no-video"]) == 0
    assert _run_cli(monkeypatch, ["run", str(wf_dir), "--record-video"]) == 0


def test_parse_inputs():
    parsed = _parse_inputs(["k=1", "s=hello", "obj={\"a\": 2}", "flag=true"])
    assert parsed == {"k": 1, "s": "hello", "obj": {"a": 2}, "flag": True}
    with pytest.raises(SystemExit):
        _parse_inputs(["novalue"])


# ---------------------------------------------------------------------------
# gap skills
# ---------------------------------------------------------------------------


def test_skills_list_fixtures(monkeypatch, capsys):
    code = _run_cli(monkeypatch, ["skills", "list", "--skills", str(_FIXTURES)])
    assert code == 0
    out = capsys.readouterr().out
    assert "fixture-tool" in out
    assert "fixture-skill" in out
    assert "echo" in out  # tools column (SKILL.md gap.tools short names)
    assert "tool" in out and "skill" in out  # kind column


def test_skills_check_fixtures(monkeypatch, capsys):
    code = _run_cli(monkeypatch, ["skills", "check", "--skills", str(_FIXTURES)])
    assert code == 0
    out = capsys.readouterr().out
    assert "[tool] fixture-tool: PASS" in out
    assert "[skill] fixture-skill: PASS" in out
    assert "2 PASS, 0 WARN, 0 FAIL" in out


def test_skills_check_format_failure_is_nonzero(tmp_path, monkeypatch, capsys):
    """Format errors (not just import errors) FAIL the check."""
    bundle = tmp_path / "skills" / "bad-skill"
    (bundle / "scripts").mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        "---\n"
        "name: bad-skill\n"
        "description: A skill bundle without exit conditions. Use when testing.\n"
        "gap:\n"
        "  canonical_scripts:\n"
        "    - missing: scripts/does_not_exist.py\n"
        "---\n\n# bad-skill\n"
    )
    code = _run_cli(monkeypatch, ["skills", "check", "--skills", str(tmp_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "[skill] bad-skill: FAIL" in out
    assert "exit_conditions" in out
    assert "does_not_exist.py" in out


def test_skills_table_pretty_and_markdown_and_json(monkeypatch, capsys):
    code = _run_cli(monkeypatch, ["skills", "table", "--skills", str(_FIXTURES)])
    assert code == 0
    out = capsys.readouterr().out
    assert "fixture-tool" in out and "fixture-skill" in out
    assert "Bundle" in out and "Kind" in out and "Extra" in out

    code = _run_cli(
        monkeypatch,
        ["skills", "table", "--skills", str(_FIXTURES), "--format", "markdown"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| Bundle | Kind | Description | Tools | Extra |" in out
    assert "| [fixture-tool](tools/fixture-tool/) | tool |" in out
    assert "`fixture-tool.echo`" in out

    # --kind filters rows and drops the Kind column from markdown output.
    code = _run_cli(
        monkeypatch,
        ["skills", "table", "--skills", str(_FIXTURES), "--format", "markdown",
         "--kind", "skill"],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "| Bundle | Description | Tools | Extra |" in out
    assert "[fixture-skill](skills/fixture-skill/)" in out
    assert "fixture-tool" not in out

    code = _run_cli(
        monkeypatch,
        ["skills", "table", "--skills", str(_FIXTURES), "--format", "json"],
    )
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    by_name = {r["name"]: r for r in rows}
    assert by_name["fixture-tool"]["kind"] == "tool"
    assert by_name["fixture-tool"]["tools"] == ["fixture-tool.echo"]
    assert by_name["fixture-skill"]["kind"] == "skill"
    # First sentence only, no pyproject in the fixture checkout -> no extra.
    assert by_name["fixture-skill"]["description"].endswith(".")
    assert by_name["fixture-skill"]["extra"] == ""


def test_skills_check_download_reports_no_weights(monkeypatch, capsys):
    code = _run_cli(
        monkeypatch, ["skills", "check", "--skills", str(_FIXTURES), "--download"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "declares no weights" in out


def test_skills_check_reports_missing_extras(tmp_path, monkeypatch, capsys):
    """A bundle whose tools.py import fails maps to its pip extra."""
    bundle = tmp_path / "tools" / "broken-bundle"
    bundle.mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        "---\n"
        "name: broken-bundle\n"
        "description: A bundle with a missing dependency.\n"
        "---\n\n# broken-bundle\n"
    )
    (bundle / "tools.py").write_text("import not_a_real_module_xyz\n")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = \"open-robot-skills\"\nversion = \"0\"\n"
        "[project.optional-dependencies]\n"
        "broken-bundle = [\"not-a-real-module-xyz\"]\n"
    )
    code = _run_cli(monkeypatch, ["skills", "check", "--skills", str(tmp_path)])
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "not_a_real_module_xyz" in out
    assert "pip install 'open-robot-skills[broken-bundle]'" in out


def test_skills_new_scaffolds_loadable_bundles(tmp_path, monkeypatch, capsys):
    code = _run_cli(
        monkeypatch,
        ["skills", "new", "my-tool", "--kind", "tool", "--skills", str(tmp_path)],
    )
    assert code == 0
    assert (tmp_path / "tools" / "my-tool" / "SKILL.md").is_file()
    assert (tmp_path / "tools" / "my-tool" / "tools.py").is_file()

    code = _run_cli(
        monkeypatch,
        ["skills", "new", "my-skill", "--kind", "skill", "--skills", str(tmp_path)],
    )
    assert code == 0
    assert (tmp_path / "skills" / "my-skill" / "SKILL.md").is_file()
    assert (tmp_path / "skills" / "my-skill" / "scripts" / "example.py").is_file()

    # The scaffolded bundles must pass the loader (gap skills check).
    code = _run_cli(monkeypatch, ["skills", "check", "--skills", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "[tool] my-tool: PASS" in out
    assert "[skill] my-skill: PASS" in out

    # Refuses to overwrite.
    code = _run_cli(
        monkeypatch,
        ["skills", "new", "my-tool", "--kind", "tool", "--skills", str(tmp_path)],
    )
    assert code == 1
