"""The rehearsal loop offline: contract checks, workspace layout, and the
broker answering requests written by the agent-side tool. The rehearsal
subprocess is replaced by a stub that writes what a rehearsal writes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gap.rehearse.loop import Broker, GraphRejected, check_graph, init_loop
from gap.rehearse.loop import broker as broker_module
from gap.rehearse.loop.protocol import graph_files, script_issues
from gap.rehearse.loop.workspace import read_ledger

WORKFLOW = {
    "version": 3, "meta": {},
    "nodes": {"close": {"type": "tool", "tool": "robot.close_gripper", "inputs": {}},
              "done": {"type": "end", "status": "success"}},
    "edges": [["START", "close"], ["close", "done"]], "conditional_edges": {}, "subgraphs": {},
}

# Stands in for `gap run --validate-only` and `gap rehearse`.
STUB = r'''
import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "run":
    print("OK: 0 errors, 0 warning(s)")
    sys.exit(0)
out = Path(args[args.index("--out") + 1])
graph = args[1]
cases = [int(c) for c in args[args.index("--cases") + 1].split(",")]
(out / "feedback").mkdir(parents=True)
(out / "perception_cache").mkdir()
(out / "perception_cache" / "blob.pkl").write_text("cache")
rows = [{"case": c, "success": c % 2 == 1} for c in cases]
fb = {"summary": {"cases": len(rows), "successes": sum(r["success"] for r in rows)}, "cases": rows,
      "previous": "--previous" in args}
(out / "feedback.json").write_text(json.dumps(fb))
(out / "feedback" / "main.md").write_text(f"Graph `{graph}`, results in {out}\n")
for c in cases:
    d = out / "cases" / f"case_{c:04d}"
    d.mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"case": c, "trace_dir": str(d / "trace")}))
'''


def _graph(root: Path, scripts: dict[str, str] | None = None, workflow: dict | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "workflow.json").write_text(json.dumps(workflow or WORKFLOW))
    for rel, source in (scripts or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(source)
    return root


@pytest.fixture
def loop(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(broker_module, "_GAP_MAIN", STUB)
    monkeypatch.setattr(broker_module, "pick_gpu", lambda: None)
    skills = tmp_path / "skills"
    (skills / "demo" / "scripts").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text("# demo")
    (skills / "demo" / "scripts" / "library.py").write_text("import os\n\ndef run(ctx):\n    return {}\n")
    start = _graph(tmp_path / "start", {"scripts/a.py": "def run(ctx):\n    return {}\n"})
    config = init_loop(
        tmp_path / "ws", tmp_path / "trusted", graph=start, sim="fake/0",
        instruction="put the can in the basket", visible_cases=[1, 2, 3], holdout_cases=[41, 42],
        rehearsal_budget=2, skills=[skills], python=False, confine=False,
    )
    return Path(config["workspace"]), Path(config["trusted"])


def _tool(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(workspace / "runtime" / "tools" / "submit.py"), *args],
                          capture_output=True, text=True, cwd=workspace)


# --- contract --------------------------------------------------------------


def test_script_contract_names_each_violation():
    source = (
        "import os\nfrom pathlib import Path\nimport numpy as np\nfrom gap.connector.sim import sim\n"
        "def run(ctx, x):\n"
        "    open('/etc/passwd')\n"
        "    ctx.tool('sim.check_success')\n"
        "    ctx._tool_registry\n"
        "    x.__class__\n"
        "    return ctx.tool('robot.get_ee_pose')\n"
    )
    issues = script_issues(source, "s.py")
    assert [i.split(": ", 1)[1] for i in issues] == [
        "import of 'os' is not allowed",
        "import from 'pathlib' is not allowed",
        "import from 'gap.connector.sim' is not allowed",
        "call to 'open' is not allowed",
        "tool 'sim.check_success' is not allowed",
        "private attribute 'ctx._tool_registry' is not allowed",
        "access to '__class__' is not allowed",
    ]
    assert script_issues("import numpy as np\n\ndef run(ctx, a):\n    return {'b': np.asarray(a).sum()}\n", "ok.py") == []
    assert "syntax error" in script_issues("def run(:\n", "bad.py")[0]


def test_check_graph_accepts_trusted_scripts_and_rejects_the_rest(tmp_path: Path):
    graph = _graph(tmp_path / "g", {"scripts/a.py": "import os\n\ndef run(ctx):\n    return {}\n"})
    with pytest.raises(GraphRejected, match="import of 'os'"):
        check_graph(graph)
    files = graph_files(graph)
    assert check_graph(graph, trusted_hashes={files["scripts/a.py"]}) == files

    sim_tool = dict(WORKFLOW, nodes={**WORKFLOW["nodes"], "peek": {"type": "tool", "tool": "sim.check_success"}})
    with pytest.raises(GraphRejected, match="main.peek uses tool 'sim.check_success'"):
        check_graph(_graph(tmp_path / "g2", workflow=sim_tool))

    linked = _graph(tmp_path / "g3")
    (linked / "scripts").mkdir()
    (linked / "scripts" / "link.py").symlink_to("/etc/hostname")
    with pytest.raises(GraphRejected, match="symbolic links"):
        check_graph(linked)
    with pytest.raises(GraphRejected, match="file type not accepted"):
        check_graph(_graph(tmp_path / "g4", {"data.pkl": "x"}))
    with pytest.raises(GraphRejected, match="no workflow.json"):
        (tmp_path / "g5").mkdir()
        check_graph(tmp_path / "g5")


# --- workspace -------------------------------------------------------------


def test_init_creates_both_sides_and_keeps_heldout_out_of_the_workspace(loop):
    workspace, trusted = loop
    for rel in ("results/graph/workflow.json", "results/notes", ".cache/tmp", ".requests", "eval/run",
                "eval/status.json", "runtime/tools/submit.py", "runtime/tools/status.py",
                "runtime/skills/demo/SKILL.md", "AGENT_PROMPT.md"):
        assert (workspace / rel).exists(), rel
    prompt = (workspace / "AGENT_PROMPT.md").read_text()
    assert "put the can in the basket" in prompt and "Visible cases: 1, 2, 3." in prompt
    assert "**2 rehearsals**" in prompt and "{" not in prompt.split("## The workspace")[0]
    config = json.loads((trusted / "config.json").read_text())
    assert config["holdout_cases"] == [41, 42]
    for path in workspace.rglob("*"):
        # runtime/ holds copies of the documentation and the skill library.
        if path.is_file() and path.suffix in (".json", ".md") and "runtime" not in path.relative_to(workspace).parts:
            text = path.read_text()
            assert "41" not in text and "holdout" not in text.lower(), path


def test_init_refuses_overlap_and_existing_directories(tmp_path: Path, loop):
    workspace, trusted = loop
    start = _graph(tmp_path / "s2")
    kw = dict(graph=start, sim="fake/0", instruction="x", python=False, confine=False)
    with pytest.raises(FileExistsError):
        init_loop(workspace, tmp_path / "t2", visible_cases=[1], holdout_cases=[], **kw)
    with pytest.raises(ValueError, match="both visible and held out"):
        init_loop(tmp_path / "w3", tmp_path / "t3", visible_cases=[1, 2], holdout_cases=[2], **kw)
    with pytest.raises(ValueError, match="must not contain each other"):
        init_loop(tmp_path / "w4", tmp_path / "w4" / "trusted", visible_cases=[1], holdout_cases=[], **kw)


# --- broker ----------------------------------------------------------------


def test_rehearse_request_round_trip(loop):
    workspace, trusted = loop
    sent = _tool(workspace, "rehearse", "--timeout", "0.2")
    assert sent.returncode == 0 and "is still running" in sent.stdout
    request = sent.stdout.split()[3]
    # Editing the working graph now does not change the submitted snapshot.
    (workspace / "results" / "graph" / "workflow.json").write_text("{}")

    Broker(trusted).serve(once=True)
    done = _tool(workspace, "wait", request, "--timeout", "5")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "accepted" in done.stdout and "round: 0" in done.stdout
    assert "task success on the visible cases: 2 of 3" in done.stdout
    assert "remaining_rehearsals: 1" in done.stdout

    round_dir = workspace / "eval" / "rounds" / "round_00"
    assert json.loads((round_dir / "graph" / "workflow.json").read_text()) == WORKFLOW
    assert sorted(p.name for p in (round_dir / "cases").iterdir()) == ["case_0001", "case_0002", "case_0003"]
    assert not (round_dir / "perception_cache").exists()
    main = (round_dir / "feedback" / "main.md").read_text()
    assert main == "Graph `eval/rounds/round_00/graph`, results in eval/rounds/round_00\n"
    case = json.loads((round_dir / "cases" / "case_0001" / "case.json").read_text())
    assert case["trace_dir"] == "eval/rounds/round_00/cases/case_0001/trace"

    # Held-out results exist on the trusted side only.
    ledger = read_ledger(trusted)
    assert ledger[0]["visible"]["successes"] == 2 and ledger[0]["holdout"] == {
        "cases": 2, "successes": 1, "per_case": {"41": True, "42": False}}
    assert (trusted / "rounds" / "round_00" / "holdout" / "feedback.json").exists()
    for path in workspace.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".md", ".jsonl"):
            text = path.read_text()
            assert str(trusted) not in text and "case_0041" not in text and "holdout" not in text.lower(), path
    status = json.loads((workspace / "eval" / "status.json").read_text())
    assert status["rehearsals_used"] == 1 and status["rounds"] == [
        {"round": 0, "cases": 3, "successes": 2, "results": "eval/rounds/round_00"}]


def test_second_round_compares_with_the_first_and_the_budget_is_enforced(loop):
    workspace, trusted = loop
    broker = Broker(trusted)
    for expected_round in (0, 1):
        _tool(workspace, "rehearse", "--timeout", "0.1")
        broker.serve(once=True)
        fb = json.loads((trusted / "rounds" / f"round_{expected_round:02d}" / "visible" / "feedback.json").read_text())
        assert fb["previous"] is (expected_round == 1)
    sent = _tool(workspace, "rehearse", "--timeout", "0.1")
    broker.serve(once=True)
    over = _tool(workspace, "wait", sent.stdout.split()[3], "--timeout", "5")
    assert over.returncode == 1 and "REJECTED" in over.stdout and "budget of 2 is used up" in over.stdout
    assert len(read_ledger(trusted)) == 2


def test_rejected_graph_is_not_charged_and_library_scripts_pass(loop):
    workspace, trusted = loop
    graph = workspace / "results" / "graph"
    (graph / "scripts" / "new.py").write_text("import subprocess\n\ndef run(ctx):\n    return {}\n")
    sent = _tool(workspace, "validate", "--timeout", "0.1")
    Broker(trusted).serve(once=True)
    bad = _tool(workspace, "wait", sent.stdout.split()[3], "--timeout", "5")
    assert bad.returncode == 1 and "import of 'subprocess' is not allowed" in bad.stdout
    assert read_ledger(trusted) == []

    # A byte-identical copy of a library script is accepted even though it imports os.
    (graph / "scripts" / "new.py").write_text("import os\n\ndef run(ctx):\n    return {}\n")
    sent = _tool(workspace, "validate", "--timeout", "0.1")
    Broker(trusted).serve(once=True)
    ok = _tool(workspace, "wait", sent.stdout.split()[3], "--timeout", "5")
    assert ok.returncode == 0 and "the graph validates" in ok.stdout and "remaining_rehearsals: 2" in ok.stdout


def test_infrastructure_failure_is_reported_and_not_charged(loop, monkeypatch):
    workspace, trusted = loop
    monkeypatch.setattr(broker_module, "_GAP_MAIN",
                        "import sys\nif sys.argv[1] == 'run':\n    print('OK'); sys.exit(0)\nprint('sim crashed'); sys.exit(3)")
    sent = _tool(workspace, "rehearse", "--timeout", "0.1")
    Broker(trusted).serve(once=True)
    out = _tool(workspace, "wait", sent.stdout.split()[3], "--timeout", "5")
    assert out.returncode == 1 and "infrastructure error, not a fault of the graph" in out.stdout
    assert "sim crashed" in out.stdout and read_ledger(trusted) == []
    assert not (workspace / "eval" / "rounds" / "round_00").exists()
