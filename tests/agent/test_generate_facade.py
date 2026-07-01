"""End-to-end test for the gap.agent.generate() facade.

generate_sync runs against the REAL open-robot-skills checkout with the
stubbed LLM (canned coordinator/subgraph/checkpoint responses) and the
resulting GeneratedGraph is checked for path contents + code map.
"""

from __future__ import annotations

import json

import pytest

import gap.agent as agent

from .conftest import PIPELINE_RESPONSES


@pytest.fixture
def generated(skills_root, stub_llm, tmp_path):
    stub = stub_llm(list(PIPELINE_RESPONSES))
    graph = agent.generate_sync(
        "Pick up the alphabet soup",
        skills=skills_root,
        out_dir=tmp_path / "out",
    )
    return graph, stub, tmp_path


class TestGenerateFacade:
    def test_generated_graph_path(self, generated):
        graph, _, tmp_path = generated
        assert graph.path == tmp_path / "out" / "task_00"
        assert (graph.path / "workflow.json").is_file()
        assert (graph.path / "scripts/perceive_target/make_obb.py").is_file()
        assert (graph.path / "scripts/perceive_target/perceive_dino_vlm.py").is_file()
        assert (graph.path / "scripts/grasp_target/compute_align_pose.py").is_file()
        assert (graph.path / "checkpoints/perceive_target.py").is_file()
        assert (graph.path / "checkpoints/grasp_target.py").is_file()
        meta = json.loads((graph.path / "multi_agent_meta.json").read_text())
        assert meta["pipeline"] == "gap_multi_agent"

    def test_workflow_dict_matches_file(self, generated):
        graph, _, _ = generated
        on_disk = json.loads((graph.path / "workflow.json").read_text())
        assert graph.workflow == on_disk
        assert set(graph.workflow["subgraphs"]) == {"perceive_target", "grasp_target"}
        assert graph.workflow["meta"]["description"] == "Pick up the alphabet soup"

    def test_code_map(self, generated):
        graph, _, _ = generated
        assert "scripts/perceive_target/make_obb.py" in graph.code
        assert "checkpoints/perceive_target.py" in graph.code
        assert "checkpoints/grasp_target.py" in graph.code
        # The code map mirrors what's on disk.
        for rel, content in graph.code.items():
            assert (graph.path / rel).read_text() == content

    def test_generated_workflow_loads(self, generated):
        graph, _, _ = generated
        from gap.runtime.workflow import load_workflow

        wf = load_workflow(graph.path / "workflow.json")
        assert set(wf.subgraphs) == {"perceive_target", "grasp_target"}

    def test_model_and_provider_overrides_reach_llm(
        self, skills_root, stub_llm, tmp_path,
    ):
        stub = stub_llm(list(PIPELINE_RESPONSES))
        agent.generate_sync(
            "Pick up the alphabet soup",
            skills=skills_root,
            model="gemini-2.5-flash",
            provider="vertex",
            out_dir=tmp_path / "out",
        )
        cfgs = [c["cfg"] for c in stub.calls]
        assert all(c.provider == "vertex" for c in cfgs)
        assert all(c.model == "gemini-2.5-flash" for c in cfgs)

    def test_failure_raises(self, skills_root, stub_llm, tmp_path):
        # Three garbage coordinator responses exhaust the retry budget.
        stub_llm(["no code here"] * 3)
        with pytest.raises(RuntimeError, match="graph generation failed"):
            agent.generate_sync(
                "Pick up the alphabet soup",
                skills=skills_root,
                out_dir=tmp_path / "out",
            )

    def test_str_renders_workflow(self, generated):
        graph, _, _ = generated
        from gap.viz.text import to_text

        rendered = str(graph)
        assert rendered == to_text(graph.workflow)
        assert "┌─" in rendered
        assert "\x1b" not in rendered  # plain text, never ANSI


def test_cli_generate_prints_graph(monkeypatch, tmp_path, capsys):
    """`gap generate` prints the box-drawing rendering after the OK line."""
    import argparse

    from gap.cli import generate as cli_generate

    wf = {
        "version": 3,
        "meta": {"name": "stub"},
        "nodes": {
            "act": {"type": "tool", "tool": "robot.move_to_pose"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "act"], ["act", "done"]],
        "conditional_edges": {},
    }
    stub = agent.GeneratedGraph(path=tmp_path / "task_00", workflow=wf, code={})
    monkeypatch.setattr(agent, "generate_sync", lambda *a, **k: stub)

    args = argparse.Namespace(
        instruction="pick it up", skills=tmp_path, provider=None, model=None,
        out=None, config=None, verbose=False,
    )
    assert cli_generate._handle(args) == 0
    out = capsys.readouterr().out
    assert "OK: wrote" in out
    assert "┌─ act " in out
    assert "\x1b" not in out  # capsys is not a TTY → no color
