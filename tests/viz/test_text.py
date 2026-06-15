"""to_text terminal-rendering tests: golden output, layout invariants, inputs."""

from __future__ import annotations

import copy
import re
from pathlib import Path

from gap.viz.text import to_text

from .conftest import streaming_graph, two_stage_graph, write_workflow

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

GOLDEN_GRASP = """\
graph_obb_policy_grasp
Long-horizon clean-all-items loop -- VLA-gr…

START
  │
  ▼
┌─ capture ────────────────────── generic ─┐
│ observe ⚙ robot.get_observation          │
└──────────────────────────────────────────┘
  │ done                    failed ▶ ✗ abort
  ▼
┌─ container ─ perceiving-objects-oneshot ─┐
│ observe ─▶ perceive ─▶ filter_obb        │
└──────────────────────────────────────────┘
  │ found                not_found ▶ ✗ abort
  ▼
┌─ reset ──────────────────────── generic ─┐
│ open_gripper ─▶ move_to_initial          │
└──────────────────────────────────────────┘
  │ done                    failed ▶ ✗ abort
  ▼
┌─ target ──── perceiving-objects-oneshot ─┐
│ observe ─▶ perceive ─▶ filter_obb        │
└──────────────────────────────────────────┘
  │ found                 not_found ▶ ✓ done
  ▼
┌─ approach ───────────────────── generic ─┐
│ approach_above ƒ approach_above_target   │
└──────────────────────────────────────────┘
  │ ready                   failed ▶ ✗ abort
  ▼
┌─ grasp ──────────────────────── generic ─┐
│ run_policy ⚙ {{policy_id}}.run           │
└──────────────────────────────────────────┘
  │ grasped                 failed ▶ ✗ abort
  ▼
┌─ place ──────────────────────── generic ─┐
│ place ƒ place_above_basket               │
└──────────────────────────────────────────┘
  │ done ↺ reset            failed ▶ ✗ abort

✓ done (success)
✗ abort (failure, recovery: open_gripper)"""


def _body_lines(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if line.startswith("│")]


class TestGolden:
    def test_steered_policy_grasp(self):
        assert to_text(EXAMPLES / "steered_policy/graph_grasp") == GOLDEN_GRASP

    def test_spine_order_and_loop(self):
        out = to_text(EXAMPLES / "steered_policy/graph_loop")
        headers = [out.index(f"┌─ {n} ")
                   for n in ("capture", "target", "approach", "run", "reset")]
        assert headers == sorted(headers)
        assert "↺ target" in out
        assert "(from" not in out  # single spine, no orphan segments


class TestStructure:
    def test_two_stage(self):
        out = to_text(two_stage_graph())
        assert "detect ─▶ check" in out
        assert "plan ─▶ move" in out
        assert "not_found ▶ ✗ abort" in out
        assert "grasped ▶ ✓ done" in out
        assert "✗ abort (failure, recovery: open_gripper)" in out
        # noop markers are exit labels, never body nodes; the only internal
        # conditional mapping targets a noop, so no branch line either.
        assert "↳" not in out
        for line in _body_lines(out):
            assert "found" not in line and "grasped" not in line

    def test_internal_branch_line(self):
        wf = copy.deepcopy(two_stage_graph())
        mapping = wf["subgraphs"]["perceive_sg"]["conditional_edges"]["check"]
        mapping["mapping"]["retry"] = "detect"  # visible → visible branch
        out = to_text(wf)
        assert "↳ check: retry ▶ detect" in out

    def test_streaming_parallel_separator(self):
        out = to_text(streaming_graph())
        assert "track · servo" in out  # parallel sources, no false arrow

    def test_flat_graph(self):
        wf = {
            "version": 3,
            "meta": {"name": "flat"},
            "nodes": {
                "observe": {"type": "tool", "tool": "robot.get_observation"},
                "decide": {"type": "script", "script": "scripts/decide.py"},
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "observe"], ["observe", "decide"],
                      ["decide", "done"]],
            "conditional_edges": {},
        }
        out = to_text(wf)
        assert out.count("┌─") == 2  # one card per non-end node
        assert "observe ⚙ get_observation" in out
        assert "decide ƒ decide" in out
        assert "▶ ✓ done" in out

    def test_orphan_segment(self):
        sg = {
            "skill": "s", "inputs": {}, "outputs": {},
            "nodes": {"act": {"type": "tool", "tool": "robot.move_to_pose"},
                      "ok": {"type": "noop"}},
            "edges": [["START", "act"], ["act", "ok"], ["ok", "END"]],
            "conditional_edges": {},
            "exit": {"router_field": None, "success_values": ["ok"]},
            "on_error": "failed",
        }
        wf = {
            "version": 3,
            "meta": {"name": "orphan_demo"},
            "nodes": {
                "work": {"type": "subgraph", "ref": "work_sg"},
                "recover": {"type": "subgraph", "ref": "recover_sg"},
                "done": {"type": "end", "status": "success"},
                "abort": {"type": "end", "status": "failure"},
            },
            "edges": [["START", "work"]],
            "conditional_edges": {
                "work": {"router_field": "exit",
                         "mapping": {"ok": "done", "failed": "recover"}},
                "recover": {"router_field": "exit",
                            "mapping": {"ok": "done", "failed": "abort"}},
            },
            "subgraphs": {"work_sg": sg, "recover_sg": copy.deepcopy(sg)},
        }
        out = to_text(wf)
        intro = out.index("(from work ▶ failed)")
        assert intro < out.index("┌─ recover ")
        assert out.index("┌─ work ") < intro


class TestLayout:
    def test_plain_by_default(self):
        assert "\x1b" not in to_text(two_stage_graph())

    def test_color_layout_invariant(self):
        wf = two_stage_graph()
        colored = to_text(wf, color=True)
        assert "\x1b[" in colored
        assert ANSI_RE.sub("", colored) == to_text(wf)

    def test_width_cap(self):
        sample = EXAMPLES / "grocery_fulfillment/sample_generated_graph"
        for line in to_text(sample).splitlines():
            assert len(line) <= 76
        for line in to_text(sample, width=60).splitlines():
            assert len(line) <= 60
        # Degenerate widths clamp to the floor instead of raising.
        for line in to_text(sample, width=10).splitlines():
            assert len(line) <= 24

    def test_input_forms(self, tmp_path):
        wf = two_stage_graph()
        write_workflow(tmp_path, wf)
        expected = to_text(wf)
        assert to_text(tmp_path) == expected
        assert to_text(tmp_path / "workflow.json") == expected
        assert to_text(str(tmp_path / "workflow.json")) == expected

    def test_lazy_export(self):
        import gap.viz

        assert gap.viz.to_text is to_text
