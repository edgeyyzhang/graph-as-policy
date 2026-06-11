"""Tests for the ``Subgraph.add_checkpoint`` API + sidecar roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from gap.builder import END, START, Subgraph


def _minimal_subgraph(name: str = "approach") -> Subgraph:
    sg = Subgraph(name=name, skill="motion")
    sg.add_node("step", type="tool", tool="robot.open_gripper")
    sg.add_exit("at_pose")
    sg.add_edge(START, "step")
    sg.add_edge("step", "at_pose")
    sg.add_edge("at_pose", END)
    sg.set_on_error("blocked")
    return sg


def test_add_checkpoint_appends():
    sg = _minimal_subgraph()
    sg.add_checkpoint(
        "ee_above_target",
        predicate=lambda w: True,
        rationale="placeholder",
        validate=True,
    )
    assert len(sg._checkpoints) == 1
    assert sg._checkpoints[0].name == "ee_above_target"


def test_add_checkpoint_validate_defaults_true():
    sg = _minimal_subgraph()
    sg.add_checkpoint("c1", predicate=lambda w: True)
    assert sg._checkpoints[0].validate is True


def test_add_checkpoint_validate_false_recorded():
    sg = _minimal_subgraph()
    sg.add_checkpoint("probe", predicate=lambda w: True, validate=False)
    assert sg._checkpoints[0].validate is False


def test_add_checkpoint_rejects_duplicate_name():
    sg = _minimal_subgraph()
    sg.add_checkpoint("c1", predicate=lambda w: True)
    with pytest.raises(Exception, match="already declared"):
        sg.add_checkpoint("c1", predicate=lambda w: False)


def test_add_checkpoint_rejects_non_callable():
    sg = _minimal_subgraph()
    with pytest.raises(Exception, match="must be callable"):
        sg.add_checkpoint("c1", predicate="not a function")


def test_to_dict_excludes_checkpoints():
    """Checkpoints must not be serialized into workflow.json."""
    sg = _minimal_subgraph()
    sg.add_checkpoint("c1", predicate=lambda w: True, rationale="r")
    d = sg.to_dict()
    # Walk every value — no "checkpoints" key at any nesting level.
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert "checkpoint" not in str(k).lower(), (
                    f"to_dict() leaked checkpoint key: {k}"
                )
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
    _walk(d)


def test_dump_checkpoints_module_writes_sidecar(tmp_path: Path):
    sg = _minimal_subgraph()
    sg.add_checkpoint(
        "ee_above_target",
        predicate=lambda w: True,
        rationale="placeholder",
        validate=True,
    )
    # The builder needs a source block to render the sidecar.
    source = (
        "from gap.builder import Subgraph, Ref, START, END\n"
        "sg = Subgraph(name='approach', skill='motion')\n"
        "sg.add_node('step', type='tool', tool='robot.open_gripper')\n"
        "sg.add_exit('at_pose')\n"
        "sg.add_edge(START, 'step')\n"
        "sg.add_edge('step', 'at_pose')\n"
        "sg.add_edge('at_pose', END)\n"
        "sg.set_on_error('blocked')\n"
        "sg.add_checkpoint('ee_above_target', predicate=lambda w: True,\n"
        "                  rationale='placeholder', validate=True)\n"
    )
    sg._set_source_block(source)
    out_path = tmp_path / "checkpoints" / "approach.py"
    written = sg.dump_checkpoints_module(out_path)
    assert written is not None
    assert written.exists()
    text = written.read_text()
    assert "CHECKPOINTS" in text
    assert "ee_above_target" in text
    assert "validate=_c.validate" in text
    assert "from gap.builder import" in text
    assert "from gap.runtime.verify import Checkpoint" in text


def test_no_checkpoints_returns_none(tmp_path: Path):
    sg = _minimal_subgraph()
    sg._set_source_block("# placeholder\n")
    # No add_checkpoint calls — dump returns None and writes nothing.
    out_path = tmp_path / "checkpoints" / "approach.py"
    written = sg.dump_checkpoints_module(out_path)
    assert written is None
    assert not out_path.exists()


def test_sidecar_re_execs_to_yield_checkpoints(tmp_path: Path):
    """The harness's contract: import the sidecar → get CHECKPOINTS."""
    verify = pytest.importorskip("gap.runtime.verify")
    sg = _minimal_subgraph()
    sg.add_checkpoint(
        "c1", predicate=lambda w: True,
        rationale="passes always", validate=True,
    )
    source = (
        "from gap.builder import Subgraph, START, END\n"
        "sg = Subgraph(name='approach', skill='motion')\n"
        "sg.add_node('step', type='tool', tool='robot.open_gripper')\n"
        "sg.add_exit('at_pose')\n"
        "sg.add_edge(START, 'step')\n"
        "sg.add_edge('step', 'at_pose')\n"
        "sg.add_edge('at_pose', END)\n"
        "sg.set_on_error('blocked')\n"
        "sg.add_checkpoint('c1', predicate=lambda w: True,\n"
        "                  rationale='passes always', validate=True)\n"
    )
    sg._set_source_block(source)
    out_path = tmp_path / "checkpoints" / "approach.py"
    sg.dump_checkpoints_module(out_path)

    cps = verify.load_checkpoints(out_path)
    assert len(cps) == 1
    assert cps[0].name == "c1"
    assert cps[0].subgraph == "approach"
    assert cps[0].rationale == "passes always"
    assert cps[0].validate is True


def test_validate_false_round_trips_through_sidecar(tmp_path: Path):
    verify = pytest.importorskip("gap.runtime.verify")
    sg = _minimal_subgraph()
    sg.add_checkpoint(
        "probe", predicate=lambda w: True,
        rationale="diagnostic only", validate=False,
    )
    source = (
        "from gap.builder import Subgraph, START, END\n"
        "sg = Subgraph(name='approach', skill='motion')\n"
        "sg.add_node('step', type='tool', tool='robot.open_gripper')\n"
        "sg.add_exit('at_pose')\n"
        "sg.add_edge(START, 'step')\n"
        "sg.add_edge('step', 'at_pose')\n"
        "sg.add_edge('at_pose', END)\n"
        "sg.set_on_error('blocked')\n"
        "sg.add_checkpoint('probe', predicate=lambda w: True,\n"
        "                  rationale='diagnostic only', validate=False)\n"
    )
    sg._set_source_block(source)
    out_path = tmp_path / "checkpoints" / "approach.py"
    sg.dump_checkpoints_module(out_path)

    cps = verify.load_checkpoints(out_path)
    assert len(cps) == 1 and cps[0].validate is False
