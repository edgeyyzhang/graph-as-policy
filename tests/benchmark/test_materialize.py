"""Workflow materialization templating (launcher + benchmark wrapper)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gap.agent.launcher import materialize_workflow
from gap.benchmark.workflow_materialize import (
    materialize_for_task,
    parse_target_container,
)


def _make_template(parent: Path, *, with_target: bool = True) -> Path:
    tdir = parent / "template"
    (tdir / "scripts").mkdir(parents=True)
    wf = {
        "version": 3,
        "meta": {"name": "tmpl", "description": "{{policy_id}}"},
        "nodes": {},
        "edges": [],
        "conditional_edges": {},
        "subgraphs": {},
    }
    if with_target:
        wf["meta"]["target"] = "{{target}}"
        wf["meta"]["target_full"] = "{{target_full}}"
        wf["meta"]["container"] = "{{container}}"
    (tdir / "workflow.json").write_text(json.dumps(wf))
    (tdir / "scripts" / "helper.py").write_text("X = 1\n")
    return tdir


# --------------------------------------------------------------------------
# launcher.materialize_workflow
# --------------------------------------------------------------------------


def test_materialize_substitutes_and_copies_scripts(tmp_path) -> None:
    tdir = _make_template(tmp_path)
    out = materialize_workflow(
        str(tdir),
        {"target": "milk", "target_full": "the milk carton",
         "container": "basket", "policy_id": "libero_pi05"},
        tmp_path / "dest",
    )
    out_dir = Path(out)
    assert out_dir == (tmp_path / "dest" / "workflow").resolve()
    wf = json.loads((out_dir / "workflow.json").read_text())
    assert wf["meta"]["target"] == "milk"
    assert wf["meta"]["target_full"] == "the milk carton"
    assert wf["meta"]["description"] == "libero_pi05"
    # scripts tree copied verbatim
    assert (out_dir / "scripts" / "helper.py").read_text() == "X = 1\n"


def test_materialize_invalid_json_fails_fast(tmp_path) -> None:
    tdir = tmp_path / "bad"
    tdir.mkdir()
    (tdir / "workflow.json").write_text('{"meta": "{{target}}"')  # truncated
    with pytest.raises(ValueError, match="invalid JSON"):
        materialize_workflow(str(tdir), {"target": "x"}, tmp_path / "dest")


def test_materialize_missing_workflow_json(tmp_path) -> None:
    tdir = tmp_path / "empty"
    tdir.mkdir()
    with pytest.raises(FileNotFoundError):
        materialize_workflow(str(tdir), {}, tmp_path / "dest")


def test_materialize_overwrites_previous_copy(tmp_path) -> None:
    tdir = _make_template(tmp_path)
    dest = tmp_path / "dest"
    materialize_workflow(str(tdir), {"target": "a", "target_full": "a",
                                     "container": "b", "policy_id": "p"}, dest)
    out = materialize_workflow(
        str(tdir), {"target": "ketchup", "target_full": "ketchup",
                    "container": "basket", "policy_id": "p"}, dest,
    )
    wf = json.loads((Path(out) / "workflow.json").read_text())
    assert wf["meta"]["target"] == "ketchup"


# --------------------------------------------------------------------------
# prompt grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "target", "container"),
    [
        ("pick up the milk and place it in the basket",
         "milk", "basket"),
        ("Pick the blue and yellow alphabet soup can and place it in the basket.",
         "blue and yellow alphabet soup can", "basket"),
        ("pick up the ketchup and place it in the tray.",
         "ketchup", "tray"),
    ],
)
def test_parse_target_container(prompt, target, container) -> None:
    assert parse_target_container(prompt) == (target, container)


def test_parse_target_container_no_match() -> None:
    assert parse_target_container("Pack every item into the basket") == ("", "")


# --------------------------------------------------------------------------
# materialize_for_task
# --------------------------------------------------------------------------


def test_materialize_for_task_with_explicit_prompt(tmp_path) -> None:
    tdir = _make_template(tmp_path)
    out = materialize_for_task(
        template_dir=tdir,
        dest_parent=tmp_path / "cell" / "task_00",
        suite_name="libero_object_all_variance",
        task_id=0,
        task_prompt="pick up the milk and place it in the basket",
        policy_id="molmoact",
    )
    wf = json.loads((Path(out) / "workflow.json").read_text())
    assert wf["meta"]["target"] == "milk"
    assert wf["meta"]["container"] == "basket"
    assert wf["meta"]["description"] == "molmoact"


def test_materialize_for_task_unparseable_prompt_with_target_template(
    tmp_path,
) -> None:
    tdir = _make_template(tmp_path, with_target=True)
    with pytest.raises(ValueError, match="could not parse a target"):
        materialize_for_task(
            template_dir=tdir,
            dest_parent=tmp_path / "t",
            suite_name="s",
            task_id=0,
            task_prompt="Pack every item from the floor into the basket",
        )


def test_materialize_for_task_targetless_template_tolerates_any_prompt(
    tmp_path,
) -> None:
    """A {{policy_id}}-only template (grocery cyclic graph) is fine with
    an unparseable prompt."""
    tdir = _make_template(tmp_path, with_target=False)
    out = materialize_for_task(
        template_dir=tdir,
        dest_parent=tmp_path / "t",
        suite_name="s",
        task_id=0,
        task_prompt="Pack every item from the floor into the basket",
    )
    wf = json.loads((Path(out) / "workflow.json").read_text())
    assert wf["meta"]["description"] == "pi05-libero"  # default policy
