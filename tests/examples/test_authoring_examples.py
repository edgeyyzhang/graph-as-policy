"""Default-suite tests for the authoring examples.

- ``examples/build_a_graph``: the build function produces a workflow that
  passes ``validate_workflow`` (with the real open-robot-skills checkout in the
  loop when present), and the written artifact has the
  ``gap generate``-shaped layout.
- ``examples/generate_a_graph``: the sample code imports and parses — no
  live LLM calls in the default suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


def _import_example(rel: str, module_name: str):
    path = EXAMPLES / rel
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _skills_root() -> Path:
    candidate = REPO_ROOT.parent / "open-robot-skills"
    if not (candidate / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {candidate}")
    return candidate


@pytest.fixture(scope="module")
def build_graph_module():
    return _import_example("build_a_graph/build_graph.py", "_example_build_graph")


# ---------------------------------------------------------------------------
# build_a_graph
# ---------------------------------------------------------------------------


def test_build_workflow_passes_validate_workflow(build_graph_module, tmp_path):
    """The builder output (workflow + copied scripts) validates cleanly."""
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    skills_root = _skills_root()
    wf = build_graph_module.build_workflow()
    out = build_graph_module.write_graph(wf, tmp_path / "graph", skills_root)

    wf_def = load_workflow(out / "workflow.json")
    issues = validate_workflow(wf_def, skill_registry=load_skills(skills_root))
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, [str(i) for i in errors]


def test_build_workflow_mirrors_quickstart_structure(build_graph_module):
    wf = build_graph_module.build_workflow()
    d = wf.to_dict()

    assert d["version"] == 3
    assert set(d["subgraphs"]) == {
        "target_sg", "container_sg", "grasp_sg", "transport_sg",
    }
    assert d["subgraphs"]["target_sg"]["skill"] == "perceiving-objects"
    assert d["subgraphs"]["grasp_sg"]["skill"] == "grasping-direct-ik"
    assert d["subgraphs"]["transport_sg"]["skill"] == "transporting-objects"

    # Success path: target -> container -> grasp -> transport -> done;
    # every failure exit routes to the recovering abort end.
    cond = d["conditional_edges"]
    assert cond["target"]["mapping"] == {"found": "container", "not_found": "abort"}
    assert cond["container"]["mapping"] == {"found": "grasp", "not_found": "abort"}
    assert cond["grasp"]["mapping"] == {"grasped": "transport", "failed": "abort"}
    assert cond["transport"]["mapping"] == {"placed": "done", "blocked": "abort"}
    assert d["nodes"]["abort"]["recovery"][0]["tool"] == "robot.open_gripper"

    # Cross-subgraph dataflow binds by name: grasp consumes the OBB the
    # target perception produces.
    assert "target_obb" in d["subgraphs"]["target_sg"]["outputs"]
    assert d["subgraphs"]["grasp_sg"]["inputs"] == {
        "target_obb": "OrientedBoundingBox",
    }


def test_written_artifact_layout(build_graph_module, tmp_path):
    """The on-disk layout is the `gap generate` artifact shape."""
    skills_root = _skills_root()
    wf = build_graph_module.build_workflow()
    out = build_graph_module.write_graph(wf, tmp_path / "graph", skills_root)

    assert (out / "workflow.json").is_file()
    for rel in build_graph_module.SCRIPT_SOURCES:
        assert (out / rel).is_file(), rel
    sidecar = (out / "checkpoints" / "grasp_sg.py").read_text()
    assert "CHECKPOINTS" in sidecar
    assert build_graph_module.TARGET_BODY in sidecar

    # The serialized workflow round-trips through json.
    parsed = json.loads((out / "workflow.json").read_text())
    assert parsed["meta"]["name"] == "pick_into_basket"


# ---------------------------------------------------------------------------
# generate_a_graph (no live LLM calls — import/parse only)
# ---------------------------------------------------------------------------


def test_generate_example_imports_and_parses():
    module = _import_example("generate_a_graph/generate.py", "_example_generate")
    assert callable(module.main)
    assert module.DEFAULT_INSTRUCTION
    # --help exercises the argparse wiring without touching an LLM.
    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])
    assert exc_info.value.code == 0
