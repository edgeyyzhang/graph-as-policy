"""Default-suite tests for ``examples/hello_graph`` — the zero-friction
front door must keep working CPU-only, end to end (build → validate →
render PNG)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"


def _skills_root() -> Path:
    candidate = REPO_ROOT.parent / "open-robot-skills"
    if not (candidate / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {candidate}")
    return candidate


@pytest.fixture(scope="module")
def hello_module():
    path = EXAMPLES / "hello_graph" / "hello.py"
    spec = importlib.util.spec_from_file_location("_example_hello", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules["_example_hello"] = module
    spec.loader.exec_module(module)
    return module


def test_hello_runs_end_to_end(hello_module, tmp_path):
    """One command: build + validate + render, exit 0, PNG exists."""
    skills_root = _skills_root()
    out = tmp_path / "out"

    assert hello_module.main(
        ["--out", str(out), "--skills", str(skills_root)]) == 0

    assert (out / "workflow.json").is_file()
    assert (out / "scripts" / "perceive_dino_vlm.py").is_file()
    png = out / "graph.png"
    assert png.is_file() and png.stat().st_size > 1024
    with Image.open(png) as im:
        im.load()
        assert im.width > 0


def test_hello_graph_structure(hello_module):
    d = hello_module.build_workflow().to_dict()

    assert d["version"] == 3
    assert set(d["subgraphs"]) == {"perceive_sg", "grasp_sg"}
    assert d["subgraphs"]["perceive_sg"]["skill"] == "perceiving-objects"
    assert d["subgraphs"]["grasp_sg"]["skill"] == "grasping-direct-ik"

    # Failure exits both route to abort; success path ends at done.
    cond = d["conditional_edges"]
    assert cond["perceive"]["mapping"] == {"found": "grasp", "not_found": "abort"}
    assert cond["grasp"]["mapping"] == {"grasped": "done", "failed": "abort"}

    # Cross-subgraph dataflow: grasp consumes the OBB perception produces.
    assert "target_obb" in d["subgraphs"]["perceive_sg"]["outputs"]
    assert d["subgraphs"]["grasp_sg"]["inputs"] == {
        "target_obb": "OrientedBoundingBox",
    }
