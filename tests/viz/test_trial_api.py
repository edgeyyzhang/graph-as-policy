"""Trial loading + FastAPI surface tests over a REAL executed trace.

A tiny stub-tool workflow runs through ``gap.execute`` with
``trace_dir=tmp`` (same pattern as tests/runtime/test_executor.py); the
viz layer must then discover the trial and serve its data with the exact
filenames/keys the landed tracer writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import gap
from gap.tools import ToolRegistry
from gap.viz import api as viz_api
from gap.viz.server import create_app
from gap.viz.trial_loader import (
    discover_trials,
    list_node_assets,
    list_node_subcalls,
    load_trial,
    load_viz_trial,
)

# ---------------------------------------------------------------------------
# Fixture: execute a tiny workflow and trace it
# ---------------------------------------------------------------------------


def _trial_workflow() -> dict:
    return {
        "version": 3,
        "meta": {"name": "viz_trial"},
        "nodes": {
            "run_sg": {"type": "subgraph", "ref": "work_sg"},
            "done": {"type": "end", "status": "success"},
            "failed": {"type": "end", "status": "failure"},
        },
        "edges": [["START", "run_sg"]],
        "conditional_edges": {
            "run_sg": {
                "router_field": "exit",
                "mapping": {"ok": "done", "boom": "failed"},
            },
        },
        "subgraphs": {
            "work_sg": {
                "skill": "stub_skill",
                "inputs": {},
                "outputs": {"value": {"$ref": "compute.value"}},
                "nodes": {
                    "capture": {"type": "tool", "tool": "cam.capture", "inputs": {}},
                    "compute": {
                        "type": "tool", "tool": "stub.compute",
                        "inputs": {"x": {"$ref": "capture.value"}},
                    },
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "capture"], ["capture", "compute"],
                    ["compute", "ok"], ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
                "on_error": "boom",
            },
        },
    }


def _capture() -> dict:
    return {
        "value": 3,
        "rgb": np.full((8, 8, 3), 7, dtype=np.uint8),
    }


def _registry(compute=None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_callable("cam.capture", _capture, summary="capture")
    reg.register_callable(
        "stub.compute", compute or (lambda x: {"value": x * 2}), summary="compute",
    )
    return reg


def _run_trial(root: Path, name: str, *, compute=None) -> Path:
    """Execute the stub workflow; trace lands in root/<name>."""
    wf_dir = root / f"wf_{name}"
    wf_dir.mkdir(parents=True)
    (wf_dir / "workflow.json").write_text(json.dumps(_trial_workflow()))
    trace_dir = root / name
    conn = SimpleNamespace(tool_registry=_registry(compute))
    result = gap.execute(wf_dir, conn, trace_dir=trace_dir)
    assert result.trace_path == trace_dir
    return trace_dir


@pytest.fixture()
def trial_root(tmp_path: Path) -> Path:
    """One executed trial under <root>/trial_a."""
    root = tmp_path / "outputs"
    trace = _run_trial(root, "trial_a")
    assert (trace / "dag_trace.json").exists()
    assert (trace / "workflow.json").exists()  # copied by the tracer
    return root


# ---------------------------------------------------------------------------
# trial_loader against the real on-disk layout
# ---------------------------------------------------------------------------


def test_discover_trials_finds_trace_dir(trial_root: Path) -> None:
    trials = discover_trials(trial_root)
    assert "trial_a" in trials
    # The source workflow dir (workflow.json only) is also discoverable.
    assert "wf_trial_a" in trials


def test_load_trial_nodes_and_workflow(trial_root: Path) -> None:
    trial = load_trial(trial_root / "trial_a")

    nodes = {n.node_id: n for n in trial.nodes}
    # Trace node names are fully-qualified inside subgraphs.
    assert {"work_sg.capture", "work_sg.compute", "work_sg.ok"} <= set(nodes)
    assert nodes["work_sg.capture"].status == "ok"
    assert nodes["work_sg.compute"].status == "ok"
    assert nodes["work_sg.compute"].has_inputs
    assert nodes["work_sg.compute"].has_output
    # dag_trace.json keeps service/method as null-able fields.
    assert nodes["work_sg.compute"].service is None
    assert nodes["work_sg.compute"].method is None
    # Extracted image asset recorded on the capture node.
    assert any("rgb" in a for a in nodes["work_sg.capture"].assets)
    assert trial.total_duration_ms > 0

    state_ids = {s.state_id for s in trial.workflow.states}
    assert {"work_sg.capture", "work_sg.compute", "work_sg.ok"} <= state_ids


def test_load_viz_trial_builds_execution_view(trial_root: Path) -> None:
    viz = load_viz_trial(trial_root / "trial_a", trial_path="trial_a")
    assert viz.meta.trial_path == "trial_a"
    assert not viz.execution.degraded  # events came from the real tracer
    step_nodes = {s.node_id for s in viz.execution.steps}
    assert {"work_sg.capture", "work_sg.compute"} <= step_nodes
    lane_ids = {lane.id for lane in viz.execution.lanes}
    assert "work_sg" in lane_ids


def test_node_assets_listing(trial_root: Path) -> None:
    assets = list_node_assets(trial_root / "trial_a", "work_sg.capture")
    assert assets, "capture output should have extracted an rgb asset"
    assert all(a.endswith(".png") for a in assets)


def test_subcalls_empty_for_plain_tool_nodes(trial_root: Path) -> None:
    assert list_node_subcalls(trial_root / "trial_a", "work_sg.compute") == []


# ---------------------------------------------------------------------------
# FastAPI surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(trial_root: Path) -> TestClient:
    app = create_app(root_dir=trial_root)
    return TestClient(app)


def test_api_trials_list(client: TestClient) -> None:
    res = client.get("/api/trials")
    assert res.status_code == 200
    assert "trial_a" in res.json()


def test_api_trial_detail(client: TestClient) -> None:
    res = client.get("/api/trial", params={"trial": "trial_a"})
    assert res.status_code == 200
    body = res.json()
    assert body["total_duration_ms"] > 0
    node_ids = {n["node_id"]: n for n in body["nodes"]}
    assert node_ids["work_sg.compute"]["status"] == "ok"
    state_ids = {s["state_id"] for s in body["workflow"]["states"]}
    assert "work_sg.compute" in state_ids


def test_api_viz_trial(client: TestClient) -> None:
    res = client.get("/api/viz/trial", params={"trial": "trial_a"})
    assert res.status_code == 200
    body = res.json()
    assert body["meta"]["trial_path"] == "trial_a"
    assert body["execution"]["steps"], "executed nodes must become steps"
    assert body["provenance"]["edges"], "data edges must surface as provenance"


def test_api_workflow(client: TestClient) -> None:
    res = client.get("/api/workflow", params={"trial": "trial_a"})
    assert res.status_code == 200
    assert res.json()["begin"] == "work_sg"


def test_api_node_data(client: TestClient) -> None:
    res = client.get("/api/node/work_sg.compute/inputs", params={"trial": "trial_a"})
    assert res.status_code == 200
    assert res.json() == {"x": 3}

    res = client.get("/api/node/work_sg.compute/output", params={"trial": "trial_a"})
    assert res.status_code == 200
    assert res.json() == {"value": 6}

    # Missing data → 404, not 500.
    res = client.get("/api/node/work_sg.ok/output", params={"trial": "trial_a"})
    assert res.status_code == 404


def test_api_asset_serving(client: TestClient, trial_root: Path) -> None:
    res = client.get("/api/node/work_sg.capture/assets", params={"trial": "trial_a"})
    assert res.status_code == 200
    assets = res.json()
    assert assets

    res = client.get(
        f"/api/node/work_sg.capture/asset/{assets[0]}",
        params={"trial": "trial_a"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_multiple_trials_requires_param(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _run_trial(root, "trial_a")
    _run_trial(root, "trial_b")
    client = TestClient(create_app(root_dir=root))
    res = client.get("/api/trial")
    assert res.status_code == 400  # ambiguous without ?trial=

    res = client.get("/api/trial", params={"trial": "trial_b"})
    assert res.status_code == 200


def test_api_unknown_trial_404(client: TestClient) -> None:
    res = client.get("/api/trial", params={"trial": "nope"})
    assert res.status_code == 404


def test_configure_is_idempotent(trial_root: Path) -> None:
    viz_api.configure(root_dir=trial_root)
    first = list(viz_api._trial_paths)
    viz_api.configure(root_dir=trial_root)
    assert viz_api._trial_paths == first
