"""Static-render smoke tests: PDF/PNG outputs + go_home hiding."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.text
import pytest
from PIL import Image

import gap.viz.render as render_mod
from gap.viz.render import render

from .conftest import go_home_graph, streaming_graph, two_stage_graph


@pytest.mark.parametrize(
    "graph_fn", [two_stage_graph, streaming_graph, go_home_graph],
    ids=["two_stage", "streaming", "go_home"],
)
def test_render_smoke(tmp_path: Path, graph_fn) -> None:
    out = render(graph_fn(), tmp_path / "fig.pdf")

    assert out == tmp_path / "fig.pdf"
    pdf_bytes = out.read_bytes()
    assert len(pdf_bytes) > 1024
    assert pdf_bytes.startswith(b"%PDF")

    # A sibling PNG is written by default and must decode.
    png = tmp_path / "fig.png"
    assert png.exists()
    with Image.open(png) as im:
        im.load()
        assert im.width > 0 and im.height > 0


def test_render_accepts_path_and_dir(tmp_path: Path) -> None:
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(two_stage_graph()))

    out1 = render(wf_dir, tmp_path / "from_dir.png", also_png=False, legend=False)
    out2 = render(wf_dir / "workflow.json", tmp_path / "from_file.png", also_png=False)
    for p in (out1, out2):
        with Image.open(p) as im:
            im.load()


def test_go_home_nodes_hidden(tmp_path: Path, monkeypatch) -> None:
    """go_home reset states are dropped from the figure entirely."""
    captured: list = []
    real_close = render_mod.plt.close
    monkeypatch.setattr(render_mod.plt, "close", captured.append)
    try:
        render(go_home_graph(), tmp_path / "fig.pdf", also_png=False)
        assert len(captured) == 1
        fig = captured[0]
        texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not any("go_home" in t for t in texts)
        # The surviving nodes still render.
        assert any("act" in t for t in texts)
        assert any("done" in t for t in texts)
    finally:
        for fig in list(captured):
            real_close(fig)


def test_hidden_node_rewires_control_flow() -> None:
    """Bypassing go_home keeps preds connected to succs."""
    stripped = render_mod._strip_hidden(go_home_graph())
    sg = stripped["subgraphs"]["work_sg"]
    assert "go_home_reset" not in sg["nodes"]
    assert ["act", "ok"] in sg["edges"]
    # Top-level: conditional mapping retargets through the hidden node.
    assert "go_home" not in stripped["nodes"]
    mapping = stripped["conditional_edges"]["work"]["mapping"]
    assert mapping["ok"] == "done"


def test_data_edges_overlay(tmp_path: Path) -> None:
    out = render(two_stage_graph(), tmp_path / "fig.pdf", data_edges=True, legend=True)
    assert out.exists() and out.stat().st_size > 1024
