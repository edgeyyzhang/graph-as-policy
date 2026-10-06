#!/usr/bin/env python3
"""Two-panel video of one rehearsed case: the simulator on the left, the graph
with the active node highlighted on the right.

    compose_video.py CASE_DIR GRAPH_DIR OUT.mp4 [--height 720] [--speed 1.0] [--title TEXT]

CASE_DIR is a ``cases/case_NNNN`` directory of a rehearsal run with ``--video``:
it holds ``video.mp4`` (one frame per N simulator steps), ``video.mp4.frames.json``
(the trajectory sample index of every frame) and ``trajectory.jsonl`` (which node
was active at every sample). GRAPH_DIR holds the ``workflow.json`` that was run.

Run with the rehearsal Python: ``examples/rehearse_loop/gap.sh`` sets nothing this
needs, so the checkout's ``.venv/bin/python`` (``uv sync``) with PYTHONPATH on the checkout
and gap-core is enough (see ``show_graph.sh``).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def node_of_sample(sample: dict) -> str | None:
    """The renderer's highlight name for a trajectory sample: ``lane.node`` or ``node``."""
    node, unit = sample.get("node"), sample.get("unit")
    if not node:
        return None
    if unit and node.startswith(unit + "."):
        return f"{unit}.{node[len(unit) + 1:]}"
    return node


def render_graph_images(graph: Path, names: set[str | None], workdir: Path, height: int) -> dict[str | None, Image.Image]:
    """One graph image per distinct active node (and one with nothing active)."""
    from gap.viz import render

    images: dict[str | None, Image.Image] = {}
    for name in names:
        stem = "none" if name is None else name.replace("/", "_")
        out = workdir / f"graph_{stem}.png"
        render(graph / "workflow.json", out, highlight=name, legend=False, also_png=False)
        img = Image.open(out).convert("RGB")
        scale = height / img.height
        images[name] = img.resize((max(2, int(round(img.width * scale))), height), Image.LANCZOS)
    return images


def compose(case_dir: Path, graph: Path, out: Path, *, height: int = 720, speed: float = 1.0,
            title: str | None = None, quality: int = 7) -> Path:
    meta = json.loads((case_dir / "video.mp4.frames.json").read_text())
    samples = {}
    with (case_dir / "trajectory.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                samples[row["i"]] = row
    frame_nodes = [node_of_sample(samples.get(i, {})) for i in meta["samples"]]
    frame_times = [(samples.get(i, {}).get("state") or {}).get("time_s") for i in meta["samples"]]

    font, small = _font(max(12, height // 36)), _font(max(10, height // 48))
    bar = int(height * 0.075)
    with tempfile.TemporaryDirectory() as tmp:
        graphs = render_graph_images(graph, set(frame_nodes), Path(tmp), height)
        writer = None
        try:
            for k, sim in enumerate(iio.imiter(case_dir / "video.mp4", plugin="pyav")):
                left = Image.fromarray(np.asarray(sim)).convert("RGB")
                left = left.resize((int(round(left.width * height / left.height)), height), Image.LANCZOS)
                right = graphs[frame_nodes[k]]
                width = left.width + right.width + 3 * 12
                canvas = Image.new("RGB", (width + width % 2, height + bar + (height + bar) % 2), "white")
                canvas.paste(left, (12, 0))
                canvas.paste(right, (left.width + 24, 0))
                draw = ImageDraw.Draw(canvas)
                node = frame_nodes[k] or "between nodes"
                t = frame_times[k]
                text = f"{title + '   ' if title else ''}node: {node}"
                draw.text((14, height + 4), text, fill="#1c1c22", font=font)
                stamp = f"t = {t:.1f} s   step {meta['samples'][k]}" if t is not None else f"frame {k}"
                draw.text((14, height + 4 + font.size + 2), stamp, fill="#555560", font=small)
                if writer is None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = iio2.get_writer(str(out), fps=meta["fps"] * speed, codec="libx264",
                                             quality=quality, macro_block_size=1)
                writer.append_data(np.asarray(canvas))
        finally:
            if writer is not None:
                writer.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("graph_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--height", type=int, default=720, help="Panel height in pixels (default 720)")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed; 1.0 is real time")
    ap.add_argument("--title", default=None, help="Text shown before the node name")
    ap.add_argument("--quality", type=int, default=7, help="Encoder quality 1-10 (default 7)")
    args = ap.parse_args()
    out = compose(args.case_dir, args.graph_dir, args.out, height=args.height, speed=args.speed, title=args.title,
                  quality=args.quality)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
