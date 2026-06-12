#!/usr/bin/env bash
# Regenerate the README media in docs/assets/ — run from the repo root.
#
# Provenance: every asset is produced from THIS repo. The graph figures
# are rendered from checked-in public graphs with gap.viz.render; the
# rollout GIF is cut from a passing trial video recorded by
# `gap benchmark` on the quickstart task (libero_object_all_variance/0).
# Point SRC at any fresh `videos/*__pass.mp4` to refresh it.
set -euo pipefail

SRC=${SRC:-outputs/g1_rename10/20260611_072437/videos/suites__libero_object_all_variance__task_00__trial_01__pass.mp4}
if [[ ! -f "$SRC" ]]; then
  echo "SRC not found: $SRC" >&2
  echo "Set SRC=<path to a videos/*__pass.mp4 from a gap benchmark run> (outputs/ is not in git)." >&2
  exit 1
fi

mkdir -p docs/assets

# 1) The quickstart workflow figure, from the checked-in graph.
uv run python -c "
from gap.viz.render import render
print(render('examples/libero_quickstart/graph', 'docs/assets/quickstart_graph.png'))
"

# 2) The hello_graph figure, from the example itself.
uv run python examples/hello_graph/hello.py --out /tmp/hello_assets >/dev/null
cp /tmp/hello_assets/graph.png docs/assets/hello_graph.png

# 3) The rollout GIF (bundled ffmpeg; palette pass keeps it < 5 MB).
FF=$(uv run python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
F="setpts=PTS/1.4,fps=12,scale=800:-1:flags=lanczos"
"$FF" -loglevel error -i "$SRC" -vf "$F,palettegen" -y /tmp/gap_palette.png
"$FF" -loglevel error -i "$SRC" -i /tmp/gap_palette.png \
  -lavfi "$F [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5" \
  -loop 0 -y docs/assets/quickstart_rollout.gif

ls -la docs/assets/
