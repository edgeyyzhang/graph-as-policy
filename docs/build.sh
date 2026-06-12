#!/usr/bin/env bash
# Build the docs. Warnings are errors (same policy as SkyPilot's build.sh).
#
#   ./build.sh                 one-shot HTML build into docs/build/html
#   ./build.sh --watch [PORT]  live-reload dev server (sphinx-autobuild)
#   ./build.sh --clean         remove build artifacts
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf build
  echo "Removed docs/build."
  exit 0
fi

if [[ "${1:-}" == "--watch" ]]; then
  PORT="${2:-8000}"
  exec sphinx-autobuild source build/html --port "$PORT" --open-browser
fi

sphinx-build -W --keep-going -b html source build/html
echo
echo "Docs built: file://$(pwd)/build/html/index.html"
