#!/usr/bin/env bash
# Run any `gap` command with the environment this checkout needs.
#
#   examples/rehearse_loop/gap.sh rehearse GRAPH --sim SUITE/TASK --cases 1-8 --out DIR
#   examples/rehearse_loop/gap.sh rehearse-loop serve TRUSTED_DIR
#
# The OpenRouter key is read from ~/.config/gap/credentials.env and is never printed.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gap=$(cd "$here/../.." && pwd)
python=${GAP_PYTHON:-$gap/.venv/bin/python}

cred=${GAP_CREDENTIALS:-$HOME/.config/gap/credentials.env}
# shellcheck disable=SC1090
[ -f "$cred" ] && source "$cred"

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$gap/third_party/Variational-Automation-Benchmark/libero:$gap:$gap/gap-core/src:$gap/third_party/robosuite${GAP_EXTRA_PYTHONPATH:+:$GAP_EXTRA_PYTHONPATH}
export MUJOCO_GL=egl JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$(cd "$gap/.." && pwd)/.libero}

exec "$python" -c 'import sys; from gap.cli import main; sys.argv = ["gap", *sys.argv[1:]]; sys.exit(main())' "$@"
