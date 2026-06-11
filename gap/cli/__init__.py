"""Unified ``gap`` CLI — single entry point for all gap commands.

Subcommands are registered lazily so that heavy imports (jax, mujoco, …)
only happen when the subcommand is actually invoked.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gap",
        description="gap — graph as policy: typed, verified robot skill graphs",
    )
    sub = parser.add_subparsers(dest="command")

    # Import each subcommand module and let it register its parser.
    # These modules must NOT import heavy deps at module level.
    from .benchmark import register as _reg_benchmark
    from .generate import register as _reg_generate
    from .policy import register as _reg_policy
    from .run import register as _reg_run
    from .skills import register as _reg_skills
    from .trace_diff import register as _reg_trace_diff
    from .viz import register as _reg_viz

    _reg_run(sub)
    _reg_skills(sub)
    _reg_generate(sub)
    _reg_viz(sub)
    _reg_trace_diff(sub)
    _reg_benchmark(sub)
    _reg_policy(sub)

    args = parser.parse_args()

    if hasattr(args, "func"):
        result = args.func(args)
        if isinstance(result, int):
            sys.exit(result)
    else:
        parser.print_help()
        sys.exit(1)
