#!/usr/bin/env python
"""Generate a workflow graph from a natural-language instruction.

The Python flow behind ``gap generate``: one ``generate_sync`` call runs
the coordinator → subgraph agents → checkpoint agent pipeline and writes
a validated workflow directory (``workflow.json`` + ``scripts/`` +
``checkpoints/``).

Needs an LLM credential (``OPENROUTER_API_KEY`` by default — see the
README for vertex). The open-robot-skills checkout is auto-discovered
(``$GAP_SKILLS_PATH`` or the checkout next to the graph-as-policy checkout).

    python examples/generate_a_graph/generate.py \\
        "pick up the alphabet soup can and place it in the basket" \\
        --out my_graph
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_INSTRUCTION = "pick up the alphabet soup can and place it in the basket"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("instruction", nargs="?", default=DEFAULT_INSTRUCTION,
                        help=f"the task (default: {DEFAULT_INSTRUCTION!r})")
    parser.add_argument("--out", default="my_graph", metavar="DIR",
                        help="output directory (default: my_graph)")
    parser.add_argument("--skills", default=None,
                        help="open-robot-skills checkout (default: auto-discovered)")
    parser.add_argument("--provider", default=None,
                        choices=["openrouter", "vertex"],
                        help="LLM provider (default: openrouter)")
    parser.add_argument("--model", default=None,
                        help="model override (default: the provider default)")
    args = parser.parse_args(argv)

    import gap

    graph = gap.agent.generate_sync(
        args.instruction,
        skills=args.skills,          # None -> find_skills_path() discovery
        provider=args.provider,
        model=args.model,
        out_dir=args.out,
    )

    print(f"wrote {graph.path} ({len(graph.code)} generated file(s))")
    print()
    print(graph)                     # the workflow as box-drawing text
    print("\nvalidate / run it with:")
    print(f"  gap run {graph.path} --validate-only")
    print(f"  MUJOCO_GL=egl gap run {graph.path} --sim libero_object_all_variance/0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
