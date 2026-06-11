"""``gap generate`` subcommand — LLM graph generation from an instruction."""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "generate",
        help="Generate a workflow graph from a natural-language instruction",
    )
    sp.add_argument(
        "instruction",
        help='The task, e.g. "pick up the alphabet soup and put it in the basket"',
    )
    sp.add_argument(
        "--skills", default=None,
        help="Path to an open-robot-skills checkout (bundle discovery root). "
             "Default: auto-discovered — $GAP_SKILLS_PATH or an open-robot-skills "
             "checkout next to the graph-as-policy checkout; --config skills: also works",
    )
    sp.add_argument(
        "--provider", default=None, choices=["anthropic", "openai", "vertex"],
        help="LLM provider override (default: anthropic)",
    )
    sp.add_argument(
        "--model", default=None,
        help="LLM model override (default: the provider default)",
    )
    sp.add_argument(
        "--out", default=None, metavar="DIR",
        help="Output directory (default: outputs/generated_<timestamp>)",
    )
    sp.add_argument(
        "--config", default=None, metavar="YAML",
        help="Optional pipeline config YAML (llm/composition/skills knobs)",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    import logging

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from gap.agent import PipelineConfig, generate_sync
    from gap.skills import find_skills_path

    config = None
    skills = args.skills
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        if skills is None and config.skills is not None:
            skills = config.skills
    if skills is None:
        try:
            skills = find_skills_path(required=True)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2

    try:
        graph = generate_sync(
            args.instruction,
            skills=skills,
            model=args.model,
            provider=args.provider,
            out_dir=args.out,
            config=config,
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    n_subgraphs = len(graph.workflow.get("subgraphs", {}))
    print(f"OK: wrote {graph.path} ({n_subgraphs} subgraph(s), {len(graph.code)} generated file(s))")
    print(f"run it with: gap run {graph.path}")
    return 0
