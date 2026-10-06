"""``gap rehearse`` — run a graph over fixed sim cases and write feedback.

    gap rehearse GRAPH --sim libero_10/0 --cases 1-8 --out outputs/rehearse/r01
    gap rehearse GRAPH --sim libero_10/0 --cases 1-8 --out outputs/rehearse/r02 \\
        --previous outputs/rehearse/r01

Per case the harness resets the simulator to that initial state, runs the
graph, and records the privileged world before and after every node, every
subgraph exit, checkpoint results, and the native task predicate. The run
directory then holds:

- ``feedback/main.md`` and ``feedback/subgraphs/<name>.md``: one file per
  graph, all in one format (see :mod:`gap.rehearse.graph_feedback`);
- ``cases/case_NNNN/trajectory.jsonl``: the privileged state after every
  simulator step, and ``trajectories/case_NNNN/*.md``, its readable views;
- ``feedback.json`` and ``feedback.md``: the whole-run summary;
- one ``cases/case_NNNN/`` directory per case with the GaP trace, and
  ``workflow.snapshot.json`` for the next round's diff.
"""

from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "rehearse",
        help="Run a graph over fixed simulator cases and write per-node feedback",
    )
    sp.add_argument("graph", help="Workflow directory or workflow.json path")
    sp.add_argument("--sim", required=True, metavar="SUITE/TASK",
                    help="LIBERO sim task, e.g. libero_10/0")
    sp.add_argument("--env", default="libero", help="Sim env registry name (default: libero)")
    sp.add_argument("--cases", required=True,
                    help="Initial-state cases as seeds: '1-8' or '1,3,5' (seed k -> init (k-1) mod n)")
    sp.add_argument("--out", required=True, help="Output directory for this round")
    sp.add_argument("--previous", default=None,
                    help="Output directory of the previous round, for per-case changes and the graph diff")
    sp.add_argument("--skills", action="append", default=None, metavar="PATH",
                    help="Skill registry root(s); repeatable (default: resolved registries)")
    sp.add_argument("--tools", default=None, metavar="MODULE:FUNCTION",
                    help="Plugin called with the connector before running, to register extra tools "
                         "(e.g. privileged geometry stand-ins for perception)")
    sp.add_argument("--ik", default=None, choices=["curobo", "pyroki"],
                    help="Inverse-kinematics backend (default: the connector's, cuRobo)")
    sp.add_argument("--checkpoints", default="warn", choices=["off", "warn", "raise"])
    sp.add_argument("--objects", default=None,
                    help="Comma-separated body names to track (default: every non-robot, non-region body)")
    sp.add_argument("--frames", action="store_true",
                    help="Save one camera frame per node exit under cases/<case>/frames/")
    sp.add_argument("--no-trajectory", action="store_true",
                    help="Do not sample the privileged state at every simulator step")
    sp.add_argument("--trajectory-interval", type=int, default=5, metavar="N",
                    help="Rows in the trajectory views: one every N simulator steps (default: 5)")
    sp.add_argument("--video", action="store_true",
                    help="Write cases/<case>/video.mp4 from the exterior camera, one frame per --video-interval steps")
    sp.add_argument("--video-interval", type=int, default=5, metavar="N",
                    help="Simulator steps between video frames (default: 5, i.e. 4 fps at 20 Hz)")
    sp.add_argument("--claims", default=None, metavar="JSON",
                    help="JSON object mapping exit values to 'held' / 'not_held' claims, merged over the defaults")
    sp.add_argument("--inputs", nargs="*", default=[], metavar="K=V", help="Initial workflow inputs")
    sp.add_argument("--max-node-workers", type=int, default=1)
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    from gap.cli.run import _parse_inputs
    from gap.rehearse import rehearse

    claims = json.loads(args.claims) if args.claims else None
    objects = [s for s in (args.objects or "").split(",") if s] or None
    fb = rehearse(
        args.graph,
        sim=args.sim,
        env=args.env,
        cases=args.cases,
        out=args.out,
        previous=args.previous,
        skills=args.skills,
        tools=args.tools,
        checkpoints=args.checkpoints,
        objects=objects,
        frames=args.frames,
        claims=claims,
        inputs=_parse_inputs(args.inputs) or None,
        max_node_workers=args.max_node_workers,
        ik=args.ik,
        trajectory=not args.no_trajectory,
        trajectory_interval=args.trajectory_interval,
        video=args.video,
        video_interval=args.video_interval,
    )
    s = fb["summary"]
    line = f"rehearsal complete: {s['successes']}/{s['cases']} native successes"
    if s.get("previous_successes") is not None:
        line += f" (previous {s['previous_successes']}; fixed {s['fixed']}, broken {s['broken']})"
    print(line)
    if fb["disagreements"]:
        print(f"{len(fb['disagreements'])} verdict/world disagreement pattern(s); see feedback.md")
    print(f"feedback: {args.out}/feedback/main.md")
    return 0
