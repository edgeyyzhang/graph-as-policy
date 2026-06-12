#!/usr/bin/env python
"""hello_graph — build, validate, and *picture* a real robot-skill graph.

CPU-only, no GPU, no API key, no extras: ``uv sync`` is enough.

    uv run python examples/hello_graph/hello.py

This is the same artifact pipeline the LLM agent uses: ``gap.builder``
emits a workflow directory (``workflow.json`` + ``scripts/``) that goes
through the same parser and validator as ``gap generate`` output, and
that ``gap run`` executes. Here we build the smallest real graph —
perceive a target, then grasp it — and render it to a PNG you can open.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

TARGET = "blue and yellow alphabet soup can"   # the quickstart's target

#: Canonical bundle script the perceive node references, copied verbatim
#: from the open-robot-skills checkout — exactly what `gap generate` emits.
PERCEIVE_SCRIPT = "skills/perceiving-objects/scripts/perceive_dino_vlm.py"


def build_workflow(target: str = TARGET) -> Workflow:
    """Two subgraphs, each owned by a real skill from open-robot-skills."""
    # 1) perceive: observe → DINO + VLM + SAM3 script → OBB fit
    see = Subgraph(name="perceive_sg", skill="perceiving-objects")
    see.add_node("observe", type="tool", tool="robot.get_observation")
    see.add_node(
        "perceive", type="script", script="scripts/perceive_dino_vlm.py",
        inputs={"cameras": Ref("observe.cameras"), "object_name": target},
    )
    see.add_node(
        "fit_obb", type="tool", tool="geometry.filter_and_compute_obb",
        inputs={"points": Ref("perceive.cloud")},
    )
    see.add_exit("found")                      # noop success marker
    see.set_on_error("not_found")              # any raise → "not_found" exit
    for src, dst in [("START", "observe"), ("observe", "perceive"),
                     ("perceive", "fit_obb"), ("fit_obb", "found"),
                     ("found", "END")]:
        see.add_edge(src, dst)
    see.set_outputs(target_obb=Ref("fit_obb.obb"))

    # 2) grasp: open → top-down candidates → descend → close
    grab = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
    grab.add_input("target_obb", type_name="OrientedBoundingBox")
    grab.add_node("open", type="tool", tool="robot.open_gripper")
    grab.add_node(
        "candidates", type="tool", tool="geometry.top_down_grasp_candidates",
        inputs={"obb": Ref("in.target_obb")},
    )
    grab.add_node("descend", type="tool", tool="robot.go_to_pose",
                  inputs={"pose": Ref("candidates.candidates.poses.0")})
    grab.add_node("close", type="tool", tool="robot.close_gripper")
    grab.add_exit("grasped")
    grab.set_on_error("failed")
    for src, dst in [("START", "open"), ("open", "candidates"),
                     ("candidates", "descend"), ("descend", "close"),
                     ("close", "grasped"), ("grasped", "END")]:
        grab.add_edge(src, dst)

    # Top level: perceive, then grasp; every failure exit routes to abort.
    wf = Workflow(name="hello_graph",
                  description=f"Perceive the {target}, then grasp it.")
    wf.add_subgraph(see)
    wf.add_subgraph(grab)
    wf.add_node("perceive", type="subgraph", ref="perceive_sg")
    wf.add_node("grasp", type="subgraph", ref="grasp_sg")
    wf.add_node("done", type="end", status="success")
    wf.add_node("abort", type="end", status="failure")
    wf.add_edge("START", "perceive")
    wf.add_conditional_edges(
        "perceive", {"found": "grasp", "not_found": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "grasp", {"grasped": "done", "failed": "abort"},
        router_field="exit")
    return wf


def main(argv: list[str] | None = None) -> int:
    # Keep the emitted artifact pristine: validation imports the copied
    # script, which would otherwise drop a __pycache__/ into scripts/.
    sys.dont_write_bytecode = True

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", default="outputs/hello_graph", metavar="DIR",
                        help="output workflow directory (default: outputs/hello_graph)")
    parser.add_argument("--skills", default=None,
                        help="open-robot-skills checkout (default: auto-discovered "
                             "via $GAP_SKILLS_PATH or the sibling checkout)")
    args = parser.parse_args(argv)

    from gap.skills import find_skills_path, load_skills

    skills_root = find_skills_path(args.skills, required=True)

    out = Path(args.out)
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(skills_root / PERCEIVE_SCRIPT,
                 out / "scripts/perceive_dino_vlm.py")

    wf = build_workflow()
    wf.save(out / "workflow.json")             # parse + structural validation

    # Registry-in-the-loop validation — the same checks as
    # `gap run <dir> --validate-only`.
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow

    issues = validate_workflow(load_workflow(out / "workflow.json"),
                               skill_registry=load_skills(skills_root))
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        for issue in errors:
            print(f"  {issue}")
        return 1

    from gap.viz.render import render

    png = render(out / "workflow.json", out / "graph.png")

    print(f"""
Built a real, validated robot-skill graph — no GPU, no API key.

  {out}/workflow.json   the graph (2 subgraphs, validated, 0 errors)
  {png}   <-- open this

Next steps:
  uv run gap run {out} --validate-only    # the CLI does what this script just did
  examples/build_a_graph/      the full authoring example (checkpoints, recovery, --execute)
  examples/generate_a_graph/   let the LLM write a graph like this from one instruction
  examples/libero_quickstart/  run a graph like this on the LIBERO sim (GPU)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
