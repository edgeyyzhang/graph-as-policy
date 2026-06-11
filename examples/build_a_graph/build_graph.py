#!/usr/bin/env python
"""Author a complete pick-and-place workflow in Python with ``gap.builder``.

The builder is the same artifact pipeline as ``gap generate``: it produces
a workflow directory (``workflow.json`` + ``scripts/`` + ``checkpoints/``)
that goes through the same parser and structural validation as
LLM-generated graphs, and runs with the same ``gap run`` / ``gap.execute``.

The graph built here mirrors the quickstart's structure — and it is real,
not a toy: perceive the target and the container with the
``perceiving-objects`` bundle's canonical DINO + VLM + SAM3 script, grasp
with the planner-free ``grasping-direct-ik`` recipe (align above the OBB,
descend, close), then transport into the container with the
``transporting-objects`` scripts. A ground-truth ``target_held``
postcondition checkpoint guards the grasp.

Build + validate (CPU-only, no sim)::

    python examples/build_a_graph/build_graph.py --out my_graph

Execute on the LIBERO sim (needs ``[libero]``, ``open-robot-skills[quickstart]``
with downloaded weights, a VLM credential, and ``MUJOCO_GL=egl``)::

    MUJOCO_GL=egl python examples/build_a_graph/build_graph.py \\
        --out my_graph --execute

or run the built artifact like any other graph::

    MUJOCO_GL=egl gap run my_graph --sim libero_object_all_variance/0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from gap.builder import Ref, Subgraph, Workflow

# Perception prompts (what the vision pipeline looks for) and the sim body
# name the grasp checkpoint verifies against ground truth — the same task
# as the quickstart: libero_object_all_variance/0.
TARGET_PROMPT = "blue and yellow alphabet soup can"
CONTAINER_PROMPT = "basket"
TARGET_BODY = "alphabet soup"

#: Canonical bundle scripts the graph's ``type: script`` nodes reference,
#: copied verbatim from the open-robot-skills checkout into ``<out>/scripts/`` —
#: exactly the per-workflow copies ``gap generate`` emits.
SCRIPT_SOURCES: dict[str, str] = {
    "scripts/perceive_dino_vlm.py":
        "skills/perceiving-objects/scripts/perceive_dino_vlm.py",
    "scripts/compute_align_pose.py":
        "skills/grasping-direct-ik/scripts/compute_align_pose.py",
    "scripts/compute_drop_pose.py":
        "skills/transporting-objects/scripts/compute_drop_pose.py",
    "scripts/waypoint_move.py":
        "skills/transporting-objects/scripts/waypoint_move.py",
    "scripts/descend_release.py":
        "skills/transporting-objects/scripts/descend_release.py",
}


# ---------------------------------------------------------------------------
# Subgraphs
# ---------------------------------------------------------------------------


def perception_subgraph(name: str, *, object_name: str, prefix: str) -> Subgraph:
    """observe → perceive (DINO + VLM + SAM3 script) → OBB fit → found.

    Owned by the ``perceiving-objects`` skill: the canonical script returns
    a world-frame point cloud + mask, and ``geometry.filter_and_compute_obb``
    fits the oriented bounding box downstream nodes consume.
    """
    sg = Subgraph(name=name, skill="perceiving-objects")
    sg.add_node("observe", type="tool", tool="robot.get_observation")
    sg.add_node(
        "perceive", type="script", script="scripts/perceive_dino_vlm.py",
        inputs={"cameras": Ref("observe.cameras"), "object_name": object_name},
    )
    sg.add_node(
        "filter_obb", type="tool", tool="geometry.filter_and_compute_obb",
        inputs={"points": Ref("perceive.cloud")},
    )
    sg.add_exit("found")                      # noop success marker
    sg.set_on_error("not_found")              # any raise → "not_found" exit
    for src, dst in [("START", "observe"), ("observe", "perceive"),
                     ("perceive", "filter_obb"), ("filter_obb", "found"),
                     ("found", "END")]:
        sg.add_edge(src, dst)
    # Cross-subgraph outputs; downstream subgraph inputs bind by name.
    sg.set_outputs(**{
        f"{prefix}_obb": Ref("filter_obb.obb"),
        f"{prefix}_mask": Ref("perceive.mask"),
        f"{prefix}_cloud": Ref("perceive.cloud"),
    })
    return sg


def grasp_subgraph() -> Subgraph:
    """The ``grasping-direct-ik`` recipe: open → top-down candidates →
    pre-rotate at altitude → straight-line descend → close.

    No trajectory planner: ``robot.go_to_pose`` runs the connector's
    in-process IK. The align-then-descend split avoids twisting the
    gripper against the object while closing.
    """
    sg = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
    sg.add_input("target_obb", type_name="OrientedBoundingBox")
    sg.add_node("open", type="tool", tool="robot.open_gripper",
                inputs={"settle_steps": 40})
    sg.add_node("compute_grasp", type="tool",
                tool="geometry.top_down_grasp_candidates",
                inputs={"obb": Ref("in.target_obb")})
    sg.add_node(
        "compute_align", type="script", script="scripts/compute_align_pose.py",
        inputs={"grasp_pose": Ref("compute_grasp.candidates.poses.0"),
                "target_obb": Ref("in.target_obb")},
    )
    sg.add_node("rotate_align", type="tool", tool="robot.go_to_pose",
                inputs={"pose": Ref("compute_align.align_pose")})
    sg.add_node("descend", type="tool", tool="robot.go_to_pose",
                inputs={"pose": Ref("compute_grasp.candidates.poses.0")})
    sg.add_node("close", type="tool", tool="robot.close_gripper",
                inputs={"settle_steps": 60})
    sg.add_exit("grasped")
    sg.set_on_error("failed")
    for src, dst in [("START", "open"), ("open", "compute_grasp"),
                     ("compute_grasp", "compute_align"),
                     ("compute_align", "rotate_align"),
                     ("rotate_align", "descend"), ("descend", "close"),
                     ("close", "grasped"), ("grasped", "END")]:
        sg.add_edge(src, dst)
    return sg


def transport_subgraph() -> Subgraph:
    """``transporting-objects``: drop pose from the container OBB → lift +
    lateral waypoint move → descend, release, retract."""
    sg = Subgraph(name="transport_sg", skill="transporting-objects")
    sg.add_input("container_obb", type_name="OrientedBoundingBox")
    sg.add_node(
        "compute_drop", type="script", script="scripts/compute_drop_pose.py",
        inputs={"container_obb": Ref("in.container_obb")},
    )
    sg.add_node(
        "move_above", type="script", script="scripts/waypoint_move.py",
        inputs={"drop_x": Ref("compute_drop.drop_position.x"),
                "drop_y": Ref("compute_drop.drop_position.y"),
                "drop_rotation": Ref("compute_drop.drop_pose.rotation")},
    )
    sg.add_node(
        "release", type="script", script="scripts/descend_release.py",
        inputs={"drop_position": Ref("compute_drop.drop_position"),
                "drop_rotation": Ref("compute_drop.drop_pose.rotation")},
    )
    sg.add_exit("placed")
    sg.set_on_error("blocked")
    for src, dst in [("START", "compute_drop"), ("compute_drop", "move_above"),
                     ("move_above", "release"), ("release", "placed"),
                     ("placed", "END")]:
        sg.add_edge(src, dst)
    sg.set_outputs(drop_pose=Ref("compute_drop.drop_pose"))
    return sg


# ---------------------------------------------------------------------------
# Top-level workflow
# ---------------------------------------------------------------------------


def build_workflow(
    target: str = TARGET_PROMPT,
    container: str = CONTAINER_PROMPT,
) -> Workflow:
    """Assemble the full pick-and-place workflow (pure, no file I/O)."""
    wf = Workflow(
        name="pick_into_basket",
        description=(
            f"Pick the {target} and place it in the {container} "
            f"(direct-IK align-then-descend grasp; authored with gap.builder)."
        ),
    )
    wf.add_subgraph(perception_subgraph(
        "target_sg", object_name=target, prefix="target"))
    wf.add_subgraph(perception_subgraph(
        "container_sg", object_name=container, prefix="container"))
    wf.add_subgraph(grasp_subgraph())
    wf.add_subgraph(transport_subgraph())

    wf.add_node("target", type="subgraph", ref="target_sg")
    wf.add_node("container", type="subgraph", ref="container_sg")
    wf.add_node("grasp", type="subgraph", ref="grasp_sg")
    wf.add_node("transport", type="subgraph", ref="transport_sg")
    wf.add_node("done", type="end", status="success")
    wf.add_node(
        "abort", type="end", status="failure",
        recovery=[{"tool": "robot.open_gripper", "inputs": {}},
                  {"tool": "robot.go_home", "inputs": {}}],
    )

    wf.add_edge("START", "target")
    wf.add_conditional_edges(
        "target", {"found": "container", "not_found": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "container", {"found": "grasp", "not_found": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "grasp", {"grasped": "transport", "failed": "abort"},
        router_field="exit")
    wf.add_conditional_edges(
        "transport", {"placed": "done", "blocked": "abort"},
        router_field="exit")
    return wf


# ---------------------------------------------------------------------------
# Artifact assembly (the same on-disk layout `gap generate` writes)
# ---------------------------------------------------------------------------

_CHECKPOINT_SIDECAR = '''\
"""Postcondition checkpoint for `grasp_sg` (authored by build_graph.py).

Loaded by the executor's checkpoint hook (`--checkpoints warn|raise`) and
evaluated against the sim's ground-truth world snapshot every time
`grasp_sg` exits on its success path.
"""

from gap.runtime.verify import Checkpoint


def _target_held(world) -> bool:
    """The target is in contact with a robot finger link after `close`."""
    return world.body({target_body!r}).is_grasped()


CHECKPOINTS = [
    Checkpoint(
        name="target_held",
        subgraph="grasp_sg",
        predicate=_target_held,
        rationale=(
            "After close, the grasp target must actually be held — "
            "ground-truth contacts prove the gripper closed ON the object, "
            "not on air."
        ),
        validate=True,
    ),
]
'''


def write_graph(
    wf: Workflow,
    out_dir: str | Path,
    skills_root: str | Path,
    *,
    target_body: str = TARGET_BODY,
) -> Path:
    """Materialize the workflow directory: scripts + workflow.json + checkpoints.

    Scripts are copied first so ``wf.save`` (which parses and structurally
    validates, importing each script for schema introspection) sees the
    complete artifact.
    """
    out_dir = Path(out_dir)
    skills_root = Path(skills_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dest_rel, src_rel in SCRIPT_SOURCES.items():
        src = skills_root / src_rel
        if not src.is_file():
            raise FileNotFoundError(
                f"canonical script not found: {src} — is {skills_root} a "
                f"open-robot-skills checkout?"
            )
        dest = out_dir / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    wf.save(out_dir / "workflow.json")        # parse + W/S validation

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    (checkpoints_dir / "grasp_sg.py").write_text(
        _CHECKPOINT_SIDECAR.format(target_body=target_body), encoding="utf-8",
    )
    return out_dir


def validate_graph(out_dir: Path, skills_root: Path) -> int:
    """Re-validate the written artifact with the skill registry in the loop
    (the same checks as ``gap run <dir> --validate-only``)."""
    from gap.runtime.validate import validate_workflow
    from gap.runtime.workflow import load_workflow
    from gap.skills import load_skills

    wf_def = load_workflow(out_dir / "workflow.json")
    issues = validate_workflow(wf_def, skill_registry=load_skills(skills_root))
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        print(f"  {issue}")
    print(f"validate: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)")
    return len(errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="my_graph", metavar="DIR",
                        help="output workflow directory (default: my_graph)")
    parser.add_argument("--skills", default=None,
                        help="open-robot-skills checkout (default: auto-discovered "
                             "via $GAP_SKILLS_PATH or the sibling checkout)")
    parser.add_argument("--target", default=TARGET_PROMPT,
                        help="perception prompt for the object to pick")
    parser.add_argument("--container", default=CONTAINER_PROMPT,
                        help="perception prompt for the destination container")
    parser.add_argument("--target-body", default=TARGET_BODY,
                        help="sim body name the grasp checkpoint verifies")
    parser.add_argument("--execute", action="store_true",
                        help="after building, run the graph on the LIBERO sim")
    parser.add_argument("--task", default="libero_object_all_variance/0",
                        metavar="SUITE/ID", help="sim task for --execute")
    args = parser.parse_args(argv)

    from gap.skills import find_skills_path

    skills_root = find_skills_path(args.skills, required=True)

    wf = build_workflow(target=args.target, container=args.container)
    out_dir = write_graph(wf, args.out, skills_root, target_body=args.target_body)
    print(f"built {out_dir}/  (workflow.json + {len(SCRIPT_SOURCES)} scripts "
          f"+ checkpoints/grasp_sg.py)")

    n_errors = validate_graph(out_dir, skills_root)
    if n_errors:
        return 1

    if not args.execute:
        print(f"\nrun it with:\n  MUJOCO_GL=egl gap run {out_dir} --sim {args.task}")
        return 0

    import gap

    conn = gap.connector.sim("libero", task=args.task)
    try:
        result = gap.execute(out_dir, conn, skills=skills_root)
    finally:
        conn.close()
    print(f"{'SUCCESS' if result.success else 'FAILURE'} "
          f"(exit={result.exit_status}, {result.duration_s:.1f}s)")
    for cp in result.checkpoint_results:
        print(f"checkpoint: {cp}")
    if result.trace_path is not None:
        print(f"trace: {result.trace_path}  (browse with `gap viz`)")
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
