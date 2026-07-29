#!/usr/bin/env python
"""Perception generalization spot-check: yellow bowl / gray bowl instead of
yellow tape / gray duct tape (see LIBERO-YAM/libero_yam/bddl_files/
libero_yam_tabletop/two_bowl_perceive.bddl and the flat-color bowl assets
under LIBERO-YAM/libero_yam/assets/objects/libero/{yellow_bowl,gray_bowl}/).

Runs both perceive-tape-cv and perceive-tape-sam against the SAME scene via a
minimal observe->perceive graph, copying each skill's CURRENT canonical
scripts into a temp dir at runtime (never a stale checked-in copy).

    uv run --no-sync python examples/tape_handover_yam/perceive_bowl_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gap
from gap.connector.libero_yam import libero_yam

HERE = Path(__file__).resolve().parent
TSH_SKILLS = HERE.parents[1] / "tsh-skills"
OPEN_ROBOT_SKILLS = HERE.parents[2] / "open-robot-skills"
BDDL = (HERE.parents[2] / "LIBERO-YAM" / "libero_yam" / "bddl_files"
        / "libero_yam_tabletop" / "two_bowl_perceive.bddl")
SKILLS = [str(OPEN_ROBOT_SKILLS), str(TSH_SKILLS)]

_SKILL_FILES = {
    "perceive-tape-cv": ["perceive_object_cv.py", "_perceive.py", "_perceive_cv.py", "constants.py"],
    "perceive-tape-sam": ["perceive_object.py", "_perceive.py", "constants.py"],
}
_ENTRY_SCRIPT = {
    "perceive-tape-cv": "scripts/perceive_object_cv.py",
    "perceive-tape-sam": "scripts/perceive_object.py",
}


def _materialize(skill: str, query: str) -> str:
    out = Path(tempfile.mkdtemp(prefix=f"bowl_{skill}_"))
    (out / "scripts").mkdir()
    for fname in _SKILL_FILES[skill]:
        shutil.copy(TSH_SKILLS / "skills" / skill / "scripts" / fname, out / "scripts" / fname)
    graph = {
        "version": 3,
        "meta": {"name": "bowl_perceive", "description": f"{skill} on {query!r}"},
        "nodes": {"perceive_sg": {"type": "subgraph", "ref": "perceive_sg"},
                  "done": {"type": "end", "status": "success"},
                  "abort": {"type": "end", "status": "failure"}},
        "edges": [["START", "perceive_sg"]],
        "conditional_edges": {"perceive_sg": {"router_field": "exit",
                                               "mapping": {"perceived": "done", "not_found": "abort"}}},
        "subgraphs": {"perceive_sg": {
            "skill": skill, "inputs": {}, "outputs": {"found": {"$ref": "perceive.found"}},
            "nodes": {
                "observe": {"type": "tool", "tool": "robot.get_observation"},
                "perceive": {"type": "script",
                             "inputs": {"cameras": {"$ref": "observe.cameras"}, "object_query": query},
                             "script": _ENTRY_SCRIPT[skill]},
                "perceived": {"type": "noop"},
            },
            "edges": [["START", "observe"], ["observe", "perceive"], ["perceive", "perceived"], ["perceived", "END"]],
            "conditional_edges": {}, "exit": {"router_field": None, "success_values": ["perceived"]},
            "on_error": "not_found",
        }},
    }
    (out / "workflow.json").write_text(json.dumps(graph, indent=2))
    return str(out)


def main() -> None:
    conn = libero_yam(bddl=str(BDDL), cameras=["agentview"])
    conn.reset()

    for skill, query in (("perceive-tape-cv", "yellow bowl"), ("perceive-tape-cv", "gray bowl"),
                         ("perceive-tape-sam", "yellow bowl"), ("perceive-tape-sam", "gray bowl")):
        graph_dir = _materialize(skill, query)
        t0 = time.perf_counter()
        r = gap.execute(graph_dir, conn, skills=SKILLS)
        dt = time.perf_counter() - t0
        found = r.outputs.get("perceive_sg", {}).get("found") if r.success else None
        print(f"{skill:18s} query={query!r:14s} found={found}  wall={dt:.2f}s"
              + (f"  error={r.error}" if r.error else ""))


if __name__ == "__main__":
    main()
