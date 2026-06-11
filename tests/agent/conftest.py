"""Shared fixtures for the gap.agent test suite.

``stub_llm`` replaces the provider layer with a canned-response queue —
no live LLM anywhere in this suite. The canned responses below are
realistic coordinator / subgraph_agent / checkpoint_agent outputs
authored against gap.builder for a tiny 2-subgraph workflow
(perceiving-objects → grasping-direct-ik).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gap.agent import llm as llm_mod

#: The real open-robot-skills checkout (sibling of the gap repo).
SKILLS_ROOT = Path(__file__).resolve().parents[3] / "open-robot-skills"


@pytest.fixture
def skills_root() -> Path:
    if not (SKILLS_ROOT / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {SKILLS_ROOT}")
    return SKILLS_ROOT


class StubLLM:
    """FIFO canned-response stand-in for gap.agent.llm."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, cfg, *, system, messages, **kw):
        self.calls.append({
            "kind": "complete", "cfg": cfg,
            "system": system, "messages": list(messages),
        })
        return self._next()

    async def complete_with_tools(self, cfg, *, system, messages, tools,
                                  tool_handler, max_rounds=6, **kw):
        self.calls.append({
            "kind": "tools", "cfg": cfg,
            "system": system, "messages": list(messages), "tools": list(tools),
        })
        return self._next()

    def _next(self) -> str:
        if not self.responses:
            raise AssertionError("stub LLM response queue exhausted")
        return self.responses.pop(0)


@pytest.fixture
def stub_llm(monkeypatch):
    """Install a StubLLM over gap.agent.llm; returns the installer."""

    def install(responses: list[str]) -> StubLLM:
        stub = StubLLM(responses)
        monkeypatch.setattr(llm_mod, "complete", stub.complete)
        monkeypatch.setattr(llm_mod, "complete_with_tools", stub.complete_with_tools)
        return stub

    return install


# ---------------------------------------------------------------------------
# Canned agent responses (authored against gap.builder)
# ---------------------------------------------------------------------------

COORDINATOR_RESPONSE = '''\
I'll decompose this into perception followed by a direct-IK grasp.

```python
from gap.builder import WorkflowSpec, START

spec = WorkflowSpec(name="pick_soup", description="Pick up the alphabet soup")

spec.declare_subgraph(
    "perceive_target",
    skill="perceiving-objects",
    description="Locate the alphabet soup",
    outputs={
        "target_obb": "OrientedBoundingBox",
        "target_mask": "Mask",
        "target_cloud": "PointCloud",
    },
    exit_success_values=["found"],
    on_error="not_found",
)
spec.declare_subgraph(
    "grasp_target",
    skill="grasping-direct-ik",
    description="Grasp the alphabet soup",
    inputs={"target_obb": "OrientedBoundingBox"},
    outputs={"grasp_pose": "Se3Pose"},
    exit_success_values=["grasped"],
    on_error="failed",
    stage="grasp",
)

spec.add_subgraph_node("perceive_target", ref="perceive_target")
spec.add_subgraph_node("grasp_target", ref="grasp_target")
spec.add_end("done", status="success")
spec.add_end(
    "abort",
    status="failure",
    recovery=[
        {"tool": "robot.open_gripper", "inputs": {}},
        {"tool": "robot.go_home", "inputs": {}},
    ],
)

spec.add_edge(START, "perceive_target")
spec.add_conditional_edges(
    "perceive_target",
    {"found": "grasp_target", "not_found": "abort"},
    router_field="exit",
)
spec.add_conditional_edges(
    "grasp_target",
    {"grasped": "done", "failed": "abort"},
    router_field="exit",
)
```
'''

MAKE_OBB_SCRIPT = '''\
"""Compute an axis-aligned OBB from a point cloud."""

from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap.types import OrientedBoundingBox, PointCloud


class Output(TypedDict):
    obb: OrientedBoundingBox


def run(ctx: NodeContext, cloud: PointCloud) -> Output:
    pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return {"obb": {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "extent": {"x": float(half[0]), "y": float(half[1]), "z": float(half[2])},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }}
'''

PERCEIVE_SUBGRAPH_RESPONSE = f'''\
```python
from gap.builder import Subgraph, Ref, START, END

sg = Subgraph(name="perceive_target", skill="perceiving-objects")

sg.add_node("observe", type="tool", tool="robot.get_observation")
sg.add_node("perceive", type="script",
            script="scripts/perceive_target/perceive_dino_vlm.py",
            inputs={{"cameras": Ref("observe.cameras"),
                     "object_name": "alphabet soup"}})
sg.add_node("make_obb", type="script",
            script="scripts/perceive_target/make_obb.py",
            inputs={{"cloud": Ref("perceive.cloud")}})

sg.add_exit("found")
sg.add_edge(START, "observe")
sg.add_edge("observe", "perceive")
sg.add_edge("perceive", "make_obb")
sg.add_edge("make_obb", "found")
sg.add_edge("found", END)

sg.set_outputs(
    target_obb=Ref("make_obb.obb"),
    target_mask=Ref("perceive.mask"),
    target_cloud=Ref("perceive.cloud"),
)
sg.set_on_error("not_found")
```

```python:scripts/perceive_target/make_obb.py
{MAKE_OBB_SCRIPT}
```
'''

GRASP_SUBGRAPH_RESPONSE = '''\
```python
from gap.builder import Subgraph, Ref, START, END

sg = Subgraph(name="grasp_target", skill="grasping-direct-ik")
sg.add_input("target_obb", type_name="OrientedBoundingBox")

sg.add_node("compute_grasp", type="tool",
            tool="geometry.top_down_grasp_candidates",
            inputs={"obb": Ref("in.target_obb")})
sg.add_node("compute_align", type="script",
            script="scripts/grasp_target/compute_align_pose.py",
            inputs={"grasp_pose": Ref("compute_grasp.candidates.poses.0"),
                    "target_obb": Ref("in.target_obb")})
sg.add_node("align", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("compute_align.align_pose")})
sg.add_node("descend", type="tool", tool="robot.go_to_pose",
            inputs={"pose": Ref("compute_grasp.candidates.poses.0")})
sg.add_node("close", type="tool", tool="robot.close_gripper")

sg.add_exit("grasped")
sg.add_edge(START, "compute_grasp")
sg.add_edge("compute_grasp", "compute_align")
sg.add_edge("compute_align", "align")
sg.add_edge("align", "descend")
sg.add_edge("descend", "close")
sg.add_edge("close", "grasped")
sg.add_edge("grasped", END)

sg.set_outputs(grasp_pose=Ref("compute_grasp.candidates.poses.0"))
sg.set_on_error("failed")
```
'''

#: Missing the required z-axis checkpoint on the grasp subgraph (which
#: binds `grasp_pose`) — the runner's per-skill rule rejects it and
#: re-prompts.
CHECKPOINT_RESPONSE_BAD = '''\
```python
subgraphs["perceive_target"].add_checkpoint(
    "target_obb_sane",
    predicate=lambda w, o: 0.0 < o["target_obb"]["extent"]["x"] < 0.5,
    rationale="perceived OBB is finite and non-degenerate",
    validate=True,
)
subgraphs["grasp_target"].add_checkpoint(
    "target_held",
    predicate=lambda w: w.body("alphabet soup").is_grasped(),
    rationale="target in contact with a robot link after close",
    validate=True,
)
```
'''

CHECKPOINT_RESPONSE_GOOD = '''\
```python
subgraphs["perceive_target"].add_checkpoint(
    "target_obb_sane",
    predicate=lambda w, o: 0.0 < o["target_obb"]["extent"]["x"] < 0.5,
    rationale="perceived OBB is finite and non-degenerate",
    validate=True,
)
subgraphs["grasp_target"].add_checkpoint(
    "grasp_pose_above_table",
    predicate=lambda w, o: o["grasp_pose"]["position"]["z"] > 0.01,
    rationale="computed grasp pose is above the tabletop",
    validate=True,
)
subgraphs["grasp_target"].add_checkpoint(
    "target_held",
    predicate=lambda w: w.body("alphabet soup").is_grasped(),
    rationale="target in contact with a robot link after close",
    validate=True,
)
```
'''

FIXED_MAKE_OBB_RESPONSE = '''\
```python
"""Compute an axis-aligned OBB from a point cloud (fixed)."""

from typing import TypedDict

import numpy as np

from gap import NodeContext
from gap.types import OrientedBoundingBox, PointCloud

FIXED_MARKER = True


class Output(TypedDict):
    obb: OrientedBoundingBox


def run(ctx: NodeContext, cloud: PointCloud) -> Output:
    pts = np.asarray(cloud["points"], dtype=np.float64).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return {"obb": {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "extent": {"x": float(half[0]), "y": float(half[1]), "z": float(half[2])},
        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    }}
```
'''

#: The standard full-pipeline queue: coordinator → 2 subgraphs →
#: checkpoint agent (one bad attempt, then a good one).
PIPELINE_RESPONSES = [
    COORDINATOR_RESPONSE,
    PERCEIVE_SUBGRAPH_RESPONSE,
    GRASP_SUBGRAPH_RESPONSE,
    CHECKPOINT_RESPONSE_BAD,
    CHECKPOINT_RESPONSE_GOOD,
]
