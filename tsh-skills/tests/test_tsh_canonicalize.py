"""Unit tests for the tsh-canonicalize skill bundle (CPU-only via FakeContext)."""

import pytest
from gap.testing import FakeContext


def _script(skills_registry):
    info = skills_registry.get("tsh-canonicalize")
    return info.canonical_scripts["canonical_reach"].module


def test_canonical_reach_applies_offset_and_reaches(skills_registry):
    script = _script(skills_registry)
    ctx = FakeContext({"robot.go_to_pose": None})

    obb = {"center": {"x": 0.1, "y": 0.3, "z": 0.0},
           "extent": {"x": 0.03, "y": 0.03, "z": 0.02}}
    out = script.run(ctx, target_obb=obb, canonical_offset=[0.0, 0.0, 0.15], arm_id=1)

    assert out["reached"] is True
    pos = out["target_pose"]["position"]
    assert (pos["x"], pos["y"], pos["z"]) == (0.1, 0.3, 0.15)  # center + offset

    [call] = ctx.calls_to("robot.go_to_pose")
    assert call.kwargs["arm_id"] == 1
    assert call.kwargs["pose"]["position"]["z"] == 0.15


def test_canonical_reach_requires_offset(skills_registry):
    script = _script(skills_registry)
    ctx = FakeContext({"robot.go_to_pose": None})
    obb = {"center": {"x": 0.0, "y": 0.0, "z": 0.0},
           "extent": {"x": 0.0, "y": 0.0, "z": 0.0}}
    with pytest.raises(NotImplementedError):
        script.run(ctx, target_obb=obb, canonical_offset=None)
