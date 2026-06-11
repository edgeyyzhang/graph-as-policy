"""Tests for gap.testing.FakeContext."""

import pytest

from gap.errors import PerceptionFailed, ToolError
from gap.testing import FakeContext


def test_scripted_value_and_call_log():
    ctx = FakeContext(tool_responses={"vlm.query": "A"})
    assert ctx.tool("vlm.query", prompt="pick one") == "A"
    assert ctx.tool("vlm.query", prompt="again") == "A"
    assert ctx.call_count("vlm.query") == 2
    assert ctx.calls_to("vlm.query")[0].kwargs == {"prompt": "pick one"}


def test_scripted_callable_receives_kwargs_and_can_raise():
    def detect(query: str):
        if query == "missing":
            raise PerceptionFailed("not found")
        return {"detections": [query]}

    ctx = FakeContext(tool_responses={"grounding-dino.detect": detect})
    assert ctx.tool("grounding-dino.detect", query="can") == {"detections": ["can"]}
    with pytest.raises(PerceptionFailed):
        ctx.tool("grounding-dino.detect", query="missing")


def test_scripted_sequence_pops_then_errors():
    ctx = FakeContext(tool_responses={"sam3.segment_box": [{"mask": 1}, {"mask": 2}]})
    assert ctx.tool("sam3.segment_box") == {"mask": 1}
    assert ctx.tool("sam3.segment_box") == {"mask": 2}
    with pytest.raises(ToolError, match="exhausted"):
        ctx.tool("sam3.segment_box")


def test_unscripted_tool_raises_loudly():
    ctx = FakeContext()
    with pytest.raises(ToolError, match="no scripted response"):
        ctx.tool("robot.go_home")


def test_publish_and_cancel_token():
    ctx = FakeContext()
    ctx.publish({"tick": 1})
    assert ctx.published == [{"tick": 1}]
    ctx.cancel_token.raise_if_set()  # not set: no raise
