"""Live anthropic smoke tests — skipped without ANTHROPIC_API_KEY.

Run with: pytest -m llm tests/agent/test_llm_live.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

from gap.agent.llm import LlmConfig, complete, complete_with_tools

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]

# Cheap, fast model for the smoke round-trip; the pipeline default
# (claude-opus-4-8) is exercised by the mocked contract tests.
_CFG = LlmConfig(provider="anthropic", model="claude-haiku-4-5", max_tokens=256)


def test_tiny_completion():
    out = asyncio.run(complete(
        _CFG,
        system="You reply with exactly one word.",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    ))
    assert "pong" in out.lower()


def test_one_tool_use_round():
    calls: list = []

    def handler(name, kwargs):
        calls.append((name, kwargs))
        return {"echoed": kwargs.get("text", "")}

    out = asyncio.run(complete_with_tools(
        _CFG,
        system=(
            "You MUST call the echo tool exactly once with text='hi', then "
            "reply with the word done."
        ),
        messages=[{"role": "user", "content": "Use the echo tool now."}],
        tools=[{
            "name": "echo",
            "description": "Echo the given text back. Call this when asked.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }],
        tool_handler=handler,
    ))
    assert calls and calls[0][0] == "echo"
    assert isinstance(out, str) and out.strip()
