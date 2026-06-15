"""Fixture tool bundle — one ``@tool`` function for the loader tests.

Imported by the registry through the synthetic package
``gap_skills.tools.fixture-tool.tools``; the decorator below must land in
``gap.tools._registry._PENDING_TOOLS`` for the tool registry to drain.
"""

from gap_core.tools import tool


@tool(name="fixture-tool.echo", summary="Echo a string back, uppercased.")
def echo(text: str) -> str:
    return text.upper()
