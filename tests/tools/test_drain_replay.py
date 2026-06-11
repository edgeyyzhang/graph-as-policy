"""Regression: a registry built AFTER another drained the pendings still gets bundle tools."""
from gap.tools import ToolRegistry, tool


def test_second_registry_gets_tools_after_first_drain():
    @tool(name="testdrain.fn", summary="t", tags=())
    def fn(x: int = 0) -> dict:
        return {"x": x}

    r1 = ToolRegistry()
    r1.discover_pending()
    assert "testdrain.fn" in r1
    # The pending queue is now empty — a later registry in the SAME
    # process (sequential benchmark path) must still see the tool.
    r2 = ToolRegistry()
    r2.discover_pending()
    assert "testdrain.fn" in r2
    # And with a per-worker catalog (pool path) it composes too.
    cat: list = []
    r3 = ToolRegistry()
    r3.discover_pending(catalog=cat)
    assert "testdrain.fn" in r3
