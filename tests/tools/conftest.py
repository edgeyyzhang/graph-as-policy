"""Isolate the process-global @tool pending/drained state between tests.

Production semantics deliberately replay every drained registration into
every new registry (the sequential-benchmark path depends on it); tests
that assert exact registry contents need a clean slate instead.
"""

import gap_core.tools._registry as _registry
import pytest


@pytest.fixture(autouse=True)
def _isolated_tool_state():
    pending = list(_registry._PENDING_TOOLS)
    drained = list(_registry._DRAINED_TOOLS)
    _registry._PENDING_TOOLS.clear()
    _registry._DRAINED_TOOLS.clear()
    try:
        yield
    finally:
        _registry._PENDING_TOOLS[:] = pending
        _registry._DRAINED_TOOLS[:] = drained
