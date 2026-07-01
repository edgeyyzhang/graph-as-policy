# graph-as-policy-core

The bundle-author surface for gap (graph-as-policy). Every per-bundle `.venv`
(tool servers, policy servers) installs this distribution alone — they get
the `@tool` decorator, the typed value vocabulary, the error hierarchy, the
skill metadata dataclasses, and the msgpack-RPC primitives, but none of the
heavy runtime stack (no fastapi, no JAX, no MuJoCo, no
opencv/matplotlib/viser).

## Public modules

```python
from gap_core.tools import tool                              # @tool decorator
from gap_core.types import Se3Pose, Mask, BoundingBox2D      # typed values
from gap_core.errors import PerceptionFailed, PlanningFailed # exceptions
from gap_core.skills import Skill, SkillMeta, Param, Serving # skill metadata
from gap_core.schema import TYPE_REGISTRY                    # type-name registry
# (PR 5+) from gap_core.rpc.client import ToolClient
# (PR 5+) from gap_core.rpc.server import main as tool_server_main
```

`NodeContext` is intentionally NOT here — it lives in `gap.runtime.context`
(the runtime distribution). Tool bundles never instantiate it; gap-runtime
constructs it and passes it as the first arg to tool functions that declare
a `ctx` parameter.

## Install

Inside the dev workspace at `/home/kych/graph-as-policy/`, `uv sync` makes
this package editable alongside the full `graph-as-policy` runtime. Bundle
authors who want a slim per-bundle venv install just this distribution:

```bash
uv venv .venv-bundle
uv pip install --python .venv-bundle/bin/python graph-as-policy-core
```

The wheel is ~0.1 MB; the resolved deps (numpy + scipy + msgpack) are
~150 MB on Linux.
