"""Top-level router: loop to the next item, or finish.

TSH is a single-item task (``total_items=1`` → done after the first placed
item). A sorting port raises ``total_items`` (or replaces this counter with
a perceive-not_found loop exit) and the same edge re-enters at
``perceive_dest`` — re-perceiving the destination each item also makes the
growing stack height come for free. The counter is per-process (one run =
one episode).
"""

import itertools

from gap import NodeContext

_COUNT = itertools.count(1)


def run(ctx: NodeContext, *, total_items: int = 1) -> dict:
    placed = next(_COUNT)
    return {"route": "done" if placed >= int(total_items) else "next"}
