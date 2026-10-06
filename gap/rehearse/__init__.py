"""gap.rehearse — run a fixed graph over fixed cases and report per-node feedback.

The paper's self-learning loop rehearses a graph on sampled task instances,
records robot and object state around every node, and hands failure
summaries back to the authoring agents. This package is the released-code
version of the recording and reporting half: it executes a workflow through
the ordinary executor, captures the privileged world at every node boundary
and subgraph exit, and writes ``feedback.json`` / ``feedback.md``. Who reads
the feedback and how the graph is edited is left to the caller.

Each graph also gets its own feedback file in one shared format
(``feedback/main.md``, ``feedback/subgraphs/<name>.md``), and the privileged
state is sampled at every simulator step into ``trajectory.jsonl``.
"""

from .graph_feedback import build_graph_feedback, write_graph_feedback
from .report import DEFAULT_CLAIMS, build_feedback, render_markdown, write_feedback
from .run import graph_manifest, parse_cases, rehearse, run_case
from .trajectory import StepSampler
from .trajectory_view import write_views
from .world_state import diff as world_diff
from .world_state import summarize as summarize_world

__all__ = [
    "rehearse", "run_case", "parse_cases", "graph_manifest",
    "build_feedback", "render_markdown", "write_feedback", "DEFAULT_CLAIMS",
    "summarize_world", "world_diff",
    "build_graph_feedback", "write_graph_feedback", "StepSampler", "write_views",
]
