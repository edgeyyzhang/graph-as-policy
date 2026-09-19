"""gap.runtime.verify — World/checkpoint vocabulary for sim-side verification.

LLM-authored checkpoint predicates are written against :class:`World` /
:class:`Body` / :class:`Robot` snapshots; :func:`evaluate_checkpoint` runs
a :class:`Checkpoint` with eval-error capture and diagnostics. A sim
backend adapter (e.g. the LIBERO world adapter) builds ``World`` snapshots
from plain numpy/python data.
"""

from __future__ import annotations

from gap.runtime.verify.checkpoints import (
    Checkpoint,
    CheckpointResult,
    evaluate_checkpoint,
    load_checkpoints,
)
from gap.runtime.verify.world import (
    Articulation,
    Body,
    BodyNotFoundError,
    Robot,
    StubWorld,
    World,
    always,
    at_end,
    contacts_from_pairs,
    eventually,
    stub_world_with_history,
)

__all__ = [
    "Articulation",
    "Body",
    "BodyNotFoundError",
    "Checkpoint",
    "CheckpointResult",
    "Robot",
    "StubWorld",
    "World",
    "always",
    "at_end",
    "contacts_from_pairs",
    "evaluate_checkpoint",
    "eventually",
    "load_checkpoints",
    "stub_world_with_history",
]
