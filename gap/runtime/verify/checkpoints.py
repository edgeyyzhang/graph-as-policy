"""Checkpoint primitives (per-subgraph postcondition predicates).

A :class:`Checkpoint` wraps one predicate over a :class:`World` snapshot;
:func:`evaluate_checkpoint` runs it with eval-error capture and optional
diagnostics; :func:`load_checkpoints` loads a sidecar ``checkpoints/<sg>.py``
module exposing ``CHECKPOINTS: list[Checkpoint]``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gap.runtime.verify.world import World

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Checkpoint:
    """One postcondition predicate over a :class:`World` snapshot.

    Authored by the subgraph builder via ``Subgraph.add_checkpoint(...)``
    and materialized to a sidecar ``checkpoints/<sg>.py`` module the
    harness imports. ``predicate`` is called once per visit to the owning
    subgraph against the post-exit ``World`` snapshot.

    ``validate=True`` checkpoints are hard postconditions: they contribute
    to coverage metrics and drive triage, and are the ones a real run
    enforces. ``validate=False`` checkpoints are probes — they surface in
    the feedback prompt but never gate downstream subgraphs and are never
    enforced.
    """

    name: str
    subgraph: str
    predicate: Callable[..., bool]
    """``predicate(world)`` (1-arg) for the classic privileged-only check
    or ``predicate(world, outputs)`` (2-arg) for predicates that compare
    a subgraph's bound outputs (the ``set_outputs(...)`` dict) against
    the privileged ``World``. The evaluator introspects the callable's
    arity and dispatches accordingly so old single-arg predicates keep
    working unchanged."""
    rationale: str = ""
    validate: bool = True
    weight: float = 1.0
    diagnostics_fn: Callable[..., dict] | None = None
    """Same arity rules as ``predicate``."""


@dataclass
class CheckpointResult:
    """Outcome of one checkpoint evaluation."""

    name: str
    subgraph: str
    passed: bool
    eval_error: str | None = None
    diagnostics: dict | None = None
    eval_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subgraph": self.subgraph,
            "passed": self.passed,
            "eval_error": self.eval_error,
            "diagnostics": self.diagnostics,
            "eval_time_s": self.eval_time_s,
        }


def evaluate_checkpoint(
    checkpoint: Checkpoint,
    world: World,
    outputs: dict | None = None,
) -> CheckpointResult:
    """Evaluate one :class:`Checkpoint` against a :class:`World` snapshot.

    ``outputs`` is the subgraph's ``bound_outputs`` dict (the values
    declared via ``Subgraph.set_outputs(...)``, resolved at the moment
    the subgraph exits). It is forwarded to predicates whose signature
    is ``(world, outputs)`` — i.e. predicates that compare authored
    workflow outputs (e.g. a perception OBB) against the privileged
    ``World``. Classic 1-arg predicates (``(world,) -> bool``) keep
    working unchanged; arity is detected per-call via
    :func:`inspect.signature`.

    A predicate that raises is reported as ``passed=False, eval_error=...``
    so the failure surfaces in the next iter's feedback prompt instead of
    crashing the harness. A diagnostics function that raises is treated
    as a missing diagnostic (``diagnostics=None``) — the predicate result
    is still authoritative.
    """
    t0 = time.perf_counter()
    outputs = outputs or {}
    passed = False
    eval_error: str | None = None
    try:
        passed = bool(_call_with_optional_outputs(
            checkpoint.predicate, world, outputs,
        ))
    except Exception as exc:
        eval_error = f"{type(exc).__name__}: {exc}"

    diagnostics: dict | None = None
    if checkpoint.diagnostics_fn is not None:
        try:
            diag = _call_with_optional_outputs(
                checkpoint.diagnostics_fn, world, outputs,
            )
            if isinstance(diag, dict):
                diagnostics = diag
        except Exception as exc:
            logger.debug(
                "checkpoint %s.%s diagnostics raised: %s",
                checkpoint.subgraph, checkpoint.name, exc,
            )
    return CheckpointResult(
        name=checkpoint.name,
        subgraph=checkpoint.subgraph,
        passed=passed,
        eval_error=eval_error,
        diagnostics=diagnostics,
        eval_time_s=time.perf_counter() - t0,
    )


def _call_with_optional_outputs(fn: Callable, world: World, outputs: dict) -> Any:
    """Dispatch a checkpoint callable on its declared arity.

    1-arg: ``fn(world)`` — the historical signature, kept working.
    2-arg: ``fn(world, outputs)`` — the new shape, lets predicates
    compare workflow-edge outputs (the subgraph's ``set_outputs(...)``
    dict) against the privileged ``World``. ``*args`` callables are
    treated as 1-arg by default — pass ``outputs`` as the second
    positional only when the signature explicitly declares a second
    positional parameter.
    """
    try:
        import inspect
        sig = inspect.signature(fn)
        # Count positional-capable parameters (POSITIONAL_OR_KEYWORD or
        # POSITIONAL_ONLY); ignore *args/**kwargs/keyword-only.
        positional_kinds = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }
        positional = [
            p for p in sig.parameters.values() if p.kind in positional_kinds
        ]
        if len(positional) >= 2:
            return fn(world, outputs)
        return fn(world)
    except (TypeError, ValueError):
        # Builtins or C-level callables sometimes refuse introspection.
        # Fall back to the historical 1-arg signature.
        return fn(world)


def load_checkpoints(path: str | Any) -> list[Checkpoint]:
    """Load a checkpoint module written by ``Subgraph.dump_checkpoints_module``.

    The module is expected to expose ``CHECKPOINTS: list[Checkpoint]``.
    Re-executed each call so lambdas re-close over a fresh scope (the
    harness can reload after every workflow rewrite without stale
    references to torn-down Subgraph objects).
    """
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint module {p} not found")
    source = p.read_text()
    sandbox: dict[str, Any] = {"__name__": f"_checkpoints_{p.stem}"}
    exec(compile(source, str(p), "exec"), sandbox)
    cps = sandbox.get("CHECKPOINTS")
    if not isinstance(cps, list):
        raise ValueError(
            f"checkpoint module {p} did not export CHECKPOINTS as a list"
        )
    out: list[Checkpoint] = []
    for c in cps:
        if not isinstance(c, Checkpoint):
            raise ValueError(
                f"checkpoint module {p}: CHECKPOINTS[{len(out)}] is not a "
                f"Checkpoint (got {type(c).__name__})"
            )
        out.append(c)
    return out


__all__ = [
    "Checkpoint",
    "CheckpointResult",
    "evaluate_checkpoint",
    "load_checkpoints",
]
