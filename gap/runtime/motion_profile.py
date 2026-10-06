"""Invocation-local additive adjustments to exposed motion inputs and gains."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from gap_core.errors import WorkflowValidationError as _WorkflowValidationError


class WorkflowValidationError(_WorkflowValidationError):
    """A profile that cannot be applied to its node: a configuration error.

    ``terminal`` keeps a subgraph from routing it to its ``on_error`` exit as if
    the robot had tried something and failed (see ``gap_core.errors.is_terminal``).
    """

    terminal = True


MotionProfileError = WorkflowValidationError


def _scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise WorkflowValidationError(f"{label} must be a finite scalar")
    return float(value)


@dataclass(frozen=True)
class ControllerGainOffsets:
    kp_offset: float = 0.0
    kd_offset: float = 0.0

    def __post_init__(self):
        for name in ("kp_offset", "kd_offset"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))


@dataclass(frozen=True)
class MotionProfile:
    controller: ControllerGainOffsets | None = None
    input_offsets: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.controller is not None and not isinstance(self.controller, ControllerGainOffsets):
            raise WorkflowValidationError("controller must be ControllerGainOffsets")
        offsets = {}
        for path, value in self.input_offsets.items():
            if not isinstance(path, str) or not path or any(not p for p in path.split('.')):
                raise WorkflowValidationError("Input offset requires a nonempty dotted path")
            offsets[path] = _scalar(value, path)
        object.__setattr__(self, "input_offsets", MappingProxyType(offsets))

    def to_dict(self):
        return dict(controller=None if self.controller is None else dict(
            kp_offset=self.controller.kp_offset, kd_offset=self.controller.kd_offset),
            input_offsets=dict(self.input_offsets))


def prepare_inputs(profile, resolved, fields, *, ctx, resolver=None, validator=None, trace=None, node_id=""):
    """Validate bindings, materialize defaults, and adjust a private input copy.

    An optional primitive-owned ``resolve_motion_profile_inputs(ctx, inputs,
    paths)`` resolves automatic/sentinel values before addition. It must not
    execute motion. Schemas alone cannot infer the semantics of numeric sentinels.
    """
    if profile is None:
        return resolved
    if not isinstance(profile, MotionProfile):
        raise WorkflowValidationError("Profile resolver must return MotionProfile or None")
    nominal = copy.deepcopy(resolved)
    for path in profile.input_offsets:
        root = path.split('.')[0]
        if root not in fields:
            raise WorkflowValidationError(f"{node_id}: unknown profiled input {root!r}")
        if root not in nominal:
            info = fields[root]
            if info.required:
                raise WorkflowValidationError(f"{node_id}: missing profiled input {root!r}")
            nominal[root] = copy.deepcopy(info.default)
    if resolver is not None and profile.input_offsets:
        nominal = resolver(ctx, nominal, tuple(profile.input_offsets))
    effective = copy.deepcopy(nominal)
    for path, offset in profile.input_offsets.items():
        parts = path.split('.')
        target = effective
        try:
            for part in parts[:-1]:
                target = target[part]
            original = target[parts[-1]]
        except (KeyError, TypeError) as exc:
            raise WorkflowValidationError(f"{node_id}: invalid input path {path!r}") from exc
        # Respect the declared domain: JSON may spell a float coordinate as 0.
        # Counts and IDs declared as integers remain outside this interface.
        declared = fields[parts[0]].python_type
        if isinstance(original, bool) or (len(parts) == 1 and declared in (int, bool)):
            raise WorkflowValidationError(f"{node_id}: {path} is not a continuous scalar input")
        target[parts[-1]] = _scalar(_scalar(original, path) + offset, path)
    if validator is not None and profile.input_offsets:
        validator(ctx, effective, tuple(profile.input_offsets))
    if trace is not None:
        trace.record_motion_profile(node_id, profile.to_dict(), nominal, effective)
        trace.record_resolved_inputs(node_id, effective)
    return effective
