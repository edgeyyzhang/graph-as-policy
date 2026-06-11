"""Test plumbing for the skill-bundle loader.

``gap.skills`` codes against :mod:`gap.tools` (the ``@tool`` decorator +
the ``gap.tools._registry._PENDING_TOOLS`` drain list and
``gap.tools.schema.extract_schema``), which is ported in parallel. When it
isn't importable yet, install an interface-equivalent stub in
``sys.modules`` so the loader tests exercise the real integration seam.
The stub is skipped automatically once the real ``gap.tools`` lands.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import typing
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import ModuleType
from typing import Any


def _real_gap_tools_present() -> bool:
    try:
        return importlib.util.find_spec("gap.tools") is not None
    except ModuleNotFoundError:
        return False


def _install_gap_tools_stub() -> None:
    if _real_gap_tools_present() or "gap.tools" in sys.modules:
        return

    tools = ModuleType("gap.tools")
    tools.__path__ = []  # type: ignore[attr-defined]  # mark as package
    registry = ModuleType("gap.tools._registry")
    registry._PENDING_TOOLS = []  # type: ignore[attr-defined]

    def tool(*, name: str, summary: str, scope: str = "runtime", tags: tuple = ()):
        """Stub of gap.tools.tool — same registration contract."""

        def _wrap(fn):
            registry._PENDING_TOOLS.append({  # type: ignore[attr-defined]
                "name": name,
                "summary": summary,
                "scope": scope,
                "tags": tuple(tags),
                "fn": fn,
            })
            return fn

        return _wrap

    registry.tool = tool  # type: ignore[attr-defined]
    tools._registry = registry  # type: ignore[attr-defined]
    tools.tool = tool  # type: ignore[attr-defined]

    schema = ModuleType("gap.tools.schema")

    @dataclass
    class FieldInfo:
        name: str
        python_type: Any
        type_str: str
        required: bool
        default: Any = None
        description: str = ""

    @dataclass
    class UnitSchema:
        name: str
        description: str = ""
        inputs: dict[str, FieldInfo] = dc_field(default_factory=dict)
        outputs: dict[str, FieldInfo] = dc_field(default_factory=dict)

    def extract_schema(module: Any, meta: Any | None = None) -> UnitSchema:
        """Trimmed stub of gap.tools.schema.extract_schema (same shape)."""
        run_fn = getattr(module, "run", None)
        if run_fn is None or not callable(run_fn):
            raise ValueError(
                f"Module {getattr(module, '__name__', module)} has no callable run()"
            )
        sig = inspect.signature(run_fn)
        hints = typing.get_type_hints(run_fn)
        if meta is None:
            meta = getattr(module, "_meta", None)

        inputs: dict[str, FieldInfo] = {}
        for pname, param in sig.parameters.items():
            if pname in ("ctx", "self"):
                continue
            has_default = param.default is not inspect.Parameter.empty
            hint = hints.get(pname, Any)
            inputs[pname] = FieldInfo(
                name=pname,
                python_type=hint,
                type_str=getattr(hint, "__name__", str(hint)),
                required=not has_default,
                default=param.default if has_default else None,
            )

        outputs: dict[str, FieldInfo] = {}
        return_hint = hints.get("return")
        if return_hint is not None and hasattr(return_hint, "__annotations__"):
            for fname, ftype in typing.get_type_hints(return_hint).items():
                outputs[fname] = FieldInfo(
                    name=fname,
                    python_type=ftype,
                    type_str=getattr(ftype, "__name__", str(ftype)),
                    required=True,
                )

        return UnitSchema(
            name=getattr(module, "__name__", "unknown"),
            description=getattr(meta, "description", "") if meta else "",
            inputs=inputs,
            outputs=outputs,
        )

    schema.FieldInfo = FieldInfo  # type: ignore[attr-defined]
    schema.UnitSchema = UnitSchema  # type: ignore[attr-defined]
    schema.extract_schema = extract_schema  # type: ignore[attr-defined]

    tools.schema = schema  # type: ignore[attr-defined]
    sys.modules["gap.tools"] = tools
    sys.modules["gap.tools._registry"] = registry
    sys.modules["gap.tools.schema"] = schema


_install_gap_tools_stub()
