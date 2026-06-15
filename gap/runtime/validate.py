"""v3 structural validator for workflow graphs.

Runs structural rules against a parsed v3 ``Workflow`` dataclass.
Returns a list of ``ValidationIssue`` objects; the caller decides
whether to hard-fail.

Rules (v3):

  Workflow level:
    W1.  ``version == 3``.
    W2.  At least one edge from ``START``.
    W3.  Every edge endpoint is either ``START``, ``END``, a declared
         node, or a target reachable through ``conditional_edges``.
    W4.  Every non-``END`` node is reachable from ``START``.
    W5.  Every node has at least one outgoing edge or is the source of
         a ``conditional_edges`` entry, or is type=end, or is streaming.
    W6.  ``conditional_edges`` source must be a declared node; every
         mapping target must be a declared node or ``END``.
    W7.  Every ``subgraph`` node's ``ref`` must resolve to a declared
         subgraph.
    W8.  Every reachable subgraph's declared inputs must be available
         either via the executor-injected ``observation_stream`` or
         from an upstream subgraph that declares an output of the same
         name (rule equivalent to v2 §6 R24).

  Subgraph level:
    S1.  ``edges`` from ``START`` to at least one declared node.
    S2.  Every non-``END`` node reachable from ``START``.
    S3.  Streaming nodes must have **no outgoing edges** and no
         conditional-edge entry.
    S4.  Streaming-flag/skill-contract consistency: a node with
         ``streaming: true`` must invoke a skill whose contract has
         ``streaming: true``, and vice versa (when skill_registry is
         provided).
    S5.  ``$ref`` heads must reference declared nodes (or ``in.<name>``
         for declared subgraph inputs / observation_stream).
    S6.  Subgraph ``outputs`` bind to declared nodes (not ``END``).
    S7.  ``exit.success_values`` must be a non-empty list.
    S8.  Conditional-edges from a non-router source must declare a
         ``router_field`` (the field on the source's output to switch
         on); from a router source ``router_field`` is null.
    S9.  ``on_error``, when set, must not be the name of a declared
         node — failure exits surface only via the on_error symbol.
    S10. ``on_error`` must not appear as any ``conditional_edges``
         mapping target.
    S11. Success-value/router_field consistency: when
         ``exit.router_field`` is null, every entry in
         ``exit.success_values`` must be a declared ``noop`` node;
         when ``router_field`` is set, success values are returned
         field-strings and must not collide with node names.

Declared subgraph input types ("PointCloud", "Se3Pose", ...) resolve
through the :mod:`gap.schema` type registry; an unknown type name
surfaces as an error-level issue. The TypedDict introspection helpers
describe what each node consumes and produces, which is unchanged in v3.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gap_core.errors import ValidationIssue
from gap_core.schema import FieldInfo, type_fields

from .workflow import (
    END,
    OBSERVATION_STREAM_INPUT_NAME,
    RESERVED_INPUT_PSEUDOSTATE,
    START,
    NodeDef,
    Ref,
    SubgraphDef,
    Workflow,
)

logger = logging.getLogger(__name__)

# Type names that accept (or produce) anything — the open-dict escape
# hatch the proto era spelled "google.protobuf.Struct".
_ANY_TYPE_NAMES = frozenset({"Any", "dict"})
_NUMERIC_TYPE_NAMES = frozenset({"int", "float"})
_OBSERVATION_STREAM_TYPE_NAME = "ObservationStream"


# ---------------------------------------------------------------------------
# Schema dataclasses (reused for viz / replay)
# ---------------------------------------------------------------------------


@dataclass
class FieldSchema:
    """One input/output field of a node, normalized for cross-binding checks.

    ``type_str`` is the bare type name used across the graph surface: a
    scalar name ("str", "int", ...), a registered :mod:`gap.types` name
    ("PointCloud"), "dict"/"Any" for open dictionaries, or
    "ObservationStream". An empty ``type_str`` marks an unresolvable
    field (treated leniently). ``is_message`` marks structured
    (TypedDict) values; ``is_repeated`` marks ``list[...]`` fields.
    """

    name: str
    type_str: str
    is_message: bool = False
    is_repeated: bool = False
    required: bool = False


@dataclass
class NodeSchema:
    node_id: str
    inputs: dict[str, FieldSchema] = field(default_factory=dict)
    outputs: dict[str, FieldSchema] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema introspection helpers (typing logic; declared names via gap.schema)
# ---------------------------------------------------------------------------


def _field_info_to_schema(fi: FieldInfo) -> FieldSchema:
    """Map a :class:`gap.schema.FieldInfo` row to the validator's FieldSchema."""
    return FieldSchema(
        name=fi.name,
        type_str=fi.type_str,
        is_message=fi.is_message,
        is_repeated=fi.is_repeated,
        required=fi.required,
    )


def type_name_to_field_schemas(type_name: str) -> dict[str, FieldSchema]:
    """FieldSchema rows for a declared type name (subgraph inputs, viz ports).

    Resolves through the :mod:`gap.schema` registry; raises ``KeyError``
    (with the known names listed) when the name is unknown. Non-TypedDict
    registrations (Mask, scalars) introspect to an empty dict.
    """
    return {fi.name: _field_info_to_schema(fi) for fi in type_fields(type_name)}


def _is_typeddict(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__annotations__") and hasattr(
        tp, "__total__"
    )


def _hint_to_field_schema(name: str, hint: Any) -> FieldSchema:
    origin = typing.get_origin(hint)

    if hint is Any:
        return FieldSchema(name=name, type_str="Any", is_message=True)

    if origin is dict:
        return FieldSchema(name=name, type_str="dict", is_message=True)

    if origin is list:
        args = typing.get_args(hint)
        if not args:
            raise ValueError(
                f"list type hint for '{name}' must be parameterized"
            )
        inner = _hint_to_field_schema(name, args[0])
        return FieldSchema(
            name=name, type_str=inner.type_str,
            is_message=inner.is_message, is_repeated=True,
        )

    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return _hint_to_field_schema(name, args[0])
        raise ValueError(
            f"union type hint for '{name}' must be T | None, got {hint}"
        )

    try:
        from .observation_stream import ObservationStream as _ObservationStream
    except Exception:
        _ObservationStream = None  # type: ignore[assignment]
    if _ObservationStream is not None and isinstance(hint, type) and issubclass(
        hint, _ObservationStream,
    ):
        return FieldSchema(
            name=name, type_str=_OBSERVATION_STREAM_TYPE_NAME, is_message=True,
        )

    if _is_typeddict(hint):
        return FieldSchema(name=name, type_str=hint.__name__, is_message=True)

    if isinstance(hint, type):
        # Scalars (str/int/float/bool/bytes) and bare classes (np.ndarray)
        # carry their class name, matching gap.schema.FieldInfo.type_str.
        return FieldSchema(name=name, type_str=hint.__name__)

    raise ValueError(
        f"cannot map type hint '{hint}' for '{name}' to a field schema"
    )


def build_tool_node_schema(
    node_id: str, node: NodeDef, tool_registry: Any,
) -> NodeSchema:
    """Build a NodeSchema for a `type: tool` node from the tool registry.

    ``tool_registry.get(name)`` returns a descriptor whose ``schema``
    attribute is a ``UnitSchema`` (see :mod:`gap.tools.schema`):
    ``inputs`` / ``outputs`` dicts of ``FieldInfo`` rows carrying the
    original ``python_type`` hint. Fields whose hints cannot be mapped
    degrade to an unresolvable (empty ``type_str``) FieldSchema.
    """
    tool_name = node.tool or ""
    if not tool_name:
        raise ValueError(f"tool node {node_id!r} has no `tool` field")
    descriptor = tool_registry.get(tool_name)

    inputs: dict[str, FieldSchema] = {}
    schema = descriptor.schema
    for fname, finfo in schema.inputs.items():
        try:
            inputs[fname] = _hint_to_field_schema(fname, finfo.python_type)
        except Exception:
            inputs[fname] = FieldSchema(name=fname, type_str="")
    outputs: dict[str, FieldSchema] = {}
    for fname, finfo in schema.outputs.items():
        try:
            outputs[fname] = _hint_to_field_schema(fname, finfo.python_type)
        except Exception:
            outputs[fname] = FieldSchema(name=fname, type_str="")
    return NodeSchema(node_id=node_id, inputs=inputs, outputs=outputs)


def build_script_node_schema(
    node_id: str, node: NodeDef, workflow_dir: Path,
) -> NodeSchema:
    script_rel = node.script or ""
    script_path = workflow_dir / script_rel
    module = _import_script_for_validation(script_path, node_id)

    run_fn = getattr(module, "run", None)
    if run_fn is None or not callable(run_fn):
        raise ValueError(f"script '{script_rel}' missing run() function")

    try:
        hints = typing.get_type_hints(run_fn, globalns=vars(module))
    except Exception as e:
        raise ValueError(f"cannot resolve type hints on run(): {e}") from e

    sig = inspect.signature(run_fn)
    inputs: dict[str, FieldSchema] = {}
    for param_name in sig.parameters:
        if param_name == "ctx":
            continue
        hint = hints.get(param_name)
        if hint is None:
            raise ValueError(
                f"parameter '{param_name}' on run() missing type annotation"
            )
        inputs[param_name] = _hint_to_field_schema(param_name, hint)

    outputs: dict[str, FieldSchema] = {}
    return_hint = hints.get("return")
    if return_hint is not None and return_hint is not type(None):
        if not hasattr(return_hint, "__annotations__"):
            raise ValueError(
                f"run() return type must be a TypedDict or None, got {return_hint}"
            )
        try:
            output_hints = typing.get_type_hints(return_hint)
        except Exception as e:
            raise ValueError(f"cannot resolve return TypedDict hints: {e}") from e
        for field_name, field_hint in output_hints.items():
            outputs[field_name] = _hint_to_field_schema(field_name, field_hint)

    return NodeSchema(node_id=node_id, inputs=inputs, outputs=outputs)


def _import_script_for_validation(path: Path, node_id: str):
    # Flatten dots in node_id so the synthetic module_name doesn't get
    # interpreted as a dotted package path (same fix as the runtime
    # script loader in gap/runtime/nodes.py).
    _safe_node_id = str(node_id).replace(".", "_")
    module_name = f"_gap_validate_{_safe_node_id}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _types_compatible(upstream: FieldSchema, downstream: FieldSchema) -> bool:
    if (upstream.type_str in _ANY_TYPE_NAMES
            or downstream.type_str in _ANY_TYPE_NAMES):
        return True
    if upstream.is_message and downstream.is_message:
        if upstream.type_str and downstream.type_str:
            return upstream.type_str == downstream.type_str
        return True
    if not upstream.is_message and not downstream.is_message:
        if upstream.type_str == downstream.type_str:
            return True
        if (upstream.type_str in _NUMERIC_TYPE_NAMES
                and downstream.type_str in _NUMERIC_TYPE_NAMES):
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# v3 structural validator
# ---------------------------------------------------------------------------


def validate_workflow(
    wf: Workflow,
    *,
    agent_registry: dict[str, dict[str, str]] | None = None,
    skill_registry: Any | None = None,
    tool_registry: Any | None = None,
) -> list[ValidationIssue]:
    """Run structural rules against a parsed v3 Workflow.

    Args:
        wf: Loaded ``Workflow`` dataclass.
        agent_registry: Optional ``{agent_name: exit_conditions_dict}``.
            When provided, used to validate that subgraph success/on_error
            match the agent's declared exit conditions. Codegen pipeline
            supplies this; the executor omits it.
        skill_registry: Optional skill registry. Duck-typed: must support
            ``name in registry`` and ``registry.get(name)`` returning an
            object whose ``meta`` carries the streaming contract field.
            When provided, enables streaming-contract enforcement on
            `type: tool` nodes whose tool name matches a registered
            atomic skill.
        tool_registry: Optional tool registry. Duck-typed: must support
            ``name in registry`` and ``registry.get(name)`` returning a
            descriptor whose ``schema`` attribute is a
            ``gap.tools.schema.UnitSchema``. When provided, enables
            schema introspection on `type: tool` nodes for cross-binding
            type checks. Unknown tools (e.g. `robot.*` connector tools
            not yet registered at validate time) degrade to a
            warning-level issue.
    """
    issues: list[ValidationIssue] = []

    # Workflow-level rules
    issues.extend(_check_workflow_level(wf))

    # Per-subgraph rules + node-schema introspection
    schemas: dict[tuple[str, str], NodeSchema] = {}
    for sg_name, sg in wf.subgraphs.items():
        issues.extend(_check_subgraph_level(
            sg_name, sg, skill_registry, agent_registry,
        ))
        for node_name, node in sg.nodes.items():
            sch = _try_build_node_schema(
                sg_name, node_name, node, wf.workflow_dir,
                skill_registry, tool_registry, issues,
            )
            if sch is not None:
                schemas[(sg_name, node_name)] = sch

    # Cross-subgraph I/O
    issues.extend(_check_cross_subgraph_io(wf, schemas))

    return issues


# ---------------------------------------------------------------------------
# Workflow-level rules
# ---------------------------------------------------------------------------


def _check_workflow_level(wf: Workflow) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if wf.version != 3:
        issues.append(_issue(
            "error", "workflow.version",
            f"unsupported version {wf.version}, expected 3 (W1)",
        ))

    begin_targets = wf.begin_targets()
    if not begin_targets:
        issues.append(_issue(
            "error", "workflow.edges",
            "workflow has no edge from START (W2)",
        ))

    valid_endpoints = {START, END} | set(wf.nodes.keys())
    for src, dst in wf.edges:
        if src not in valid_endpoints:
            issues.append(_issue(
                "error", "workflow.edges",
                f"edge source {src!r} is not a declared node (W3)",
            ))
        if dst not in valid_endpoints:
            issues.append(_issue(
                "error", "workflow.edges",
                f"edge target {dst!r} is not a declared node (W3)",
            ))

    # Subgraph-ref resolution
    for node_name, node in wf.nodes.items():
        if node.type == "subgraph":
            if node.ref not in wf.subgraphs:
                issues.append(_issue(
                    "error", f"workflow.nodes.{node_name}",
                    f"subgraph node references unknown subgraph "
                    f"{node.ref!r} (W7)",
                ))

    # Conditional-edges sources and mapping targets
    for src, ce in wf.conditional_edges.items():
        if src not in wf.nodes:
            issues.append(_issue(
                "error", f"workflow.conditional_edges.{src}",
                f"source {src!r} is not a declared node (W6)",
            ))
        for cond, tgt in ce.mapping.items():
            if tgt not in valid_endpoints:
                issues.append(_issue(
                    "error", f"workflow.conditional_edges.{src}.mapping.{cond}",
                    f"target {tgt!r} is not a declared node (W6)",
                ))

    # Reachability
    reachable = _reachable_from_start(wf.edges, wf.conditional_edges, wf.nodes.keys())
    for node_name, node in wf.nodes.items():
        if node_name not in reachable and node.type != "end":
            issues.append(_issue(
                "error", f"workflow.nodes.{node_name}",
                f"node {node_name!r} is unreachable from START (W4)",
            ))

    # Outgoing-edge requirement
    has_outgoing: set[str] = {src for src, _ in wf.edges} | set(wf.conditional_edges.keys())
    for node_name, node in wf.nodes.items():
        if node.type == "end":
            continue
        if node_name not in has_outgoing:
            issues.append(_issue(
                "error", f"workflow.nodes.{node_name}",
                f"node {node_name!r} has no outgoing edge or conditional "
                f"dispatch (W5)",
            ))

    # At least one end node exists
    if not any(n.type == "end" for n in wf.nodes.values()):
        issues.append(_issue(
            "error", "workflow.nodes",
            "workflow has no end node",
        ))

    return issues


def _reachable_from_start(
    edges: tuple[tuple[str, str], ...],
    cond_edges: dict[str, Any],
    declared: Any,
) -> set[str]:
    out: dict[str, list[str]] = {}
    for src, dst in edges:
        out.setdefault(src, []).append(dst)
    for src, ce in cond_edges.items():
        for tgt in ce.mapping.values():
            out.setdefault(src, []).append(tgt)

    reachable = {START}
    stack = [START]
    while stack:
        cur = stack.pop()
        for nxt in out.get(cur, []):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)
    reachable.discard(START)
    reachable.discard(END)
    return reachable


# ---------------------------------------------------------------------------
# Subgraph-level rules
# ---------------------------------------------------------------------------


def _check_subgraph_level(
    sg_name: str,
    sg: SubgraphDef,
    skill_registry: Any | None,
    agent_registry: dict[str, dict[str, str]] | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    loc = f"subgraphs.{sg_name}"
    valid = {START, END} | set(sg.nodes.keys())

    # S1: edge from START
    starts = [dst for src, dst in sg.edges if src == START]
    if not starts:
        issues.append(_issue(
            "error", f"{loc}.edges",
            "subgraph has no edge from START (S1)",
        ))

    # S3: streaming nodes must have no outgoing edges or conditional entry
    streaming_nodes = {
        n for n, nd in sg.nodes.items() if nd.streaming
    }
    for src, dst in sg.edges:
        if src in streaming_nodes:
            issues.append(_issue(
                "error", f"{loc}.edges",
                f"streaming node {src!r} has outgoing edge to {dst!r}; "
                f"streaming nodes must be pure sources (S3)",
            ))
    for src in sg.conditional_edges:
        if src in streaming_nodes:
            issues.append(_issue(
                "error", f"{loc}.conditional_edges",
                f"streaming node {src!r} has a conditional-edges entry; "
                f"streaming nodes must be pure sources (S3)",
            ))

    # S4: streaming-flag/skill-contract consistency. Atomic skills are
    # invoked via `type: tool` (the runtime auto-registers them as flat-name
    # tools); composite skills are expanded into subgraphs and never appear
    # as nodes. So we look up the skill metadata via the tool name, which
    # for an atomic-skill tool equals the skill name.
    if skill_registry is not None:
        for node_name, node in sg.nodes.items():
            if node.type != "tool":
                continue
            skill_name = node.tool or ""
            if skill_name not in skill_registry:
                # Not an atomic skill (e.g. it's a `robot.*` connector
                # tool); streaming-contract check doesn't apply.
                continue
            try:
                info = skill_registry.get(skill_name)
                meta = getattr(info, "meta", None) or info
                contract = getattr(meta, "contract", None) or {}
                if hasattr(contract, "get"):
                    contract_streaming = bool(contract.get("streaming", False))
                else:
                    contract_streaming = bool(getattr(contract, "streaming", False))
            except Exception:
                contract_streaming = False
            if node.streaming and not contract_streaming:
                issues.append(_issue(
                    "error", f"{loc}.nodes.{node_name}",
                    f"node has streaming=true but skill {skill_name!r}'s "
                    f"contract has streaming=false (S4)",
                ))
            if (not node.streaming) and contract_streaming:
                issues.append(_issue(
                    "error", f"{loc}.nodes.{node_name}",
                    f"skill {skill_name!r}'s contract has streaming=true "
                    f"but the node does not declare streaming=true (S4)",
                ))

    # Edge endpoints declared
    for src, dst in sg.edges:
        if src not in valid:
            issues.append(_issue(
                "error", f"{loc}.edges",
                f"edge source {src!r} not declared (S5)",
            ))
        if dst not in valid:
            issues.append(_issue(
                "error", f"{loc}.edges",
                f"edge target {dst!r} not declared (S5)",
            ))

    # Conditional-edges
    for src, ce in sg.conditional_edges.items():
        if src not in sg.nodes:
            issues.append(_issue(
                "error", f"{loc}.conditional_edges.{src}",
                f"source {src!r} not declared",
            ))
        else:
            src_node = sg.nodes[src]
            if src_node.type == "router":
                if ce.router_field is not None:
                    issues.append(_issue(
                        "error", f"{loc}.conditional_edges.{src}",
                        f"router source {src!r} must have router_field=null "
                        f"(routing function returns target directly) (S8)",
                    ))
            else:
                if ce.router_field is None:
                    issues.append(_issue(
                        "error", f"{loc}.conditional_edges.{src}",
                        f"non-router source {src!r} must declare router_field "
                        f"naming the output field to switch on (S8)",
                    ))
        for cond, tgt in ce.mapping.items():
            if tgt not in valid:
                issues.append(_issue(
                    "error", f"{loc}.conditional_edges.{src}.mapping.{cond}",
                    f"target {tgt!r} not declared (S5)",
                ))

    # S2: reachability
    reachable = _reachable_from_start(
        sg.edges, sg.conditional_edges, sg.nodes.keys(),
    )
    # Streaming nodes reached via START fan-out are considered reachable.
    # A streaming-only graph (just [START, tracker]) would otherwise miss
    # reachability since they have no outgoing edges.
    for node_name in sg.nodes:
        if node_name in reachable:
            continue
        # Streaming nodes that are direct START targets are reachable.
        is_start_target = any(
            src == START and dst == node_name for src, dst in sg.edges
        )
        if is_start_target:
            continue
        issues.append(_issue(
            "error", f"{loc}.nodes.{node_name}",
            f"node {node_name!r} unreachable from START (S2)",
        ))

    # Outgoing-edge requirement (non-streaming, non-end)
    has_outgoing: set[str] = {src for src, _ in sg.edges} | set(sg.conditional_edges.keys())
    for node_name, node in sg.nodes.items():
        if node.type == "end" or node.streaming:
            continue
        if node_name not in has_outgoing:
            issues.append(_issue(
                "error", f"{loc}.nodes.{node_name}",
                f"non-streaming node {node_name!r} has no outgoing edge "
                f"or conditional dispatch",
            ))

    # S6: subgraph outputs
    for out_name, ref in sg.outputs.items():
        head = ref.parts()[0] if ref.parts() else ""
        if head not in sg.nodes:
            issues.append(_issue(
                "error", f"{loc}.outputs.{out_name}",
                f"output binding {ref.path!r} references unknown node "
                f"{head!r} (S6)",
            ))
            continue
        if sg.nodes[head].type == "end":
            issues.append(_issue(
                "error", f"{loc}.outputs.{out_name}",
                f"output binding {ref.path!r} cannot reference an end "
                f"node (S6)",
            ))

    # S7: exit.success_values non-empty
    if not sg.exit.success_values:
        issues.append(_issue(
            "error", f"{loc}.exit",
            "subgraph.exit.success_values must be a non-empty list (S7)",
        ))

    # Declared input types must resolve in the gap.schema type registry
    # (replaces the proto-era message-FQN resolution; a typo'd type name
    # must fail at validate time, not mid-execution).
    for in_name, in_type in sg.inputs.items():
        if in_type.endswith(_OBSERVATION_STREAM_TYPE_NAME):
            continue
        try:
            type_name_to_field_schemas(in_type)
        except KeyError as e:
            issues.append(_issue(
                "error", f"{loc}.inputs.{in_name}",
                str(e.args[0]) if e.args else str(e),
            ))

    # S9: on_error must NOT be a node name (failure exits live only as the
    #     on_error symbol — never as a declared node).
    if sg.on_error is not None and sg.on_error in sg.nodes:
        issues.append(_issue(
            "error", f"{loc}.on_error",
            f"on_error={sg.on_error!r} must not be a declared node — "
            f"failure exits surface only via on_error, not as nodes (S9)",
        ))

    # S10: on_error must NOT appear as any conditional_edges mapping target.
    if sg.on_error is not None:
        for src, ce in sg.conditional_edges.items():
            for cond, tgt in ce.mapping.items():
                if tgt == sg.on_error:
                    issues.append(_issue(
                        "error",
                        f"{loc}.conditional_edges.{src}.mapping.{cond}",
                        f"target {tgt!r} is the subgraph's on_error symbol "
                        f"and cannot be a conditional_edges target — failure "
                        f"exits surface only via on_error (S10)",
                    ))

    # S11: success_values vs router_field consistency.
    if sg.exit.router_field is None:
        # Each success value must be a declared noop node.
        for sv in sg.exit.success_values:
            node = sg.nodes.get(sv)
            if node is None:
                issues.append(_issue(
                    "error", f"{loc}.exit.success_values",
                    f"success value {sv!r} must be a declared noop node "
                    f"when router_field=null (S11)",
                ))
            elif node.type != "noop":
                issues.append(_issue(
                    "error", f"{loc}.exit.success_values",
                    f"success value {sv!r} is declared as type={node.type!r}; "
                    f"must be type=noop when router_field=null (S11)",
                ))
    else:
        # When router_field is set, success values are string field-values
        # the terminal node returns — they must NOT collide with node names.
        for sv in sg.exit.success_values:
            if sv in sg.nodes:
                issues.append(_issue(
                    "error", f"{loc}.exit.success_values",
                    f"success value {sv!r} collides with a declared node; "
                    f"when router_field is set, success values are returned "
                    f"field-strings and must not be node names (S11)",
                ))

    # Agent-registry exit-condition match (codegen-time only).
    # The catalog declares exit_conditions covering both success and failure;
    # we must reconcile against (success_values ∪ {on_error}).
    if agent_registry is not None and sg.skill in agent_registry:
        declared = set(agent_registry[sg.skill].keys())
        actual: set[str] = set(sg.exit.success_values)
        if sg.on_error is not None:
            actual.add(sg.on_error)
        for missing in sorted(declared - actual):
            issues.append(_issue(
                "error", f"{loc}.exit",
                f"missing exit condition {missing!r} declared in "
                f"{sg.skill}.exit_conditions",
            ))
        for extra in sorted(actual - declared):
            issues.append(_issue(
                "error", f"{loc}.exit",
                f"unexpected exit condition {extra!r} not in "
                f"{sg.skill}.exit_conditions",
            ))

    # S5: $ref heads referencing declared nodes / declared inputs
    for node_name, node in sg.nodes.items():
        for in_name, value in node.inputs.items():
            if not isinstance(value, Ref):
                continue
            parts = value.parts()
            if not parts:
                continue
            head = parts[0]
            if head == RESERVED_INPUT_PSEUDOSTATE:
                if len(parts) < 2:
                    issues.append(_issue(
                        "error",
                        f"{loc}.nodes.{node_name}.inputs.{in_name}",
                        f"$ref {value.path!r}: 'in' must be followed by an "
                        f"input name (S5)",
                    ))
                    continue
                bound = parts[1]
                if bound == OBSERVATION_STREAM_INPUT_NAME:
                    continue
                if bound not in sg.inputs:
                    issues.append(_issue(
                        "error",
                        f"{loc}.nodes.{node_name}.inputs.{in_name}",
                        f"$ref {value.path!r}: bound input {bound!r} not "
                        f"declared in subgraph inputs (S5)",
                    ))
                continue
            if head not in sg.nodes:
                issues.append(_issue(
                    "error",
                    f"{loc}.nodes.{node_name}.inputs.{in_name}",
                    f"$ref {value.path!r}: unknown node {head!r} (S5)",
                ))

    return issues


# ---------------------------------------------------------------------------
# Cross-subgraph I/O (W8)
# ---------------------------------------------------------------------------


def _check_cross_subgraph_io(
    wf: Workflow,
    schemas: dict[tuple[str, str], NodeSchema],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    # Build the set of subgraphs reachable from START via subgraph nodes.
    reachable_sgs: set[str] = set()
    out: dict[str, list[str]] = {}
    for src, dst in wf.edges:
        out.setdefault(src, []).append(dst)
    for src, ce in wf.conditional_edges.items():
        for tgt in ce.mapping.values():
            out.setdefault(src, []).append(tgt)
    stack = [START]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in out.get(cur, []):
            if nxt in (START, END) or nxt in seen:
                continue
            node = wf.nodes.get(nxt)
            if node is None:
                continue
            if node.type == "subgraph" and node.ref:
                reachable_sgs.add(node.ref)
            stack.append(nxt)

    # Producers: subgraphs that declare an output of a given name.
    producers: dict[str, list[str]] = {}
    for sg_name in reachable_sgs:
        sg = wf.subgraphs.get(sg_name)
        if sg is None:
            continue
        for out_name in sg.outputs:
            producers.setdefault(out_name, []).append(sg_name)

    # Consumers: each reachable subgraph's declared inputs must have a producer.
    for sg_name in reachable_sgs:
        sg = wf.subgraphs.get(sg_name)
        if sg is None:
            continue
        for in_name, in_type in sg.inputs.items():
            if (
                in_name == OBSERVATION_STREAM_INPUT_NAME
                and in_type.endswith(_OBSERVATION_STREAM_TYPE_NAME)
            ):
                continue
            if in_name not in producers:
                issues.append(_issue(
                    "error",
                    f"subgraphs.{sg_name}.inputs.{in_name}",
                    f"input {in_name!r} (type {in_type!r}) has no upstream "
                    f"producer subgraph that declares an output named "
                    f"{in_name!r} (W8)",
                ))

    return issues


# ---------------------------------------------------------------------------
# Per-node schema build
# ---------------------------------------------------------------------------


def _try_build_node_schema(
    sg_name: str,
    node_name: str,
    node: NodeDef,
    workflow_dir: Path,
    skill_registry: Any | None,
    tool_registry: Any | None,
    issues: list[ValidationIssue],
) -> NodeSchema | None:
    """Introspect the schema for a node so cross-binding type checks can run.

    The validator only sees `type: tool` / `type: script` / `type: noop` /
    `type: router` / `type: subgraph` / `type: end` after the dispatch
    surface was unified (legacy `type: service`/`skill`/`policy` are
    rejected upstream by the workflow loader). Schema introspection routes
    `tool` through `build_tool_node_schema`, which reads the registered
    tool's ``UnitSchema``; tools missing from the registry at validate
    time (e.g. `robot.*` connector tools registered only at execution)
    degrade to a warning-level issue.
    """
    full_id = f"{sg_name}.{node_name}"
    try:
        if node.type == "script":
            return build_script_node_schema(full_id, node, workflow_dir)
        if node.type == "tool":
            if tool_registry is None:
                return None
            tool_name = node.tool or ""
            if not tool_name or tool_name not in tool_registry:
                try:
                    available = ", ".join(sorted(tool_registry._tools.keys()))
                except Exception:
                    available = ""
                issues.append(_issue(
                    "warning",
                    f"subgraphs.{sg_name}.nodes.{node_name}",
                    f"tool {tool_name!r} is not registered; skipping schema "
                    f"introspection."
                    + (f" Available tools: {available[:300]}..." if available else ""),
                ))
                return None
            return build_tool_node_schema(full_id, node, tool_registry)
    except ValueError as e:
        issues.append(_issue(
            "warning",
            f"subgraphs.{sg_name}.nodes.{node_name}",
            f"could not introspect schema: {e}",
        ))
    except Exception as e:
        issues.append(_issue(
            "warning",
            f"subgraphs.{sg_name}.nodes.{node_name}",
            f"could not introspect schema: {e}",
        ))
    return None


def _issue(severity: str, location: str, message: str) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        node_id=location,
        field=None,
        message=message,
    )


# ---------------------------------------------------------------------------
# Legacy v1 compatibility shim — preserved for in-flight callers that
# haven't migrated yet. Returns ``(issues, {})`` to match the old shape.
# ---------------------------------------------------------------------------


def validate_graph(
    workflow: Any,
    workflow_dir: Path,
    skill_registry: Any | None = None,
) -> tuple[list[ValidationIssue], dict[str, NodeSchema]]:
    logger.warning(
        "validate_graph() is a legacy shim; use validate_workflow() directly"
    )
    if isinstance(workflow, Workflow):
        issues = validate_workflow(
            workflow, agent_registry=None, skill_registry=skill_registry,
        )
        return issues, {}
    return [], {}
