"""PromptAssembler — composes codegen subagent system prompts.

Each prompt is assembled from five sections:

  [agent .md body]                          (gap/agent/prompts/<name>.md)
+ [included shared docs]                    (_workflow_spec.md, ...)
+ [per-skill content for subgraph_agent]    (SKILL.md body + canonical scripts)
+ [tool catalog]                            (gap.tools descriptors: name —
                                             summary + typed I/O fields)
+ [per-call context]                        (subgraph spec, upstream outputs)

The tool catalogs are rendered from :class:`gap.tools.ToolRegistry`
descriptors (UnitSchema field tables) — the coordinator sees the flat
runtime catalog; the subgraph_agent sees only the chosen skill's
``allowed_tools``. The coordinator's Skills catalog lists ONLY
``kind == "skill"`` bundles — tool bundles surface through the flat tool
catalog instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from gap.skills import SkillInfo, SkillsRegistry
from gap.tools import ToolDescriptor, ToolRegistry
from gap.tools.schema import FieldInfo

from ._registry import AgentRegistry, AgentSpec, read_include


@dataclass
class AssembledPrompt:
    system_prompt: str
    """The full system message text."""

    bound_codegen_tools: list[ToolDescriptor] = field(default_factory=list)
    """The whitelist of codegen-scope tools the LLM client should bind via
    the provider ``tools=[...]`` parameter. Per-agent from
    ``frontmatter.tools``."""

    user_prompt: str = ""
    """Optional initial user message. Used by agents whose per-call
    context is too large to inline in the system prompt — e.g. the
    checkpoint_agent's per-subgraph builder sources + bound outputs
    tables. ``""`` means the runner supplies its own user message
    (the common case)."""


class PromptAssembler:
    """Compose agent system prompts from skill bundles, the flat tool
    catalog, and shared docs."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        skills_registry: SkillsRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        self.agents = agent_registry
        self.skills = skills_registry
        self.tools = tool_registry

    def assemble_coordinator(self, task_prompt: str) -> AssembledPrompt:
        """Build the coordinator's system prompt: agent body + shared specs +
        Available Skills table + flat tool catalog + task description."""
        spec = self.agents.get("coordinator")
        parts: list[str] = [spec.body, ""]
        parts.extend(self._render_includes(spec))
        parts.append(self._render_skills_catalog())
        parts.append(self._render_tools_catalog())
        parts.append("## Task\n")
        parts.append(task_prompt)
        bound = self._bind_codegen_tools(spec)
        return AssembledPrompt(system_prompt="\n".join(parts), bound_codegen_tools=bound)

    def assemble_subgraph_agent(
        self,
        skill_name: str,
        subgraph_spec: dict,
        upstream_outputs: dict[str, dict[str, str]] | None = None,
    ) -> AssembledPrompt:
        """Build the subgraph_agent prompt for a specific skill instance.

        Args:
            skill_name: The skill bundle the subgraph uses (chosen by the
                coordinator).
            subgraph_spec: Per-call dict with the subgraph's ``name``,
                ``description``, ``inputs``, ``outputs``, ``exit``,
                ``context``.
            upstream_outputs: ``{upstream_subgraph_name: {output_name: type}}``
                map of outputs already declared by earlier subgraphs.
        """
        spec = self.agents.get("subgraph_agent")
        skill_info = self.skills.get(skill_name)

        parts: list[str] = [spec.body, ""]
        parts.extend(self._render_includes(spec))
        parts.append(self._render_skill_body(skill_info))
        parts.append(self._render_filtered_tools_for_skill(skill_info))
        parts.append(self._render_subgraph_context(subgraph_spec, skill_info, upstream_outputs))

        bound = self._bind_codegen_tools(spec)
        return AssembledPrompt(system_prompt="\n".join(parts), bound_codegen_tools=bound)

    def assemble_coder(self, coder_spec: dict) -> AssembledPrompt:
        """Build the coder's prompt: agent body + script contract + per-call spec."""
        spec = self.agents.get("coder")
        parts: list[str] = [spec.body, ""]
        parts.extend(self._render_includes(spec))
        parts.append("## Script spec\n")
        parts.append(f"- **name:** `{coder_spec['name']}`")
        parts.append(f"- **signature:** `{coder_spec['signature']}`")
        parts.append(f"- **purpose:** {coder_spec['purpose']}")
        if coder_spec.get("body_hint"):
            parts.append(f"- **body hint:**\n\n```\n{coder_spec['body_hint']}\n```")
        parts.append("")
        parts.append(
            f"Emit a single ` ```python:scripts/{coder_spec.get('subgraph_name', '_global')}/"
            f"{coder_spec['name']}.py` fenced block."
        )
        bound = self._bind_codegen_tools(spec)
        return AssembledPrompt(system_prompt="\n".join(parts), bound_codegen_tools=bound)

    def assemble_checkpoint_agent(
        self,
        workflow_dict: dict,
        sg_sources: dict[str, str],
        task_prompt: str,
        subgraphs: dict[str, Any],
    ) -> AssembledPrompt:
        """Build the checkpoint_agent prompt — one whole-workflow call.

        System prompt = agent body + ``_checkpoints_api.md`` include +
        the per-skill canonical-checkpoint table.
        User prompt = task description + per-subgraph builder source +
        per-subgraph bound-outputs table.
        """
        spec = self.agents.get("checkpoint_agent")
        system_parts: list[str] = [spec.body, ""]
        system_parts.extend(self._render_includes(spec))
        system_parts.append(
            self._render_canonical_checkpoints_for_workflow(workflow_dict, subgraphs)
        )

        user_parts: list[str] = [
            "## Task",
            task_prompt.strip(),
            "",
            "## Workflow subgraphs",
            "",
            "Each subgraph below was just generated by `subgraph_agent`. "
            "Attach checkpoints to it via "
            "`subgraphs[\"<sg>\"].add_checkpoint(...)` in your single "
            "output block.",
            "",
        ]
        sgs_dict = workflow_dict.get("subgraphs") or {}
        for sg_name in sg_sources.keys():
            sg = subgraphs.get(sg_name)
            sg_meta = sgs_dict.get(sg_name) or {}
            skill = sg_meta.get("skill") or "—"
            user_parts.append(f"### `{sg_name}`  (skill: `{skill}`)")
            user_parts.append("")
            user_parts.append("Builder source (already authored, do NOT re-emit):")
            user_parts.append("```python")
            user_parts.append(sg_sources[sg_name].rstrip())
            user_parts.append("```")
            user_parts.append("")
            user_parts.append(
                "Bound outputs (the keys readable as `o[...]` in 2-arg predicates):"
            )
            user_parts.append(self._render_bound_outputs_table(sg, sg_meta))
            user_parts.append("")

        user_parts.append("")
        user_parts.append(
            "Emit ONE ```python``` block containing only "
            "`subgraphs[\"<sg>\"].add_checkpoint(...)` statements — at "
            "least one with `validate=True` per subgraph. Prefer 2-arg "
            "`lambda w, o: ...` predicates whenever the subgraph has a "
            "bound output to compare against privileged ground truth."
        )

        bound = self._bind_codegen_tools(spec)
        return AssembledPrompt(
            system_prompt="\n".join(system_parts),
            user_prompt="\n".join(user_parts),
            bound_codegen_tools=bound,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _render_includes(self, spec: AgentSpec) -> list[str]:
        out: list[str] = []
        for inc in spec.includes:
            try:
                out.append(read_include(inc))
                out.append("")
            except FileNotFoundError:
                continue
        return out

    def _render_skills_catalog(self) -> str:
        """The coordinator's Available Skills table — ONLY kind=="skill"
        bundles. Tool bundles never own subgraphs; their functions appear
        in the flat tool catalog instead."""
        visible = sorted(s.name for s in self.skills.list_skills(kind="skill"))

        lines = ["## Available Skills\n"]
        if not visible:
            lines.append("(no skill bundles registered)")
            return "\n".join(lines)
        lines.append(
            "| Skill | Tags | Exit conditions | Required inputs | Produces outputs |"
        )
        lines.append("|-------|------|-----------------|-----------------|------------------|")
        for name in visible:
            info = self.skills.get(name)
            tags = ", ".join(info.meta.tags) or "—"
            ec = ", ".join(info.meta.exit_conditions.keys()) or "—"
            ins = ", ".join(f"{k}: {v}" for k, v in info.meta.required_inputs.items()) or "—"
            outs = ", ".join(f"{k}: {v}" for k, v in info.meta.produces_outputs.items()) or "—"
            lines.append(f"| `{name}` | {tags} | {ec} | {ins} | {outs} |")
        lines.append("")
        # Also emit each skill's description, so the coordinator can pick.
        for name in visible:
            info = self.skills.get(name)
            lines.append(f"### `{name}`\n")
            lines.append(info.meta.description.strip())
            lines.append("")
        return "\n".join(lines)

    def _render_tool_catalog(self, tool_names: Iterable[str]) -> list[str]:
        """Per-tool entry: ``name — summary`` + typed input/output field
        lines rendered from the descriptor's UnitSchema."""
        lines: list[str] = []
        for name in tool_names:
            try:
                desc = self.tools.get(name)
            except KeyError:
                lines.append(
                    f"- `{name}` — (not registered at codegen time; schema unavailable)"
                )
                continue
            summary = (desc.summary or "").strip()
            lines.append(f"- `{name}` — {summary}" if summary else f"- `{name}`")
            ins = self._render_fields(desc.schema.inputs.values(), show_defaults=True)
            outs = self._render_fields(desc.schema.outputs.values())
            lines.append(f"  - inputs: {ins or '(none)'}")
            lines.append(f"  - outputs: {outs or '(none)'}")
        return lines

    @staticmethod
    def _render_fields(fields: Iterable[FieldInfo], *, show_defaults: bool = False) -> str:
        parts: list[str] = []
        for f in fields:
            entry = f"{f.name}: {f.type_str}"
            if show_defaults and not f.required:
                entry += f" = {f.default!r}"
            parts.append(entry)
        return ", ".join(parts)

    def _render_tools_catalog(self) -> str:
        """The coordinator's flat catalog: every runtime tool with schema."""
        lines = ["## Available Tools (flat catalog)\n"]
        runtime = sorted(self.tools.runtime_tools().keys())
        if not runtime:
            lines.append("(no runtime tools registered)")
            return "\n".join(lines)
        lines.extend(self._render_tool_catalog(runtime))
        lines.append("")
        return "\n".join(lines)

    def _render_skill_body(self, info: SkillInfo) -> str:
        """Emit the chosen skill's SKILL.md body verbatim plus a header."""
        lines = [f"## Skill in scope: `{info.name}`\n"]
        body = info.meta.body.strip()
        if body:
            lines.append(body)
            lines.append("")
        if info.canonical_scripts:
            lines.append("### Canonical scripts (you may emit as `type: script` states)\n")
            lines.append("| Logical name | Path | Inputs | Outputs |")
            lines.append("|--------------|------|--------|---------|")
            for sname, sinfo in info.canonical_scripts.items():
                ins = ", ".join(f"{n}: {f.type_str}" for n, f in sinfo.schema.inputs.items()) or "—"
                outs = ", ".join(f"{n}: {f.type_str}" for n, f in sinfo.schema.outputs.items()) or "—"
                lines.append(f"| `{sname}` | `{sinfo.bundle_relative}` | {ins} | {outs} |")
            lines.append("")
        return "\n".join(lines)

    def _render_filtered_tools_for_skill(self, info: SkillInfo) -> str:
        """Show only the tools the skill's frontmatter authorizes, with
        their typed schemas (connector tools render from the static
        codegen catalog)."""
        allowed = list(info.meta.allowed_tools)
        lines = ["## Tools available in this subgraph\n"]
        if not allowed:
            # No whitelist — fall back to the full runtime catalog.
            allowed = sorted(self.tools.runtime_tools().keys())
        if not allowed:
            lines.append("(none — see report_missing_capability)")
            return "\n".join(lines)
        lines.extend(self._render_tool_catalog(allowed))
        lines.append("")
        return "\n".join(lines)

    def _render_subgraph_context(
        self,
        spec: dict,
        skill_info: SkillInfo,
        upstream_outputs: dict[str, dict[str, str]] | None,
    ) -> str:
        lines = [f"## Your subgraph: `{spec.get('name', '<unnamed>')}`\n"]
        if "description" in spec:
            lines.append(spec["description"])
            lines.append("")

        if "context" in spec and spec["context"]:
            lines.append("### Context")
            for k, v in spec["context"].items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        if spec.get("inputs"):
            lines.append("### Bound inputs")
            lines.append("Reference these as `{\"$ref\": \"in.<name>\"}`:\n")
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for n, t in spec["inputs"].items():
                lines.append(f"| {n} | `{t}` |")
            lines.append("")

        if spec.get("outputs"):
            lines.append("### Required outputs")
            lines.append(
                "Bind these from internal state fields via "
                "`{\"$ref\": \"<state>.<field>\"}` (or `{\"$ref\": \"<state>\"}`"
                " for whole-output returns):\n"
            )
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for n, t in spec["outputs"].items():
                lines.append(f"| {n} | `{t}` |")
            lines.append("")

        if skill_info.meta.exit_conditions:
            lines.append("### Required end states")
            lines.append(
                "Your subgraph must contain exactly these end states. Each "
                "success state is a `noop` marker reached via the edge "
                "list; the failure state is the `on_error` symbol.\n"
            )
            lines.append("| End state | Meaning |")
            lines.append("|-----------|---------|")
            for k, v in skill_info.meta.exit_conditions.items():
                lines.append(f"| `{k}` | {v} |")
            lines.append("")

        if upstream_outputs:
            lines.append("### Upstream outputs (already produced by earlier subgraphs)")
            lines.append("| Subgraph | Output | Type |")
            lines.append("|----------|--------|------|")
            for sg, outs in upstream_outputs.items():
                for n, t in outs.items():
                    lines.append(f"| {sg} | {n} | `{t}` |")
            lines.append("")

        return "\n".join(lines)

    def _bind_codegen_tools(self, spec: AgentSpec) -> list[ToolDescriptor]:
        """Resolve the agent's frontmatter ``tools:`` list into ToolDescriptors."""
        codegen = self.tools.codegen_tools()
        out: list[ToolDescriptor] = []
        for name in spec.tools:
            if name in codegen:
                out.append(codegen[name])
        return out

    # ------------------------------------------------------------------
    # checkpoint_agent helpers
    # ------------------------------------------------------------------

    def _render_bound_outputs_table(self, sg: Any, sg_meta: dict) -> str:
        """Markdown table of (output_name, type) for one subgraph.

        The output names come from ``sg._outputs`` (the in-memory
        Subgraph builder object); types come from the workflow's
        declared outputs on its subgraph entry if available.
        """
        out_keys = list((getattr(sg, "_outputs", None) or {}).keys())
        if not out_keys:
            return (
                "*(this subgraph declares no `set_outputs(...)` — use "
                "1-arg `lambda w: ...` predicates only.)*"
            )
        declared_types = sg_meta.get("outputs") or {}
        if isinstance(declared_types, dict):
            type_by_name = {k: str(v) for k, v in declared_types.items()}
        else:
            type_by_name = {}
        lines = ["| Output key | Type |", "|------------|------|"]
        for k in out_keys:
            lines.append(f"| `{k}` | `{type_by_name.get(k, '?')}` |")
        return "\n".join(lines)

    def _render_canonical_checkpoints_for_workflow(
        self, workflow_dict: dict, subgraphs: dict[str, Any],
    ) -> str:
        """Per-skill canonical postcondition guidance for the
        checkpoint_agent prompt. Lists the 2-arg shape FIRST and the
        1-arg fallback second, so output-anchored predicates are the
        default choice."""
        sgs_dict = workflow_dict.get("subgraphs") or {}
        seen_skills: set[str] = set()
        skill_order: list[str] = []
        for sg_name in subgraphs.keys():
            skill = (sgs_dict.get(sg_name) or {}).get("skill")
            if skill and skill not in seen_skills:
                seen_skills.add(skill)
                skill_order.append(skill)

        lines = ["## Canonical checkpoint shapes by skill\n"]
        lines.append(
            "For each subgraph in this workflow, the shapes below are the "
            "*recommended* postconditions. **2-arg shapes are listed first; "
            "use them when the subgraph has a matching bound output.** 1-arg "
            "shapes are the fallback when no output is suitable. The `o[...]` "
            "values are gap.types TypedDicts — index them with string keys "
            "(`o['grasp_pose']['position']['z']`), never attribute access.\n"
        )
        for skill in skill_order:
            shapes = _CANONICAL_CHECKPOINTS_BY_SKILL.get(skill) or []
            if not shapes:
                continue
            lines.append(f"### `{skill}`\n")
            for s in shapes:
                lines.append(f"- **shape**: `{s['shape']}`")
                lines.append(f"  - *rationale*: {s['rationale']}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical postcondition shapes per skill — the 2-arg-first guidance that
# replaces the stripped `## Checkpoints` sections in SKILL.md files. Each
# entry's `shape` is a one-line predicate hint; `rationale` is the one-line
# postcondition meaning. The LLM picks names + body names per task; this
# table just nudges the *shape* of the predicate. NOTE: workflow outputs
# (`o[...]`) are gap.types TypedDicts → dict subscripts; the privileged
# `World`/`Body` side keeps attribute access.
# ---------------------------------------------------------------------------

_PERCEPTION_SHAPES: list[dict[str, str]] = [
    {
        "shape": "lambda w, o: abs(o['<name>_obb']['center']['x'] - "
                 "w.body('<name>').position[0]) < 0.03 and ...",
        "rationale": "perception OBB center within 3cm of the privileged "
                     "body pose (output-anchored)",
    },
    {
        "shape": "lambda w: w.has_body('<name>')",
        "rationale": "the target body exists in the scene (presence-only "
                     "fallback when no OBB output is bound)",
    },
]

_GRASP_SHAPES: list[dict[str, str]] = [
    {
        "shape": "lambda w, o: w.body('<target>').aabb_lower[0] < "
                 "o['ee_pose_at_grasp']['position']['x'] < "
                 "w.body('<target>').aabb_upper[0] and ... (also for y)",
        "rationale": "the EE was over the target's xy footprint at grasp "
                     "(output-anchored)",
    },
    {
        "shape": "lambda w, o: o['grasp_pose']['position']['z'] > 0.01",
        "rationale": "the *computed* grasp pose is above the table — "
                     "catches a low-profile-object bug where the grasp "
                     "candidate generator emits a grasp z below the "
                     "tabletop. REQUIRED for every grasp subgraph that "
                     "binds `grasp_pose` as an output. Use "
                     "`o['grasp_pose']` (the planner target), NOT "
                     "`o['ee_pose_at_grasp']` (the live EE pose, which "
                     "is the approach pose at z≈0.35 and would pass "
                     "trivially).",
    },
    {
        "shape": "lambda w: w.body('<target>').is_grasped()",
        "rationale": "the target body is in contact with a robot link "
                     "after close",
    },
]

_CANONICAL_CHECKPOINTS_BY_SKILL: dict[str, list[dict[str, str]]] = {
    "perceiving-objects": _PERCEPTION_SHAPES,
    "perceiving-objects-oneshot": _PERCEPTION_SHAPES,
    "perceiving-objects-multiview": _PERCEPTION_SHAPES,
    # A SUBPART (handle, rim, spout) has NO ground-truth body of its own —
    # only the parent object does. Comparing the subpart OBB to
    # `w.body('<parent>').position` is therefore GUARANTEED to fail: the
    # subpart is offset from the parent body origin by ~object-radius
    # (e.g. a pan handle sits ~10-15cm from the pan-body centroid). Such a
    # checkpoint blinds the whole feedback loop — it is the perception
    # subgraph's only gate, so it pins triage on the perception subgraph
    # forever even when perception is actually fine. Use a NON-privileged
    # sanity predicate over the bound OBB output instead (no
    # `w.body(...)`); true subpart-localization correctness is validated
    # implicitly by the downstream grasp checkpoint.
    "perceiving-object-parts": [
        {
            "shape": "lambda w, o: 0.0 < o['<name>_obb']['extent']['x'] < 0.5 "
                     "and 0.0 < o['<name>_obb']['extent']['y'] < 0.5 and 0.0 < "
                     "o['<name>_obb']['extent']['z'] < 0.5 and "
                     "o['<name>_obb']['center']['z'] > 0.0",
            "rationale": "perceived subpart OBB is finite, non-degenerate "
                         "and above the table (NON-privileged sanity — a "
                         "subpart has no GT body to match; do NOT compare "
                         "to w.body('<parent>').position, that is "
                         "guaranteed-false and blinds the loop)",
        },
    ],
    "grasping-with-planner": _GRASP_SHAPES,
    "grasping-short-axis": _GRASP_SHAPES,
    "grasping-direct-ik": _GRASP_SHAPES,
    "transporting-objects": [
        {
            "shape": "lambda w, o: w.body('<container>').cavity_lower[0] < "
                     "o['drop_position']['x'] < "
                     "w.body('<container>').cavity_upper[0] and ... (also for y)",
            "rationale": "planned drop xy lands inside the container's "
                         "privileged cavity AABB (output-anchored)",
        },
        {
            "shape": "lambda w, o: o['drop_position']['z'] > "
                     "w.body('<container>').cavity_lower[2] - 0.01",
            "rationale": "planned drop z is at or above the container floor "
                         "(catches mis-computed drop heights before release).",
        },
        {
            "shape": "lambda w: w.body('<target>').is_in(w.body('<container>'))",
            "rationale": "target settled inside the container cavity after "
                         "release",
        },
    ],
    "running-policies": [
        {
            "shape": "lambda w: w.body('<target>').is_in(w.body('<container>'))",
            "rationale": "task-level success predicate over the privileged "
                         "world (policy black-box)",
        },
    ],
    "tracking-objects": [
        {
            "shape": "lambda w: w.has_body('<target>')",
            "rationale": "the tracked body is still present in the scene at "
                         "subgraph exit",
        },
    ],
}
