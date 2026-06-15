"""Capability probing — "can this tool bundle run here?" and what follows.

Powers ``gap check``. The atomic question is per tool bundle: are its
deps importable, are its declared :class:`~gap.skills.meta.SkillRequires`
met (GPU, env vars), are its weights cached? Skill runnability *derives*
from that: a skill is blocked exactly by the not-ready bundles that own
its ``allowed_tools`` (``robot.*``/``sim.*`` are connector-provided and
always satisfied once a connector is attached).

Everything here is fast and offline: static SKILL.md parses, one
``nvidia-smi`` subprocess, env-var lookups, per-bundle import probes
(bundles lazy-load their models, so importing ``tools.py`` never loads
weights), and the optional filesystem-only ``weights_cached()`` hook.
No model loads, no downloads, no network.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from gap_core.skills.meta import SkillMeta
from .registries import RegistrySet, RegistrySpec
from .validate import load_checkout_extras, validate_checkout

__all__ = [
    "BundleCapability",
    "CheckReport",
    "EnvironmentReport",
    "ProbeResult",
    "SkillRunnability",
    "build_check_report",
    "dep_fix_hint",
    "probe_bundle_import",
    "probe_gpu",
    "probe_llm_providers",
]

ProbeStatus = Literal["ok", "missing", "unknown", "error"]

#: Tool-name prefixes registered by the live connector at runtime — never
#: importable statically, always satisfied once a connector is attached.
CONNECTOR_PREFIXES = ("robot", "sim")


@dataclass
class ProbeResult:
    """One probe's verdict."""

    status: ProbeStatus
    detail: str = ""
    fix_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json_dict(self) -> dict[str, str]:
        d = {"status": self.status}
        if self.detail:
            d["detail"] = self.detail
        if self.fix_hint:
            d["fix_hint"] = self.fix_hint
        return d


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def probe_gpu(*, timeout: float = 5.0) -> ProbeResult:
    """NVIDIA GPU presence via ``nvidia-smi`` — deliberately torch-free."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return ProbeResult(
            "missing", "nvidia-smi not on PATH",
            fix_hint="install the NVIDIA driver (gpu bundles need one)",
        )
    except subprocess.TimeoutExpired:
        return ProbeResult("error", f"nvidia-smi timed out after {timeout:g}s")
    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout).strip().splitlines()
        return ProbeResult(
            "missing", snippet[0] if snippet else
            f"nvidia-smi exited {proc.returncode}",
        )
    gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not gpus:
        return ProbeResult("missing", "nvidia-smi reports no GPUs")
    return ProbeResult("ok", "; ".join(gpus))


def _adc_path() -> Path:
    """Google Application Default Credentials file location."""
    cloudsdk = os.environ.get("CLOUDSDK_CONFIG", "").strip()
    base = Path(cloudsdk).expanduser() if cloudsdk else Path.home() / ".config" / "gcloud"
    return base / "application_default_credentials.json"


def probe_llm_providers() -> dict[str, ProbeResult]:
    """Key/credential presence per LLM provider, as gap.agent.llm reads them.

    - ``anthropic`` (default provider): the SDK reads ``ANTHROPIC_API_KEY``.
    - ``openai``: ``OPENAI_API_KEY`` (custom OpenAI-compatible ``endpoint:``
      configs may not need it — noted in the detail).
    - ``vertex``: native Google SDKs authenticate via Application Default
      Credentials (``$GOOGLE_APPLICATION_CREDENTIALS`` or the gcloud ADC
      file); project/region come from the llm config, not env vars.
    """
    results: dict[str, ProbeResult] = {}

    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        results["anthropic"] = ProbeResult("ok", "ANTHROPIC_API_KEY set")
    else:
        results["anthropic"] = ProbeResult(
            "missing", "ANTHROPIC_API_KEY not set",
            fix_hint="export ANTHROPIC_API_KEY=...",
        )

    if os.environ.get("OPENAI_API_KEY", "").strip():
        results["openai"] = ProbeResult("ok", "OPENAI_API_KEY set")
    else:
        results["openai"] = ProbeResult(
            "missing",
            "OPENAI_API_KEY not set (custom `endpoint:` configs may not need it)",
            fix_hint="export OPENAI_API_KEY=...",
        )

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if gac:
        if Path(gac).expanduser().is_file():
            results["vertex"] = ProbeResult(
                "ok", "ADC via $GOOGLE_APPLICATION_CREDENTIALS",
            )
        else:
            results["vertex"] = ProbeResult(
                "error",
                f"$GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {gac}",
            )
    elif _adc_path().is_file():
        results["vertex"] = ProbeResult("ok", f"gcloud ADC ({_adc_path()})")
    else:
        results["vertex"] = ProbeResult(
            "missing", "no Application Default Credentials",
            fix_hint="gcloud auth application-default login",
        )
    if results["vertex"].ok:
        import importlib.util

        if importlib.util.find_spec("google") is None or (
            importlib.util.find_spec("google.genai") is None
        ):
            results["vertex"].detail += (
                "; google-genai not installed (gemini-* models need "
                "`pip install 'graph-as-policy[vertex]'`)"
            )
    return results


@dataclass
class EnvironmentReport:
    python_version: str
    gap_version: str
    gpu: ProbeResult
    llm_providers: dict[str, ProbeResult]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "gap_version": self.gap_version,
            "gpu": self.gpu.to_json_dict(),
            "llm_providers": {
                name: probe.to_json_dict()
                for name, probe in self.llm_providers.items()
            },
        }


def probe_environment() -> EnvironmentReport:
    import gap

    return EnvironmentReport(
        python_version=platform.python_version(),
        gap_version=getattr(gap, "__version__", "unknown"),
        gpu=probe_gpu(),
        llm_providers=probe_llm_providers(),
    )


# ---------------------------------------------------------------------------
# Per-bundle probes
# ---------------------------------------------------------------------------


def dep_fix_hint(
    registry_root: Path,
    *,
    dist_name: str | None,
    bundle: str,
    has_extra: bool,
) -> str:
    """How to install a bundle's missing deps, for *this* registry."""
    if not has_extra:
        return (
            f"declare a {bundle!r} extra in {registry_root}/pyproject.toml "
            f"and install it"
        )
    if (registry_root / "uv.lock").is_file():
        return f"uv sync --extra {bundle}  (in {registry_root})"
    if dist_name:
        return f"pip install '{dist_name}[{bundle}]'"
    return f"install the {bundle!r} extra's dependencies (see {registry_root}/pyproject.toml)"


def probe_bundle_import(
    name: str,
    bundle_dir: Path,
    *,
    kind: Literal["tool", "skill", "policy"],
    fix_hint: str = "",
) -> tuple[ProbeResult, Any | None]:
    """Import-probe one bundle in isolation.

    Registers the bundle into a throwaway registry so one broken bundle
    never masks the rest. Returns the probe verdict and, on success, the
    :class:`~gap.skills._registry.SkillInfo` (whose ``tools_module`` /
    ``module`` host the optional ``weights_cached()`` / ``prefetch()``
    hooks).
    """
    from ._registry import SkillsRegistry

    reg = SkillsRegistry()
    try:
        reg.register_bundle(name, bundle_dir, kind=kind)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        return ProbeResult(
            "missing", f"missing {missing}", fix_hint=fix_hint,
        ), None
    except Exception as exc:
        return ProbeResult("error", f"import probe failed: {exc}"), None
    return ProbeResult("ok"), reg.get(name)


def _probe_requirements(
    meta: SkillMeta, *, gpu: ProbeResult,
) -> list[tuple[str, ProbeResult]]:
    """Evaluate the bundle's declared ``gap.requires`` block."""
    req = meta.requires
    if req is None:
        return []
    results: list[tuple[str, ProbeResult]] = []
    if req.gpu:
        results.append(("gpu", gpu))
    for var in req.env:
        if os.environ.get(var, "").strip():
            results.append((f"env:{var}", ProbeResult("ok")))
        else:
            results.append((f"env:{var}", ProbeResult(
                "missing", f"{var} not set",
                fix_hint=f"export {var}=...  (see the bundle's SKILL.md)",
            )))
    if req.env_any:
        set_vars = [v for v in req.env_any if os.environ.get(v, "").strip()]
        label = f"env_any:{'|'.join(req.env_any)}"
        if set_vars:
            results.append((label, ProbeResult("ok", f"{set_vars[0]} set")))
        else:
            results.append((label, ProbeResult(
                "missing", f"none of {', '.join(req.env_any)} set",
                fix_hint=(
                    f"export {req.env_any[0]}=...  (any one of these works; "
                    f"see the bundle's SKILL.md)"
                ),
            )))
    return results


def _probe_weights(meta: SkillMeta, info: Any | None) -> ProbeResult:
    """The optional filesystem-only ``weights_cached()`` hook."""
    req = meta.requires
    if req is None or not req.weights:
        return ProbeResult("ok", "no weights declared")
    if info is None:
        return ProbeResult(
            "unknown", "deps not importable",
            fix_hint="run `gap skills check --download` after installing deps",
        )
    hook = None
    for module in (getattr(info, "tools_module", None), getattr(info, "module", None)):
        fn = getattr(module, "weights_cached", None) if module is not None else None
        if callable(fn):
            hook = fn
            break
    if hook is None:
        return ProbeResult(
            "unknown", "bundle has no weights_cached() hook",
            fix_hint="run `gap skills check --download` to (pre)fetch",
        )
    try:
        cached = hook()
    except Exception as exc:
        return ProbeResult("error", f"weights_cached() failed: {exc}")
    if cached is True:
        return ProbeResult("ok", "cached")
    if cached is False:
        return ProbeResult(
            "missing", "not cached (downloads on first use)",
            fix_hint="gap skills check --download",
        )
    return ProbeResult(
        "unknown", "weights_cached() returned None",
        fix_hint="run `gap skills check --download` to (pre)fetch",
    )


@dataclass
class BundleCapability:
    """Operational status of one bundle in one registry."""

    name: str
    kind: Literal["tool", "skill", "policy"]
    registry: str
    bundle_dir: Path
    deps: ProbeResult
    requirements: list[tuple[str, ProbeResult]] = field(default_factory=list)
    weights: ProbeResult = field(default_factory=lambda: ProbeResult("ok"))
    shadowed_by: str = ""
    """Set when a higher-precedence registry already claims this bundle
    name — the bundle is inert (never imported, never dispatched)."""

    @property
    def status(self) -> Literal["ready", "not-ready", "shadowed"]:
        if self.shadowed_by:
            return "shadowed"
        if not self.deps.ok:
            return "not-ready"
        if any(not probe.ok for _, probe in self.requirements):
            return "not-ready"
        return "ready"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "registry": self.registry,
            "bundle_dir": str(self.bundle_dir),
            "status": self.status,
            "deps": self.deps.to_json_dict(),
            "requirements": {
                label: probe.to_json_dict()
                for label, probe in self.requirements
            },
            "weights": self.weights.to_json_dict(),
            **({"shadowed_by": self.shadowed_by} if self.shadowed_by else {}),
        }


@dataclass
class SkillRunnability:
    """A skill bundle's rollup: blocked exactly by not-ready tool bundles."""

    name: str
    registry: str
    tools: dict[str, str] = field(default_factory=dict)
    """``{allowed_tool_name: owner}`` where owner is a bundle name,
    ``"connector"`` (robot.*/sim.* — satisfied at run time), or
    ``"unknown"`` (unresolvable prefix — flagged)."""
    blocked_by: list[str] = field(default_factory=list)
    unknown_tools: list[str] = field(default_factory=list)
    self_ready: bool = True
    """The skill bundle's own deps/requirements verdict."""

    @property
    def status(self) -> Literal["ready", "blocked"]:
        return "ready" if self.self_ready and not self.blocked_by else "blocked"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "registry": self.registry,
            "status": self.status,
            "tools": dict(self.tools),
            "blocked_by": list(self.blocked_by),
            "unknown_tools": list(self.unknown_tools),
        }


@dataclass
class CheckReport:
    """Everything ``gap check`` reports; ``to_json_dict()`` is the stable
    ``--format json`` schema (bump ``schema_version`` on breaking change)."""

    environment: EnvironmentReport
    registries: list[RegistrySpec] = field(default_factory=list)
    bundles: list[BundleCapability] = field(default_factory=list)
    skills: list[SkillRunnability] = field(default_factory=list)
    schema_version: int = 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_json_dict(),
            "registries": [
                {
                    "name": spec.name,
                    "path": str(spec.path),
                    "source": spec.source,
                    "dist_name": spec.dist_name,
                    "origin": spec.origin,
                }
                for spec in self.registries
            ],
            "bundles": [b.to_json_dict() for b in self.bundles],
            "skills": [s.to_json_dict() for s in self.skills],
        }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _tool_owner(tool_name: str, owner_by_prefix: dict[str, str]) -> str:
    prefix = tool_name.split(".", 1)[0]
    if prefix in CONNECTOR_PREFIXES:
        return "connector"
    return owner_by_prefix.get(prefix, "unknown")


def build_check_report(registry_set: RegistrySet) -> CheckReport:
    """Probe every bundle of every active registry and roll up skills."""
    environment = probe_environment()
    gpu = environment.gpu

    bundles: list[BundleCapability] = []
    metas: dict[str, SkillMeta] = {}          # winner per bundle name
    claimed: dict[str, str] = {}              # bundle name -> registry name

    for spec in registry_set:
        extras = load_checkout_extras(spec.path) or {}
        for report in validate_checkout(spec.path):
            shadowed_by = claimed.get(report.name, "")
            if not shadowed_by:
                claimed[report.name] = spec.name

            if report.meta is None:
                bundles.append(BundleCapability(
                    name=report.name, kind=report.kind, registry=spec.name,
                    bundle_dir=report.bundle_dir,
                    deps=ProbeResult("error", "SKILL.md rejected — run `gap skills check`"),
                    shadowed_by=shadowed_by,
                ))
                continue

            if shadowed_by:
                # Inert: never imported (the synthetic gap_skills.* namespace
                # belongs to the winner), so no probes either.
                bundles.append(BundleCapability(
                    name=report.name, kind=report.kind, registry=spec.name,
                    bundle_dir=report.bundle_dir, deps=ProbeResult("ok"),
                    shadowed_by=shadowed_by,
                ))
                continue

            metas[report.name] = report.meta
            hint = dep_fix_hint(
                spec.path, dist_name=spec.dist_name, bundle=report.name,
                has_extra=report.name in extras,
            )
            deps, info = probe_bundle_import(
                report.name, report.bundle_dir, kind=report.kind,
                fix_hint=hint,
            )
            bundles.append(BundleCapability(
                name=report.name, kind=report.kind, registry=spec.name,
                bundle_dir=report.bundle_dir, deps=deps,
                requirements=_probe_requirements(report.meta, gpu=gpu),
                weights=_probe_weights(report.meta, info),
            ))

    # Skill rollup — owners resolved statically from declared gap.tools
    # names (works even when a bundle's deps are not installed).
    owner_by_prefix: dict[str, str] = {}
    for name, meta in metas.items():
        for tool_name in meta.tools:
            owner_by_prefix.setdefault(tool_name.split(".", 1)[0], name)

    capability_by_name = {
        b.name: b for b in bundles if not b.shadowed_by
    }
    skills: list[SkillRunnability] = []
    for bundle in bundles:
        if bundle.kind != "skill" or bundle.shadowed_by:
            continue
        meta = metas.get(bundle.name)
        if meta is None:
            continue
        tools: dict[str, str] = {}
        blocked: list[str] = []
        unknown: list[str] = []
        for tool_name in meta.allowed_tools:
            owner = _tool_owner(tool_name, owner_by_prefix)
            tools[tool_name] = owner
            if owner == "unknown":
                unknown.append(tool_name)
            elif owner not in ("connector",):
                owner_cap = capability_by_name.get(owner)
                if owner_cap is not None and owner_cap.status == "not-ready":
                    if owner not in blocked:
                        blocked.append(owner)
        skills.append(SkillRunnability(
            name=bundle.name, registry=bundle.registry, tools=tools,
            blocked_by=blocked, unknown_tools=unknown,
            self_ready=bundle.status == "ready",
        ))

    return CheckReport(
        environment=environment,
        registries=list(registry_set),
        bundles=bundles,
        skills=skills,
    )
