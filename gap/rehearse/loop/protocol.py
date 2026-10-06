"""Checks a submitted graph must pass before it is rehearsed.

Graph scripts run inside the rehearsal process, next to the simulator. The
checks keep a script to what a graph may use on a robot: its inputs and the
tools reached through ``ctx.tool``. They are a screen for mistakes and for
the obvious shortcuts, not a security boundary; the held-out cases are what
exposes a graph that fits the visible cases by other means.

A script that is byte-identical to one in the starting graph or in the skill
library is accepted as is.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable

VERSION = 1

ALLOWED_SUFFIXES = (".json", ".py", ".md", ".txt", ".yaml", ".yml")
MAX_FILES = 200
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024

#: Modules a new or modified script may not import: file, process and network
#: access, and anything that reaches the simulator or the rehearsal harness.
DENIED_MODULES = (
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "glob", "tempfile", "pickle",
    "importlib", "ctypes", "multiprocessing", "threading", "signal", "inspect", "builtins", "gc",
    "requests", "httpx", "urllib", "http", "mujoco", "robosuite", "libero",
    "gap.connector", "gap.rehearse", "gap.envs", "gap.runtime.verify", "gap.runtime.execute",
)
DENIED_CALLS = ("open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "vars")
DENIED_ATTRIBUTES = ("__globals__", "__subclasses__", "__builtins__", "__dict__", "__class__", "__code__")
#: Tools that read or drive the simulator directly instead of the robot.
DENIED_TOOL_PREFIXES = ("sim.",)


class GraphRejected(ValueError):
    """The submitted graph breaks the contract; the message says how."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_files(root: Path) -> dict[str, str]:
    """Relative path -> content hash of every file of a graph directory."""
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise GraphRejected(f"symbolic links are not accepted: {path.relative_to(root)}")
        if path.is_file():
            out[str(path.relative_to(root))] = sha256(path)
    return out


def library_hashes(skill_roots: Iterable[Path]) -> set[str]:
    """Content hashes of every Python file in the skill library."""
    out: set[str] = set()
    for root in skill_roots:
        for path in Path(root).rglob("*.py"):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                out.add(sha256(path))
            except OSError:
                continue
    return out


def _denied_module(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in DENIED_MODULES)


def script_issues(source: str, name: str) -> list[str]:
    """Contract violations in one script, as readable messages."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{name}: syntax error on line {exc.lineno}: {exc.msg}"]
    issues: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _denied_module(alias.name):
                    issues.append(f"{name}:{line}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or _denied_module(module):
                issues.append(f"{name}:{line}: import from '{'.' * node.level}{module}' is not allowed")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in DENIED_CALLS:
                issues.append(f"{name}:{line}: call to '{fn.id}' is not allowed")
            for arg in node.args[:1]:
                if (isinstance(fn, ast.Attribute) and fn.attr in ("tool", "tool_call")
                        and isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        and arg.value.startswith(DENIED_TOOL_PREFIXES)):
                    issues.append(f"{name}:{line}: tool '{arg.value}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in DENIED_ATTRIBUTES:
                issues.append(f"{name}:{line}: access to '{node.attr}' is not allowed")
            elif (node.attr.startswith("_") and not node.attr.startswith("__")
                  and isinstance(node.value, ast.Name) and node.value.id == "ctx"):
                issues.append(f"{name}:{line}: private attribute 'ctx.{node.attr}' is not allowed")
    return issues


def check_graph(root: Path, *, trusted_hashes: set[str] | None = None) -> dict[str, str]:
    """Raise :class:`GraphRejected` unless ``root`` is an acceptable graph.

    Returns the graph's file manifest. ``trusted_hashes`` are the content
    hashes accepted without inspection (starting graph, skill library).
    """
    import json

    root = Path(root)
    files = graph_files(root)
    if "workflow.json" not in files:
        raise GraphRejected("the graph has no workflow.json")
    if len(files) > MAX_FILES:
        raise GraphRejected(f"the graph has {len(files)} files; the limit is {MAX_FILES}")
    total = 0
    for rel in files:
        path = root / rel
        if path.suffix not in ALLOWED_SUFFIXES:
            raise GraphRejected(f"file type not accepted: {rel} (accepted: {', '.join(ALLOWED_SUFFIXES)})")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise GraphRejected(f"{rel} is {size} bytes; the limit per file is {MAX_FILE_BYTES}")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise GraphRejected(f"the graph is {total} bytes; the limit is {MAX_TOTAL_BYTES}")

    try:
        workflow = json.loads((root / "workflow.json").read_text())
    except json.JSONDecodeError as exc:
        raise GraphRejected(f"workflow.json is not valid JSON: {exc}") from exc
    issues: list[str] = []
    graphs = [("main", workflow)] + list((workflow.get("subgraphs") or {}).items())
    for graph_name, graph in graphs:
        for node_name, node in (graph.get("nodes") or {}).items():
            tool = node.get("tool") or ""
            if isinstance(tool, str) and tool.startswith(DENIED_TOOL_PREFIXES):
                issues.append(f"workflow.json: node {graph_name}.{node_name} uses tool '{tool}', which is not allowed")

    trusted = trusted_hashes or set()
    for rel, digest in files.items():
        if rel.endswith(".py") and digest not in trusted:
            issues += script_issues((root / rel).read_text(), rel)
    if issues:
        raise GraphRejected("the graph breaks the script contract:\n  " + "\n  ".join(issues))
    return files


__all__ = [
    "VERSION", "GraphRejected", "check_graph", "graph_files", "library_hashes", "script_issues", "sha256",
    "DENIED_MODULES", "DENIED_CALLS", "DENIED_TOOL_PREFIXES",
]
