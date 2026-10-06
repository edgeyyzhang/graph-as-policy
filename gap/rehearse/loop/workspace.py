"""Create the two sides of a rehearsal loop.

``init_loop`` writes:

- the **workspace**, handed to the authoring agent. Its layout follows what
  the ``confine`` helper expects: the whole workspace is readable, and only
  ``results/``, ``.cache/`` and ``.requests/`` are writable;
- the **trusted directory**, which holds the configuration (including the
  held-out cases), every round's full output, and the ledger.

The workspace carries its own Python (NumPy, SciPy, Pillow) because confined
commands can read nothing outside it except the system directories.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from . import protocol

logger = logging.getLogger(__name__)

AGENT_FILES = Path(__file__).resolve().parent / "agent"
GAP_ROOT = Path(__file__).resolve().parents[3]

#: Documentation copied into the workspace: the graph format and what a graph may use.
DOCS = (
    "docs/source/reference/workflow-schema.md",
    "docs/source/reference/connector-tools.md",
    "docs/source/reference/executor.md",
    "docs/source/authoring/patterns.md",
    "docs/source/running/checkpoints.md",
    "docs/source/skills/skill-catalog.md",
    "docs/source/skills/tool-catalog.md",
)
PYTHON_PACKAGES = ("numpy", "scipy", "pillow")


def _copy_graph(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".llm_cache"))


def _find_python() -> Path | None:
    """A relocatable Python installed by uv, to copy into the workspace."""
    base = Path.home() / ".local/share/uv/python"
    found = sorted(p for p in base.glob("cpython-3.11.*-linux-x86_64-gnu") if p.is_dir() and not p.is_symlink())
    return found[-1] if found else None


def install_python(workspace: Path, *, source: Path | None = None) -> Path:
    """Put a Python with the analysis packages inside the workspace."""
    source = source or _find_python()
    if source is None:
        raise FileNotFoundError(
            "no uv-managed Python 3.11 found under ~/.local/share/uv/python; run `uv python install 3.11`")
    home = workspace / "runtime" / "python"
    if not home.exists():
        shutil.copytree(source, home, symlinks=True)
    interpreter = home / "bin" / "python3.11"
    venv = workspace / ".venv"
    if not venv.exists():
        subprocess.run([str(interpreter), "-m", "venv", "--without-pip", str(venv)], check=True)
        subprocess.run(
            ["uv", "pip", "install", "--quiet", "--python", str(venv / "bin" / "python"),
             "--link-mode", "copy", *PYTHON_PACKAGES],
            check=True,
        )
    return venv


def build_confine(workspace: Path) -> Path:
    """Compile the confinement helper into the workspace."""
    tools = workspace / "runtime" / "tools"
    target = tools / "confine"
    subprocess.run(["cc", "-O2", "-Wall", "-Wextra", str(AGENT_FILES / "confine.c"), "-o", str(target)], check=True)
    return target


def init_loop(
    workspace: str | Path,
    trusted: str | Path,
    *,
    graph: str | Path,
    sim: str,
    instruction: str,
    visible_cases: list[int],
    holdout_cases: list[int],
    rehearsal_budget: int = 8,
    env: str = "libero",
    ik: str | None = "pyroki",
    skills: list[str | Path] | None = None,
    frames: bool = True,
    trajectory_interval: int = 5,
    python: bool = True,
    confine: bool = True,
) -> dict[str, Any]:
    """Create the workspace and the trusted directory. Both must not exist yet."""
    workspace, trusted, graph = Path(workspace).resolve(), Path(trusted).resolve(), Path(graph).resolve()
    if workspace.exists() or trusted.exists():
        raise FileExistsError(f"refusing to overwrite an existing loop: {workspace} or {trusted}")
    if workspace in trusted.parents or trusted in workspace.parents:
        raise ValueError("the workspace and the trusted directory must not contain each other")
    overlap = sorted(set(visible_cases) & set(holdout_cases))
    if overlap:
        raise ValueError(f"cases {overlap} are both visible and held out")
    if not visible_cases:
        raise ValueError("at least one visible case is needed")
    if skills is None:
        skills = [GAP_ROOT.parent / "open-robot-skills" / "skills"]
    skills = [Path(s).resolve() for s in skills]

    # The starting graph is accepted as given: all of its files are trusted.
    baseline = protocol.check_graph(graph, trusted_hashes=set(protocol.graph_files(graph).values()))

    # --- workspace ---------------------------------------------------------
    for sub in ("results/notes", ".cache/tmp", ".requests", "eval/rounds", "runtime/tools", "runtime/docs"):
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    _copy_graph(graph, workspace / "results" / "graph")
    for name in ("submit.py", "status.py"):
        shutil.copy2(AGENT_FILES / name, workspace / "runtime" / "tools" / name)
    shutil.copy2(AGENT_FILES / "run", workspace / "eval" / "run")
    (workspace / "eval" / "run").chmod(0o755)
    for root in skills:
        if root.is_dir():
            shutil.copytree(
                root, workspace / "runtime" / "skills", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".llm_cache"),
            )
    for rel in DOCS:
        src = GAP_ROOT / rel
        if src.is_file():
            shutil.copy2(src, workspace / "runtime" / "docs" / src.name)
    prompt = (AGENT_FILES / "AGENT_PROMPT.md").read_text()
    prompt = (prompt.replace("{instruction}", instruction).replace("{sim}", sim)
              .replace("{visible_cases}", ", ".join(str(c) for c in visible_cases))
              .replace("{budget}", str(rehearsal_budget)))
    (workspace / "AGENT_PROMPT.md").write_text(prompt)
    if confine:
        build_confine(workspace)
    if python:
        install_python(workspace)

    # --- trusted side ------------------------------------------------------
    (trusted / "rounds").mkdir(parents=True)
    config = {
        "version": protocol.VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace": str(workspace),
        "trusted": str(trusted),
        "instruction": instruction,
        "sim": sim,
        "env": env,
        "ik": ik,
        "visible_cases": list(visible_cases),
        "holdout_cases": list(holdout_cases),
        "rehearsal_budget": int(rehearsal_budget),
        "skills": [str(s) for s in skills],
        "frames": bool(frames),
        "trajectory_interval": int(trajectory_interval),
        "start_graph": str(graph),
        "baseline_hashes": sorted(set(baseline.values())),
    }
    (trusted / "config.json").write_text(json.dumps(config, indent=2))
    (trusted / "ledger.jsonl").write_text("")
    write_status(config, [])
    return config


def write_status(config: dict[str, Any], ledger: list[dict[str, Any]]) -> None:
    """What the agent may know: the visible results and the remaining budget."""
    rounds = [
        {"round": row["round"], "cases": row["visible"]["cases"], "successes": row["visible"]["successes"],
         "results": f"eval/rounds/round_{row['round']:02d}"}
        for row in ledger
    ]
    status = {
        "instruction": config["instruction"],
        "sim": config["sim"],
        "visible_cases": config["visible_cases"],
        "rehearsal_budget": config["rehearsal_budget"],
        "rehearsals_used": len(ledger),
        "rounds": rounds,
    }
    path = Path(config["workspace"]) / "eval" / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(path)


def read_ledger(trusted: Path) -> list[dict[str, Any]]:
    path = Path(trusted) / "ledger.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


__all__ = ["init_loop", "install_python", "build_confine", "write_status", "read_ledger", "DOCS"]
