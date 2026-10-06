"""Serve the agent's requests: one at a time, from request files.

For a ``validate`` request the broker checks the snapshot against the script
contract and runs the structural validator. For a ``rehearse`` request it
also rehearses the snapshot on the visible cases and on the held-out cases,
in two subprocesses. The visible results are copied into the workspace; the
held-out results stay in the trusted directory and appear only in the ledger.

The broker must be started with the environment a rehearsal needs (Python
path, credentials, LIBERO configuration); the subprocesses inherit it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import protocol
from .workspace import read_ledger, write_status

logger = logging.getLogger(__name__)

_GAP_MAIN = "import sys; from gap.cli import main; sys.argv = ['gap', *sys.argv[1:]]; sys.exit(main())"
#: Not copied into the workspace: caches, and nothing the agent needs.
_SKIP = shutil.ignore_patterns("perception_cache", "__pycache__", "*.pyc")
_TEXT_SUFFIXES = (".json", ".md", ".jsonl", ".txt")


def pick_gpu() -> str | None:
    """The GPU every run of a round shares: ``CUDA_VISIBLE_DEVICES`` when the
    broker was started with it, else the least-used GPU by memory. ``None``
    without nvidia-smi.

    Both rehearsals of a round fit on one GPU (about 9 GB each), and keeping
    them together leaves the other GPUs to other users.
    """
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    if pinned:
        return pinned
    gpus = pick_gpus(1)
    return gpus[0] if gpus else None


def pick_gpus(n: int) -> list[str]:
    """The ``n`` least-used GPUs by memory, as nvidia-smi indices. Empty without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            rows.append((float(parts[1]), parts[0]))
    ids = [i for _, i in sorted(rows)]
    return [ids[k % len(ids)] for k in range(n)] if ids else []


def _rewrite_paths(root: Path, replacements: list[tuple[str, str]]) -> None:
    """Replace trusted-side paths in the text files copied into the workspace."""
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES or path.name == "trajectory.jsonl":
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        for old, replacement in replacements:
            new = new.replace(old, replacement)
        if new != text:
            path.write_text(new)


class Broker:
    """One loop's request server."""

    def __init__(self, trusted: str | Path, *, python: str | None = None):
        self.trusted = Path(trusted).resolve()
        self.config: dict[str, Any] = json.loads((self.trusted / "config.json").read_text())
        self.workspace = Path(self.config["workspace"])
        self.requests = self.workspace / ".requests"
        self.python = python or sys.executable
        hashes = set(self.config.get("baseline_hashes") or [])
        hashes |= protocol.library_hashes(Path(s) for s in self.config.get("skills") or [])
        self.trusted_hashes = hashes

    # -- requests ----------------------------------------------------------

    def pending(self) -> list[Path]:
        if not self.requests.is_dir():
            return []
        return sorted(
            p for p in self.requests.iterdir()
            if p.is_dir() and not p.name.endswith(".tmp")
            and (p / "request.json").is_file() and not (p / "response.json").exists()
        )

    def respond(self, request: Path, answer: dict[str, Any]) -> None:
        tmp = request / "response.json.tmp"
        tmp.write_text(json.dumps(answer, indent=2))
        tmp.replace(request / "response.json")

    def serve(self, *, poll_s: float = 2.0, once: bool = False, idle_timeout_s: float | None = None) -> None:
        """Handle requests until interrupted. ``once`` returns after one pass."""
        write_status(self.config, read_ledger(self.trusted))
        logger.info("broker serving %s", self.workspace)
        idle_since = time.time()
        while True:
            handled = False
            for request in self.pending():
                self.handle(request)
                handled = True
            if handled:
                idle_since = time.time()
            if once:
                return
            if idle_timeout_s is not None and time.time() - idle_since > idle_timeout_s:
                logger.info("broker idle for %.0f s; stopping", idle_timeout_s)
                return
            time.sleep(poll_s)

    def handle(self, request: Path) -> dict[str, Any]:
        try:
            meta = json.loads((request / "request.json").read_text())
            kind = meta.get("type")
            if kind not in ("validate", "rehearse"):
                raise protocol.GraphRejected(f"unknown request type {kind!r}")
            answer = self._handle(request, kind)
        except protocol.GraphRejected as exc:
            answer = {"ok": False, "message": str(exc)}
        except Exception as exc:  # infrastructure failure: report it, charge nothing
            logger.exception("request %s failed", request.name)
            answer = {"ok": False, "message": f"infrastructure error, not a fault of the graph: "
                                              f"{type(exc).__name__}: {exc}"}
        answer["id"] = request.name
        self.respond(request, answer)
        logger.info("request %s: %s", request.name, "ok" if answer.get("ok") else answer.get("message"))
        return answer

    # -- handling ----------------------------------------------------------

    def _handle(self, request: Path, kind: str) -> dict[str, Any]:
        ledger = read_ledger(self.trusted)
        budget = int(self.config["rehearsal_budget"])
        if kind == "rehearse" and len(ledger) >= budget:
            raise protocol.GraphRejected(f"the rehearsal budget of {budget} is used up")

        # Work from a copy the agent cannot change while it runs.
        snapshot = self.trusted / "requests" / request.name / "graph"
        if snapshot.parent.exists():
            shutil.rmtree(snapshot.parent)
        snapshot.parent.mkdir(parents=True)
        source = request / "graph"
        if not source.is_dir():
            raise protocol.GraphRejected("the request holds no graph")
        protocol.graph_files(source)  # rejects symbolic links before anything is copied
        shutil.copytree(source, snapshot, symlinks=True)
        manifest = protocol.check_graph(snapshot, trusted_hashes=self.trusted_hashes)

        code, output = self._validate(snapshot)
        if code != 0:
            raise protocol.GraphRejected("the graph does not validate:\n" + output.strip())
        if kind == "validate":
            return {"ok": True, "message": "the graph validates and meets the script contract",
                    "output": output, "remaining_rehearsals": budget - len(ledger)}
        return self._rehearse(request, snapshot, manifest, ledger)

    def _gap(self, args: list[str], *, env: dict[str, str] | None = None, log: Path | None = None,
             ) -> subprocess.Popen:
        full_env = {**os.environ, **(env or {})}
        stream = log.open("w") if log is not None else subprocess.PIPE
        return subprocess.Popen(
            [self.python, "-c", _GAP_MAIN, *args],
            stdout=stream, stderr=subprocess.STDOUT if log is not None else subprocess.PIPE,
            text=True, env=full_env, cwd=str(self.trusted),
        )

    def _validate(self, graph: Path) -> tuple[int, str]:
        # The skill registries are resolved the same way as for a rehearsal.
        proc = self._gap(["run", str(graph), "--validate-only"])
        out, _ = proc.communicate(timeout=300)
        return proc.returncode, (out or "").replace(str(graph), "graph")

    def _rehearse_args(self, graph: Path, cases: list[int], out: Path, previous: Path | None) -> list[str]:
        args = ["rehearse", str(graph), "--sim", self.config["sim"], "--env", self.config.get("env", "libero"),
                "--cases", ",".join(str(c) for c in cases), "--out", str(out),
                "--trajectory-interval", str(self.config.get("trajectory_interval", 5))]
        if self.config.get("ik"):
            args += ["--ik", self.config["ik"]]
        if self.config.get("frames"):
            args.append("--frames")
        if previous is not None and (previous / "feedback.json").exists():
            args += ["--previous", str(previous)]
        return args

    def _rehearse(self, request: Path, snapshot: Path, manifest: dict[str, str],
                  ledger: list[dict[str, Any]]) -> dict[str, Any]:
        number = len(ledger)
        name = f"round_{number:02d}"
        round_dir = self.trusted / "rounds" / name
        if round_dir.exists():
            shutil.rmtree(round_dir)
        round_dir.mkdir(parents=True)
        graph = round_dir / "graph"
        shutil.copytree(snapshot, graph)
        previous = self.trusted / "rounds" / f"round_{number - 1:02d}" if number else None

        runs = [("visible", self.config["visible_cases"])]
        if self.config.get("holdout_cases"):
            runs.append(("holdout", self.config["holdout_cases"]))
        gpu = pick_gpu()
        procs = []
        for label, cases in runs:
            out = round_dir / label
            env = {"GAP_PERCEPTION_CACHE_DIR": str(out / "perception_cache")}
            if gpu is not None:
                # nvidia-smi numbers GPUs by bus id; CUDA does not by default.
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = gpu
            prev = previous / label if previous is not None else None
            procs.append((label, out, self._gap(
                self._rehearse_args(graph, cases, out, prev), env=env, log=round_dir / f"{label}.log")))
        results: dict[str, Any] = {}
        for label, out, proc in procs:
            code = proc.wait()
            feedback = out / "feedback.json"
            if code != 0 or not feedback.exists():
                tail = "\n".join((round_dir / f"{label}.log").read_text().splitlines()[-15:])
                raise RuntimeError(f"the {label} rehearsal did not finish (exit {code}):\n{tail}")
            fb = json.loads(feedback.read_text())
            results[label] = {
                "cases": fb["summary"]["cases"], "successes": fb["summary"]["successes"],
                "per_case": {str(c["case"]): c["success"] for c in fb["cases"]},
            }

        # Visible results go to the workspace; held-out results do not.
        target = self.workspace / "eval" / "rounds" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(round_dir / "visible", target, ignore=_SKIP)
        shutil.copytree(graph, target / "graph")
        _rewrite_paths(target, [
            (str(round_dir / "visible"), f"eval/rounds/{name}"),
            (str(graph), f"eval/rounds/{name}/graph"),
            (str(self.trusted), "<trusted>"),
        ])

        row = {
            "round": number, "request": request.name, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "graph_files": manifest, "visible": results["visible"], "holdout": results.get("holdout"),
        }
        with (self.trusted / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        ledger = ledger + [row]
        write_status(self.config, ledger)
        visible = results["visible"]
        return {
            "ok": True, "round": number, "results": f"eval/rounds/{name}",
            "visible": {"cases": visible["cases"], "successes": visible["successes"]},
            "remaining_rehearsals": int(self.config["rehearsal_budget"]) - len(ledger),
            "message": f"task success on the visible cases: {visible['successes']} of {visible['cases']}",
        }


__all__ = ["Broker", "pick_gpu", "pick_gpus"]
