"""Shared fixtures for the gap.benchmark test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples" / "benchmark"

#: The real open-robot-skills checkout (sibling of the gap repo). Example YAMLs
#: reference it via ``skills:``; parsing them needs it on disk.
SKILLS_ROOT = REPO.parent / "open-robot-skills"


@pytest.fixture
def skills_root() -> Path:
    if not (SKILLS_ROOT / "skills").is_dir():
        pytest.skip(f"open-robot-skills checkout not found at {SKILLS_ROOT}")
    return SKILLS_ROOT


def stub_workflow_dict() -> dict:
    """A minimal models-free v3 workflow: check success then exit ok."""
    return {
        "version": 3,
        "meta": {"name": "stub"},
        "nodes": {
            "m": {"type": "subgraph", "ref": "m"},
            "done": {"type": "end", "status": "success"},
        },
        "edges": [["START", "m"]],
        "conditional_edges": {
            "m": {"router_field": "exit", "mapping": {"ok": "done"}},
        },
        "subgraphs": {
            "m": {
                "skill": "generic",
                "inputs": {},
                "outputs": {},
                "nodes": {
                    "check": {"type": "tool", "tool": "sim.check_success"},
                    "ok": {"type": "noop"},
                },
                "edges": [["START", "check"], ["check", "ok"], ["ok", "END"]],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }


def write_stub_workflow(parent: Path) -> Path:
    """Materialize the stub workflow folder under *parent*; returns it."""
    wf_dir = parent / "workflow"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "workflow.json").write_text(json.dumps(stub_workflow_dict()))
    return wf_dir


class StubConnector:
    """Duck-typed connector: tool registry + reset/obs/success, no sim."""

    def __init__(self, *, succeed: bool = True, reward: float = 1.0) -> None:
        from gap.tools import ToolRegistry

        self.succeed = succeed
        self.reward = reward
        self.reset_seeds: list[int | None] = []
        self.closed = False
        reg = ToolRegistry()
        reg.register_callable(
            "sim.check_success",
            lambda: {"task_completed": self.succeed, "reward": self.reward},
            summary="stub success check",
        )
        self._registry = reg

    @property
    def tool_registry(self):
        return self._registry

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return {}

    def get_observation(self):
        return {}

    def check_success(self):
        return self.succeed, self.reward

    def save_video(self, output_path: str, fps: int = 20, clear: bool = False):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x00mp4")
        return {"success": True, "file_path": output_path, "num_frames": 1}

    def close(self):
        self.closed = True


def stub_connector_factory(suite_name, task_id, config):
    """Module-level factory (picklable by reference) building a stub."""
    return StubConnector()


class FlakyConnector(StubConnector):
    """Stub connector that kills or wedges its worker on chosen seeds.

    Drives the pool-lifecycle regression tests: trial seeds listed in
    ``GAP_TEST_CRASH_SEEDS`` hard-kill the worker process mid-trial
    (simulating a native sim/perception crash — the 2026-06-10 livelock
    trigger); seeds in ``GAP_TEST_HANG_SEEDS`` block forever (simulating
    a wedged MuJoCo/EGL call, exercised by the watchdog). Both env vars
    reach spawned workers via ``WorkerSetupConfig.extra_env``.
    """

    @staticmethod
    def _seeds(var: str) -> set[int]:
        import os

        raw = os.environ.get(var, "")
        return {int(s) for s in raw.split(",") if s.strip()}

    def reset(self, seed=None):
        import os
        import time

        super().reset(seed)
        sleep_s = float(os.environ.get("GAP_TEST_TRIAL_SLEEP", "0") or 0)
        if sleep_s:
            time.sleep(sleep_s)  # make trials slow enough to observe
        if seed in self._seeds("GAP_TEST_CRASH_SEEDS"):
            os._exit(139)  # die like a segfault: no result, no traceback
        if seed in self._seeds("GAP_TEST_CRASH_ONCE_SEEDS"):
            # Transient crash: kill the worker on the FIRST attempt only.
            # A filesystem marker survives across worker processes.
            marker = (
                Path(os.environ["GAP_TEST_CRASH_MARKER_DIR"])
                / f"crashed_{seed}"
            )
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                os._exit(139)
        if seed in self._seeds("GAP_TEST_HANG_SEEDS"):
            time.sleep(600.0)  # wedge until the watchdog hard-kills us
        return {}


def flaky_connector_factory(suite_name, task_id, config):
    """Module-level factory (picklable by dotted path) for FlakyConnector."""
    return FlakyConnector()
