"""Unit tests for the in-process parallel worker infrastructure.

No sim, no GPU: the worker loop runs sequentially against a stub
connector factory; the watchdog test uses a hanging trial body. The
pool-lifecycle section spawns real worker processes (still simless)
against the stub/flaky connector factories — regression coverage for
the 2026-06-10 respawn livelock.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import gap.agent.parallel as parallel_mod
from gap.agent.parallel import (
    TrialResult,
    WorkerSetupConfig,
    WorkItem,
    required_policy_ids,
    run_parallel_trials,
    run_trial_on_worker,
    worker_setup,
)

from .conftest import StubConnector, stub_connector_factory, write_stub_workflow

STUB_FACTORY = "tests.benchmark.conftest:stub_connector_factory"
FLAKY_FACTORY = "tests.benchmark.conftest:flaky_connector_factory"

# --------------------------------------------------------------------------
# Pickling (the spawn boundary)
# --------------------------------------------------------------------------


def test_worker_setup_config_pickles_roundtrip() -> None:
    cfg = WorkerSetupConfig(
        suite_name="libero_object_all_variance",
        skills_path="/tmp/open-robot-skills",
        camera_names=["agentview"],
        record_video=True,
        task_timeout_secs=600.0,
        output_dir="/tmp/out",
        safety_limits={"perception": 50, "planning": 20, "sim_step": 5000},
        policies={"libero_pi05": {"url": "ws://127.0.0.1:9100"}},
        policy_manager={"startup_timeout_s": 900},
        extra_env={"GOOGLE_CLOUD_PROJECT": "bc-y7-06"},
        connector_factory="tests.benchmark.conftest:stub_connector_factory",
    )
    clone = pickle.loads(pickle.dumps(cfg))
    assert clone == cfg


def test_work_item_and_trial_result_pickle() -> None:
    item = WorkItem(task_id=3, trial_id=7, codegen_id=1, workflow_dir="/wf")
    assert pickle.loads(pickle.dumps(item)) == item
    res = TrialResult(trial_id=7, task_id=3, task_completed=True, reward=1.0)
    assert pickle.loads(pickle.dumps(res)) == res


# --------------------------------------------------------------------------
# Sequential worker loop with a stub connector
# --------------------------------------------------------------------------


def test_run_parallel_trials_sequential_with_stub(tmp_path) -> None:
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="stub_suite",
        output_dir=str(tmp_path / "out"),
    )
    items = [
        WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
        WorkItem(task_id=0, trial_id=2, workflow_dir=str(wf_dir)),
    ]
    seen = []
    results = run_parallel_trials(
        items, 1, cfg,
        on_complete=seen.append,
        connector_factory=stub_connector_factory,
    )
    assert len(results) == 2 and len(seen) == 2
    for r in results:
        assert r.exit_code == 0
        assert r.task_completed is True
        assert r.reward == 1.0
        assert r.completion_rate == 1.0
    # Per-trial result.json + codegen copy are written immediately.
    for trial_id in (1, 2):
        tdir = tmp_path / "out" / "task_00" / f"trial_{trial_id:02d}"
        data = json.loads((tdir / "result.json").read_text())
        assert data["task_completed"] is True
        assert data["seed"] == trial_id
        assert (tdir / "codegen" / "workflow.json").is_file()


def test_task_reuse_keeps_connector_across_same_task(tmp_path) -> None:
    """Same (suite, task) trials reuse one connector; new task rebuilds."""
    built: list[StubConnector] = []

    def factory(suite_name, task_id, config):
        conn = StubConnector()
        built.append(conn)
        return conn

    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(suite_name="s", output_dir=str(tmp_path / "out"))
    items = [
        WorkItem(task_id=1, trial_id=2, workflow_dir=str(wf_dir)),
        WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
        WorkItem(task_id=0, trial_id=2, workflow_dir=str(wf_dir)),
    ]
    results = run_parallel_trials(items, 1, cfg, connector_factory=factory)
    assert len(results) == 3
    # Items are sorted by task_id: task 0 (x2, one connector) then task 1.
    assert len(built) == 2
    assert built[0].reset_seeds == [1, 2]  # reused across both task-0 trials
    assert built[0].closed is True         # closed when task switched
    assert built[1].reset_seeds == [2]


def test_failed_workflow_records_failure_not_raise(tmp_path) -> None:
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text("{not json")
    cfg = WorkerSetupConfig(suite_name="s", output_dir=str(tmp_path / "out"))
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
    )
    assert result.exit_code == 1
    assert result.task_completed is False
    assert result.execution_stderr


def test_missing_workflow_dir_is_failed_trial(tmp_path) -> None:
    cfg = WorkerSetupConfig(suite_name="s", output_dir=str(tmp_path / "out"))
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=""),
    )
    assert result.exit_code == -1
    assert "no workflow_dir" in result.execution_stderr


# --------------------------------------------------------------------------
# Watchdog
# --------------------------------------------------------------------------


def test_watchdog_kills_hanging_trial(tmp_path, monkeypatch) -> None:
    """A hanging trial body -> exit_code 124, failed result, result.json
    written, and the connector poisoned for the next trial."""
    hang_started = []

    def _hang(state, item, trial_result, trial_dir):
        hang_started.append(item.trial_id)
        time.sleep(60.0)

    monkeypatch.setattr(parallel_mod, "_execute_trial", _hang)

    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        task_timeout_secs=0.3,
    )
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    state.connector = StubConnector()  # pretend a live env from a prior trial
    state.last_init = ("s", 0)

    wf_dir = write_stub_workflow(tmp_path)
    t0 = time.monotonic()
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
    )
    elapsed = time.monotonic() - t0

    assert hang_started == [1]
    assert elapsed < 10.0  # came back at the cap, not after the hang
    assert result.exit_code == 124
    assert result.task_completed is False
    assert result.reward == 0.0
    assert "wall-clock cap" in result.execution_stderr
    # The failed result is on disk.
    data = json.loads(
        (tmp_path / "out" / "task_00" / "trial_01" / "result.json").read_text()
    )
    assert data["exit_code"] == 124
    # Connector poisoned: the next trial must rebuild a fresh env.
    assert state.connector is None
    assert state.last_init is None
    # The wedged runner thread is abandoned too: the next trial gets a
    # fresh runner (the old thread is stuck inside the hung body).
    assert state.trial_runner is None


def test_watchdog_trials_share_one_runner_thread(tmp_path, monkeypatch) -> None:
    """EGL thread-affinity regression (the incident's worker killer):
    with the watchdog enabled, every trial body of a worker must run on
    the SAME persistent thread — thread-per-trial made the second trial
    on a reused connector destroy the env from a new thread (SIGSEGV)."""
    import threading

    thread_ids: list[int] = []
    real_execute = parallel_mod._execute_trial

    def _spy(state, item, trial_result, trial_dir):
        thread_ids.append(threading.get_ident())
        real_execute(state, item, trial_result, trial_dir)

    monkeypatch.setattr(parallel_mod, "_execute_trial", _spy)

    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        task_timeout_secs=30.0,
    )
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    wf_dir = write_stub_workflow(tmp_path)
    for t in (1, 2, 3):
        result = run_trial_on_worker(
            state, WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir)),
        )
        assert result.exit_code == 0
    assert len(thread_ids) == 3
    assert len(set(thread_ids)) == 1, "trial bodies migrated threads"
    assert thread_ids[0] != threading.get_ident()  # runner, not the caller
    # Same connector reused across all three trials (the optimization
    # the affinity bug used to kill).
    assert state.connector is not None
    assert state.connector.reset_seeds == [1, 2, 3]


def test_watchdog_disabled_runs_inline(tmp_path) -> None:
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s", output_dir=str(tmp_path / "out"),
        task_timeout_secs=0.0,
    )
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
    )
    assert result.exit_code == 0 and result.task_completed


def test_watchdog_fast_trial_unaffected(tmp_path) -> None:
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s", output_dir=str(tmp_path / "out"),
        task_timeout_secs=30.0,
    )
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
    )
    assert result.exit_code == 0 and result.task_completed
    # Connector kept for reuse (no poisoning on the happy path).
    assert state.connector is not None


# --------------------------------------------------------------------------
# Spawn-pool lifecycle (regression: the 2026-06-10 respawn livelock).
#
# Real spawned worker processes, no sim: the stub/flaky connector
# factories cross the spawn boundary as dotted paths. The incident
# pattern was: a worker dies mid-trial -> its item is lost forever ->
# `remaining` never reaches 0 -> the pool respawns workers (each respawn
# adding a poison pill) that immediately eat a pill and exit -> infinite
# respawn churn (679 respawns / 50 min, run never concluded).
# --------------------------------------------------------------------------


@pytest.fixture
def fast_pool(monkeypatch):
    """Speed up pool polling/stagger so crash recovery is test-fast."""
    monkeypatch.setattr(parallel_mod, "_POLL_SECS", 0.3)
    monkeypatch.setattr(parallel_mod, "_STAGGER_SECS", 0.05)


def _respawn_count(caplog) -> int:
    return len(re.findall(r"Respawning worker slot", caplog.text))


def _by_trial(results) -> dict[int, TrialResult]:
    return {r.trial_id: r for r in results}


def test_pool_completes_promptly_with_more_workers_than_trials(
    tmp_path, fast_pool, caplog,
) -> None:
    """Idle workers must not churn: trials < workers -> pool returns as
    soon as all results are in, with zero respawns."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="stub_suite",
        output_dir=str(tmp_path / "out"),
        connector_factory=STUB_FACTORY,
    )
    items = [
        WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
        for t in (1, 2)
    ]
    t0 = time.monotonic()
    results = run_parallel_trials(items, 4, cfg)
    elapsed = time.monotonic() - t0

    assert len(results) == 2
    assert all(r.exit_code == 0 and r.task_completed for r in results)
    assert _respawn_count(caplog) == 0
    # Conclusion is positive (all results in), not a liveness timeout
    # spiral. Bound generously for slow CI (spawn + import per worker).
    assert elapsed < 60.0


def test_pool_recovers_lost_trial_from_crashed_worker(
    tmp_path, fast_pool, caplog,
) -> None:
    """Incident core: a worker dying mid-trial must not lose the trial.

    Seed 2 kills its worker on the first attempt only; the pool must
    re-enqueue the item and complete ALL trials successfully (the
    incident left such trials bare on disk forever).
    """
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        connector_factory=FLAKY_FACTORY,
        extra_env={
            "GAP_TEST_CRASH_ONCE_SEEDS": "2",
            "GAP_TEST_CRASH_MARKER_DIR": str(tmp_path / "markers"),
        },
    )
    items = [
        WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
        for t in (1, 2, 3)
    ]
    results = run_parallel_trials(items, 2, cfg)
    by_trial = _by_trial(results)
    assert set(by_trial) == {1, 2, 3}
    assert all(r.exit_code == 0 and r.task_completed for r in results)
    # The re-run trial's result is on disk like any other.
    data = json.loads(
        (tmp_path / "out" / "task_00" / "trial_02" / "result.json").read_text()
    )
    assert data["task_completed"] is True


def test_pool_crash_looping_trial_fails_with_capped_retries(
    tmp_path, fast_pool, caplog,
) -> None:
    """A trial that kills its worker on EVERY attempt must come back as
    a synthesized failure after the retry budget — other trials still
    complete and the pool concludes (no lost items, no churn)."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        connector_factory=FLAKY_FACTORY,
        extra_env={"GAP_TEST_CRASH_SEEDS": "2"},
    )
    items = [
        WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
        for t in (1, 2, 3)
    ]
    results = run_parallel_trials(
        items, 2, cfg, max_item_retries=1, max_respawns_per_slot=3,
    )
    by_trial = _by_trial(results)
    assert set(by_trial) == {1, 2, 3}
    assert by_trial[1].exit_code == 0
    assert by_trial[3].exit_code == 0
    assert by_trial[2].exit_code == parallel_mod._CRASHED_EXIT_CODE
    assert "retry budget" in by_trial[2].execution_stderr
    # result.json for the synthesized failure is on disk.
    data = json.loads(
        (tmp_path / "out" / "task_00" / "trial_02" / "result.json").read_text()
    )
    assert data["exit_code"] == parallel_mod._CRASHED_EXIT_CODE
    # Respawns (if capacity was ever short) stayed under the slot cap;
    # a surviving idle worker covering the retry needs none at all.
    assert _respawn_count(caplog) <= 2 * 3


def test_pool_crash_loop_exhausts_respawn_budget_and_concludes(
    tmp_path, fast_pool, caplog,
) -> None:
    """Every trial crash-loops: the pool must retire its slots at the
    respawn cap, fail the remaining trials, and RETURN — the livelock
    ran forever respawning workers that only ever ate poison pills."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        connector_factory=FLAKY_FACTORY,
        extra_env={"GAP_TEST_CRASH_SEEDS": "1,2"},
    )
    items = [
        WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
        for t in (1, 2)
    ]
    t0 = time.monotonic()
    results = run_parallel_trials(
        items, 2, cfg,
        max_respawns_per_slot=1,
        max_item_retries=10,  # item budget never binds; the slot cap must
    )
    elapsed = time.monotonic() - t0

    assert len(results) == 2  # every expected trial got a result
    assert all(
        r.exit_code == parallel_mod._CRASHED_EXIT_CODE for r in results
    )
    assert _respawn_count(caplog) <= 2 * 1  # cap honored
    assert "Worker pool exhausted" in caplog.text
    assert elapsed < 90.0  # concluded promptly — no infinite churn


def test_pool_watchdog_hard_exit_respawns_and_completes(
    tmp_path, fast_pool, caplog,
) -> None:
    """Watchdog semantics kept: a wedged trial is hard-killed
    (exit_code=124, result delivered, worker os._exit) and a replacement
    worker is spawned while queued work remains — every trial scores."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        connector_factory=FLAKY_FACTORY,
        task_timeout_secs=3.0,
        extra_env={
            "GAP_TEST_HANG_SEEDS": "1",
            # Slow every trial down (well under the watchdog cap) so the
            # queue is still non-empty when the watchdog kills the wedged
            # worker -> a respawn must occur.
            "GAP_TEST_TRIAL_SLEEP": "1.0",
        },
    )
    trial_ids = (1, 2, 3, 4, 5, 6, 7)
    items = [
        WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
        for t in trial_ids
    ]
    results = run_parallel_trials(items, 2, cfg)
    by_trial = _by_trial(results)
    assert set(by_trial) == set(trial_ids)
    assert by_trial[1].exit_code == 124
    assert "wall-clock cap" in by_trial[1].execution_stderr
    for t in trial_ids[1:]:
        assert by_trial[t].exit_code == 0 and by_trial[t].task_completed
    # The hard-killed slot was replaced (bounded), never churned.
    assert 1 <= _respawn_count(caplog) <= 2 * 3
    # The timeout result reached disk before the os._exit(124).
    data = json.loads(
        (tmp_path / "out" / "task_00" / "trial_01" / "result.json").read_text()
    )
    assert data["exit_code"] == 124


def test_two_concurrent_pools_complete_independently(
    tmp_path, fast_pool, caplog,
) -> None:
    """Two cells run concurrently (the launcher's asyncio.gather shape):
    both pools must conclude with full results and no respawn churn."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)

    def _run_cell(name: str):
        cfg = WorkerSetupConfig(
            suite_name=f"suite_{name}",
            output_dir=str(tmp_path / name),
            connector_factory=STUB_FACTORY,
        )
        items = [
            WorkItem(task_id=0, trial_id=t, workflow_dir=str(wf_dir))
            for t in (1, 2, 3, 4)
        ]
        return run_parallel_trials(items, 2, cfg)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_run_cell, name) for name in ("a", "b")]
        cell_a, cell_b = (f.result(timeout=120) for f in futs)

    for results, name in ((cell_a, "a"), (cell_b, "b")):
        assert len(results) == 4
        assert all(r.exit_code == 0 and r.task_completed for r in results)
        for t in (1, 2, 3, 4):
            rj = tmp_path / name / "task_00" / f"trial_{t:02d}" / "result.json"
            assert json.loads(rj.read_text())["task_completed"] is True
    assert _respawn_count(caplog) == 0


def test_pool_setup_crash_loop_fails_cell_not_livelock(
    tmp_path, fast_pool, caplog,
) -> None:
    """Workers that die during setup (the concurrent-cell suspect) must
    burn the bounded respawn budget and fail the cell promptly."""
    caplog.set_level(logging.INFO, logger="gap.agent.parallel")
    wf_dir = write_stub_workflow(tmp_path)
    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        connector_factory="tests.benchmark.conftest:no_such_factory",
    )
    items = [WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir))]
    t0 = time.monotonic()
    results = run_parallel_trials(
        items, 2, cfg, max_respawns_per_slot=1,
    )
    elapsed = time.monotonic() - t0
    assert len(results) == 1
    assert results[0].exit_code == parallel_mod._CRASHED_EXIT_CODE
    assert "exhausted" in results[0].execution_stderr
    assert "Worker setup failed" in caplog.text
    assert elapsed < 90.0


def test_bundle_tools_survive_connector_rebuild(tmp_path, monkeypatch) -> None:
    """The @tool pending queue drains once per process; a persistent
    worker switching tasks (new connector -> new registry) must keep the
    bundle tools via the WorkerState catalog."""
    import gap.tools._registry as registry_mod

    monkeypatch.setattr(registry_mod, "_PENDING_TOOLS", [{
        "name": "vision.fake_bundle_tool",
        "summary": "test tool",
        "scope": "runtime",
        "tags": (),
        "fn": lambda: "ok",
    }])

    cfg = WorkerSetupConfig(suite_name="s", output_dir=str(tmp_path / "out"))
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)

    parallel_mod._ensure_connector(state, "s", 0)
    assert "vision.fake_bundle_tool" in state.tool_registry

    parallel_mod._ensure_connector(state, "s", 1)  # task switch: rebuild
    assert "vision.fake_bundle_tool" in state.tool_registry


# --------------------------------------------------------------------------
# EGL spread + env knobs
# --------------------------------------------------------------------------


def test_egl_devices_round_robin(monkeypatch) -> None:
    monkeypatch.setenv(parallel_mod.EGL_DEVICES_ENV, "1, 2,3")
    for wid, expected in [(0, "1"), (1, "2"), (2, "3"), (3, "1")]:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
        worker_setup(
            wid, WorkerSetupConfig(),
            connector_factory=stub_connector_factory,
        )
        assert (
            parallel_mod.os.environ["MUJOCO_EGL_DEVICE_ID"] == expected
        ), f"worker {wid}"


def test_policies_toggle_joint_motion_mode(monkeypatch) -> None:
    monkeypatch.delenv("GAP_LIBERO_JOINT_MOTION_MODE", raising=False)
    worker_setup(
        0, WorkerSetupConfig(),
        connector_factory=stub_connector_factory,
    )
    import os
    assert os.environ["GAP_LIBERO_JOINT_MOTION_MODE"] == "teleport"
    worker_setup(
        0, WorkerSetupConfig(policies={"p": {"url": "ws://h:1"}}),
        connector_factory=stub_connector_factory,
    )
    assert os.environ["GAP_LIBERO_JOINT_MOTION_MODE"] == "closed_loop"


def test_extra_env_propagated(monkeypatch) -> None:
    monkeypatch.delenv("GAP_TEST_VLM_KEY", raising=False)
    worker_setup(
        0, WorkerSetupConfig(extra_env={"GAP_TEST_VLM_KEY": "abc"}),
        connector_factory=stub_connector_factory,
    )
    import os
    assert os.environ["GAP_TEST_VLM_KEY"] == "abc"


# --------------------------------------------------------------------------
# Policy preflight scan (workflow -> required policy ids)
# --------------------------------------------------------------------------


def test_required_policy_ids_both_forms(tmp_path) -> None:
    wf = {
        "version": 3,
        "meta": {"name": "p"},
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
                    # canonical form
                    "vla": {
                        "type": "tool", "tool": "run_policy",
                        "inputs": {"policy_id": "libero_pi05"},
                    },
                    # legacy form: tool name IS the policy id
                    "legacy": {"type": "tool", "tool": "molmoact"},
                    # unrelated tool
                    "check": {"type": "tool", "tool": "sim.check_success"},
                    "ok": {"type": "noop"},
                },
                "edges": [
                    ["START", "vla"], ["vla", "legacy"],
                    ["legacy", "check"], ["check", "ok"], ["ok", "END"],
                ],
                "conditional_edges": {},
                "exit": {"router_field": None, "success_values": ["ok"]},
            },
        },
    }
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(wf))

    configured = {"libero_pi05", "molmoact", "unused"}
    assert required_policy_ids(wf_dir, configured) == {
        "libero_pi05", "molmoact",
    }
    # Nothing configured -> nothing required.
    assert required_policy_ids(wf_dir, set()) == set()
    # Unreadable workflow -> empty (caller fails later with clear error).
    assert required_policy_ids(tmp_path / "nope", configured) == set()


def test_required_policies_boot_through_manager(tmp_path) -> None:
    """A workflow referencing a configured external policy validates its
    URL through the PolicyManager during the trial."""
    wf = {
        "version": 3,
        "meta": {"name": "p"},
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
    # Reference a policy via the legacy form so the scan finds it, but
    # give it a malformed URL -> the trial records the failure.
    wf["subgraphs"]["m"]["nodes"]["vla"] = {"type": "tool", "tool": "badpol"}
    wf["subgraphs"]["m"]["edges"] = [
        ["START", "vla"], ["vla", "check"], ["check", "ok"], ["ok", "END"],
    ]
    wf_dir = tmp_path / "wf"
    wf_dir.mkdir()
    (wf_dir / "workflow.json").write_text(json.dumps(wf))

    cfg = WorkerSetupConfig(
        suite_name="s",
        output_dir=str(tmp_path / "out"),
        policies={"badpol": {"url": "not-a-url"}},
    )
    state = worker_setup(0, cfg, connector_factory=stub_connector_factory)
    result = run_trial_on_worker(
        state, WorkItem(task_id=0, trial_id=1, workflow_dir=str(wf_dir)),
    )
    assert result.exit_code != 0
    assert "url" in result.execution_stderr.lower()
