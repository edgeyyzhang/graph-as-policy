"""Static lifecycle manager for learned-policy servers.

Graph is the source of truth. After workflow validation the executor
collects the set of ``policy_id``s referenced by policy states and asks
the manager to boot them all. If any required policy cannot be started
— unknown id, subprocess crash, readiness timeout, port conflict —
every subprocess this manager started is torn down and the error
propagates so ``gap benchmark`` (or ``gap run``) exits with a clear
message. There is no LRU, no eviction, no runtime registration of new
policies. Call :meth:`shutdown_all` in a ``finally`` to make sure no
server outlives the run.

Registry entries are bundle-owned: :func:`gap.runtime.policy_boot.boot_policies`
walks the workflow, discovers each ``kind='policy'`` skill bundle it
references, and translates the bundle's SKILL.md ``gap.serving:`` block
into one entry per preset. A user-supplied ``policies:`` block in
task.yaml can override the bundle's recipe (or point at an external
server). Each entry is either:

- **managed**: carries ``command`` (a list[str], NOT a shell string) and
  optionally ``bundle_dir``. When ``bundle_dir`` is set, the manager
  prepends ``uv run --project <bundle_dir> --`` so the spawn runs in
  the bundle's own venv. A ``{port}`` token in any ``command`` element
  is substituted at spawn time.
- **external**: carries ``url``. Already running; the manager only
  records the URL and never touches the process.

Specifying both ``url`` and ``command`` is a config error.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class PolicyConfigError(ValueError):
    """Raised when a policy registry entry or referenced id is invalid."""


class PolicyStartupError(RuntimeError):
    """Raised when a managed policy subprocess fails to come up."""


@dataclass
class _Managed:
    """Internal record for a subprocess the manager spawned."""

    policy_id: str
    proc: subprocess.Popen
    url: str
    port: int
    started_at: float = field(default_factory=time.time)


def _allocate_free_port() -> int:
    """Ask the OS for a free TCP port by binding to 0 and reading back.

    There's an inherent race between us closing the socket and the
    spawned subprocess binding to the same port, but the window is tiny
    and the alternative (binding the port ourselves and passing the fd
    to the child) doesn't generalize across ``scripts/serve_policy.py``'s
    CLI shape.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tcp_ready(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if ``host:port`` accepts a TCP connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_ws_url(url: str) -> tuple[str, int]:
    """Parse ``ws://host:port`` (or ``wss://`` / bare ``host:port``) into a tuple."""
    stripped = url
    for scheme in ("ws://", "wss://", "http://", "https://"):
        if stripped.startswith(scheme):
            stripped = stripped[len(scheme):]
            break
    if "/" in stripped:
        stripped = stripped.split("/", 1)[0]
    if ":" not in stripped:
        raise PolicyConfigError(
            f"url {url!r}: expected ws://host:port (missing port)"
        )
    host, port_str = stripped.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError as exc:
        raise PolicyConfigError(
            f"url {url!r}: port {port_str!r} is not an integer"
        ) from exc
    return host, port


class PolicyManager:
    """Boot all graph-required policies at compile time; tear them all down on exit.

    Typical use (from the launcher)::

        manager = PolicyManager(
            entries=task_cfg.get("policies", {}),
            startup_timeout_s=task_cfg.get("policy_manager", {})
                .get("startup_timeout_s", 120.0),
        )
        manager.boot_all(required_ids)    # raises on failure
        try:
            executor.execute()
        finally:
            manager.shutdown_all()
    """

    def __init__(
        self,
        entries: dict[str, dict[str, Any]] | None,
        startup_timeout_s: float = 120.0,
        evict_grace_s: float = 10.0,
    ) -> None:
        self._entries: dict[str, dict[str, Any]] = dict(entries or {})
        self._timeout = float(startup_timeout_s)
        self._grace = float(evict_grace_s)
        self._managed: dict[str, _Managed] = {}
        self._urls: dict[str, str] = {}
        self._lock = threading.Lock()
        self._booted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def boot_all(self, required_ids: Iterable[str]) -> None:
        """Resolve, validate, and start every required policy in parallel.

        On ANY failure, any subprocess already started by this manager is
        terminated and the original exception is re-raised.
        """
        required = set(required_ids)
        if not required:
            self._booted = True
            return

        # 1. Resolve entries + split managed vs external. Fail early on
        #    unknown ids, ambiguous entries, and malformed urls.
        resolved: dict[str, dict[str, Any]] = {}
        for pid in sorted(required):
            entry = self._entries.get(pid)
            if entry is None:
                raise PolicyConfigError(
                    f"workflow references policy {pid!r} but no entry "
                    f"exists. Either ship a `kind='policy'` skill bundle "
                    f"named {pid!r} (boot_policies auto-resolves its "
                    f"`gap.serving:` block), or add an entry to the task "
                    f"config's `policies:` block with `command:` (managed) "
                    f"or `url:` (external)."
                )
            if not isinstance(entry, dict):
                raise PolicyConfigError(
                    f"policy {pid!r}: entry must be a mapping, got "
                    f"{type(entry).__name__}"
                )
            if "url" in entry and "command" in entry:
                raise PolicyConfigError(
                    f"policy {pid!r}: specify either 'url' or 'command', "
                    f"not both"
                )
            if "url" not in entry and "command" not in entry:
                raise PolicyConfigError(
                    f"policy {pid!r}: entry must declare 'url' (external) "
                    f"or 'command' (managed)"
                )
            resolved[pid] = entry

        to_start: list[tuple[str, dict[str, Any]]] = []
        for pid, entry in resolved.items():
            if "url" in entry:
                # Validate the URL shape eagerly so we fail fast.
                _parse_ws_url(str(entry["url"]))
                self._urls[pid] = str(entry["url"])
            else:
                to_start.append((pid, entry))

        if not to_start:
            self._booted = True
            return

        # 2. Start managed policies in parallel. Any failure terminates
        #    every started subprocess and re-raises.
        logger.info(
            "PolicyManager: booting %d managed polic%s (%s)",
            len(to_start),
            "y" if len(to_start) == 1 else "ies",
            ", ".join(pid for pid, _ in to_start),
        )
        failures: list[tuple[str, BaseException]] = []
        started: list[_Managed] = []
        with ThreadPoolExecutor(max_workers=max(1, len(to_start))) as pool:
            futures = {
                pool.submit(self._start_one, pid, entry): pid
                for pid, entry in to_start
            }
            try:
                # Global wall-clock cap so a single stuck start doesn't hold
                # the whole fleet forever. Per-start waits use the same
                # budget internally.
                for fut in as_completed(futures, timeout=self._timeout):
                    pid = futures[fut]
                    try:
                        m = fut.result()
                    except BaseException as exc:  # includes KeyboardInterrupt
                        failures.append((pid, exc))
                    else:
                        started.append(m)
            except TimeoutError:
                # as_completed timed out waiting on one or more starts.
                for fut, pid in futures.items():
                    if not fut.done():
                        failures.append((
                            pid,
                            PolicyStartupError(
                                f"policy {pid!r}: did not become ready "
                                f"within {self._timeout}s"
                            ),
                        ))

        if failures:
            # Best-effort cleanup of anything that did start, then surface
            # the first failure (attach all for context).
            with self._lock:
                for m in started:
                    self._managed[m.policy_id] = m
                    self._urls[m.policy_id] = m.url
            self.shutdown_all()
            first_pid, first_exc = failures[0]
            msg = (
                f"failed to boot {len(failures)} polic"
                f"{'y' if len(failures) == 1 else 'ies'}: "
                + "; ".join(f"{pid} ({exc})" for pid, exc in failures)
            )
            if isinstance(first_exc, PolicyConfigError):
                raise PolicyConfigError(msg) from first_exc
            raise PolicyStartupError(msg) from first_exc

        # 3. Record started procs. ``started`` already excludes externals.
        with self._lock:
            for m in started:
                self._managed[m.policy_id] = m
                self._urls[m.policy_id] = m.url
        logger.info(
            "PolicyManager: all %d polic%s ready",
            len(resolved), "y" if len(resolved) == 1 else "ies",
        )
        self._booted = True

    def url_for(self, policy_id: str) -> str:
        """Return the websocket URL for a policy loaded via :meth:`boot_all`."""
        try:
            return self._urls[policy_id]
        except KeyError as exc:
            raise PolicyConfigError(
                f"policy {policy_id!r}: not loaded; PolicyManager.boot_all "
                f"must be called with this id before url_for() is used"
            ) from exc

    def loaded_ids(self) -> list[str]:
        """Return the list of policy ids with a known URL."""
        return sorted(self._urls.keys())

    def shutdown_all(self) -> None:
        """Terminate every subprocess this manager started. Idempotent.

        External (``url:``) entries are never touched; we didn't start
        them. The URL map is also cleared for managed ids so post-shutdown
        ``loaded_ids()`` only reports still-live external entries.
        """
        with self._lock:
            managed = list(self._managed.values())
            self._managed.clear()
            for m in managed:
                self._urls.pop(m.policy_id, None)

        if not managed:
            return

        logger.info(
            "PolicyManager: shutting down %d managed polic%s",
            len(managed), "y" if len(managed) == 1 else "ies",
        )
        for m in managed:
            _terminate_group(m.proc, self._grace)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_one(self, policy_id: str, entry: dict[str, Any]) -> _Managed:
        """Spawn one subprocess and wait for its port to accept connections.

        Returns the managed record; raises ``PolicyStartupError`` on any
        failure. Guaranteed not to leak a subprocess: on any error path
        before return, the started process is terminated.

        Spawn shape:
          - ``entry["command"]`` MUST be a list[str]. A ``{port}`` token
            in any element is substituted at spawn time.
          - If ``entry["bundle_dir"]`` is set, the manager prepends
            ``uv run --project <bundle_dir> --`` so the spawn activates
            the bundle's own venv (no shell, no env-var expansion).
          - Without ``bundle_dir``, the command is spawned directly via
            ``subprocess.Popen(argv, shell=False)`` — useful when the
            user provides a self-contained absolute-path command in
            task.yaml.
        """
        command = entry["command"]
        if not isinstance(command, list) or not command:
            raise PolicyConfigError(
                f"policy {policy_id!r}: 'command' must be a non-empty "
                f"list[str] (NOT a shell string)"
            )
        env_overrides = entry.get("env") or {}
        if not isinstance(env_overrides, dict):
            raise PolicyConfigError(
                f"policy {policy_id!r}: 'env' must be a mapping of "
                f"str -> str, got {type(env_overrides).__name__}"
            )

        port = _allocate_free_port()
        argv = [str(arg).format(port=port) if "{port}" in str(arg) else str(arg)
                for arg in command]
        bundle_dir = entry.get("bundle_dir")
        cwd = None
        if bundle_dir is not None:
            argv = ["uv", "run", "--project", str(bundle_dir), "--", *argv]
            # cwd at the bundle root: the bundle's `command` can reference
            # its own files by relative path (e.g. `server.py`) without
            # depending on env-var expansion.
            cwd = str(bundle_dir)
        env = {**os.environ, **{str(k): str(v) for k, v in env_overrides.items()}}

        logger.info(
            "[policy:%s] spawning on port %d: %s",
            policy_id, port, " ".join(argv),
        )
        try:
            proc = subprocess.Popen(
                argv,
                shell=False,
                env=env,
                cwd=cwd,
                start_new_session=True,  # own process group for clean teardown
                stdout=None, stderr=None,
            )
        except OSError as exc:
            raise PolicyStartupError(
                f"policy {policy_id!r}: failed to spawn subprocess "
                f"({type(exc).__name__}: {exc}); argv was {argv!r}"
            ) from exc

        url = f"ws://127.0.0.1:{port}"
        try:
            self._wait_until_ready(policy_id, proc, "127.0.0.1", port)
        except BaseException:
            _terminate_group(proc, self._grace)
            raise

        return _Managed(policy_id=policy_id, proc=proc, url=url, port=port)

    def _wait_until_ready(
        self,
        policy_id: str,
        proc: subprocess.Popen,
        host: str,
        port: int,
    ) -> None:
        """Block until ``host:port`` accepts TCP connections or timeout hits.

        If the subprocess exits during the wait, surface its return code
        immediately — no point waiting the full timeout for a dead proc.
        """
        deadline = time.time() + self._timeout
        poll_interval = 0.5
        while True:
            rc = proc.poll()
            if rc is not None:
                raise PolicyStartupError(
                    f"policy {policy_id!r}: subprocess exited early with "
                    f"returncode {rc} before port {port} became ready"
                )
            if _tcp_ready(host, port, timeout=0.5):
                logger.info(
                    "[policy:%s] ready on %s:%d", policy_id, host, port,
                )
                return
            if time.time() >= deadline:
                raise PolicyStartupError(
                    f"policy {policy_id!r}: port {port} did not open within "
                    f"{self._timeout}s"
                )
            time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _terminate_group(proc: subprocess.Popen, grace_s: float) -> None:
    """SIGTERM the process group, wait up to ``grace_s``, then SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=max(0.1, grace_s))
        return
    except subprocess.TimeoutExpired:
        pass

    # Force-kill the whole group.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning(
            "policy subprocess pid=%d did not exit after SIGKILL",
            proc.pid,
        )


__all__ = [
    "PolicyManager",
    "PolicyConfigError",
    "PolicyStartupError",
]
