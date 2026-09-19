"""Environment registry — env names → lazy factories + per-env config.

The registry is the seam between the connector layer and the simulation
envs: :func:`resolve` maps a user-facing env name (``"libero_object"``,
``"libero_object_all_variance"``, ``"libero_grocery_packing_object"``)
to a factory callable plus the canonical suite key to hand it.

Factories are registered as *lazy dotted paths* (``"gap.envs.libero_env:
make_env"``) so importing this module never pulls mujoco / robosuite /
libero — the heavy sim stack only loads when a factory is actually
resolved and called. Keep it that way: no top-level imports beyond the
stdlib.

Factory contract::

    make_env(suite_name, task_id, camera_names, enable_render, **extra)
        -> (env, EnvConfig)

where ``env`` satisfies the :class:`gap.envs.base_env.BaseEnv` surface and
``EnvConfig`` carries the static robot/control metadata the connector
needs (action mode, DOF, home joints, TCP, cameras).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperSpec:
    """What the hand on this arm is, in the terms skills reason about.

    Declared, not measured: the numbers come from the hand's datasheet or
    MJCF, and a skill that reads them stays portable because it never has to
    know *which* hand it is holding. Defaults describe the Panda parallel jaw
    that robosuite/LIBERO ship.
    """

    name: str = "panda_hand"
    #: Widest jaw gap, metres. The refusal threshold for grasp proposals.
    span_m: float = 0.08
    #: Narrowest useful gap — below this the jaws are effectively shut.
    min_grasp_width_m: float = 0.0
    #: Unit axis, in EE frame, the jaws translate ALONG as they close.
    close_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    #: Unit axis, in EE frame, pointing out of the palm toward the object.
    approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: Fingertip reach past the TCP, metres. ``None`` = unstated, which
    #: tells the grasp skills to skip their support-surface correction
    #: rather than apply it with a made-up number.
    finger_reach_m: float | None = None
    #: Margin kept between fingertip and support surface, metres.
    finger_clearance_m: float = 0.005
    #: Measured gap at fraction 0. ``None`` falls back to the span model.
    width_at_closed_m: float | None = None


@dataclass(frozen=True)
class WorkspaceSpec:
    """Where the work happens, in world-frame metres.

    ``surface_z`` is the one number most worth overriding per scene; a sim
    connector measures it from the table body instead (see
    :meth:`gap.connector.sim.SimConnector._workspace_payload`).
    """

    #: Height of the primary support surface.
    surface_z: float = 0.0
    #: Cruise height for carrying an object across the scene. ``None``
    #: derives ``surface_z + 0.25``.
    transport_z: float | None = None
    #: Hover clearance above an object before descending onto it.
    align_clearance_m: float = 0.08


@dataclass(frozen=True)
class EnvConfig:
    """Static per-env robot/control metadata consumed by the connector."""

    arm_dof: int = 7
    num_arms: int = 1
    action_mode: str = "absolute_joints"
    control_freq: float = 20.0
    home_joints: tuple | None = None
    joint_names: tuple[str, ...] | None = None
    joint_limits: tuple[tuple[float, float], ...] | None = None
    tcp_offset: tuple | None = None
    tcp_rotation_z: float | None = None
    arm_bases: tuple | None = None
    robot_urdf_path: str | None = None
    default_cameras: tuple = ("agentview", "robot0_eye_in_hand")
    is_real: bool = False
    #: Embodiment description surfaced by ``robot.describe_gripper`` /
    #: ``robot.describe_workspace``. Geometry that is not explicitly
    #: declared stays unstated; concrete env factories should override it
    #: when they have a measured embodiment value.
    gripper: GripperSpec = GripperSpec()
    workspace: WorkspaceSpec = WorkspaceSpec()


@dataclass(frozen=True)
class _Entry:
    """One registry row: where the factory lives and what key it gets."""

    factory_path: str  # "module.path:attr" — imported lazily by resolve()
    key: str  # canonical suite name passed to the factory
    prefix: bool  # True: entry also matches any env_name it prefixes


_REGISTRY: dict[str, _Entry] = {}


def register_env(
    name: str,
    factory_dotted_path: str,
    prefix: bool = False,
    *,
    key: str | None = None,
) -> None:
    """Register an env name.

    Args:
        name: User-facing env name (exact match) or prefix when
            ``prefix=True``.
        factory_dotted_path: Lazy ``"module.path:attr"`` reference to the
            factory; imported only when :func:`resolve` is called.
        prefix: When True the entry also matches any ``env_name`` that
            starts with ``name`` (the matched env_name itself becomes the
            suite key, e.g. ``"libero_object_with_mug"`` under the
            ``"libero"`` prefix entry).
        key: Canonical suite name handed to the factory; defaults to
            ``name``. Use for aliases (dev configs say
            ``libero_grocery_packing_object``, the vab task dir is
            ``libero_object_packing``).
    """
    _REGISTRY[name] = _Entry(factory_dotted_path, key or name, prefix)


def _load_factory(dotted_path: str) -> Callable:
    module_name, _, attr = dotted_path.partition(":")
    if not attr:
        raise ValueError(
            f"factory path {dotted_path!r} must be 'module.path:attr'"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def resolve(env_name: str) -> tuple[Callable, str]:
    """Resolve an env name to ``(factory, suite_key)``.

    Exact registrations win; otherwise the longest matching ``prefix=True``
    entry is used with ``env_name`` itself as the suite key. Raises
    ``KeyError`` for unknown names.
    """
    entry = _REGISTRY.get(env_name)
    if entry is not None:
        return _load_factory(entry.factory_path), entry.key

    prefix_matches = [
        name
        for name, e in _REGISTRY.items()
        if e.prefix and env_name.startswith(name)
    ]
    if prefix_matches:
        best = max(prefix_matches, key=len)
        return _load_factory(_REGISTRY[best].factory_path), env_name

    raise KeyError(
        f"Unknown env {env_name!r}. Registered: {sorted(_REGISTRY)}"
    )


def registered_envs() -> dict[str, str]:
    """Mapping of registered names to their canonical suite keys."""
    return {name: entry.key for name, entry in _REGISTRY.items()}


# ---------------------------------------------------------------------------
# Pre-registrations
# ---------------------------------------------------------------------------

_LIBERO_FACTORY = "gap.envs.libero_env:make_env"

# Default: any libero* suite name routes to the libero factory; the loader
# then decides per suite between the vab task-dir format and the LIBERO-PRO
# benchmark registry (see gap.envs.loader.load_libero_task).
register_env("libero", _LIBERO_FACTORY, prefix=True)

# Variational-Automation-Benchmark (vab) suites — self-contained YAML task
# dirs under third_party/Variational-Automation-Benchmark/tasks/<name>/.
for _suite in (
    "libero_object_all_variance",
    "libero_object_target_pos_var20x20",
    "libero_object_target_permutation_variance",
    "libero_object_target_basket_swap_variance",
    "libero_object_packing",
    "permutation_packing",
):
    register_env(_suite, _LIBERO_FACTORY)

# Dev-config aliases for the packing suites.
register_env(
    "libero_grocery_packing_object", _LIBERO_FACTORY, key="libero_object_packing"
)
register_env(
    "libero_grocery_packing_permutation", _LIBERO_FACTORY, key="permutation_packing"
)

# Real hardware. ``franka_real`` is the robots_realtime msgpack bridge
# (the env binds the server; gap.connector.real() optionally spawns the
# rr-session client); ``ur_zed`` is the perception-only UR + ZED env
# (direct pyzed capture + read-only RTDE joint state).
register_env("franka_real", "gap.envs.franka_real_env:make_env")
register_env("ur_zed", "gap.envs.ur_zed_env:make_env")

# Classic LIBERO-PRO benchmark-registry suites. The "libero" prefix entry
# already covers every libero_* name; these exact rows pin the canonical
# benchmark suites for discoverability (registered_envs()).
for _suite in (
    "libero_object",
    "libero_spatial",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_object_swap",
    "libero_object_basket_swap",
):
    register_env(_suite, _LIBERO_FACTORY)
