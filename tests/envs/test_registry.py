"""gap.envs.registry tests — no sim stack required.

The registry is the connector's seam onto the env layer: these tests pin
the resolve/prefix/alias semantics, EnvConfig's shape, and the lazy-import
contract (the registry module must work with mujoco absent — factories are
dotted paths, imported only on resolve).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import pytest

from gap.envs.registry import _REGISTRY, EnvConfig, _Entry, register_env, resolve

# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------


def test_envconfig_defaults():
    cfg = EnvConfig()
    assert cfg.arm_dof == 7
    assert cfg.num_arms == 1
    assert cfg.action_mode == "absolute_joints"
    assert cfg.control_freq == 20.0
    assert cfg.home_joints is None
    assert cfg.tcp_offset is None
    assert cfg.tcp_rotation_z is None
    assert cfg.arm_bases is None
    assert cfg.robot_urdf_path is None
    assert cfg.default_cameras == ("agentview", "robot0_eye_in_hand")
    assert cfg.is_real is False


def test_envconfig_application():
    """Field overrides apply and the dataclass stays frozen/hashable."""
    cfg = EnvConfig(
        arm_dof=6,
        num_arms=2,
        action_mode="velocity_joints",
        control_freq=30.0,
        home_joints=(0.0, 1.047, 1.047, 0.0, 0.0, 0.0),
        tcp_offset=(0.0, 0.0, -0.1),
        tcp_rotation_z=0.785,
        arm_bases=((0.25, 0.31, 0.75), (0.25, -0.31, 0.75)),
        robot_urdf_path="panda_description",
        default_cameras=("agentview",),
        is_real=True,
    )
    assert cfg.arm_dof == 6
    assert cfg.arm_bases[1] == (0.25, -0.31, 0.75)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.arm_dof = 7  # type: ignore[misc]
    # frozen dataclass of hashable fields → usable as a dict key
    assert hash(cfg) == hash(dataclasses.replace(cfg))


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


def test_resolve_exact_suite():
    factory, key = resolve("libero_object")
    assert callable(factory)
    assert factory.__name__ == "make_env"
    assert key == "libero_object"


def test_resolve_vab_suites_registered():
    for suite in (
        "libero_object_all_variance",
        "libero_object_target_pos_var20x20",
        "libero_object_target_permutation_variance",
        "libero_object_target_basket_swap_variance",
        "libero_object_packing",
        "permutation_packing",
    ):
        factory, key = resolve(suite)
        assert callable(factory)
        assert key == suite


def test_resolve_aliases_map_to_canonical_suite():
    _, key = resolve("libero_grocery_packing_object")
    assert key == "libero_object_packing"
    _, key = resolve("libero_grocery_packing_permutation")
    assert key == "permutation_packing"


def test_resolve_prefix_passes_name_through():
    """Unregistered libero_* names fall back to the "libero" prefix entry."""
    factory, key = resolve("libero_object_with_mug")
    assert callable(factory)
    assert key == "libero_object_with_mug"


def test_resolve_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        resolve("droid_sim_scene1")


def test_register_env_prefix_and_key(monkeypatch):
    monkeypatch.setitem(
        _REGISTRY, "demo", _Entry("gap.envs.libero_env:make_env", "demo", True)
    )
    register_env("demo_alias", "gap.envs.libero_env:make_env", key="demo_canonical")
    try:
        _, key = resolve("demo_alias")
        assert key == "demo_canonical"
        _, key = resolve("demo_something")
        assert key == "demo_something"
    finally:
        _REGISTRY.pop("demo_alias", None)


# ---------------------------------------------------------------------------
# Lazy-import contract
# ---------------------------------------------------------------------------

_NO_MUJOCO_PROBE = textwrap.dedent(
    """
    import sys

    # Make any import of the sim stack raise ImportError.
    for blocked in ("mujoco", "robosuite", "libero", "torch"):
        sys.modules[blocked] = None

    import gap.envs.registry as registry

    # Registering + EnvConfig never touch the sim stack.
    registry.register_env("probe", "gap.envs.libero_env:make_env")
    assert registry.EnvConfig().arm_dof == 7

    # Importing the registry must not drag in the env modules.
    assert "gap.envs.libero_env" not in sys.modules

    # resolve() imports the factory *module* (gymnasium/viser) but still
    # no mujoco/libero/robosuite.
    factory, key = registry.resolve("libero_object_all_variance")
    assert callable(factory) and key == "libero_object_all_variance"
    assert "gap.envs.libero_env" in sys.modules

    print("REGISTRY_OK")
    """
)


def test_registry_imports_with_mujoco_absent():
    result = subprocess.run(
        [sys.executable, "-c", _NO_MUJOCO_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "REGISTRY_OK" in result.stdout
