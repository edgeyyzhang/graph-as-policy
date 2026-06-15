"""Pure (no-GPU, CI-safe) tests for the benchmark grid config."""

from __future__ import annotations

from pathlib import Path

import pytest

from gap.benchmark.config import (
    FAMILY_SUITES,
    KNOWN_FAMILIES,
    KNOWN_MODES,
    POSVAR_SUITES,
    BenchmarkConfig,
)

from .conftest import EXAMPLES

SMOKE_YAML = EXAMPLES / "smoke.yaml"
POSVAR_YAML = EXAMPLES / "posvar.yaml"
ACCEPT_YAML = EXAMPLES / "grocery_acceptance.yaml"
ACCEPT_SMOKE_YAML = EXAMPLES / "grocery_acceptance_smoke.yaml"


# --------------------------------------------------------------------------
# family -> suite expansion
# --------------------------------------------------------------------------


def test_posvar_suites_are_the_four_variations() -> None:
    assert set(POSVAR_SUITES) == {"pos_var", "permutation", "basket_swap", "all"}
    assert POSVAR_SUITES["pos_var"] == "libero_object_target_pos_var20x20"
    assert POSVAR_SUITES["all"] == "libero_object_all_variance"


def test_family_suites_shape() -> None:
    assert set(KNOWN_FAMILIES) == {
        "libero", "libero_pro", "posvar", "grocery_packing",
    }
    assert FAMILY_SUITES["libero"] == {"object": "libero_object"}
    assert FAMILY_SUITES["libero_pro"] == {"object_swap": "libero_object_swap"}
    assert FAMILY_SUITES["posvar"] is POSVAR_SUITES
    # grocery_packing maps to the canonical gap suite names.
    assert FAMILY_SUITES["grocery_packing"] == {
        "object": "libero_object_packing",
        "permutation": "permutation_packing",
    }


def test_grocery_suites_resolve_through_env_registry_aliases() -> None:
    """The family's suites and the dev aliases land on the same keys."""
    from gap.envs.registry import registered_envs

    envs = registered_envs()
    for canonical in FAMILY_SUITES["grocery_packing"].values():
        assert envs[canonical] == canonical
    # The dev recipes say libero_grocery_packing_*; the registry aliases
    # them onto the same canonical suites the family map uses.
    assert envs["libero_grocery_packing_object"] == "libero_object_packing"
    assert envs["libero_grocery_packing_permutation"] == "permutation_packing"


def test_grid_cells_union_over_families() -> None:
    cfg = BenchmarkConfig(
        source_yaml=Path("x.yaml"),
        families=["libero", "grocery_packing"],
    )
    assert cfg.grid_cells() == [
        ("libero", "object", "libero_object"),
        ("grocery_packing", "object", "libero_object_packing"),
        ("grocery_packing", "permutation", "permutation_packing"),
    ]


def test_variations_filter_within_family() -> None:
    cfg = BenchmarkConfig(
        source_yaml=Path("x.yaml"),
        families=["libero_pro", "grocery_packing"],
        variations=["object_swap", "permutation"],
    )
    assert cfg.grid_cells() == [
        ("libero_pro", "object_swap", "libero_object_swap"),
        ("grocery_packing", "permutation", "permutation_packing"),
    ]


def test_monolithic_mode_deleted() -> None:
    assert "monolithic" not in KNOWN_MODES
    with pytest.raises(ValueError, match="unknown mode"):
        BenchmarkConfig(source_yaml=Path("x.yaml"), modes=["monolithic"])


def test_unknown_family_variation_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        BenchmarkConfig(source_yaml=Path("x.yaml"), families=["bogus"])
    with pytest.raises(ValueError, match="unknown variation"):
        BenchmarkConfig(
            source_yaml=Path("x.yaml"),
            families=["libero"],
            variations=["pos_var"],  # posvar-only, not in libero
        )
    with pytest.raises(ValueError, match="unknown mode"):
        BenchmarkConfig(source_yaml=Path("x.yaml"), modes=["bogus"])


def test_resolved_task_ids() -> None:
    cfg = BenchmarkConfig(source_yaml=Path("x.yaml"), n_tasks=3)
    assert cfg.resolved_task_ids() == [0, 1, 2]
    cfg = BenchmarkConfig(source_yaml=Path("x.yaml"), task_ids=[4, 7])
    assert cfg.resolved_task_ids() == [4, 7]


# --------------------------------------------------------------------------
# YAML loading (grid mode)
# --------------------------------------------------------------------------


def test_smoke_yaml_parses(skills_root) -> None:
    cfg = BenchmarkConfig.from_yaml(SMOKE_YAML)
    assert cfg.suites_mode is False
    assert cfg.families == ["grocery_packing"]
    assert cfg.variations == ["object"]
    assert cfg.modes == ["llm_generation"]
    assert cfg.n_tasks == 1 and cfg.n_seeds == 1 and cfg.num_workers == 1
    assert cfg.record_video is True
    assert cfg.smoke is True
    assert cfg.grid_cells() == [
        ("grocery_packing", "object", "libero_object_packing"),
    ]
    pc = cfg.pipeline_config
    assert pc is not None and pc.skills is not None


def test_posvar_yaml_parses_with_mode_overrides(skills_root) -> None:
    cfg = BenchmarkConfig.from_yaml(POSVAR_YAML)
    assert cfg.families == ["posvar"]
    assert cfg.variations == ["pos_var", "permutation", "basket_swap", "all"]
    assert cfg.modes == ["llm_generation", "llm_plus_policy", "policy_only"]
    assert cfg.n_tasks == 10 and cfg.n_seeds == 50

    # policies block carried for the policy modes.
    pc = cfg.pipeline_config
    assert pc is not None
    assert pc.policies["pi05-libero"]["url"].startswith("ws://")

    # Override precedence + workflow_dir absolutized vs the YAML dir.
    eff_policy = cfg.effective("llm_plus_policy")
    eff_gen = cfg.effective("llm_generation")
    assert eff_policy["num_workers"] == 4            # override
    assert eff_gen["num_workers"] == cfg.num_workers  # inherited
    wf = eff_policy["workflow_dir"]
    assert wf is not None and Path(wf).is_absolute()
    # References the steered_policy example graph (the loop template).
    assert Path(wf).parts[-2:] == ("steered_policy", "graph_loop")
    assert eff_policy["n_seeds"] == cfg.n_seeds      # not overridden


def test_default_gate_threshold() -> None:
    cfg = BenchmarkConfig(source_yaml=Path("x.yaml"))
    assert cfg.gate_threshold == pytest.approx(0.90)


# --------------------------------------------------------------------------
# YAML loading (suites mode — the grocery acceptance shape)
# --------------------------------------------------------------------------


def test_grocery_acceptance_yaml_is_suites_mode(skills_root) -> None:
    cfg = BenchmarkConfig.from_yaml(ACCEPT_YAML)
    assert cfg.suites_mode is True
    assert cfg.gate_threshold == pytest.approx(0.90)
    assert cfg.grid_cells() == []  # no family grid in suites mode

    pc = cfg.pipeline_config
    assert pc is not None
    assert pc.task == "auto"
    assert len(pc.suites) == 10
    # 500 total: 10 tasks x 50 trials.
    assert pc.trials.trials_per_generation == 50
    assert pc.trials.task_ids == list(range(10))
    assert pc.trials.num_workers == 25
    assert pc.trials.record_video is True
    assert pc.trials.regenerate_code_per_trial is False
    assert pc.trials.task_timeout_secs == 900
    # The 10 hand-curated objects blocks survived the port verbatim.
    for i, suite in enumerate(pc.suites):
        assert suite.suite_name == "libero_object_all_variance"
        assert suite.task_ids == [i]
        assert i in suite.task_prompts
        assert set(suite.objects) == {"target", "expected_label", "shape_hint"}
        assert suite.num_workers == 1
    assert pc.suites[0].objects["expected_label"] == "Alphabet Soup"
    assert pc.suites[9].objects["target"] == "orange juice carton"
    # llm block: vertex + gemini flash lite preview on the right project.
    assert pc.llm.provider == "vertex"
    assert pc.llm.model == "gemini-3.1-flash-lite-preview"
    assert pc.llm.project_id == "bc-y7-06"


def test_grocery_acceptance_smoke_yaml(skills_root) -> None:
    cfg = BenchmarkConfig.from_yaml(ACCEPT_SMOKE_YAML)
    assert cfg.suites_mode is True
    pc = cfg.pipeline_config
    assert len(pc.suites) == 2
    assert pc.trials.trials_per_generation == 10  # 2 x 10 = 20 trials
    assert pc.trials.task_ids == [0, 1]
    assert pc.trials.num_workers <= 8
    assert pc.trials.task_timeout_secs == 900
    full = BenchmarkConfig.from_yaml(ACCEPT_YAML).pipeline_config
    for i, suite in enumerate(pc.suites):
        # Prompt + objects blocks identical to the full recipe.
        assert suite.task_prompts == full.suites[i].task_prompts
        assert suite.objects == full.suites[i].objects


def test_yaml_without_grid_or_suites_rejected(tmp_path) -> None:
    bad = tmp_path / "empty.yaml"
    bad.write_text("task: hello\n")
    with pytest.raises(ValueError, match="nothing to run"):
        BenchmarkConfig.from_yaml(bad)


def test_gate_threshold_from_yaml(tmp_path) -> None:
    y = tmp_path / "g.yaml"
    y.write_text(
        "task: t\n"
        "gate_threshold: 0.5\n"
        "suites:\n"
        "  - suite_name: libero_object\n"
    )
    cfg = BenchmarkConfig.from_yaml(y)
    assert cfg.gate_threshold == pytest.approx(0.5)

    y2 = tmp_path / "g2.yaml"
    y2.write_text(
        "task: t\n"
        "benchmark:\n"
        "  families: [libero]\n"
        "  gate_threshold: 0.75\n"
    )
    cfg2 = BenchmarkConfig.from_yaml(y2)
    assert cfg2.gate_threshold == pytest.approx(0.75)
