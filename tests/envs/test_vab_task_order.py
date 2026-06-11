"""``_vab_task_files`` numbering tests — no sim stack required.

The vab ``libero_object*variance`` suites must enumerate tasks in classic
LIBERO ``libero_object`` order (0 = alphabet soup, 1 = cream cheese, ...),
matching how the dev sim bridge resolved task ids through LIBERO's
benchmark registry. A bare lexicographic sort renumbers 8 of the 10 tasks
(1 becomes bbq sauce), silently desynchronizing every config that
addresses tasks by classic id — the generated graph then picks the object
the config asked for while the env scores a different target (the G1
grocery-acceptance task_01 failure).
"""

from __future__ import annotations

from pathlib import Path

from gap.envs.loader import (
    _CLASSIC_LIBERO_OBJECT_TASK_ORDER,
    _vab_task_files,
)


def _touch_all(d: Path, stems: list[str]) -> None:
    for s in stems:
        (d / f"{s}.yaml").write_text("task: stub\n")


def test_libero_object_suites_use_classic_numbering(tmp_path):
    # Files created in (and globbed back in) lexicographic order...
    _touch_all(tmp_path, sorted(_CLASSIC_LIBERO_OBJECT_TASK_ORDER))

    files = _vab_task_files(tmp_path)

    # ...but enumerated in classic LIBERO order.
    assert [f.stem for f in files] == list(_CLASSIC_LIBERO_OBJECT_TASK_ORDER)
    assert files[0].stem == "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    assert files[1].stem == "pick_up_the_cream_cheese_and_place_it_in_the_basket"
    assert files[3].stem == "pick_up_the_bbq_sauce_and_place_it_in_the_basket"


def test_classic_order_matches_libero_pro_task_map():
    """Guard the constant against drift from the vendored LIBERO fork."""
    task_map = (
        Path(__file__).resolve().parents[2]
        / "third_party" / "LIBERO-PRO" / "libero" / "libero"
        / "benchmark" / "libero_suite_task_map.py"
    )
    if not task_map.is_file():
        import pytest

        pytest.skip("LIBERO-PRO fork not materialized")
    src = task_map.read_text()
    block = src.split('"libero_object": [', 1)[1].split("]", 1)[0]
    stems = [s.strip().strip('",') for s in block.splitlines() if s.strip().strip('",')]
    assert tuple(stems) == _CLASSIC_LIBERO_OBJECT_TASK_ORDER


def test_non_object_suites_keep_lexicographic_order(tmp_path):
    stems = [f"pack_all_objects_v{i:02d}" for i in range(10)]
    _touch_all(tmp_path, stems)

    files = _vab_task_files(tmp_path)

    assert [f.stem for f in files] == stems
