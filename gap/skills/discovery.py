"""Locate an open-robot-skills checkout without hardcoded relative paths.

Every gap surface that accepts a skills path (``gap.execute``,
``gap.agent.generate``, the CLI ``--skills`` flags, benchmark configs)
defaults to :func:`find_skills_path`, so users never have to spell
``../open-robot-skills`` in commands or configs. Resolution order:

1. an explicit path argument (``--skills`` / ``skills=``);
2. the ``GAP_SKILLS_PATH`` environment variable;
3. a ``open-robot-skills`` directory found by walking up from the installed
   ``gap`` package and from the current working directory — the
   documented side-by-side checkout layout. A directory counts as a
   checkout when it has both ``tools/`` and ``skills/`` roots containing
   at least one ``SKILL.md`` bundle each.

When nothing is found the result is ``None`` (or, with
``required=True``, a :class:`FileNotFoundError` whose message lists
exactly what was tried).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Environment variable naming an open-robot-skills checkout root.
GAP_SKILLS_PATH_ENV = "GAP_SKILLS_PATH"

#: Directory name of the sibling checkout the walk looks for.
_SIBLING_DIR_NAME = "open-robot-skills"


def looks_like_skills_checkout(path: str | Path) -> bool:
    """True iff *path* is a directory with both bundle roots populated.

    An open-robot-skills checkout has ``tools/`` and ``skills/`` directories, each
    containing at least one bundle (a subdirectory with a ``SKILL.md``).
    """
    p = Path(path)
    if not p.is_dir():
        return False
    for folder in ("tools", "skills"):
        root = p / folder
        if not root.is_dir():
            return False
        try:
            has_bundle = any(
                (child / "SKILL.md").is_file()
                for child in root.iterdir()
                if child.is_dir() and not child.name.startswith("_")
            )
        except OSError:
            return False
        if not has_bundle:
            return False
    return True


def find_skills_path(
    explicit: str | Path | None = None,
    *,
    required: bool = False,
    search_from: Iterable[str | Path] | None = None,
) -> Path | None:
    """Resolve the open-robot-skills checkout to use.

    Args:
        explicit: An explicitly-provided path (CLI flag / function
            argument). Returned as-is (resolved) when given — explicit
            always wins.
        required: When True, raise :class:`FileNotFoundError` (with a
            message listing everything that was tried) instead of
            returning ``None``.
        search_from: Override the sibling-walk starting points (defaults
            to the installed ``gap`` package directory and the current
            working directory). Exposed for tests.

    Returns:
        The checkout path, or ``None`` when nothing was found and
        ``required`` is False.

    Raises:
        FileNotFoundError: nothing found and ``required=True``.
        ValueError: ``GAP_SKILLS_PATH`` is set but does not point at a
            open-robot-skills checkout (an explicitly-set env var must be valid
            — silently falling back would mask typos).
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    tried: list[str] = ["explicit path argument (not given)"]

    env = os.environ.get(GAP_SKILLS_PATH_ENV, "").strip()
    if env:
        p = Path(env).expanduser()
        if looks_like_skills_checkout(p):
            return p.resolve()
        raise ValueError(
            f"${GAP_SKILLS_PATH_ENV}={env!r} is not an open-robot-skills checkout "
            f"(expected a directory with tools/ and skills/ bundle roots, "
            f"each containing at least one SKILL.md bundle)"
        )
    tried.append(f"${GAP_SKILLS_PATH_ENV} (not set)")

    if search_from is None:
        search_from = (Path(__file__).resolve().parent, Path.cwd())

    seen: set[Path] = set()
    for start in search_from:
        start = Path(start).resolve()
        for root in (start, *start.parents):
            if root in seen:
                continue
            seen.add(root)
            # Running *inside* a checkout counts (e.g. `gap skills list`
            # from the open-robot-skills repo root).
            if looks_like_skills_checkout(root):
                return root
            sibling = root / _SIBLING_DIR_NAME
            if looks_like_skills_checkout(sibling):
                return sibling.resolve()
        tried.append(
            f"a '{_SIBLING_DIR_NAME}' directory in {start} or any of its parents"
        )

    message = (
        "no open-robot-skills checkout found. Tried, in order: "
        + "; ".join(f"({i}) {t}" for i, t in enumerate(tried, 1))
        + ". Clone open-robot-skills next to the graph-as-policy checkout, or set "
        f"${GAP_SKILLS_PATH_ENV}=/path/to/open-robot-skills, or pass an explicit "
        "skills path."
    )
    if required:
        raise FileNotFoundError(message)
    logger.debug("%s", message)
    return None
