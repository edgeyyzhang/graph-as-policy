"""Fixture canonical script — exercises load_prompt via the synthetic package.

Registered as ``gap_skills.skills.fixture-skill.scripts.hello``; the
``load_prompt(__package__, ...)`` call below resolves the bundle root by
walking up from the synthetic package to the directory holding SKILL.md.
"""

from typing import TypedDict

from gap.skills import load_prompt


class Output(TypedDict):
    greeting: str


def run(ctx, who: str = "world", excited: bool = False) -> Output:
    return {"greeting": load_prompt(__package__, "greet", who=who, excited=excited)}
