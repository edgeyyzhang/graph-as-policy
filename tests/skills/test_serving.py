"""Unit tests for the ``gap.serving`` frontmatter block.

Parsing lives in ``parse_skill_md`` and is mechanical — a typo'd serving
block must not silently disable a launcher invocation. The launcher (PR 2)
will raise if a ``kind='policy'`` bundle omits ``serving:`` at boot time;
here we only test that the parser's hard rules hold.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gap_core.skills import Serving

from gap.skills import parse_skill_md

FRONT = """\
---
name: {name}
description: A fixture bundle. Use when testing serving parsing.
metadata: {{category: test, tags: [test]}}
gap:
  tools:
    - {name}.run: A fixture tool.
{serving}---
"""


def write_bundle(
    tmp_path: Path,
    *,
    serving: str = "",
    name: str = "srv-fixture",
) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "SKILL.md").write_text(FRONT.format(name=name, serving=serving))
    return d / "SKILL.md"


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


def test_full_block_round_trips(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: [python, -m, pi05_libero.server, --port, '{port}']\n"
            "    protocol: websocket\n"
            "    env: {CUDA_VISIBLE_DEVICES: '0'}\n"
            "    requires_gpu: true\n"
            "    weights_uri: s3://openpi-assets/checkpoints/pi05_libero\n"
        ),
    )
    meta = parse_skill_md(md)
    assert meta.serving == Serving(
        command=["python", "-m", "pi05_libero.server", "--port", "{port}"],
        protocol="websocket",
        env={"CUDA_VISIBLE_DEVICES": "0"},
        requires_gpu=True,
        weights_uri="s3://openpi-assets/checkpoints/pi05_libero",
    )


def test_minimal_block_defaults(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: [python, -m, gap_tool_server, --bundle, srv-fixture]\n"
        ),
    )
    meta = parse_skill_md(md)
    assert meta.serving is not None
    assert meta.serving.command == [
        "python", "-m", "gap_tool_server", "--bundle", "srv-fixture",
    ]
    # Defaults: in-process protocol, empty env, no gpu, no weights uri.
    assert meta.serving.protocol == "in-process"
    assert meta.serving.env == {}
    assert meta.serving.requires_gpu is False
    assert meta.serving.weights_uri == ""


def test_absent_block_is_none(tmp_path: Path):
    meta = parse_skill_md(write_bundle(tmp_path, name="srv-absent"))
    assert meta.serving is None


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


def test_bare_serving_rejected(tmp_path: Path):
    # YAML null is rejected — a serving block carries a required `command`,
    # so omitting it is meaningful (not "I declare empty").
    md = write_bundle(tmp_path, serving="  serving:\n")
    with pytest.raises(ValueError, match="command"):
        parse_skill_md(md)


def test_command_must_be_list(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: cd $X && uv run scripts/serve_policy.py --port {port}\n"
        ),
    )
    with pytest.raises(ValueError, match="non-empty list"):
        parse_skill_md(md)


def test_command_must_be_non_empty(tmp_path: Path):
    md = write_bundle(
        tmp_path, serving="  serving:\n    command: []\n",
    )
    with pytest.raises(ValueError, match="non-empty list"):
        parse_skill_md(md)


def test_unknown_subkey_rejected(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: [python, -m, x]\n"
            "    cmd: extra\n"
        ),
    )
    with pytest.raises(ValueError, match="unknown keys.*cmd"):
        parse_skill_md(md)


def test_unknown_protocol_rejected(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: [python, -m, x]\n"
            "    protocol: grpc\n"
        ),
    )
    with pytest.raises(ValueError, match="protocol.*grpc"):
        parse_skill_md(md)


def test_env_must_be_mapping(tmp_path: Path):
    md = write_bundle(
        tmp_path,
        serving=(
            "  serving:\n"
            "    command: [python, -m, x]\n"
            "    env: [FOO=bar]\n"
        ),
    )
    with pytest.raises(ValueError, match="env must be a mapping"):
        parse_skill_md(md)


def test_top_level_serving_rejected_with_nesting_hint(tmp_path: Path):
    d = tmp_path / "srv-toplevel"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: srv-toplevel\ndescription: x\n"
        "serving: {command: [python, -m, x]}\n---\n"
    )
    with pytest.raises(ValueError, match="nest them under the `gap:` key"):
        parse_skill_md(d / "SKILL.md")


def test_non_mapping_serving_rejected(tmp_path: Path):
    md = write_bundle(tmp_path, serving="  serving: [foo]\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_skill_md(md)
