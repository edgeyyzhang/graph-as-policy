"""Builder-block error feedback.

A subgraph builder block that execs to an error (e.g. a literal where a
``Ref()`` was required) must surface the *real* message through
``_parse_subgraph_response`` so ``run_subgraph_agent`` can feed it back to
the agent on retry, instead of the generic "no parseable subgraph found"
guidance. Weak models otherwise repeat the same mistake every attempt.
"""

from gap.agent.subgraph_runner import (
    _exec_subgraph_builder,
    _parse_subgraph_response,
)


def test_builder_exec_error_is_threaded():
    # `set_outputs(success=True)` passes a literal where a Ref() is required
    # — the exact failure seen in the wild ("output 'success' must be a
    # Ref() dict, got True").
    raw = (
        "```python\n"
        "from gap.builder import Subgraph, Ref, START, END\n"
        "sg = Subgraph(name='relative_lift_sg', skill='lift-arm-relative')\n"
        "sg.set_outputs(success=True)\n"
        "```\n"
    )
    sg_dict, _scripts, _cp_mod, _cp_meta, block, error = _parse_subgraph_response(raw)
    assert sg_dict is None
    assert block is not None  # a python block WAS emitted — not "unparseable"
    assert error is not None
    assert "must be a Ref() dict" in error


def test_valid_builder_block_has_no_error():
    raw = (
        "```python\n"
        "from gap.builder import Subgraph, Ref, START, END\n"
        "sg = Subgraph(name='ok_sg', skill='generic')\n"
        "```\n"
    )
    sg_dict, _scripts, _cp_mod, _cp_meta, _block, error = _parse_subgraph_response(raw)
    assert sg_dict is not None
    assert error is None


def test_exec_subgraph_builder_reports_non_subgraph_binding():
    sub_dict, _cp_mod, _cp_meta, error = _exec_subgraph_builder("sg = 123")
    assert sub_dict is None
    assert error is not None
    assert "Subgraph" in error
