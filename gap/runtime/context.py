"""NodeContext — tool access interface available to skill and script nodes.

Scripts receive a NodeContext that lets them invoke any registered tool
without managing dispatch, guard limits, or trace recording themselves.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from gap.errors import TaskCancelled

if TYPE_CHECKING:
    from .tracing import DagTrace


class CancelToken:
    """Cooperative cancellation primitive for streaming nodes.

    The executor constructs one ``CancelToken`` per streaming node
    on spawn. When the enclosing scope reaches END (or aborts), the
    executor fires the token; long-running skills must check
    ``ctx.cancel_token.raise_if_set()`` periodically — typically once
    per inner iteration — and raise ``TaskCancelled`` so the scope
    teardown can drain promptly.

    A skill that doesn't check the token will not be cancelled
    cooperatively; the executor will fall back to ``Future.cancel()``
    after a grace period (best-effort).
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout=timeout)

    def raise_if_set(self) -> None:
        if self._event.is_set():
            raise TaskCancelled("cooperative cancellation requested")


class NodeContext:
    """Context available to skill and script nodes for invoking tools."""

    def __init__(
        self,
        tool_registry: Any,
        *,
        trace: DagTrace | None = None,
        node_id: str | None = None,
        policy_executor: Any = None,
        cancel_token: CancelToken | None = None,
    ):
        # Flat tool registry (gap.tools.ToolRegistry) — the single dispatch
        # surface for connector tools, bundle tools, and @tool plugins. If
        # None, ctx.tool() will bind a default one lazily.
        self._tool_registry = tool_registry
        self._trace = trace
        self._node_id = node_id
        self._call_seq = 0
        self._stream_seq = 0
        # The PolicyExecutor (when available) is needed by class-based
        # skills like run_policy that share its websocket-client cache.
        self.policy_executor = policy_executor
        # Cooperative-cancellation token. Non-None inside a parallel branch
        # whose state's ``join_policy`` may pre-empt siblings. Long-running
        # skills must call ``self.cancel_token.raise_if_set()`` periodically.
        self.cancel_token = cancel_token
        # Bound by the executor for streaming nodes — see
        # WorkflowExecutor._spawn_streaming. The skill calls
        # ``ctx.publish(value)`` on each iteration.
        self._stream_slot: Any = None

    def publish(self, value: Any) -> None:
        """Publish a snapshot value from a streaming skill.

        Only valid inside a node declared with ``streaming: true``. The
        executor binds a ``StreamSlot`` to ``ctx._stream_slot`` before
        the skill runs; downstream consumers see the latest published
        value via ``{"$ref": "<this_node_name>"}``.
        """
        if self._stream_slot is None:
            raise RuntimeError(
                "ctx.publish() called from a non-streaming node; ensure "
                "the node declares streaming=true and the skill's "
                "contract.streaming is true"
            )
        self._stream_slot.publish(value)

    def tool(self, name: str, **kwargs) -> Any:
        """Invoke a tool by its flat catalog name.

        Single dispatch surface for connector tools (e.g.
        ``robot.move_to_pose``), bundle tools, and in-process Python @tool
        plugins (e.g. ``geometry.iou``). Dispatch is resolved by the bound
        :class:`gap.tools.ToolRegistry`.

        Guard enforcement is applied automatically — call counts for
        perception-, planning-, and sim-step-tagged tools are tracked and
        limited via ``gap.tools.guards``.

        Example::

            iou = ctx.tool("geometry.iou", box_a=[0,0,2,2], box_b=[1,1,3,3])
            ctx.tool("robot.move_to_pose", pose=grasp_pose)
        """
        from gap.tools.guards import check_and_increment_if_applicable

        if self._tool_registry is None:
            from gap.tools import default_tool_registry
            self._tool_registry = default_tool_registry()

        descriptor = self._tool_registry.get(name)
        check_and_increment_if_applicable(descriptor.tags)

        # The registry handles requires_ctx injection and kwarg filtering.
        response = self._tool_registry.invoke(name, ctx=self, **kwargs)

        if self._trace and self._node_id:
            self._trace.record_subcall(
                self._node_id, self._call_seq, name, kwargs, response,
            )
            self._call_seq += 1

        return response

    def _stream_read(self, name: str, value: Any, sampled_at: float) -> None:
        """Record a stream-handle .latest() consumption into the trace.

        Called by ObservationStreamHandle on each read. Bumps a per-context
        counter so consumed values are addressable as (node_id, seq) for
        deterministic replay.
        """
        if self._trace is None or self._node_id is None:
            return
        seq = self._stream_seq
        self._stream_seq += 1
        self._trace.record_stream_read(
            self._node_id, seq, name, value, sampled_at,
        )
