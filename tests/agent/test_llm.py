"""Mocked-transport contract tests for the gap.agent.llm provider layer.

No network: anthropic is exercised through a monkeypatched AsyncAnthropic
client capturing stream kwargs; openai through httpx.MockTransport;
vertex through a monkeypatched AsyncAnthropicVertex (gemini is
import-guarded).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import anthropic as anthropic_sdk
import httpx
import pytest

from gap.agent import llm as llm_mod
from gap.agent.llm import LlmConfig, complete, complete_with_tools


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _thinking_block(text: str = "...") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text)


def _tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _message(content: list, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason)


class _FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, responses: list, captured: list):
        self._responses = responses
        self._captured = captured

    def stream(self, **kwargs):
        self._captured.append(kwargs)
        return _FakeStream(self._responses.pop(0))


class _FakeAsyncAnthropic:
    """Stands in for anthropic.AsyncAnthropic; scripted via class attrs."""

    responses: list = []
    captured: list = []
    init_kwargs: list = []

    def __init__(self, **kwargs):
        type(self).init_kwargs.append(kwargs)
        self.messages = _FakeMessages(type(self).responses, type(self).captured)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_anthropic(monkeypatch):
    _FakeAsyncAnthropic.responses = []
    _FakeAsyncAnthropic.captured = []
    _FakeAsyncAnthropic.init_kwargs = []
    monkeypatch.setattr(anthropic_sdk, "AsyncAnthropic", _FakeAsyncAnthropic)
    return _FakeAsyncAnthropic


# ---------------------------------------------------------------------------
# anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropic:
    def test_complete_request_shape_and_default_model(self, fake_anthropic):
        fake_anthropic.responses = [_message([_text_block("hello "), _text_block("world")])]
        cfg = LlmConfig(provider="anthropic", api_key="k", temperature=0.3)
        out = run(complete(cfg, system="SYS", messages=[{"role": "user", "content": "hi"}]))
        assert out == "hello world"

        (kwargs,) = fake_anthropic.captured
        assert kwargs["model"] == "claude-opus-4-8"  # provider default
        assert kwargs["system"] == "SYS"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["max_tokens"] == 20480
        # claude-opus-4-8 rejects sampling params — temperature must be omitted.
        assert "temperature" not in kwargs
        assert fake_anthropic.init_kwargs == [{"api_key": "k"}]

    def test_temperature_forwarded_on_models_that_support_it(self, fake_anthropic):
        fake_anthropic.responses = [_message([_text_block("ok")])]
        cfg = LlmConfig(provider="anthropic", model="claude-haiku-4-5", temperature=0.2)
        run(complete(cfg, system="s", messages=[{"role": "user", "content": "u"}]))
        assert fake_anthropic.captured[0]["temperature"] == 0.2

    def test_thinking_blocks_skipped(self, fake_anthropic):
        fake_anthropic.responses = [
            _message([_thinking_block(), _text_block("answer")]),
        ]
        cfg = LlmConfig(provider="anthropic")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "u"}]))
        assert out == "answer"

    def test_tool_use_round(self, fake_anthropic):
        fake_anthropic.responses = [
            _message(
                [_text_block("calling"), _tool_use_block("tu_1", "echo", {"x": 1})],
                stop_reason="tool_use",
            ),
            _message([_text_block("done")]),
        ]
        calls: list = []

        def handler(name, kwargs):
            calls.append((name, kwargs))
            return {"ok": True}

        cfg = LlmConfig(provider="anthropic")
        tools = [{"name": "echo", "description": "echo", "input_schema": {"type": "object", "properties": {}}}]
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "u"}],
            tools=tools, tool_handler=handler,
        ))
        assert out == "done"
        assert calls == [("echo", {"x": 1})]

        first, second = fake_anthropic.captured
        assert first["tools"] == tools
        # Second call carries the assistant turn + tool_result user turn.
        msgs = second["messages"]
        assert msgs[1]["role"] == "assistant"
        assert {"type": "tool_use", "id": "tu_1", "name": "echo", "input": {"x": 1}} in msgs[1]["content"]
        assert msgs[2]["role"] == "user"
        (result_block,) = msgs[2]["content"]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "tu_1"
        assert json.loads(result_block["content"]) == {"ok": True}

    def test_tool_handler_error_becomes_error_result(self, fake_anthropic):
        fake_anthropic.responses = [
            _message([_tool_use_block("tu_1", "boom", {})], stop_reason="tool_use"),
            _message([_text_block("recovered")]),
        ]

        def handler(name, kwargs):
            raise FileNotFoundError("nope")

        cfg = LlmConfig(provider="anthropic")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "u"}],
            tools=[{"name": "boom", "description": "", "input_schema": {}}],
            tool_handler=handler,
        ))
        assert out == "recovered"
        result_block = fake_anthropic.captured[1]["messages"][2]["content"][0]
        assert result_block["is_error"] is True
        assert "FileNotFoundError" in result_block["content"]

    def test_max_rounds_cap(self, fake_anthropic):
        fake_anthropic.responses = [
            _message([_tool_use_block(f"tu_{i}", "echo", {})], stop_reason="tool_use")
            for i in range(3)
        ]
        cfg = LlmConfig(provider="anthropic")
        with pytest.raises(RuntimeError, match="exceeded 2 rounds"):
            run(complete_with_tools(
                cfg, system="s", messages=[{"role": "user", "content": "u"}],
                tools=[{"name": "echo", "description": "", "input_schema": {}}],
                tool_handler=lambda n, k: "x", max_rounds=2,
            ))


# ---------------------------------------------------------------------------
# openai provider (httpx.MockTransport)
# ---------------------------------------------------------------------------


def _chat_response(content="hi", tool_calls=None):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


@pytest.fixture
def openai_transport(monkeypatch):
    """Install an httpx.MockTransport; yields the recorded request list.

    The handler is swappable via ``state['handler']``.
    """
    state: dict = {"requests": [], "handler": None}

    def default_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response("hi"))

    state["handler"] = default_handler

    def dispatch(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        return state["handler"](request)

    monkeypatch.setattr(llm_mod, "_HTTPX_TRANSPORT", httpx.MockTransport(dispatch))
    # No real sleeping in retry/backoff tests.
    monkeypatch.setattr(llm_mod, "_INITIAL_BACKOFF", 0.0)
    monkeypatch.setattr(llm_mod.random, "uniform", lambda a, b: 0.0)
    return state


class TestOpenAI:
    def test_requires_explicit_model(self):
        cfg = LlmConfig(provider="openai")
        with pytest.raises(ValueError, match="model must be set"):
            run(complete(cfg, system="s", messages=[]))

    def test_request_shaping_base_url(self, openai_transport):
        cfg = LlmConfig(
            provider="openai", model="m", endpoint="http://h:1234/v1",
            api_key="sk-test", temperature=0.5,
        )
        out = run(complete(cfg, system="SYS", messages=[{"role": "user", "content": "q"}]))
        assert out == "hi"
        (req,) = openai_transport["requests"]
        assert str(req.url) == "http://h:1234/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(req.content)
        assert body["model"] == "m"
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 20480
        assert body["messages"][0] == {"role": "system", "content": "SYS"}
        assert body["messages"][1] == {"role": "user", "content": "q"}

    def test_full_chat_completions_url_accepted(self, openai_transport):
        cfg = LlmConfig(
            provider="openai", model="m",
            endpoint="http://h:8188/v1/chat/completions",
        )
        run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert str(openai_transport["requests"][0].url) == "http://h:8188/v1/chat/completions"

    def test_api_key_from_env(self, openai_transport, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = LlmConfig(provider="openai", model="m", endpoint="http://h/v1")
        run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert openai_transport["requests"][0].headers["Authorization"] == "Bearer sk-env"

    def test_429_retry_then_success(self, openai_transport):
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_chat_response("after retry"))

        openai_transport["handler"] = handler
        cfg = LlmConfig(provider="openai", model="m", endpoint="http://h/v1")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "after retry"
        assert attempts["n"] == 2

    def test_4xx_not_retried(self, openai_transport):
        def handler(request):
            return httpx.Response(400, json={"error": "bad"})

        openai_transport["handler"] = handler
        cfg = LlmConfig(provider="openai", model="m", endpoint="http://h/v1")
        with pytest.raises(httpx.HTTPStatusError):
            run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert len(openai_transport["requests"]) == 1

    def test_reasoning_content_coalesced(self, openai_transport):
        def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": None,
                                         "reasoning_content": "thought"}}],
            })

        openai_transport["handler"] = handler
        cfg = LlmConfig(provider="openai", model="m", endpoint="http://h/v1")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "thought"

    def test_tool_loop(self, openai_transport):
        responses = [
            _chat_response(None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"x": 2})},
            }]),
            _chat_response("final"),
        ]

        def handler(request):
            return httpx.Response(200, json=responses.pop(0))

        openai_transport["handler"] = handler
        calls: list = []

        cfg = LlmConfig(provider="openai", model="m", endpoint="http://h/v1")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "echo", "description": "d",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}],
            tool_handler=lambda n, k: calls.append((n, k)) or {"y": 3},
        ))
        assert out == "final"
        assert calls == [("echo", {"x": 2})]

        first_body = json.loads(openai_transport["requests"][0].content)
        assert first_body["tools"] == [{
            "type": "function",
            "function": {"name": "echo", "description": "d",
                         "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        }]
        second_body = json.loads(openai_transport["requests"][1].content)
        roles = [m["role"] for m in second_body["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        tool_msg = second_body["messages"][3]
        assert tool_msg["tool_call_id"] == "call_1"
        assert json.loads(tool_msg["content"]) == {"y": 3}


# ---------------------------------------------------------------------------
# vertex provider
# ---------------------------------------------------------------------------


class TestVertex:
    def test_claude_on_vertex(self, monkeypatch):
        captured: list = []
        init_kwargs: list = []
        responses = [_message([_text_block("from vertex")])]

        class FakeVertex:
            def __init__(self, **kwargs):
                init_kwargs.append(kwargs)
                self.messages = _FakeMessages(responses, captured)

        monkeypatch.setattr(anthropic_sdk, "AsyncAnthropicVertex", FakeVertex, raising=False)
        monkeypatch.setattr(llm_mod, "_vertex_claude_clients", {})

        cfg = LlmConfig(
            provider="vertex", model="claude-sonnet-4-6",
            project_id="proj", region="us-central1", temperature=0.1,
        )
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "from vertex"
        assert init_kwargs == [{"project_id": "proj", "region": "us-central1"}]
        assert captured[0]["model"] == "claude-sonnet-4-6"
        assert captured[0]["temperature"] == 0.1

    def test_claude_vertex_tool_loop(self, monkeypatch):
        captured: list = []
        responses = [
            _message([_tool_use_block("tu_9", "echo", {"a": 1})], stop_reason="tool_use"),
            _message([_text_block("vertex done")]),
        ]

        class FakeVertex:
            def __init__(self, **kwargs):
                self.messages = _FakeMessages(responses, captured)

        monkeypatch.setattr(anthropic_sdk, "AsyncAnthropicVertex", FakeVertex, raising=False)
        monkeypatch.setattr(llm_mod, "_vertex_claude_clients", {})

        cfg = LlmConfig(provider="vertex", model="claude-sonnet-4-6", project_id="p")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "echo", "description": "", "input_schema": {}}],
            tool_handler=lambda n, k: "pong",
        ))
        assert out == "vertex done"
        assert captured[1]["messages"][2]["content"][0]["tool_use_id"] == "tu_9"

    def test_gemini_on_vertex_import_guarded(self, monkeypatch):
        genai = pytest.importorskip("google.genai")

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(text="gemini says")

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.models = FakeModels()

        monkeypatch.setattr(genai, "Client", FakeClient)
        monkeypatch.setattr(llm_mod, "_vertex_gemini_clients", {})

        cfg = LlmConfig(provider="vertex", model="gemini-3-flash", project_id="p", region="global")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "gemini says"


# ---------------------------------------------------------------------------
# Disk cache + provider selection
# ---------------------------------------------------------------------------


class TestCacheAndSelection:
    def test_cache_hit_skips_transport(self, fake_anthropic, tmp_path, monkeypatch):
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        fake_anthropic.responses = [_message([_text_block("first")])]
        cfg = LlmConfig(provider="anthropic", cache_dir=tmp_path)
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "first"
        # Second call: no scripted response left — must come from cache.
        assert run(complete(cfg, system="s", messages=msgs)) == "first"
        assert len(fake_anthropic.captured) == 1

    def test_cache_keyed_on_model(self, fake_anthropic, tmp_path, monkeypatch):
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        fake_anthropic.responses = [
            _message([_text_block("opus")]), _message([_text_block("haiku")]),
        ]
        msgs = [{"role": "user", "content": "q"}]
        a = LlmConfig(provider="anthropic", cache_dir=tmp_path, model="claude-opus-4-8")
        b = LlmConfig(provider="anthropic", cache_dir=tmp_path, model="claude-haiku-4-5")
        assert run(complete(a, system="s", messages=msgs)) == "opus"
        assert run(complete(b, system="s", messages=msgs)) == "haiku"
        assert len(fake_anthropic.captured) == 2

    def test_no_cache_env_bypasses(self, fake_anthropic, tmp_path, monkeypatch):
        monkeypatch.setenv("GAP_LLM_NO_CACHE", "1")
        fake_anthropic.responses = [
            _message([_text_block("a")]), _message([_text_block("b")]),
        ]
        cfg = LlmConfig(provider="anthropic", cache_dir=tmp_path)
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "a"
        assert run(complete(cfg, system="s", messages=msgs)) == "b"

    def test_cache_disabled_without_dir(self, fake_anthropic, monkeypatch):
        monkeypatch.delenv("GAP_LLM_CACHE_DIR", raising=False)
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        fake_anthropic.responses = [
            _message([_text_block("a")]), _message([_text_block("b")]),
        ]
        cfg = LlmConfig(provider="anthropic")
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "a"
        assert run(complete(cfg, system="s", messages=msgs)) == "b"

    def test_unknown_provider_rejected(self):
        cfg = LlmConfig(provider="nope", model="m")
        with pytest.raises(ValueError, match="unknown LLM provider"):
            run(complete(cfg, system="s", messages=[]))


# ---------------------------------------------------------------------------
# Vertex project resolution (config > documented env vars)
# ---------------------------------------------------------------------------


def test_vertex_project_resolution(monkeypatch):
    from gap.agent.llm import LlmConfig, _vertex_project

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
    assert _vertex_project(LlmConfig(provider="vertex")) == ""

    # The README's documented knob works without a config file...
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    assert _vertex_project(LlmConfig(provider="vertex")) == "env-proj"

    # ...the Anthropic SDK's own env var is honored as a fallback...
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "anthropic-proj")
    assert _vertex_project(LlmConfig(provider="vertex")) == "anthropic-proj"

    # ...and an explicit config project_id always wins.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    assert _vertex_project(
        LlmConfig(provider="vertex", project_id="config-proj")
    ) == "config-proj"


def test_vertex_region_resolution(monkeypatch):
    from gap.agent.llm import LlmConfig, _vertex_region

    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    assert _vertex_region(LlmConfig(provider="vertex")) == "global"

    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")
    assert _vertex_region(LlmConfig(provider="vertex")) == "us-central1"

    assert _vertex_region(
        LlmConfig(provider="vertex", region="europe-west4")
    ) == "europe-west4"


def test_provider_and_model_env_defaults(monkeypatch):
    from gap.agent.llm import LlmConfig, default_model, default_provider

    monkeypatch.delenv("GAP_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GAP_LLM_MODEL", raising=False)
    assert default_provider() == "anthropic"
    assert default_model() is None
    assert LlmConfig().provider == "anthropic"

    # A shell can pin its provider/model once; bare LlmConfig() honors it...
    monkeypatch.setenv("GAP_LLM_PROVIDER", "Vertex")
    monkeypatch.setenv("GAP_LLM_MODEL", "gemini-test")
    cfg = LlmConfig()
    assert cfg.provider == "vertex"
    assert cfg.model == "gemini-test"

    # ...explicit values still win.
    explicit = LlmConfig(provider="openai", model="m")
    assert explicit.provider == "openai" and explicit.model == "m"


def test_from_yaml_llm_env_defaults(monkeypatch, tmp_path):
    from gap.agent.config import PipelineConfig

    monkeypatch.setenv("GAP_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("GAP_LLM_MODEL", "gemini-test")
    yaml_path = tmp_path / "cfg.yaml"

    # YAML omitting llm: -> env defaults flow through from_yaml.
    yaml_path.write_text("task: t\n")
    cfg = PipelineConfig.from_yaml(yaml_path)
    assert cfg.llm.provider == "vertex"
    assert cfg.llm.model == "gemini-test"

    # YAML pinning the provider wins over the env.
    yaml_path.write_text("llm:\n  provider: anthropic\n  model: claude-x\n")
    cfg = PipelineConfig.from_yaml(yaml_path)
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-x"
