"""Mocked-transport contract tests for the gap.agent.llm provider layer.

No network: openrouter is exercised through httpx.MockTransport; vertex
(Gemini) through a monkeypatched google.genai client (import-guarded).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from gap.agent import llm as llm_mod
from gap.agent.llm import LlmConfig, complete, complete_with_tools


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# openrouter provider (httpx.MockTransport) — the default
# ---------------------------------------------------------------------------


def _chat_response(content: str | None = "hi", tool_calls=None):
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}]}


@pytest.fixture
def openrouter_transport(monkeypatch):
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


class TestOpenRouter:
    def test_defaults_endpoint_key_and_model(self, openrouter_transport, monkeypatch):
        """A bare ``openrouter`` config targets OpenRouter, reads
        OPENROUTER_API_KEY, and uses the provider-default model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-router")
        cfg = LlmConfig(provider="openrouter")  # no endpoint/model/api_key
        out = run(complete(cfg, system="SYS", messages=[{"role": "user", "content": "q"}]))
        assert out == "hi"
        (req,) = openrouter_transport["requests"]
        assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-router"
        body = json.loads(req.content)
        assert body["model"] == "gemini-3.1-flash-lite-preview"  # provider default

    def test_request_shaping_custom_endpoint(self, openrouter_transport):
        cfg = LlmConfig(
            provider="openrouter", model="m", endpoint="http://h:1234/v1",
            api_key="sk-test", temperature=0.5,
        )
        out = run(complete(cfg, system="SYS", messages=[{"role": "user", "content": "q"}]))
        assert out == "hi"
        (req,) = openrouter_transport["requests"]
        assert str(req.url) == "http://h:1234/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(req.content)
        assert body["model"] == "m"
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 20480
        assert body["messages"][0] == {"role": "system", "content": "SYS"}
        assert body["messages"][1] == {"role": "user", "content": "q"}

    def test_full_chat_completions_url_accepted(self, openrouter_transport):
        cfg = LlmConfig(
            provider="openrouter", model="m",
            endpoint="http://h:8188/v1/chat/completions",
        )
        run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert str(openrouter_transport["requests"][0].url) == "http://h:8188/v1/chat/completions"

    def test_api_key_from_env(self, openrouter_transport, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
        cfg = LlmConfig(provider="openrouter", model="m", endpoint="http://h/v1")
        run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert openrouter_transport["requests"][0].headers["Authorization"] == "Bearer sk-env"

    def test_429_retry_then_success(self, openrouter_transport):
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_chat_response("after retry"))

        openrouter_transport["handler"] = handler
        cfg = LlmConfig(provider="openrouter", model="m", endpoint="http://h/v1")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "after retry"
        assert attempts["n"] == 2

    def test_4xx_not_retried(self, openrouter_transport):
        def handler(request):
            return httpx.Response(400, json={"error": "bad"})

        openrouter_transport["handler"] = handler
        cfg = LlmConfig(provider="openrouter", model="m", endpoint="http://h/v1")
        with pytest.raises(httpx.HTTPStatusError):
            run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert len(openrouter_transport["requests"]) == 1

    def test_reasoning_content_coalesced(self, openrouter_transport):
        def handler(request):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": None,
                                         "reasoning_content": "thought"}}],
            })

        openrouter_transport["handler"] = handler
        cfg = LlmConfig(provider="openrouter", model="m", endpoint="http://h/v1")
        out = run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))
        assert out == "thought"

    def test_tool_loop(self, openrouter_transport):
        responses = [
            _chat_response(None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "echo", "arguments": json.dumps({"x": 2})},
            }]),
            _chat_response("final"),
        ]

        def handler(request):
            return httpx.Response(200, json=responses.pop(0))

        openrouter_transport["handler"] = handler
        calls: list = []

        cfg = LlmConfig(provider="openrouter", model="m", endpoint="http://h/v1")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "echo", "description": "d",
                    "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}}}],
            tool_handler=lambda n, k: calls.append((n, k)) or {"y": 3},
        ))
        assert out == "final"
        assert calls == [("echo", {"x": 2})]

        first_body = json.loads(openrouter_transport["requests"][0].content)
        assert first_body["tools"] == [{
            "type": "function",
            "function": {"name": "echo", "description": "d",
                         "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        }]
        second_body = json.loads(openrouter_transport["requests"][1].content)
        roles = [m["role"] for m in second_body["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        tool_msg = second_body["messages"][3]
        assert tool_msg["tool_call_id"] == "call_1"
        assert json.loads(tool_msg["content"]) == {"y": 3}


# ---------------------------------------------------------------------------
# vertex provider (Gemini only; claude-on-vertex was removed with anthropic)
# ---------------------------------------------------------------------------


class TestVertex:
    def test_claude_on_vertex_rejected(self):
        cfg = LlmConfig(provider="vertex", model="claude-sonnet-4-6", project_id="p")
        with pytest.raises(ValueError, match="Gemini models only"):
            run(complete(cfg, system="s", messages=[{"role": "user", "content": "q"}]))

    def test_claude_on_vertex_rejected_with_tools(self):
        cfg = LlmConfig(provider="vertex", model="claude-sonnet-4-6", project_id="p")
        with pytest.raises(ValueError, match="Gemini models only"):
            run(complete_with_tools(
                cfg, system="s", messages=[{"role": "user", "content": "q"}],
                tools=[{"name": "echo", "description": "", "input_schema": {}}],
                tool_handler=lambda n, k: "x",
            ))

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

    def test_gemini_vertex_tool_loop(self, monkeypatch):
        genai = pytest.importorskip("google.genai")

        def _resp(parts, text=None):
            content = SimpleNamespace(parts=parts)
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)

        responses = [
            _resp([SimpleNamespace(
                function_call=SimpleNamespace(name="echo", args={"a": 1}, id="fc_1"),
            )]),
            _resp([SimpleNamespace(function_call=None, text="gemini done")], text="gemini done"),
        ]

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                return responses.pop(0)

        models = FakeModels()

        class FakeClient:
            def __init__(self, **kwargs):
                self.models = models

        monkeypatch.setattr(genai, "Client", FakeClient)
        monkeypatch.setattr(llm_mod, "_vertex_gemini_clients", {})

        calls: list = []
        cfg = LlmConfig(provider="vertex", model="gemini-3-flash", project_id="p", region="global")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "echo", "description": "d",
                    "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}],
            tool_handler=lambda n, k: calls.append((n, k)) or "pong",
        ))
        assert out == "gemini done"
        assert calls == [("echo", {"a": 1})]

        # Tools declared + automatic function calling disabled on the request.
        first_cfg = models.calls[0]["config"]
        assert first_cfg.automatic_function_calling.disable is True
        assert first_cfg.tools[0].function_declarations[0].name == "echo"
        # Second round echoes the model turn and appends a function response.
        second_contents = models.calls[1]["contents"]
        assert second_contents[-1].role == "user"
        assert second_contents[-1].parts[0].function_response.name == "echo"

    def test_gemini_vertex_tool_loop_forces_final_answer(self, monkeypatch):
        """A model that never stops calling tools is coaxed into a final
        tool-free completion instead of raising."""
        genai = pytest.importorskip("google.genai")

        def _call_resp():
            content = SimpleNamespace(parts=[SimpleNamespace(
                function_call=SimpleNamespace(name="echo", args={}, id=None),
            )])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=None)

        class FakeModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                # The forced final turn disables function calling via tool_config.
                if kwargs["config"].tool_config is not None:
                    text_part = SimpleNamespace(function_call=None, text="forced final")
                    content = SimpleNamespace(parts=[text_part])
                    return SimpleNamespace(
                        candidates=[SimpleNamespace(content=content)], text="forced final",
                    )
                return _call_resp()

        models = FakeModels()

        class FakeClient:
            def __init__(self, **kwargs):
                self.models = models

        monkeypatch.setattr(genai, "Client", FakeClient)
        monkeypatch.setattr(llm_mod, "_vertex_gemini_clients", {})

        cfg = LlmConfig(provider="vertex", model="gemini-3-flash", project_id="p", region="global")
        out = run(complete_with_tools(
            cfg, system="s", messages=[{"role": "user", "content": "q"}],
            tools=[{"name": "echo", "description": "", "input_schema": {}}],
            tool_handler=lambda n, k: "pong",
            max_rounds=2,
        ))
        assert out == "forced final"
        # max_rounds normal turns + 1 forced tool-free turn.
        assert len(models.calls) == 3
        assert models.calls[-1]["config"].tool_config is not None


# ---------------------------------------------------------------------------
# Disk cache + provider selection
# ---------------------------------------------------------------------------


def _cfg(tmp_path=None, model: str = "m"):
    return LlmConfig(
        provider="openrouter", model=model, endpoint="http://h/v1",
        cache_dir=tmp_path,
    )


class TestCacheAndSelection:
    def test_cache_hit_skips_transport(self, openrouter_transport, tmp_path, monkeypatch):
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        cfg = _cfg(tmp_path)
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "hi"
        # Second call: no transport hit — the response comes from cache.
        assert run(complete(cfg, system="s", messages=msgs)) == "hi"
        assert len(openrouter_transport["requests"]) == 1

    def test_cache_keyed_on_model(self, openrouter_transport, tmp_path, monkeypatch):
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        responses = [_chat_response("m1"), _chat_response("m2")]
        openrouter_transport["handler"] = lambda req: httpx.Response(200, json=responses.pop(0))
        msgs = [{"role": "user", "content": "q"}]
        a = _cfg(tmp_path, model="model-a")
        b = _cfg(tmp_path, model="model-b")
        assert run(complete(a, system="s", messages=msgs)) == "m1"
        assert run(complete(b, system="s", messages=msgs)) == "m2"
        assert len(openrouter_transport["requests"]) == 2

    def test_no_cache_env_bypasses(self, openrouter_transport, tmp_path, monkeypatch):
        monkeypatch.setenv("GAP_LLM_NO_CACHE", "1")
        responses = [_chat_response("a"), _chat_response("b")]
        openrouter_transport["handler"] = lambda req: httpx.Response(200, json=responses.pop(0))
        cfg = _cfg(tmp_path)
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "a"
        assert run(complete(cfg, system="s", messages=msgs)) == "b"

    def test_cache_disabled_without_dir(self, openrouter_transport, monkeypatch):
        monkeypatch.delenv("GAP_LLM_CACHE_DIR", raising=False)
        monkeypatch.delenv("GAP_LLM_NO_CACHE", raising=False)
        responses = [_chat_response("a"), _chat_response("b")]
        openrouter_transport["handler"] = lambda req: httpx.Response(200, json=responses.pop(0))
        cfg = _cfg()
        msgs = [{"role": "user", "content": "q"}]
        assert run(complete(cfg, system="s", messages=msgs)) == "a"
        assert run(complete(cfg, system="s", messages=msgs)) == "b"

    def test_unknown_provider_rejected(self):
        cfg = LlmConfig(provider="nope", model="m")
        with pytest.raises(ValueError, match="unknown LLM provider"):
            run(complete(cfg, system="s", messages=[]))


# ---------------------------------------------------------------------------
# Vertex project/region resolution (config > documented env var)
# ---------------------------------------------------------------------------


def test_vertex_project_resolution(monkeypatch):
    from gap.agent.llm import LlmConfig, _vertex_project

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert _vertex_project(LlmConfig(provider="vertex")) == ""

    # The README's documented knob works without a config file...
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
    assert _vertex_project(LlmConfig(provider="vertex")) == "env-proj"

    # ...and an explicit config project_id always wins.
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
    assert default_provider() == "openrouter"
    assert default_model() is None
    assert LlmConfig().provider == "openrouter"

    # A shell can pin its provider/model once; bare LlmConfig() honors it...
    monkeypatch.setenv("GAP_LLM_PROVIDER", "Vertex")
    monkeypatch.setenv("GAP_LLM_MODEL", "gemini-test")
    cfg = LlmConfig()
    assert cfg.provider == "vertex"
    assert cfg.model == "gemini-test"

    # ...explicit values still win.
    explicit = LlmConfig(provider="openrouter", model="m")
    assert explicit.provider == "openrouter" and explicit.model == "m"


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
    yaml_path.write_text("llm:\n  provider: openrouter\n  model: model-x\n")
    cfg = PipelineConfig.from_yaml(yaml_path)
    assert cfg.llm.provider == "openrouter"
    assert cfg.llm.model == "model-x"
