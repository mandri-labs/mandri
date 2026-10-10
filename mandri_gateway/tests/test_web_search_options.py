import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.gateway import privacy_transport
from mandri.gateway.litellm_adapter import (
    AnthropicHandler,
    GeminiHandler,
    OpenAIHandler,
    ResponsesHandler,
)
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState


def route(kind: ProviderKind) -> ResolvedRoute:
    provider = Provider(
        "test-provider",
        kind,
        Url("http://127.0.0.1:9999/v1"),
        SecretRef("test-key"),
        ProviderState.VERIFIED,
    )
    prefix = "openrouter" if kind == ProviderKind.OPENROUTER else "custom_openai"
    return ResolvedRoute(
        RouteId("route-one"),
        provider,
        Model(
            kind,
            ModelRef(f"{prefix}/test-model"),
            provider.api_base,
            provider.api_key,
        ),
        conversation_id="conversation-one",
    )


@pytest.fixture
async def upstream(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response, request=request)
        response["object"] = "chat.completion.chunk"
        response["choices"] = [
            {"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}
        ]
        start = json.dumps(response)
        response.pop("usage")
        response["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        response["usage"] = {
            "prompt_tokens": 321,
            "completion_tokens": 17,
            "total_tokens": 338,
            "prompt_tokens_details": {"cached_tokens": 300},
        }
        content = f"data: {start}\n\ndata: {json.dumps(response)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(content.encode()),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    privacy_transport.install_transport_observers()
    monkeypatch.setattr(privacy_transport, "_ASYNC_SEND", send)
    yield requests
    await GLOBAL_LOGGING_WORKER.flush()


@pytest.mark.parametrize("kind", [ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("search_type", ["web_search", "web_search_preview"])
async def test_responses_preserve_search_capability_on_upstream_wire(
    upstream: list[dict[str, Any]], kind: ProviderKind, stream: bool, search_type: str
) -> None:
    response = await ResponsesHandler().responses(
        route(kind),
        {
            "input": "test",
            "stream": stream,
            "tools": [
                {"type": search_type, "search_context_size": "medium"},
                {"type": "function", "name": "read_file", "parameters": {"type": "object"}},
            ],
            "tool_choice": "auto",
            "web_search_options": {"search_context_size": "high"},
        },
    )
    if stream:
        events = [event async for event in response]
        assert any(event.type == "response.completed" for event in events)
    else:
        assert response.output[0].content[0].text == "OK"
    assert len(upstream) == 1
    assert upstream[0].get("web_search_options"), "The adapter silently removed web search"
    assert upstream[0]["tools"][0]["function"]["name"] == "read_file"
    assert len(upstream[0]["tools"]) == 1
    assert upstream[0]["tool_choice"] == "auto"
    assert upstream[0].get("stream", False) is stream


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wire", ["chat", "messages", "generate_content"])
async def test_explicit_search_options_are_omitted_on_upstream_wire(
    upstream: list[dict[str, Any]], stream: bool, wire: str
) -> None:
    body = {
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 32,
        "stream": stream,
        "web_search_options": {"search_context_size": "high"},
    }
    if wire == "chat":
        response = await OpenAIHandler().chat_completions(route(ProviderKind.OPENCODE_GO), body)
    elif wire == "messages":
        response = await AnthropicHandler().messages(route(ProviderKind.OPENCODE_GO), body)
    else:
        response = await GeminiHandler().generate_content(
            route(ProviderKind.OPENCODE_GO),
            {
                "contents": [{"role": "user", "parts": [{"text": "test"}]}],
                "generationConfig": {"web_search_options": {"search_context_size": "high"}},
                "web_search_options": {"search_context_size": "high"},
            },
            stream,
        )
    if stream:
        events = [event async for event in response]
        assert events
    assert len(upstream) == 1
    assert "web_search_options" not in upstream[0]
    assert upstream[0]["messages"] == body["messages"]
    assert upstream[0].get("stream", False) is stream


@pytest.mark.parametrize("stream", [False, True])
async def test_responses_cache_metadata_survives_chat_conversion(upstream, stream):
    response = await ResponsesHandler().responses(
        route(ProviderKind.CUSTOM),
        {
            "input": "test",
            "stream": stream,
            "prompt_cache_key": "stable-conversation",
            "prompt_cache_retention": "in_memory",
            "client_metadata": {"x-codex-turn-metadata": {"thread_source": "thread_title"}},
        },
    )
    if stream:
        events = [event.model_dump() async for event in response]
        completed = next(event for event in events if event["type"] == "response.completed")
        assert completed["response"]["usage"]["input_tokens"] == 321
        assert completed["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 300
    assert len(upstream) == 1
    assert upstream[0]["prompt_cache_key"] == "stable-conversation"
    assert upstream[0]["prompt_cache_retention"] == "in_memory"
    assert upstream[0]["client_metadata"] == {
        "x-codex-turn-metadata": {"thread_source": "thread_title"}
    }
    if stream:
        assert upstream[0]["stream_options"] == {"include_usage": True}


async def test_claude_safeguards_survive_sdk_conversion(upstream):
    safeguards = {"synthetic": True}
    await AnthropicHandler().messages(
        route(ProviderKind.CUSTOM),
        {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 32,
            "safeguards": safeguards,
        },
    )
    assert upstream[0]["safeguards"] == safeguards
