import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import EpochMs, ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.provider_headers import conversation_headers
from mandri.gateway.litellm_adapter import AnthropicHandler, OpenAIHandler, ResponsesHandler
from mandri.gateway.route_registry import ResolvedRoute, Route, RouteRegistry
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState


def route(kind, conversation_id="conversation-one", route_id="route-one"):
    provider = Provider(
        "test-provider",
        kind,
        Url("http://127.0.0.1:9999/v1"),
        SecretRef("test-key"),
        ProviderState.VERIFIED,
    )
    return ResolvedRoute(
        RouteId(route_id),
        provider,
        Model(kind, ModelRef("custom_openai/test-model"), provider.api_base, provider.api_key),
        conversation_id=conversation_id,
    )


@pytest.mark.parametrize("kind", [ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wire", ["chat", "responses", "messages"])
async def test_all_inference_formats_preserve_conversation_header(monkeypatch, kind, stream, wire):
    calls = AsyncMock(return_value={})
    resolved = route(kind)
    expected = conversation_headers(kind, "session:conversation-one")
    if wire == "chat":
        monkeypatch.setattr("litellm.acompletion", calls)
        await OpenAIHandler().chat_completions(
            resolved,
            {"stream": stream, "messages": [], "extra_headers": {"x-opencode-session": "spoofed"}},
        )
    elif wire == "responses":
        monkeypatch.setattr("litellm.aresponses", calls)
        await ResponsesHandler().responses(resolved, {"stream": stream, "input": "test"})
    else:
        monkeypatch.setattr("litellm.anthropic_interface.messages.acreate", calls)
        await AnthropicHandler().messages(resolved, {"stream": stream, "messages": []})
    headers = calls.call_args.kwargs["extra_headers"]
    assert headers["User-Agent"] == "Mandri Gateway"
    assert headers["x-opencode-client"] == "mandri"
    assert headers["x-opencode-session"] == expected["x-opencode-session"]
    assert headers["x-opencode-session-id"] == expected["x-opencode-session"]
    assert UUID(headers["x-opencode-request"]).version == 4
    assert calls.call_args.kwargs["api_key"] == "test-key"
    assert calls.call_args.kwargs["stream"] is stream


async def test_route_changes_and_gateway_restarts_keep_session_identity():
    binding = {
        "id": "conversation-one",
        "execution_backend": "host",
        "privacy_mode": "none",
        "privacy_scope_id": None,
        "deleted": 0,
    }
    db = SimpleNamespace(fetch_all=AsyncMock(return_value=[binding]))
    registry = RouteRegistry(db, SimpleNamespace())
    first = await registry._bound_conversation(
        Route(RouteId("route-old"), "test", ModelRef("custom_openai/test-model"), (), EpochMs(0))
    )
    restarted = RouteRegistry(db, SimpleNamespace())
    second = await restarted._bound_conversation(
        Route(RouteId("route-new"), "test", ModelRef("custom_openai/test-model"), (), EpochMs(0))
    )
    assert first == second == "conversation-one"
    assert db.fetch_all.call_args.args[1] == ("route-new",)


def test_contexts_are_opaque_stable_and_distinct():
    first = conversation_headers(ProviderKind.OPENCODE_GO, "session:one")
    assert first == conversation_headers(ProviderKind.OPENCODE_GO, "session:one")
    assert first != conversation_headers(ProviderKind.OPENCODE_GO, "session:two")
    assert first == conversation_headers(ProviderKind.OPENCODE, "session:one")
    assert "one" not in first["x-opencode-session"]
    assert conversation_headers(ProviderKind.OPENROUTER, "session:one") == {}


async def test_unassociated_routes_have_distinct_stable_contexts(monkeypatch):
    calls = AsyncMock(return_value={})
    monkeypatch.setattr("litellm.acompletion", calls)
    handler = OpenAIHandler()
    headers = []
    for route_id in ("first", "second", "first"):
        await handler.chat_completions(route(ProviderKind.OPENCODE, None, route_id), {})
        headers.append(calls.call_args.kwargs["extra_headers"])
    request_ids = [header.pop("x-opencode-request") for header in headers]
    assert len(set(request_ids)) == 3
    assert headers[0] == headers[2]
    assert headers[0] != headers[1]


@pytest.mark.parametrize(
    "kind", [ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO, ProviderKind.CUSTOM]
)
@pytest.mark.parametrize("wire", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_inference_headers_reach_upstream_wire(monkeypatch, kind, wire, stream):
    GLOBAL_LOGGING_WORKER.start()
    requests = []

    async def send(client, request, **kwargs):
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/v1/chat/completions"
        requests.append(request)
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
        if not stream:
            return httpx.Response(200, json=response, request=request)
        response["object"] = "chat.completion.chunk"
        response["choices"] = [
            {"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}
        ]
        start = json.dumps(response)
        response["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        return httpx.Response(
            200,
            content=f"data: {start}\n\ndata: {json.dumps(response)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    resolved = route(kind)
    body = {
        "messages": [{"role": "user", "content": "test"}],
        "stream": stream,
        "max_tokens": 100,
    }
    if wire == "chat":
        body["extra_headers"] = {"x-opencode-session": "spoofed", "User-Agent": "spoofed"}
        result = await OpenAIHandler().chat_completions(resolved, body)
    elif wire == "responses":
        result = await ResponsesHandler().responses(resolved, {"input": "test", "stream": stream})
    else:
        result = await AnthropicHandler().messages(resolved, body)
    if stream:
        assert [event async for event in result]
    await GLOBAL_LOGGING_WORKER.flush()
    assert len(requests) == 1
    headers = requests[0].headers
    if kind is ProviderKind.CUSTOM:
        if wire == "chat":
            assert headers["x-opencode-session"] == "spoofed"
            assert headers["user-agent"] == "spoofed"
        else:
            assert not any(name.startswith("x-opencode-") for name in headers)
        assert "x-opencode-request" not in headers
    else:
        expected = conversation_headers(kind, "session:conversation-one")["x-opencode-session"]
        assert headers["user-agent"] == "Mandri Gateway"
        assert headers["x-opencode-client"] == "mandri"
        assert headers["x-opencode-session"] == expected
        assert headers["x-opencode-session-id"] == expected
        assert UUID(headers["x-opencode-request"]).version == 4
