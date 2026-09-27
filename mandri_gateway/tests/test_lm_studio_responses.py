import json

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.gateway.litellm_adapter import OpenAIHandler, ResponsesHandler
from mandri.gateway.response_stream import normalize_stream
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState
from mandri.providers.verify import resolve_api_base


@pytest.mark.parametrize(
    "wire,invalid", [("responses", False), ("responses", True), ("chat", False)]
)
@pytest.mark.parametrize("effort", [None, "on", "off", "high"])
async def test_lm_studio_responses_wire(monkeypatch, invalid, effort, wire):
    requests = []

    async def send(client, request, **kwargs):
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        wire_effort = body.get("reasoning_effort")
        if wire_effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh"}:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": f"Invalid 'reasoning_effort' value: '{wire_effort}'.",
                        "type": "invalid_request_error",
                        "param": "reasoning_effort",
                        "code": "invalid_value",
                    }
                },
                request=request,
            )
        assert wire_effort == {"off": "none", "on": "medium"}.get(effort, effort)
        assert "reasoning" not in body
        if invalid:
            return httpx.Response(
                200, json={"error": "Unexpected endpoint or method"}, request=request
            )
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        first = json.dumps(chunk)
        chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            content=f"data: {first}\n\ndata: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    kind = ProviderKind.LM_STUDIO
    base = resolve_api_base(kind, Url("http://localhost:1234"))
    provider = Provider("local", kind, base, SecretRef(""), ProviderState.VERIFIED)
    route = ResolvedRoute(
        RouteId("test-route"),
        provider,
        Model(kind, ModelRef("lm_studio/test-model"), base, provider.api_key),
        reasoning_effort=effort,
    )
    try:
        if wire == "chat":
            stream = await OpenAIHandler().chat_completions(
                route,
                {"messages": [{"role": "user", "content": "Hello"}], "stream": True},
            )
            chunks = [chunk async for chunk in stream]
            assert len(requests) == 1
            assert chunks[0].choices[0].delta.content == "Hello"
            return
        stream = await ResponsesHandler().responses(route, {"input": "Hello", "stream": True})
        events = [event async for event in normalize_stream(stream)]
        assert len(requests) == 1
        if invalid:
            assert events[-1]["type"] == "response.failed", events
        else:
            assert events[-1]["type"] == "response.completed"
            assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello"
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
