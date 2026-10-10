import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import litellm
import pytest
from mandri.core.ids import ProviderKind, RouteId, SecretRef
from mandri.gateway.litellm_adapter import AnthropicHandler, OpenAIHandler, _credentials
from mandri.gateway.provider_call import provider_call
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.gateway.unauthenticated_client import unauthenticated_client
from mandri.providers.service import Provider, ProviderState, provider_model_ref


def route(kind=ProviderKind.CUSTOM, key=""):
    return ResolvedRoute(
        route_id=RouteId("local-test"),
        provider=Provider(
            name="local-test",
            kind=kind,
            api_base="http://local-provider.test/v1",
            api_key=SecretRef(key),
            state=ProviderState.VERIFIED,
        ),
        model=Model(
            provider=kind,
            model_ref=provider_model_ref(kind, "local-model"),
            api_base="http://local-provider.test/v1",
            api_key=SecretRef(key),
        ),
    )


@pytest.mark.parametrize("kind", [ProviderKind.CUSTOM, ProviderKind.LM_STUDIO])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("protocol", ["chat", "anthropic"])
async def test_keyless_provider_never_sends_ambient_credentials(
    monkeypatch, kind, streaming, protocol
):
    requests = []
    transports = []

    def respond(request):
        requests.append(request)
        assert request.url == "http://local-provider.test/v1/chat/completions"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "local-model"
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"id":"test","object":"chat.completion.chunk","created":0,'
                    '"model":"local-model","choices":[{"index":0,"delta":{"content":"ok"},'
                    '"finish_reason":null}]}\n\n'
                    'data: {"id":"test","object":"chat.completion.chunk","created":0,'
                    '"model":"local-model","choices":[],"usage":{"prompt_tokens":6,'
                    '"completion_tokens":3,"total_tokens":9}}\n\ndata: [DONE]\n\n'
                ),
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "local-model",
                "usage": {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def transport():
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        transports.append(client)
        return client

    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret-must-not-leak")
    monkeypatch.setattr(litellm, "api_key", "global-secret-must-not-leak")
    monkeypatch.setattr(litellm, "openai_key", "provider-secret-must-not-leak")
    monkeypatch.setattr(
        "mandri.gateway.unauthenticated_client.OpenAIChatCompletion._get_async_http_client",
        transport,
    )
    for _ in range(2):
        body = {"messages": [{"role": "user", "content": "test"}], "stream": streaming}
        if protocol == "anthropic":
            result = await AnthropicHandler().messages(route(kind), {**body, "max_tokens": 100})
        else:
            result = await OpenAIHandler().chat_completions(route(kind), body)
        if streaming:
            chunks = [chunk async for chunk in result]
            if protocol == "anthropic":
                assert any(b'"text": "ok"' in chunk for chunk in chunks)
                events = [
                    json.loads(line[6:])
                    for chunk in chunks
                    for line in chunk.decode().splitlines()
                    if line.startswith("data: ")
                ]
                assert any(
                    event.get("usage") == {"input_tokens": 6, "output_tokens": 3}
                    for event in events
                )
            else:
                assert any(chunk.choices[0].delta.content == "ok" for chunk in chunks)
        elif protocol == "anthropic":
            assert result["content"] == [{"type": "text", "text": "ok"}]
            assert result["usage"] == {"input_tokens": 6, "output_tokens": 3}
        else:
            assert result.choices[0].message.content == "ok"
    assert len(requests) == 2
    assert all(client.is_closed for client in transports)


@pytest.mark.parametrize("kind", [ProviderKind.CUSTOM, ProviderKind.LM_STUDIO])
def test_explicit_key_keeps_authenticated_client_selection(kind):
    credentials = _credentials(route(kind, "explicit-key"))
    assert credentials["api_key"] == "explicit-key"
    assert "client" not in credentials


def test_other_providers_and_missing_base_do_not_select_keyless_client():
    assert "client" not in _credentials(route(ProviderKind.OPENAI))
    resolved = route()
    resolved = replace(resolved, model=replace(resolved.model, api_base=None))
    assert "client" not in _credentials(resolved)


@pytest.mark.parametrize("error", [RuntimeError("failed"), asyncio.CancelledError()])
async def test_owned_client_closes_when_provider_call_fails(monkeypatch, error):
    client = AsyncMock()
    monkeypatch.setattr("mandri.gateway.provider_call._provider_call", AsyncMock(side_effect=error))
    with pytest.raises(type(error)):
        await provider_call(route(), AsyncMock(), {"client": client}, None)
    client.close.assert_awaited_once()


async def test_owned_client_and_stream_close_when_consumer_stops(monkeypatch):
    client = AsyncMock()
    closed = []

    async def events():
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    monkeypatch.setattr(
        "mandri.gateway.provider_call._provider_call", AsyncMock(return_value=events())
    )
    stream = await provider_call(route(), AsyncMock(), {"client": client}, None)
    assert await anext(stream) == "first"
    await stream.aclose()
    assert closed == [True]
    client.close.assert_awaited_once()


async def test_borrowed_litellm_transport_is_not_closed(monkeypatch):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200))
    ) as shared:
        monkeypatch.setattr(litellm, "aclient_session", shared)
        client = unauthenticated_client("http://local-provider.test/v1")
        await client.close()
        assert not shared.is_closed
