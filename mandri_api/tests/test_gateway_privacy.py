from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.responses import StreamingResponse
from mandri.api.deps import GatewayWiring, gateway_wiring
from mandri.api.routers._gateway_privacy import restore_response
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.privacy import PreparedRequest
from mandri.gateway.privacy_egress import EgressGuard
from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState

_ID = "7c3d0b19-8f69-41a8-bf8d-f3584708aa42"
_ORIGINAL = "upstream-error-73149@example.invalid"


def protected_wiring():
    kind = ProviderKind.CUSTOM
    provider = Provider(
        "test",
        kind,
        Url("https://provider.invalid/v1"),
        SecretRef("synthetic-key"),
        ProviderState.VERIFIED,
    )
    route = ResolvedRoute(
        RouteId(_ID),
        provider,
        Model(kind, ModelRef("openai/test"), provider.api_base, provider.api_key),
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="scope-one",
    )
    engine = SurrogateEngine(SurrogateScope("scope-one"))
    alias = engine.protect_text(_ORIGINAL)
    guard = EgressGuard(route, engine)
    privacy = SimpleNamespace(prepare=AsyncMock(return_value=PreparedRequest({}, guard)))
    call = AsyncMock(
        side_effect=UpstreamError(429, f"Limit for {alias}", kind, {"retry-after": "1"})
    )
    handler = SimpleNamespace(
        chat_completions=call,
        responses=call,
        messages=call,
        generate_content=call,
        count_tokens=call,
    )
    return (
        GatewayWiring(
            registry=SimpleNamespace(resolve=AsyncMock(return_value=route)),
            openai=handler,
            anthropic=handler,
            responses=handler,
            gemini=handler,
            privacy=privacy,
        ),
        call,
        alias,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "v1/chat/completions",
        "responses",
        "v1/messages",
        "v1/messages/count_tokens",
        "v1beta/models/test:generateContent",
    ],
)
def test_provider_errors_restore_original_values_in_every_wire_format(make_client, endpoint):
    wiring, call, alias = protected_wiring()
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/{endpoint}",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={},
    )
    assert response.status_code == 429
    assert _ORIGINAL in response.json()["error"]["message"]
    assert alias not in response.text
    assert response.headers["retry-after"] == "1"
    assert call.await_count == 1


def test_model_request_has_no_surrogate_byte_limit(make_client):
    wiring, call, _ = protected_wiring()
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/v1/chat/completions",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={"messages": [{"role": "user", "content": "a" * 1024}]},
    )
    assert response.status_code == 429
    call.assert_awaited_once()


def test_nested_duplicate_fields_are_rejected_before_preparation(make_client):
    wiring, call, _ = protected_wiring()
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/v1/chat/completions",
        headers={"x-api-key": wiring.auth.issue(_ID), "content-type": "application/json"},
        content=b'{"messages":[{"role":"user","content":"one","content":"two"}]}',
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "gateway_request_invalid"
    call.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["v1/embeddings", "v1/rerank", "v1/files", "v1/unknown"])
def test_unknown_model_operations_fail_without_preparing_or_sending(make_client, endpoint):
    wiring, call, _ = protected_wiring()
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/{endpoint}",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={"input": _ORIGINAL},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert _ORIGINAL not in response.text
    wiring.privacy.prepare.assert_not_awaited()
    call.assert_not_awaited()


def test_unknown_operations_authorize_before_policy_lookup(make_client):
    wiring, call, _ = protected_wiring()
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(f"/v1/gateway/llm/{_ID}/v1/embeddings", json={"input": _ORIGINAL})
    assert response.status_code == 401
    wiring.registry.resolve.assert_not_awaited()
    call.assert_not_awaited()


def test_standard_unknown_operation_retains_not_found(make_client):
    wiring, call, _ = protected_wiring()
    route = wiring.registry.resolve.return_value
    wiring.registry.resolve.return_value = replace(
        route, privacy_mode=PrivacyMode.NONE, privacy_scope_id=None
    )
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/v1/embeddings",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={"input": _ORIGINAL},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    call.assert_not_awaited()


@pytest.mark.parametrize("invalid", [False, True])
def test_complete_protected_response_restores_before_http_headers(make_client, invalid):
    wiring, call, alias = protected_wiring()
    prepared = wiring.privacy.prepare.return_value
    wiring.privacy.prepare.return_value = replace(prepared, client_stream=True)
    call.side_effect = None
    call.return_value = {
        "id": "resp_complete",
        "object": "response",
        "status": "completed",
        "output": [
            {
                "id": "rs_one",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": alias}],
            },
            {
                "id": "msg_one",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "unknown_binary" if invalid else "output_text", "text": alias}
                ],
            },
        ],
    }
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/responses",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={"stream": True},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "response.completed" in response.text
    assert "response.reasoning_text.delta" in response.text
    assert _ORIGINAL in response.text
    assert (alias in response.text) is invalid
    assert int(response.headers["content-length"]) == len(response.content)


def test_standard_response_keeps_streaming_unchanged():
    response = StreamingResponse(iter([b"data: untouched\n\n"]))
    assert restore_response(response, PreparedRequest({}), GatewayProtocol.CHAT) is response


@pytest.mark.parametrize("client_stream", [False, True])
def test_protected_preparation_failure_is_terminal_in_requested_protocol(
    make_client, client_stream
):
    wiring, call, _ = protected_wiring()
    wiring.privacy.prepare.side_effect = ProtectionError(
        "privacy_state_unavailable", "Synthetic storage failure"
    )
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{_ID}/responses",
        headers={"x-api-key": wiring.auth.issue(_ID)},
        json={"stream": client_stream},
    )
    assert response.status_code == (200 if client_stream else 422)
    if client_stream:
        assert "response.failed" in response.text
        assert "Synthetic storage failure" in response.text
    else:
        assert response.json()["error"]["code"] == "privacy_state_unavailable"
    call.assert_not_awaited()
