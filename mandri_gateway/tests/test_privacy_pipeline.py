import copy
import json
from pathlib import Path

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.core.types.config import PrivacySettings
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.database.privacy import PrivacyRepository
from mandri.database.sqlite_adapter import AiosqliteDatabase
from mandri.gateway.litellm_adapter import (
    AnthropicHandler,
    GeminiHandler,
    OpenAIHandler,
    ResponsesHandler,
)
from mandri.gateway.privacy import GatewayPrivacy, prepare_request
from mandri.gateway.privacy_protocol import GatewayProtocol, transform_content, validate_request
from mandri.gateway.privacy_scopes import PrivacyScopes
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine, SurrogateScope
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState

_EMAIL = "customer-831491@example.invalid"
_ROOT = "/srv/private-owner/Project"
_MODELS = {
    ProviderKind.CUSTOM: "openai/gpt-4o",
    ProviderKind.OPENAI: "openai/gpt-4o",
    ProviderKind.OPENROUTER: "openrouter/openai/gpt-4o",
    ProviderKind.OPENCODE: "custom_openai/gpt-4o",
    ProviderKind.OPENCODE_GO: "custom_openai/gpt-4o",
    ProviderKind.LM_STUDIO: "lm_studio/gpt-4o",
    ProviderKind.OLLAMA: "ollama_chat/synthetic-model",
    ProviderKind.ANTHROPIC: "anthropic/claude-sonnet-4-20250514",
    ProviderKind.GEMINI: "gemini/gemini-2.5-flash",
}


def route(kind: ProviderKind = ProviderKind.CUSTOM, scope_id: str = "scope-test") -> ResolvedRoute:
    base = Url(
        "https://provider.invalid/v1"
        if kind is not ProviderKind.GEMINI
        else "https://provider.invalid"
    )
    key = SecretRef("synthetic-provider-key")
    provider = Provider(
        name="test", kind=kind, api_base=base, api_key=key, state=ProviderState.VERIFIED
    )
    return ResolvedRoute(
        route_id=RouteId("route-test"),
        provider=provider,
        model=Model(kind, ModelRef(_MODELS[kind]), base, key),
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id=scope_id,
    )


class StaticKey:
    def load(self, *, create=False):
        return b"k" * 32


@pytest.fixture
async def privacy(tmp_path: Path):
    litellm.in_memory_llm_clients_cache.flush_cache()
    db = AiosqliteDatabase()
    await db.connect(tmp_path / "state.sqlite")
    await db.migrate()
    settings = PrivacySettings()
    repository = PrivacyRepository(db, StaticKey())
    scopes = PrivacyScopes(repository, settings)
    scope = SurrogateScope("scope-test")
    SurrogateEngine(scope).register_root(_ROOT)
    await repository.create(scope.scope_id, scope.to_dict())
    scope_id = scope.scope_id
    yield GatewayPrivacy(scopes), scope_id
    await db.close()


def request_body(protocol: GatewayProtocol) -> dict:
    text = f"Review {_EMAIL} in {_ROOT}/api/src/file.py"
    if protocol is GatewayProtocol.CHAT:
        return {"messages": [{"role": "user", "content": text}]}
    if protocol is GatewayProtocol.RESPONSES:
        return {"input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}]}
    if protocol is GatewayProtocol.ANTHROPIC:
        return {"messages": [{"role": "user", "content": text}], "max_tokens": 100}
    return {"contents": [{"role": "user", "parts": [{"text": text}]}]}


def response_for(request: httpx.Request, text: str) -> dict:
    if request.url.path.endswith("/api/chat"):
        return {
            "model": "synthetic-model",
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 4,
        }
    if request.url.path.endswith("/messages"):
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }
    if "generateContent" in request.url.path:
        return {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 4,
                "totalTokenCount": 9,
            },
        }
    if request.url.path.endswith("/responses"):
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        }
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }


@pytest.mark.parametrize("kind", list(_MODELS))
@pytest.mark.parametrize(
    "protocol",
    [
        GatewayProtocol.CHAT,
        GatewayProtocol.RESPONSES,
        GatewayProtocol.ANTHROPIC,
        GatewayProtocol.GEMINI,
    ],
)
async def test_real_sdk_serialization_passes_guard_before_transport(
    privacy, monkeypatch, kind, protocol
):
    service, scope_id = privacy
    resolved = route(kind, scope_id)
    prepared = await service.prepare(resolved, protocol, request_body(protocol))
    guard = prepared.guard
    observed = []

    def receive(request: httpx.Request) -> httpx.Response:
        assert guard.sends > len(observed), "SDK bypassed the mandatory egress guard"
        observed.append(request.content)
        assert _EMAIL.encode() not in request.content
        assert _ROOT.encode() not in request.content
        alias = guard.engine.protect_text(_EMAIL)
        return httpx.Response(200, json=response_for(request, alias))

    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **kwargs: httpx.MockTransport(receive)),
    )
    if protocol is GatewayProtocol.CHAT:
        result = await OpenAIHandler().chat_completions(resolved, prepared.body, guard=guard)
    elif protocol is GatewayProtocol.RESPONSES:
        result = await ResponsesHandler().responses(resolved, prepared.body, guard=guard)
    elif protocol is GatewayProtocol.ANTHROPIC:
        result = await AnthropicHandler().messages(resolved, prepared.body, guard=guard)
    else:
        result = await GeminiHandler().generate_content(resolved, prepared.body, False, guard=guard)
    assert len(observed) == 1
    serialized = result if isinstance(result, dict) else result.model_dump()
    assert _EMAIL in json.dumps(guard.engine.restore(serialized))


async def test_sdk_added_unknown_metadata_does_not_block_request(privacy, monkeypatch):
    service, scope_id = privacy
    resolved = route(scope_id=scope_id)
    prepared = await service.prepare(
        resolved, GatewayProtocol.CHAT, request_body(GatewayProtocol.CHAT)
    )
    sent = []

    def receive(request):
        sent.append(request)
        return httpx.Response(200, json=response_for(request, "Synthetic reply"))

    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **kwargs: httpx.MockTransport(receive)),
    )
    original_send = httpx.AsyncClient.send

    async def injected_send(client, request, **kwargs):
        body = json.loads(request.content)
        body["metadata"] = {"email": "injected-793641@example.invalid"}
        headers = dict(request.headers)
        headers.pop("content-length", None)
        injected = httpx.Request(
            request.method, request.url, headers=headers, json=body, extensions=request.extensions
        )
        return await original_send(client, injected, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", injected_send)
    await OpenAIHandler().chat_completions(resolved, prepared.body, guard=prepared.guard)
    assert len(sent) == 1
    assert json.loads(sent[0].content)["metadata"]["email"] == "injected-793641@example.invalid"


@pytest.mark.parametrize(
    "kind", [ProviderKind.OPENCODE_GO, ProviderKind.CUSTOM, ProviderKind.OPENAI]
)
@pytest.mark.parametrize("stream", [False, True])
async def test_view_image_result_reaches_provider_unchanged(privacy, monkeypatch, kind, stream):
    service, scope_id = privacy
    resolved = route(kind, scope_id)
    image_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
        "AwMCAO+a2WQAAAAASUVORK5CYII="
    )
    body = {
        "stream": stream,
        "input": [
            {
                "type": "function_call",
                "call_id": "call_image",
                "name": "view_image",
                "arguments": json.dumps({"path": f"{_ROOT}/image.png"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                    {"type": "input_text", "text": f"Image for {_EMAIL}"},
                ],
            },
        ],
    }
    observed = []

    def receive(request):
        payload = json.loads(request.content)
        if kind is ProviderKind.OPENAI:
            output = next(
                item for item in payload["input"] if item["type"] == "function_call_output"
            )
            assert output["output"][0]["image_url"] == image_url
        else:
            output = next(item for item in payload["messages"] if item["role"] == "tool")
            assert output["tool_call_id"] == "call_image"
            images = [
                item
                for message in payload["messages"]
                if isinstance(message.get("content"), list)
                for item in message["content"]
                if item["type"] == "image_url"
            ]
            assert len(images) == 1
            assert images[0]["image_url"]["url"] == image_url
        assert _EMAIL.encode() not in request.content
        assert _ROOT.encode() not in request.content
        observed.append(payload)
        return httpx.Response(200, json=response_for(request, "Image received"))

    transport = httpx.MockTransport(receive)
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda self, url: transport)
    for _ in range(2):
        prepared = await service.prepare(resolved, GatewayProtocol.RESPONSES, body)
        assert prepared.client_stream is stream
        await ResponsesHandler().responses(resolved, prepared.body, guard=prepared.guard)
    assert len(observed) == 2


async def test_state_is_committed_before_guard_is_returned(privacy):
    service, scope_id = privacy
    prepared = await service.prepare(
        route(scope_id=scope_id), GatewayProtocol.CHAT, request_body(GatewayProtocol.CHAT)
    )
    reopened = service.scopes.engine((await service.scopes.repository.load(scope_id)).payload)
    assert reopened.restore(prepared.body) == {
        **request_body(GatewayProtocol.CHAT),
        "stream": False,
    }


async def test_provider_credential_in_plain_text_is_prepared_without_changing_auth(privacy):
    service, scope_id = privacy
    resolved = route(scope_id=scope_id)
    credential = str(resolved.model.api_key)
    prepared = await service.prepare(
        resolved,
        GatewayProtocol.CHAT,
        {"messages": [{"role": "user", "content": f"Pasted credential: {credential}"}]},
    )
    assert credential not in json.dumps(prepared.body)
    assert resolved.model.api_key == credential
    assert credential in json.dumps(prepared.guard.engine.restore(prepared.body))


async def test_missing_privacy_service_and_direct_handler_cannot_send(monkeypatch):
    with pytest.raises(ProtectionError):
        await prepare_request(None, route(), GatewayProtocol.CHAT, {"messages": []})
    with pytest.raises(ProtectionError):
        await OpenAIHandler().chat_completions(route(), {"messages": []})


@pytest.mark.parametrize(
    "field", ["api_base", "api_key", "extra_headers", "client", "callbacks", "privacy_scope_id"]
)
def test_transport_overrides_are_rejected(field):
    with pytest.raises(ProtectionError):
        validate_request({field: "override"})


def test_nested_tool_payloads_keep_protocol_and_tree():
    engine = SurrogateEngine(SurrogateScope("scope-test"))
    engine.register_root(_ROOT)
    original = {
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": f"{_ROOT}/api/src/a.py"}]},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": json.dumps(
                    {"type": _EMAIL, "role": _EMAIL, "nested": [{"username": "synthetic-owner"}]}
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "default": f"{_ROOT}/api/src/b.py"}
                    },
                },
            }
        ],
    }
    protected = transform_content(copy.deepcopy(original), engine)
    assert protected["input"][0]["role"] == "user"
    assert protected["input"][0]["content"][0]["type"] == "input_text"
    assert _ROOT not in json.dumps(protected)
    assert _EMAIL not in json.dumps(protected)
    assert "/api/src/a.py" in protected["input"][0]["content"][0]["text"]
    assert transform_content(protected, engine, restore=True) == original


@pytest.mark.parametrize(
    "protocol", [GatewayProtocol.CHAT, GatewayProtocol.RESPONSES, GatewayProtocol.ANTHROPIC]
)
async def test_protected_requests_disable_provider_streaming(privacy, protocol):
    service, scope = privacy
    body = {**request_body(protocol), "stream": True}
    prepared = await service.prepare(route(scope_id=scope), protocol, body)
    assert prepared.body["stream"] is False
    assert prepared.client_stream is True
    assert body["stream"] is True
