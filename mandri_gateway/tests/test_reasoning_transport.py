import json

import httpx
import pytest
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef, Url
from mandri.gateway.litellm_adapter import OpenAIHandler, ResponsesHandler
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.refs import MODEL_REF_PREFIXES
from mandri.providers.service import Provider, ProviderState


@pytest.mark.parametrize(
    "kind",
    [ProviderKind.LM_STUDIO, ProviderKind.CUSTOM, ProviderKind.OLLAMA, ProviderKind.OPENROUTER],
)
@pytest.mark.parametrize("effort", [None, "on", "off", "high"])
@pytest.mark.parametrize("wire", ["chat", "responses"])
@pytest.mark.parametrize("governed", [False, True])
async def test_reasoning_reaches_provider_wire(monkeypatch, kind, effort, wire, governed):
    GLOBAL_LOGGING_WORKER.start()
    requests = []

    async def send(client, request, **kwargs):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "vendor/same-model"
        if kind is ProviderKind.OLLAMA:
            assert request.url.path == "/api/chat"
            assert body.get("think") == {"on": True, "off": False}.get(effort, effort)
            assert "reasoning_effort" not in body.get("options", {})
            response = {
                "model": "vendor/same-model",
                "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "OK"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            }
        else:
            if kind is ProviderKind.OPENROUTER:
                expected = (
                    None
                    if effort is None
                    else (
                        {"enabled": effort == "on"}
                        if effort in {"on", "off"}
                        else {"effort": effort}
                    )
                )
                assert body.get("reasoning") == expected
                assert "reasoning_effort" not in body
            else:
                assert body.get("reasoning_effort") == {"off": "none", "on": "medium"}.get(
                    effort, effort
                )
                assert "reasoning" not in body
            if wire == "responses" and kind is ProviderKind.OPENROUTER:
                assert request.url.path == "/v1/responses"
                response = {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1,
                    "model": "vendor/same-model",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                        }
                    ],
                }
            else:
                assert request.url.path == "/v1/chat/completions"
                response = {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vendor/same-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
        return httpx.Response(200, json=response, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda client, request, **kwargs: httpx.Response(200, json={}, request=request),
    )
    base = Url("http://localhost:9999" + ("" if kind is ProviderKind.OLLAMA else "/v1"))
    provider = Provider("test", kind, base, SecretRef("synthetic-key"), ProviderState.VERIFIED)
    route = ResolvedRoute(
        RouteId("test"),
        provider,
        Model(
            kind, ModelRef(MODEL_REF_PREFIXES[kind] + "vendor/same-model"), base, provider.api_key
        ),
        reasoning_effort=effort if governed else None,
    )
    incoming = "low" if governed and effort is not None else effort
    try:
        if wire == "responses":
            body = {"input": "Hello"}
            if incoming:
                body["reasoning"] = {"effort": incoming}
            response = await ResponsesHandler().responses(route, body)
            assert response.output[0].content[0].text == "OK"
        else:
            body = {"messages": [{"role": "user", "content": "Hello"}]}
            if incoming:
                body["reasoning_effort"] = incoming
            response = await OpenAIHandler().chat_completions(route, body)
            assert response.choices[0].message.content == "OK"
        assert len(requests) == 1
    finally:
        await GLOBAL_LOGGING_WORKER.flush()
