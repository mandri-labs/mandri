import json

import pytest
from mandri.api.deps import GatewayWiring, gateway_wiring
from mandri.core.ids import ModelRef, ProviderKind, RouteId, SecretRef
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState

ROUTE_ID = "28b5598e-ea47-4824-822c-bd311d7b6c9f"


class Registry:
    def __init__(self):
        self.calls = []

    async def resolve(self, route_id):
        self.calls.append(route_id)
        provider = Provider(
            "fixture", ProviderKind.OPENAI, None, SecretRef("synthetic"), ProviderState.VERIFIED
        )
        return ResolvedRoute(
            route_id=RouteId(ROUTE_ID),
            provider=provider,
            model=Model(
                ProviderKind.OPENAI, ModelRef("custom_openai/pinned"), None, SecretRef("synthetic")
            ),
        )


class Gemini:
    def __init__(self):
        self.calls = []

    async def generate_content(self, route, body, stream):
        self.calls.append((route, body, stream))
        response = {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]}
        if not stream:
            return response
        return self.events(response)

    async def events(self, response):
        yield ("data: " + json.dumps(response) + "\n\n").encode()


@pytest.mark.parametrize("stream", [False, True])
def test_google_api_key_authenticates_gemini_route(make_client, stream):
    registry = Registry()
    handler = Gemini()
    wiring = GatewayWiring(registry=registry, openai=object(), anthropic=object(), gemini=handler)
    client = make_client({gateway_wiring: lambda: wiring})
    operation = "streamGenerateContent?alt=sse" if stream else "generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": "INPUT"}]}],
        "systemInstruction": {"parts": [{"text": "SYSTEM"}]},
    }
    response = client.post(
        f"/v1/gateway/llm/{ROUTE_ID}/v1beta/models/client-alias:{operation}",
        headers={"X-Goog-Api-Key": wiring.auth.issue(ROUTE_ID)},
        json=body,
    )
    assert response.status_code == 200, response.text
    assert registry.calls == [RouteId(ROUTE_ID)]
    assert len(handler.calls) == 1
    route, received, received_stream = handler.calls[0]
    assert received == body and received_stream is stream
    assert str(route.model.model_ref) == "custom_openai/pinned"
    if stream:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert json.loads(response.text.removeprefix("data: "))["candidates"]
    else:
        assert response.json()["candidates"]


@pytest.mark.parametrize("presented", ["missing", "invalid", "another-route"])
def test_gemini_rejects_missing_or_wrong_route_token(make_client, presented):
    registry = Registry()
    handler = Gemini()
    wiring = GatewayWiring(registry=registry, openai=object(), anthropic=object(), gemini=handler)
    headers = {}
    if presented != "missing":
        headers["X-Goog-Api-Key"] = (
            wiring.auth.issue("a-different-route") if presented == "another-route" else "invalid"
        )
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        f"/v1/gateway/llm/{ROUTE_ID}/v1beta/models/client-alias:generateContent",
        json={"contents": []},
        headers=headers,
    )
    assert response.status_code == 401
    assert response.json()["error"]["status"] == "UNAUTHENTICATED"
    assert registry.calls == [] and handler.calls == []
