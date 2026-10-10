"""Effort governance tests for the gateway REST routes."""

import json
from typing import Any

import mandri.api.routers.gateway as gateway_router
from mandri.api.deps import GatewayWiring, gateway_wiring, providers_registry
from mandri.core.codex_catalog import catalog
from mandri.core.ids import EpochMs, ModelRef, ProviderKind, RouteId, SecretRef, WireFormat
from mandri.gateway.model_metadata import ModelMetadata
from mandri.gateway.reasoning_catalog import ReasoningCatalog, ReasoningInfo
from mandri.gateway.route_registry import ResolvedRoute, Route
from mandri.gateway.types.model import Model
from mandri.providers.service import Provider, ProviderState

ROUTE_ID = "5f0c8e6a-2b1d-4c3a-9e8f-7a6b5c4d3e2f"
MODEL_ARG = "openrouter/anthropic/claude-4"
CATALOG_ENTRIES = {
    ("openrouter", "anthropic/claude-4"): ReasoningInfo(
        efforts=["low", "medium", "high"], default_effort="medium"
    )
}


class StubRegistry:
    def __init__(self, resolved: ResolvedRoute | None = None) -> None:
        self.created: list[tuple[str, str, tuple[WireFormat, ...], str | None]] = []
        self.resolved = resolved

    async def create(
        self,
        provider_name: str,
        model_id: str,
        formats: tuple[WireFormat, ...],
        reasoning_effort: str | None = None,
    ) -> Route:
        self.created.append((provider_name, model_id, formats, reasoning_effort))
        return Route(
            id=RouteId(ROUTE_ID),
            provider_name=provider_name,
            model_ref=ModelRef(f"openrouter/{model_id}"),
            formats=formats,
            created_at=EpochMs(0),
            reasoning_effort=reasoning_effort,
        )

    async def resolve(self, route_id: RouteId) -> ResolvedRoute:
        assert self.resolved is not None
        return self.resolved


class StubProviders:
    def get(self, name: str) -> Provider:
        return Provider(
            name="openrouter",
            kind=ProviderKind.OPENROUTER,
            api_base=None,
            api_key=SecretRef("key"),
            state=ProviderState.VERIFIED,
        )


def make_wiring(registry: StubRegistry, catalog: ReasoningCatalog | None) -> GatewayWiring:
    return GatewayWiring(
        registry=registry, openai=object(), anthropic=object(), reasoning_catalog=catalog
    )


def make_resolved_route() -> ResolvedRoute:
    return ResolvedRoute(
        route_id=RouteId(ROUTE_ID),
        provider=StubProviders().get("openrouter"),
        model=Model(
            provider=ProviderKind.OPENROUTER,
            model_ref=ModelRef("openrouter/anthropic/claude-4"),
            api_base=None,
            api_key=SecretRef("key"),
        ),
        reasoning_effort="high",
    )


def test_create_route_threads_effort_into_registry(make_client) -> None:
    registry = StubRegistry()
    wiring = make_wiring(registry, ReasoningCatalog(CATALOG_ENTRIES))
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        "/v1/gateway/routes",
        json={"model": MODEL_ARG, "formats": ["openai"], "effort": "high"},
    )
    assert response.status_code == 201
    assert registry.created == [("openrouter", "anthropic/claude-4", (WireFormat.OPENAI,), "high")]


def test_create_route_normalizes_empty_effort(make_client) -> None:
    registry = StubRegistry()
    wiring = make_wiring(registry, ReasoningCatalog(CATALOG_ENTRIES))
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.post(
        "/v1/gateway/routes",
        json={"model": MODEL_ARG, "formats": ["openai"], "effort": ""},
    )
    assert response.status_code == 201
    assert registry.created[0][3] is None


def test_model_metadata_includes_governed_reasoning(make_client, monkeypatch) -> None:
    wiring = make_wiring(StubRegistry(), ReasoningCatalog(CATALOG_ENTRIES))

    async def fake_fetch(kind: Any, model_ref: str, api_base: Any, api_key: str) -> ModelMetadata:
        return ModelMetadata(context_window=200000, output_tokens=8192)

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    client = make_client(
        {gateway_wiring: lambda: wiring, providers_registry: lambda: StubProviders()}
    )
    response = client.get("/v1/gateway/model-metadata", params={"model": MODEL_ARG})
    assert response.status_code == 200
    payload = response.json()
    assert payload["reasoning"] == {
        "efforts": ["low", "medium", "high"],
        "default_effort": "medium",
    }


def test_model_metadata_resolves_a_named_provider(make_client, monkeypatch) -> None:
    async def fake_fetch(kind: Any, model_ref: str, api_base: Any, api_key: str) -> ModelMetadata:
        assert model_ref == "openrouter/anthropic/claude-4"
        return ModelMetadata(image_input=True)

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    client = make_client(
        {
            gateway_wiring: lambda: make_wiring(StubRegistry(), None),
            providers_registry: StubProviders,
        }
    )
    response = client.get(
        "/v1/gateway/model-metadata", params={"model": "my-provider/anthropic/claude-4"}
    )
    assert response.status_code == 200
    assert response.json()["image_input"] is True


def test_model_metadata_without_catalog_has_null_reasoning(make_client, monkeypatch) -> None:
    wiring = make_wiring(StubRegistry(), None)

    async def fake_fetch(kind: Any, model_ref: str, api_base: Any, api_key: str) -> ModelMetadata:
        return ModelMetadata(context_window=200000, output_tokens=8192)

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    client = make_client(
        {gateway_wiring: lambda: wiring, providers_registry: lambda: StubProviders()}
    )
    response = client.get("/v1/gateway/model-metadata", params={"model": MODEL_ARG})
    assert response.status_code == 200
    assert response.json()["reasoning"] is None


def test_codex_listing_receives_catalog_info(make_client, monkeypatch) -> None:
    async def fake_fetch(*args: Any) -> ModelMetadata:
        return ModelMetadata(context_window=262144)

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    registry = StubRegistry(make_resolved_route())
    wiring = make_wiring(registry, ReasoningCatalog(CATALOG_ENTRIES))
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.get(
        f"/v1/gateway/llm/{ROUTE_ID}/v1/models",
        headers={"x-api-key": wiring.auth.issue(ROUTE_ID)},
        params={"client_version": "1.2.3"},
    )
    assert response.status_code == 200
    entry = response.json()["models"][0]
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == [
        "low",
        "medium",
        "high",
    ]
    assert entry["default_reasoning_level"] == "medium"
    assert entry["context_window"] == 262144
    assert entry["model_messages"]["instructions_template"]


def test_codex_listing_without_catalog_uses_defaults(make_client, monkeypatch) -> None:
    async def fake_fetch(*args: Any) -> ModelMetadata:
        return ModelMetadata()

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    registry = StubRegistry(make_resolved_route())
    wiring = make_wiring(registry, None)
    client = make_client({gateway_wiring: lambda: wiring})
    response = client.get(
        f"/v1/gateway/llm/{ROUTE_ID}/v1/models",
        headers={"x-api-key": wiring.auth.issue(ROUTE_ID)},
        params={"client_version": "1.2.3"},
    )
    assert response.status_code == 200
    entry = response.json()["models"][0]
    native = json.loads(catalog("mandri-route", ModelMetadata()))["models"][0]
    assert entry["supported_reasoning_levels"] == native["supported_reasoning_levels"]
    assert entry["default_reasoning_level"] == native["default_reasoning_level"]


def test_gemini_discovery_requires_authorization_and_preserves_partial_limits(
    make_client, monkeypatch
) -> None:
    async def fake_fetch(*args: Any) -> ModelMetadata:
        return ModelMetadata(context_window=65536, tool_call=True)

    monkeypatch.setattr(gateway_router, "fetch_model_metadata", fake_fetch)
    wiring = make_wiring(StubRegistry(make_resolved_route()), None)
    client = make_client({gateway_wiring: lambda: wiring})
    base = f"/v1/gateway/llm/{ROUTE_ID}/v1beta/models"
    assert client.get(base).status_code == 401
    headers = {"x-goog-api-key": wiring.auth.issue(ROUTE_ID)}
    response = client.get(base, headers=headers)
    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model == {
        "name": "models/mandri-route",
        "displayName": "mandri-route",
        "inputTokenLimit": 65536,
    }
    assert client.get(base + "/mandri-route", headers=headers).json() == model
