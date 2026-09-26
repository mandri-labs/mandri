"""Reasoning catalog enrichment tests for the provider models REST route."""

from types import SimpleNamespace
from typing import Any

from mandri.api.deps import GatewayWiring, gateway_wiring, http_client, providers_registry
from mandri.core.ids import ProviderKind, SecretRef, Url
from mandri.gateway.reasoning_catalog import ReasoningCatalog, ReasoningInfo
from mandri.providers.service import Provider, ProviderState

PROVIDER_NAME = "acme"
PAYLOAD = {"data": [{"id": "gpt-5"}, {"id": "gpt-mini"}]}
CATALOG = ReasoningCatalog(
    {(PROVIDER_NAME, "gpt-5"): ReasoningInfo(efforts=["low", "high"], default_effort="low")}
)


class StubHttp:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    async def get(
        self, url: str, headers: dict[str, str] | None = None, timeout: Any = None
    ) -> Any:
        return SimpleNamespace(status_code=200, text="", json=lambda: self.payload)


class StubProviders:
    def get(self, name: str) -> Provider:
        return Provider(
            name=PROVIDER_NAME,
            kind=ProviderKind.CUSTOM,
            api_base=Url("http://127.0.0.1:1234"),
            api_key=SecretRef("key"),
            state=ProviderState.VERIFIED,
        )


def make_client_with(make_client, wiring: GatewayWiring):
    return make_client(
        {
            gateway_wiring: lambda: wiring,
            providers_registry: lambda: StubProviders(),
            http_client: lambda: StubHttp(PAYLOAD),
        }
    )


def test_provider_models_include_catalog_reasoning(make_client) -> None:
    wiring = GatewayWiring(
        registry=object(), openai=object(), anthropic=object(), reasoning_catalog=CATALOG
    )
    client = make_client_with(make_client, wiring)
    response = client.get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json() == [
        {"id": "gpt-5", "reasoning_efforts": ["low", "high"], "default_effort": "low"},
        {"id": "gpt-mini", "reasoning_efforts": [], "default_effort": None},
    ]


def test_provider_models_without_catalog_has_empty_reasoning(make_client) -> None:
    wiring = GatewayWiring(registry=object(), openai=object(), anthropic=object())
    client = make_client_with(make_client, wiring)
    response = client.get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json() == [
        {"id": "gpt-5", "reasoning_efforts": [], "default_effort": None},
        {"id": "gpt-mini", "reasoning_efforts": [], "default_effort": None},
    ]
