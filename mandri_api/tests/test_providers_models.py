"""Reasoning catalog enrichment tests for the provider models REST route."""

from types import SimpleNamespace
from typing import Any

import pytest
from mandri.api.deps import GatewayWiring, gateway_wiring, http_client, providers_registry
from mandri.core.ids import ProviderKind, SecretRef, Url
from mandri.gateway.reasoning_catalog import ReasoningCatalog, ReasoningInfo
from mandri.providers.service import Provider, ProviderState

PROVIDER_NAME = "acme"
PAYLOAD = {"data": [{"id": "gpt-5"}, {"id": "gpt-mini"}]}
CATALOG = ReasoningCatalog(
    {(PROVIDER_NAME, "gpt-5"): ReasoningInfo(efforts=["low", "high"], default_effort="low")}
)
UNKNOWN = {
    "display_name": None,
    "image_input": None,
    "tool_call": None,
    "reasoning_supported": None,
    "input_modalities": None,
}


class StubHttp:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, str], dict[str, str] | None]] = []

    async def get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        timeout: Any = None,
    ) -> Any:
        self.calls.append((url, dict(headers or {}), params))
        return SimpleNamespace(status_code=200, text="", json=lambda: self.payload)


class StubProviders:
    async def ensure_fresh(self, name: str) -> None:
        pass

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
        {
            **UNKNOWN,
            "id": "gpt-5",
            "reasoning_efforts": ["low", "high"],
            "default_effort": "low",
            "reasoning_supported": True,
        },
        {**UNKNOWN, "id": "gpt-mini", "reasoning_efforts": [], "default_effort": None},
    ]


def test_provider_models_without_catalog_has_empty_reasoning(make_client) -> None:
    wiring = GatewayWiring(registry=object(), openai=object(), anthropic=object())
    client = make_client_with(make_client, wiring)
    response = client.get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json() == [
        {**UNKNOWN, "id": "gpt-5", "reasoning_efforts": [], "default_effort": None},
        {**UNKNOWN, "id": "gpt-mini", "reasoning_efforts": [], "default_effort": None},
    ]


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/api/v1", "/api/v0/"])
def test_lm_studio_native_catalog_exposes_models_and_capabilities(make_client, suffix):
    class LocalProviders(StubProviders):
        def get(self, name):
            return Provider(
                name,
                ProviderKind.LM_STUDIO,
                Url(f"http://localhost:1234{suffix}"),
                SecretRef(""),
                ProviderState.VERIFIED,
            )

    class LocalHttp:
        async def get(self, url, **kwargs):
            assert url == "http://localhost:1234/api/v1/models"
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "models": [
                        {
                            "key": "org/local-model",
                            "type": "llm",
                            "display_name": "Local Model",
                            "capabilities": {
                                "vision": True,
                                "trained_for_tool_use": True,
                                "reasoning": {"allowed_options": ["off", "on"], "default": "on"},
                            },
                        },
                        {"key": "embed", "type": "embedding"},
                    ]
                },
            )

    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object()
            ),
            providers_registry: LocalProviders,
            http_client: LocalHttp,
        }
    )
    response = client.get("/v1/providers/local/models")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "org/local-model",
            "display_name": "Local Model",
            "image_input": True,
            "tool_call": True,
            "reasoning_supported": True,
            "input_modalities": None,
            "reasoning_efforts": ["off", "on"],
            "default_effort": "on",
        }
    ]


@pytest.mark.parametrize("payload", [{"error": "Unexpected endpoint"}, {}, [], {"data": None}])
def test_invalid_catalog_is_not_reported_as_empty(make_client, payload):
    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object()
            ),
            providers_registry: StubProviders,
            http_client: lambda: StubHttp(payload),
        }
    )
    response = client.get("/v1/providers/acme/models")
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_models_failed"


@pytest.mark.parametrize(
    "options",
    [
        {"reasoning": {"supported_efforts": ["medium"], "default": "medium"}},
        {"capabilities": {"reasoning": {"allowed_options": ["medium"], "default": "medium"}}},
    ],
)
def test_custom_live_options_override_and_refresh_validation_catalog(make_client, options):
    catalog = ReasoningCatalog({(PROVIDER_NAME, "model"): ReasoningInfo(["high"])})
    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object(), reasoning_catalog=catalog
            ),
            providers_registry: StubProviders,
            http_client: lambda: StubHttp({"data": [{"id": "model", **options}]}),
        }
    )
    response = client.get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json()[0]["reasoning_efforts"] == ["medium"]
    assert response.json()[0]["default_effort"] == "medium"
    assert catalog.lookup(PROVIDER_NAME, "model") == ReasoningInfo(["medium"], "medium")


def test_custom_props_refresh_after_server_becomes_available(make_client, monkeypatch):
    async def props(url, headers, body=None):
        assert url == "http://127.0.0.1:1234/props"
        return {"model_alias": "gpt-5", "reasoning_efforts": ["low"], "default_effort": "low"}

    monkeypatch.setattr("mandri.gateway.reasoning_catalog._fetch_json", props)
    catalog = ReasoningCatalog()
    wiring = GatewayWiring(
        registry=object(), openai=object(), anthropic=object(), reasoning_catalog=catalog
    )
    response = make_client_with(make_client, wiring).get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json()[0]["reasoning_efforts"] == ["low"]
    assert response.json()[1]["reasoning_efforts"] == []
    assert catalog.lookup(PROVIDER_NAME, "gpt-5") == ReasoningInfo(["low"], "low")


def test_new_custom_provider_uses_cached_models_dev_fallback(make_client, monkeypatch):
    async def no_props(url, headers, body=None):
        return None

    monkeypatch.setattr("mandri.gateway.reasoning_catalog._fetch_json", no_props)
    catalog = ReasoningCatalog(models_dev={"vendor": {"model": ReasoningInfo(["low", "high"])}})
    client = make_client(
        {
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object(), reasoning_catalog=catalog
            ),
            providers_registry: StubProviders,
            http_client: lambda: StubHttp({"data": [{"id": "model-iq2_xs"}]}),
        }
    )
    response = client.get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json()[0]["reasoning_efforts"] == ["low", "high"]


def test_props_disabled_reasoning_does_not_restore_static_options(make_client, monkeypatch):
    async def props(url, headers, body=None):
        return {"model_alias": "gpt-5", "reasoning": False}

    monkeypatch.setattr("mandri.gateway.reasoning_catalog._fetch_json", props)
    catalog = ReasoningCatalog({(PROVIDER_NAME, "gpt-5"): ReasoningInfo(["high"])})
    wiring = GatewayWiring(
        registry=object(), openai=object(), anthropic=object(), reasoning_catalog=catalog
    )
    response = make_client_with(make_client, wiring).get(f"/v1/providers/{PROVIDER_NAME}/models")
    assert response.status_code == 200
    assert response.json()[0]["reasoning_efforts"] == []
    assert response.json()[0]["reasoning_supported"] is False
    assert catalog.lookup(PROVIDER_NAME, "gpt-5") == ReasoningInfo([])
