from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from mandri.api.deps import (
    ChatGptWiring,
    GatewayWiring,
    chatgpt_wiring,
    gateway_wiring,
    http_client,
    providers_registry,
)
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.core.ids import ProviderKind
from mandri.core.ports.provider_verifier import ProviderVerifierPort, VerificationResult
from mandri.core.ports.routes import RouteLookupPort
from mandri.core.types.config import DaemonConfig, ProviderConfig
from mandri.providers.errors import ProviderDisabledError
from mandri.providers.service import ProvidersRegistry


@pytest.fixture
def provider_client(tmp_path, make_client):
    config = TomlConfigAdapter(tmp_path)
    config.save(
        DaemonConfig(
            providers=[ProviderConfig(name="synthetic", kind="openai", api_key="stored-key")]
        )
    )
    verifier = Mock(spec=ProviderVerifierPort)
    verifier.verify_async.return_value = VerificationResult(
        ok=True, provider=ProviderKind.OPENAI, reason=None
    )
    routes = Mock(spec=RouteLookupPort)
    routes.route_ids_for_provider.return_value = []
    providers = ProvidersRegistry(config, routes, verifier)
    registry = Mock()
    registry.list_routes = AsyncMock(return_value=[])
    registry.resolve = AsyncMock(side_effect=ProviderDisabledError("synthetic"))
    handler = Mock()
    handler.chat_completions = AsyncMock()
    handler.responses = AsyncMock()
    handler.messages = AsyncMock()
    handler.generate_content = AsyncMock()
    handler.count_tokens = AsyncMock()
    wiring = GatewayWiring(
        registry=registry, openai=handler, anthropic=handler, responses=handler, gemini=handler
    )
    http = Mock(spec=httpx.AsyncClient)
    client = make_client(
        {
            providers_registry: lambda: providers,
            chatgpt_wiring: lambda: ChatGptWiring(),
            gateway_wiring: lambda: wiring,
            http_client: lambda: http,
        }
    )
    return client, providers, config, verifier, wiring, handler, http


def test_toggle_is_persistent_and_keeps_credentials(provider_client):
    client, providers, config, verifier, _, _, _ = provider_client
    response = client.patch("/v1/providers/synthetic", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert "api_key" not in response.json()
    assert client.get("/v1/providers").json()[0]["enabled"] is False
    assert client.get("/v1/gateway/info").json()["providers"][0]["enabled"] is False
    restored = TomlConfigAdapter(config.config_path.parent).load().providers[0]
    assert not restored.enabled
    assert restored.api_key == "stored-key"
    verifier.verify_async.assert_not_awaited()
    assert client.patch("/v1/providers/synthetic", json={"enabled": True}).json()["enabled"] is True
    assert providers.get_active("synthetic").api_key == "stored-key"
    verifier.verify_async.assert_not_awaited()


def test_editing_disabled_provider_preserves_disabled_state(provider_client):
    client, providers, _, verifier, _, _, _ = provider_client
    client.patch("/v1/providers/synthetic", json={"enabled": False})
    response = client.patch("/v1/providers/synthetic", json={"api_key": "replacement"})
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert providers.get("synthetic").api_key == "replacement"
    verifier.verify_async.assert_awaited_once()


def test_null_enabled_is_rejected(provider_client):
    client, providers, *_ = provider_client
    assert client.patch("/v1/providers/synthetic", json={"enabled": None}).status_code == 422
    assert providers.get("synthetic").enabled


def test_disabled_catalog_does_not_call_upstream(provider_client):
    client, _, _, _, _, _, http = provider_client
    client.patch("/v1/providers/synthetic", json={"enabled": False})
    response = client.get("/v1/providers/synthetic/models")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "provider_disabled"
    http.get.assert_not_awaited()


@pytest.mark.parametrize(
    "path",
    [
        "v1/chat/completions",
        "responses",
        "v1/messages",
        "v1/messages/count_tokens",
        "v1beta/models/default:generateContent",
    ],
)
def test_disabled_routes_fail_before_any_inference(provider_client, path):
    client, _, _, _, wiring, handler, _ = provider_client
    route_id = "8d137bdc-8bdc-469f-b42a-36b7b04a88df"
    token = wiring.issue_child_token(route_id)
    response = client.post(
        f"/v1/gateway/llm/{route_id}/{path}", headers={"Authorization": f"Bearer {token}"}, json={}
    )
    assert response.status_code == 503
    assert "disabled" in response.json()["error"]["message"]
    handler.chat_completions.assert_not_awaited()
    handler.responses.assert_not_awaited()
    handler.messages.assert_not_awaited()
    handler.generate_content.assert_not_awaited()
    handler.count_tokens.assert_not_awaited()
