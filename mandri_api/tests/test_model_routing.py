from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.api.deps import GatewayWiring, gateway_wiring, runtime_service, sessions_service
from mandri.core.ids import HarnessKind, HarnessSessionId, ProviderKind
from mandri.core.types.config import DaemonConfig, ProviderConfig
from mandri.core.types.model_selection import ModelSource
from mandri.gateway.route_registry import RouteRegistry
from mandri.providers.service import ProvidersRegistry
from mandri.runtime.launch_preparation import LaunchPreparation
from mandri.runtime.service import RuntimeService


@pytest.fixture
def routing_client(make_client):
    database = SimpleNamespace(execute=AsyncMock())
    config = SimpleNamespace(
        load=lambda: DaemonConfig(
            providers=[
                ProviderConfig(name="configured_provider", kind=ProviderKind.OPENCODE_GO.value)
            ]
        )
    )
    providers = ProvidersRegistry(config, Mock())
    routes = RouteRegistry(database, providers)
    sessions = SimpleNamespace(
        create_session=AsyncMock(return_value=SimpleNamespace(id="session")),
        set_session_state=AsyncMock(),
    )
    token = Mock(return_value="scoped-token")
    runtime = RuntimeService(
        {kind.value: [kind.value] for kind in HarnessKind},
        sessions=sessions,
        routes=routes,
        gateway_port=8175,
        token_issuer=token,
    )
    runtime._launch = LaunchPreparation(8175, token, parent_env={})
    runtime._prepare_agy = Mock(side_effect=lambda prepared, *args: prepared)
    runtime._resolve_metadata = AsyncMock(return_value=None)
    runtime._spawn_execution = AsyncMock(return_value=SimpleNamespace(returncode=None))
    runtime._attach_feed = Mock()
    runtime._attach_control = Mock()
    runtime._attach_liveness = Mock()
    runtime._reveal_identity = AsyncMock()
    runtime._session_state("session").control = SimpleNamespace(
        capture_identity=AsyncMock(return_value=HarnessSessionId("native-session"))
    )
    runtime._lifetime.arm_zero_viewer_decision = Mock()
    client = make_client(
        {
            runtime_service: lambda: runtime,
            gateway_wiring: lambda: GatewayWiring(
                registry=routes, openai=object(), anthropic=object()
            ),
        }
    )
    return client, runtime, database, sessions, token


@pytest.mark.parametrize("harness", list(HarnessKind))
@pytest.mark.parametrize("source", [None, "gateway"])
def test_gateway_provider_selection_is_identical_across_harnesses(routing_client, harness, source):
    client, runtime, database, sessions, token = routing_client
    body = {
        "harness": harness.value,
        "model": "configured_provider/organization/model",
        "cwd": "/workspace",
        "effort": "high",
    }
    if source is not None:
        body["model_source"] = source
    response = client.post("/v1/runtime/sessions", json=body)
    assert response.status_code == 201, response.text
    route_id = response.json()["gateway_route_id"]
    assert route_id is not None
    route_inserts = [
        call
        for call in database.execute.await_args_list
        if call.args[0].startswith("INSERT INTO gateway_route ")
    ]
    assert len(route_inserts) == 1
    inserted = route_inserts[0].args[1]
    assert inserted[0] == route_id
    assert inserted[1] == "configured_provider"
    assert inserted[2].endswith("organization/model")
    assert inserted[5] == "high"
    persisted = sessions.create_session.call_args
    assert persisted.args[:3] == (harness, body["model"], route_id)
    assert persisted.kwargs.get("model_source", ModelSource.GATEWAY) is ModelSource.GATEWAY
    runtime._resolve_metadata.assert_awaited_once_with(body["model"])
    runtime._spawn_execution.assert_awaited_once()
    token.assert_called_once_with(route_id)


@pytest.mark.parametrize(
    "harness", [HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI]
)
def test_native_selection_never_looks_up_a_mandri_provider(routing_client, harness):
    client, runtime, database, sessions, token = routing_client
    response = client.post(
        "/v1/runtime/sessions",
        json={
            "harness": harness.value,
            "model": "unconfigured_provider/organization/model",
            "model_source": "native",
            "cwd": "/workspace",
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["gateway_route_id"] is None
    database.execute.assert_not_awaited()
    runtime._resolve_metadata.assert_not_awaited()
    token.assert_not_called()
    persisted = sessions.create_session.call_args
    assert persisted.args[:3] == (harness, "unconfigured_provider/organization/model", None)
    assert persisted.kwargs["model_source"] is ModelSource.NATIVE


@pytest.mark.parametrize("harness", list(HarnessKind))
def test_missing_gateway_provider_is_not_reinterpreted_as_native(routing_client, harness):
    client, runtime, database, _, token = routing_client
    response = client.post(
        "/v1/runtime/sessions",
        json={
            "harness": harness.value,
            "model": "unconfigured_provider/organization/model",
            "model_source": "gateway",
            "cwd": "/workspace",
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "provider_not_found"
    database.execute.assert_not_awaited()
    runtime._spawn_execution.assert_not_awaited()
    token.assert_not_called()


def test_every_runtime_advertises_its_model_sources(make_client):
    runtime = SimpleNamespace(host_harnesses=lambda: ["custom-harness"])
    sessions = SimpleNamespace(harness_state=lambda kind: SimpleNamespace(degraded=False))
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.get("/v1/runtimes")
    assert response.status_code == 200
    assert {row["harness"]: row["capabilities"]["model_sources"] for row in response.json()} == {
        "codex": ["gateway", "native"],
        "claude": ["gateway", "native"],
        "opencode": ["gateway"],
        "agy": ["gateway", "native"],
        "pi": ["gateway", "native"],
        "custom-harness": ["gateway"],
    }
