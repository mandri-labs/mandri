from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.api.deps import GatewayWiring, gateway_wiring, runtime_service
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.native_catalog import NativeModel


def test_native_start_accepts_ultra_without_gateway_catalog(make_client):
    runtime = SimpleNamespace(
        start_session=AsyncMock(
            return_value=SimpleNamespace(
                id="native-session",
                harness="codex",
                route_id=None,
                project_path="D:/tmp",
                mode=None,
            )
        )
    )
    client = make_client(
        {
            runtime_service: lambda: runtime,
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object()
            ),
        }
    )
    response = client.post(
        "/v1/runtime/sessions",
        json={
            "harness": "codex",
            "model": "gpt-native",
            "model_source": "native",
            "cwd": "D:/tmp",
            "effort": "ultra",
        },
    )
    assert response.status_code == 201
    assert response.json()["gateway_route_id"] is None
    assert runtime.start_session.call_args.kwargs["model_source"] is ModelSource.NATIVE
    assert runtime.start_session.call_args.kwargs["effort"] == "ultra"


def test_native_catalog_exposes_harness_efforts(make_client):
    runtime = SimpleNamespace(
        native_models=AsyncMock(
            return_value=[
                NativeModel("fable", "Fable", ("high", "max")),
            ]
        )
    )
    client = make_client({runtime_service: lambda: runtime})
    response = client.get("/v1/runtimes/claude/models", params={"cwd": "D:/tmp"})
    assert response.status_code == 200
    assert response.json()[0]["reasoning_efforts"] == ["high", "max"]
    runtime.native_models.assert_awaited_once_with("claude", "D:/tmp")


def test_native_catalog_failure_is_explicit(make_client):
    runtime = SimpleNamespace(native_models=AsyncMock(side_effect=TimeoutError("timeout")))
    client = make_client({runtime_service: lambda: runtime})
    response = client.get("/v1/runtimes/codex/models")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "native_catalog_unavailable"
