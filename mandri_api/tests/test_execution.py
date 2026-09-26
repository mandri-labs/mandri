from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import GatewayWiring, gateway_wiring, runtime_service, sessions_service
from mandri.core.types.execution import (
    ExecutionBackend,
    ExecutionPhase,
    PrivacyMode,
    ProtectionError,
    SessionPolicy,
)
from mandri.core.types.execution_generation import ExecutionGeneration
from mandri.runtime.errors.docker import DockerExecutionError
from mandri.runtime.registry import SessionRegistry

SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"


def test_runtime_installation_reports_executables_not_configured_commands(make_client):
    runtime = SimpleNamespace(
        installed_harnesses=lambda: ["codex", "claude"],
        host_harnesses=lambda: ["claude"],
    )
    sessions = SimpleNamespace(harness_state=lambda kind: SimpleNamespace(degraded=False))
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.get("/v1/runtimes")
    assert response.status_code == 200
    installed = {item["harness"]: item["installed"] for item in response.json()}
    assert installed["claude"] is True
    assert installed["codex"] is False


@pytest.mark.parametrize(
    "error,status,code",
    [
        (DockerExecutionError("docker_unavailable", "Unavailable"), 409, "docker_unavailable"),
        (DockerExecutionError("docker_image_missing", "Missing"), 409, "docker_image_missing"),
        (ProtectionError("privacy_key_unavailable", "Locked"), 400, "privacy_key_unavailable"),
        (
            ProtectionError("privacy_state_unavailable", "Unsupported"),
            400,
            "privacy_state_unavailable",
        ),
    ],
)
def test_create_reports_runtime_protection_failure(make_client, error, status, code):
    runtime = SimpleNamespace(
        start_session=AsyncMock(side_effect=error),
        creation_policy=SessionPolicy(),
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
            "model": "synthetic/model",
            "cwd": "/workspace",
            "execution_backend": "docker",
            "privacy_mode": "surrogate",
        },
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    runtime.start_session.assert_awaited_once()
    assert runtime.start_session.call_args.kwargs["execution_backend"] is ExecutionBackend.DOCKER
    assert runtime.start_session.call_args.kwargs["privacy_mode"] is PrivacyMode.SURROGATE


def test_execution_status_never_claims_binding_from_stale_database_phase(make_client):
    generation = ExecutionGeneration(
        SESSION_ID,
        2,
        "owner",
        ExecutionPhase.READY,
        7,
        '{"private":"must not reach client"}',
        1,
        2,
        "container",
    )
    record = SimpleNamespace(
        execution_backend=ExecutionBackend.DOCKER,
        privacy_mode=PrivacyMode.SURROGATE,
        policy_revision=1,
    )
    sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=record),
        execution_generation=AsyncMock(return_value=generation),
    )
    runtime = SimpleNamespace(registry=SessionRegistry())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.get(f"/v1/runtime/sessions/{SESSION_ID}/execution")
    assert response.status_code == 200
    payload = response.json()
    assert payload["generation"] == 2 and payload["revision"] == 7
    assert payload["privacy_mode"] == "surrogate" and payload["policy_revision"] == 1
    assert not payload["effective_binding"]
    assert "private" not in response.text
    runtime.registry.mark_live(SESSION_ID, SimpleNamespace(returncode=None), "codex")
    assert client.get(f"/v1/runtime/sessions/{SESSION_ID}/execution").json()["effective_binding"]


def test_cancel_start_is_explicit_and_waits_for_runtime_cleanup(make_client):
    runtime = SimpleNamespace(cancel_start=AsyncMock())
    client = make_client({runtime_service: lambda: runtime})
    response = client.post(f"/v1/runtime/operations/{SESSION_ID}/cancel")
    assert response.status_code == 204
    runtime.cancel_start.assert_awaited_once_with(SESSION_ID)


def test_resume_returns_persisted_policy_revision(make_client):
    runtime = SimpleNamespace(
        resume_session=AsyncMock(
            return_value=SimpleNamespace(
                id=SESSION_ID,
                harness="codex",
                route_id="route",
                project_path="/workspace",
                mode=None,
                execution_backend=ExecutionBackend.DOCKER,
                privacy_mode=PrivacyMode.SURROGATE,
                policy_revision=7,
            )
        )
    )
    sessions = SimpleNamespace(get_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.post(f"/v1/sessions/{SESSION_ID}/resume")
    assert response.status_code == 200
    assert response.json()["policy_revision"] == 7
    assert response.json()["privacy_mode"] == "surrogate"
    assert response.json()["execution_backend"] == "docker"


def test_create_defaults_follow_daemon_but_explicit_host_none_are_preserved(make_client):
    runtime = SimpleNamespace(
        creation_policy=SessionPolicy(ExecutionBackend.DOCKER, PrivacyMode.SURROGATE),
        start_session=AsyncMock(
            return_value=SimpleNamespace(
                id=SESSION_ID,
                harness="codex",
                route_id="route",
                project_path="/workspace",
                mode=None,
            )
        ),
    )
    client = make_client(
        {
            runtime_service: lambda: runtime,
            gateway_wiring: lambda: GatewayWiring(
                registry=object(), openai=object(), anthropic=object()
            ),
        }
    )
    body = {"harness": "codex", "model": "synthetic/model", "cwd": "/workspace"}
    assert client.post("/v1/runtime/sessions", json=body).status_code == 201
    assert runtime.start_session.call_args.kwargs["execution_backend"] is ExecutionBackend.DOCKER
    assert runtime.start_session.call_args.kwargs["privacy_mode"] is PrivacyMode.SURROGATE
    assert (
        client.post(
            "/v1/runtime/sessions",
            json={
                **body,
                "execution_backend": "host",
                "privacy_mode": "none",
            },
        ).status_code
        == 201
    )
    assert runtime.start_session.call_args.kwargs["execution_backend"] is ExecutionBackend.HOST
    assert runtime.start_session.call_args.kwargs["privacy_mode"] is PrivacyMode.NONE


def test_removed_capabilities_endpoint_does_not_probe_or_prepare_runtime(make_client):
    runtime = SimpleNamespace(
        docker_readiness=AsyncMock(),
        prepare_image=AsyncMock(),
        privacy_readiness=AsyncMock(),
        host_harnesses=lambda: [],
        creation_policy=SessionPolicy(),
    )
    client = make_client({runtime_service: lambda: runtime})
    assert client.get("/v1/runtime/capabilities").status_code == 404
    assert client.get("/v1/runtime/native-auth/bindings").status_code == 404
    runtime.prepare_image.assert_not_awaited()
    runtime.docker_readiness.assert_not_awaited()
    runtime.privacy_readiness.assert_not_awaited()
