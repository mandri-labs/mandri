from types import SimpleNamespace
from unittest.mock import AsyncMock

from mandri.api.deps import runtime_service
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError

IDENTITY = "fb84195d-6d3d-4d12-9c38-980661708aad"


def test_fork_requires_explicit_target_policy_and_never_accepts_history_paths(make_client):
    runtime = SimpleNamespace(fork_session=AsyncMock())
    client = make_client({runtime_service: lambda: runtime})
    for body in (
        {},
        {"execution_backend": "docker"},
        {
            "execution_backend": "docker",
            "privacy_mode": "surrogate",
            "rollout_path": "/private",
        },
    ):
        assert client.post(f"/v1/sessions/{IDENTITY}/fork", json=body).status_code == 422
    runtime.fork_session.assert_not_awaited()


def test_fork_returns_new_native_session_effective_policy(make_client):
    runtime = SimpleNamespace(
        fork_session=AsyncMock(
            return_value=SimpleNamespace(
                id="new",
                harness="codex",
                route_id="new-route",
                project_path="/workspace",
                mode="ask",
                execution_backend=ExecutionBackend.DOCKER,
                privacy_mode=PrivacyMode.SURROGATE,
                policy_revision=1,
            )
        )
    )
    client = make_client({runtime_service: lambda: runtime})
    response = client.post(
        f"/v1/sessions/{IDENTITY}/fork",
        json={
            "execution_backend": "docker",
            "privacy_mode": "surrogate",
        },
    )
    assert response.status_code == 201
    assert response.json()["id"] == "new" and response.json()["privacy_mode"] == "surrogate"
    runtime.fork_session.assert_awaited_once_with(
        IDENTITY,
        ExecutionBackend.DOCKER,
        PrivacyMode.SURROGATE,
        mode=None,
        operation_id=None,
    )


def test_unqualified_native_transition_is_explicit(make_client):
    runtime = SimpleNamespace(
        fork_session=AsyncMock(
            side_effect=ProtectionError(
                "session_transition_unsupported",
                "Unsupported target native history",
            )
        )
    )
    client = make_client({runtime_service: lambda: runtime})
    response = client.post(
        f"/v1/sessions/{IDENTITY}/fork",
        json={
            "execution_backend": "host",
            "privacy_mode": "none",
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_transition_unsupported"
