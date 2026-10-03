from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import runtime_service, sessions_service
from mandri.core.ids import EpochMs, HarnessKind, ProjectPath, SessionId, SessionState
from mandri.core.types.execution import PrivacyMode, ProtectionError
from mandri.core.types.sessions import Session

SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"


def test_privacy_patch_returns_same_session_and_confirmed_policy(make_client):
    record = Session(
        id=SessionId(SESSION_ID), harness=HarnessKind.CODEX, native_id=None,
        native_title=None, title_overlay=None, project_path=ProjectPath("/workspace"),
        created_at=EpochMs(0), updated_at=EpochMs(0), state=SessionState.LIVE,
        privacy_mode=PrivacyMode.SURROGATE, privacy_scope_id="scope", policy_revision=2,
        model="synthetic/model", gateway_route_id=None, deleted=False, last_synced_at=EpochMs(0),
    )
    runtime = SimpleNamespace(set_session_privacy=AsyncMock(return_value=record))
    sessions = SimpleNamespace(activity_of=lambda _: None, statuses=None)
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: sessions})
    response = client.patch(
        f"/v1/sessions/{SESSION_ID}/privacy", json={"privacy_mode": "surrogate"}
    )
    assert response.status_code == 200
    assert response.json()["id"] == SESSION_ID
    assert response.json()["privacy_mode"] == "surrogate"
    assert response.json()["policy_revision"] == 2
    runtime.set_session_privacy.assert_awaited_once_with(SESSION_ID, PrivacyMode.SURROGATE)


@pytest.mark.parametrize("mode", ["none", "surrogate"])
def test_native_privacy_patch_returns_conflict(make_client, mode):
    runtime = SimpleNamespace(set_session_privacy=AsyncMock(
        side_effect=ProtectionError("privacy_native_unsupported", "Native model")
    ))
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: object()})
    response = client.patch(f"/v1/sessions/{SESSION_ID}/privacy", json={"privacy_mode": mode})
    assert response.status_code == 409
    assert "privacy_native_unsupported" in response.text


def test_invalid_privacy_mode_is_rejected_before_runtime(make_client):
    runtime = SimpleNamespace(set_session_privacy=AsyncMock())
    client = make_client({runtime_service: lambda: runtime, sessions_service: lambda: object()})
    response = client.patch(f"/v1/sessions/{SESSION_ID}/privacy", json={"privacy_mode": "invalid"})
    assert response.status_code == 422
    runtime.set_session_privacy.assert_not_called()
