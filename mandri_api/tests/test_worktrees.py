from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.api.deps import gateway_wiring, runtime_service, sessions_service
from mandri.core.types.execution import ProtectionError
from mandri.core.types.worktrees import Worktree

SESSION_ID = "8b1f3d2a-4c5e-4f6a-9b0c-1d2e3f4a5b6c"


@pytest.mark.parametrize("name", [None, "feature/search"])
def test_start_forwards_worktree_and_returns_actual_name(make_client, name):
    worktree = Worktree(
        "actual-name", "/state/worktrees/actual", "/repo", "/repo", "refs/heads/main", "abcdef"
    )
    runtime = SimpleNamespace(
        start_session=AsyncMock(
            return_value=SimpleNamespace(
                id=SESSION_ID,
                harness="claude",
                route_id=None,
                project_path=worktree.path,
                mode=None,
                worktree=worktree,
            )
        )
    )
    client = make_client(
        {runtime_service: lambda: runtime, gateway_wiring: lambda: SimpleNamespace()}
    )
    body = {
        "harness": "claude",
        "model": "default",
        "model_source": "native",
        "cwd": "/repo",
        "worktree": True,
    }
    if name is not None:
        body["worktree_id"] = name
    response = client.post("/v1/runtime/sessions", json=body)
    assert response.status_code == 201
    assert response.json()["worktree"]["id"] == "actual-name"
    assert runtime.start_session.call_args.kwargs["worktree"] is True
    assert runtime.start_session.call_args.kwargs["worktree_id"] == name


def test_delete_requires_explicit_worktree_discard_and_preserves_error_code(make_client):
    sessions = SimpleNamespace(
        delete_session=AsyncMock(
            side_effect=ProtectionError("worktree_has_changes", "Unmerged work")
        )
    )
    client = make_client({sessions_service: lambda: sessions})
    response = client.delete(f"/v1/sessions/{SESSION_ID}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "worktree_has_changes"
    sessions.delete_session.assert_awaited_once_with(SESSION_ID, purge=False)
    sessions.delete_session.reset_mock(side_effect=True)
    response = client.delete(f"/v1/sessions/{SESSION_ID}?discard_worktree=true")
    assert response.status_code == 204
    sessions.delete_session.assert_awaited_once_with(SESSION_ID, purge=False, discard_worktree=True)


def test_rename_worktree_validates_input_and_reports_running_session(make_client):
    runtime = SimpleNamespace(
        rename_worktree=AsyncMock(side_effect=ProtectionError("worktree_missing", "Missing"))
    )
    client = make_client(
        {runtime_service: lambda: runtime, sessions_service: lambda: SimpleNamespace()}
    )
    response = client.patch(f"/v1/sessions/{SESSION_ID}/worktree", json={"id": "feature/search"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "worktree_missing"
    runtime.rename_worktree.assert_awaited_once_with(SESSION_ID, "feature/search")
    assert client.patch(f"/v1/sessions/{SESSION_ID}/worktree", json={"id": ""}).status_code == 422


def test_integration_review_requires_no_assumed_main_and_forwards_strategy(make_client):
    runtime = SimpleNamespace(
        preview_worktree=AsyncMock(
            return_value={
                "branches": ["develop", "release"],
                "default_branch": None,
            }
        )
    )
    client = make_client({runtime_service: lambda: runtime})
    response = client.get(f"/v1/sessions/{SESSION_ID}/worktree/integration")
    assert response.status_code == 200
    assert response.json()["default_branch"] is None
    runtime.preview_worktree.assert_awaited_once_with(SESSION_ID, None, "squash")
    assert (
        client.get(
            f"/v1/sessions/{SESSION_ID}/worktree/integration?strategy=unsupported"
        ).status_code
        == 422
    )


def test_integration_and_cleanup_report_protection_failures(make_client):
    runtime = SimpleNamespace(
        integrate_worktree=AsyncMock(
            side_effect=ProtectionError("worktree_preview_changed", "Stale")
        ),
        finish_worktree=AsyncMock(side_effect=ProtectionError("worktree_has_changes", "Changed")),
        resolve_worktree=AsyncMock(),
    )
    client = make_client(
        {runtime_service: lambda: runtime, sessions_service: lambda: SimpleNamespace()}
    )
    body = {"target": "trunk", "strategy": "merge", "token": "a" * 64, "message": "Feature"}
    response = client.post(f"/v1/sessions/{SESSION_ID}/worktree/integration", json=body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "worktree_preview_changed"
    runtime.integrate_worktree.assert_awaited_once_with(
        SESSION_ID, "trunk", "merge", "a" * 64, "Feature"
    )
    response = client.post(f"/v1/sessions/{SESSION_ID}/worktree/finish")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "worktree_has_changes"
    response = client.post(f"/v1/sessions/{SESSION_ID}/worktree/resolve", json=body)
    assert response.status_code == 204
    runtime.resolve_worktree.assert_awaited_once_with(SESSION_ID, "trunk", "merge", "a" * 64)
