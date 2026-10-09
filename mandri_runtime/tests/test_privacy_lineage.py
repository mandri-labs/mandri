from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode, ProtectionError
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.service import RuntimeService


async def test_resume_admission_uses_refreshed_policy_before_any_launch():
    protected = SimpleNamespace(
        native_id="native-child",
        gateway_route_id=None,
        model="provider/model",
        model_source=ModelSource.GATEWAY,
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="inherited-scope",
    )
    store = SimpleNamespace(
        ensure_session_policy=AsyncMock(return_value=protected),
        get_session=AsyncMock(side_effect=AssertionError("stale policy used")),
    )
    runtime = RuntimeService(harness_commands={})
    runtime._sessions = store
    runtime._ensure_session_available = AsyncMock()
    record = await runtime._load_resumable_record("child")
    assert record is protected
    store.ensure_session_policy.assert_awaited_once_with(SessionId("child"))
    runtime._ensure_session_available.assert_awaited_once()


async def test_lineage_failure_stops_resume_before_ownership_or_provider():
    store = SimpleNamespace(
        ensure_session_policy=AsyncMock(
            side_effect=ProtectionError("session_lineage_unavailable", "Parent unavailable")
        )
    )
    runtime = RuntimeService(harness_commands={})
    runtime._sessions = store
    runtime._ensure_session_available = AsyncMock()
    runtime._spawn_harness = AsyncMock()
    with pytest.raises(ProtectionError):
        await runtime._load_resumable_record("child")
    runtime._ensure_session_available.assert_not_called()
    runtime._spawn_harness.assert_not_called()


@pytest.mark.parametrize("available", [True, False])
async def test_resume_refreshes_current_workspace_identity_before_binding_route(available):
    record = SimpleNamespace(
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.SURROGATE,
        privacy_scope_id="existing-scope",
        project_path="selected-workspace",
        harness=HarnessKind.CODEX,
        model_source=ModelSource.GATEWAY,
    )
    scopes = SimpleNamespace(validate=AsyncMock(), add_workspace=AsyncMock())
    if not available:
        scopes.add_workspace.side_effect = ProtectionError(
            "privacy_context_unavailable", "Unavailable"
        )
    runtime = RuntimeService({"codex": ["codex"]}, privacy_scopes=scopes)
    runtime._load_resumable_record = AsyncMock(return_value=record)
    runtime._resume_launch_mode = lambda *args, **kwargs: "full-access"
    runtime._validate_policy = AsyncMock()
    runtime._resume_route = AsyncMock(side_effect=RuntimeError("binding reached"))
    with pytest.raises(RuntimeError if available else ProtectionError):
        await runtime._resume_in_workspace("existing-session")
    scopes.add_workspace.assert_awaited_once_with("existing-scope", "selected-workspace")
    if not available:
        runtime._resume_route.assert_not_awaited()
