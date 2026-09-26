from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import SessionId
from mandri.core.types.execution import PrivacyMode, ProtectionError
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
