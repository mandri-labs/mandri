from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import HarnessKind
from mandri.core.types.availability import SessionActivityState, SessionOwner
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionConflictError, SessionRunningError
from mandri.sessions.ownership.service import NativeOwnership


@pytest.fixture
def runtime():
    record = SimpleNamespace(
        harness=HarnessKind.CODEX,
        native_id="native",
        model_source=ModelSource.NATIVE,
        model="default",
        gateway_route_id=None,
    )
    sessions = SimpleNamespace(
        get_session=AsyncMock(return_value=record),
        native_ownership=AsyncMock(return_value=NativeOwnership(SessionOwner.UNOWNED)),
        external_status=AsyncMock(return_value=(False, None)),
    )
    return RuntimeService({}, sessions=sessions), sessions


@pytest.mark.parametrize("owner", list(SessionOwner))
@pytest.mark.parametrize("harness", list(HarnessKind))
async def test_availability_separates_finished_turn_from_writer_owner(runtime, owner, harness):
    service, sessions = runtime
    sessions.get_session.return_value.harness = harness
    sessions.native_ownership.return_value = NativeOwnership(owner)
    if owner is SessionOwner.MANDRI:
        service.registry.mark_live("session")
    result = await service.session_availability("session")
    assert result.owner is owner
    assert result.can_resume is (owner is SessionOwner.UNOWNED)
    assert result.can_restore is (
        owner is SessionOwner.UNOWNED
        and harness in (HarnessKind.CODEX, HarnessKind.CLAUDE, HarnessKind.AGY, HarnessKind.PI)
    )
    if owner is SessionOwner.EXTERNAL:
        assert result.activity is SessionActivityState.IDLE
        assert not result.can_release


async def test_release_requires_confirmation_before_stopping(runtime):
    service, _ = runtime
    service.registry.mark_live("session")
    service.stop_session = AsyncMock()
    with pytest.raises(SessionConflictError):
        await service.release_session("session", confirmed=False)
    service.stop_session.assert_not_awaited()


async def test_unknown_owner_cannot_be_resumed_or_released(runtime):
    service, sessions = runtime
    sessions.native_ownership.return_value = NativeOwnership(SessionOwner.UNKNOWN)
    with pytest.raises(SessionRunningError):
        await service._ensure_session_available("session")
    with pytest.raises(SessionConflictError):
        await service.release_session("session", confirmed=True)


async def test_transition_reservation_blocks_another_mutation(runtime):
    service, _ = runtime
    service._session_state("session").resuming = True
    result = await service.session_availability("session")
    assert not result.can_resume and not result.can_release and not result.can_restore
    with pytest.raises(SessionRunningError):
        await service.release_session("session", confirmed=True)
