import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, SessionId
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.errors import (
    ControlTransportError,
    HarnessNotInitializedError,
    PromptDeliveryFailedError,
)
from mandri.runtime.liveness import (
    LivenessEvidence,
    LivenessEvidenceKind,
    WorkingStateTracker,
)
from mandri.runtime.service import RuntimeService


@pytest.fixture
def runtime():
    tracker = WorkingStateTracker()
    tracker.register(SessionId("session"))
    service = RuntimeService({}, liveness=tracker)
    service.registry.mark_live("session")
    return service, tracker


@pytest.mark.parametrize("already_active", [False, True])
async def test_rejection_only_removes_the_rejected_prompt(runtime, already_active):
    service, tracker = runtime
    if already_active:
        tracker.observe(LivenessEvidence(SessionId("session"), LivenessEvidenceKind.TURN_STARTED))
    service._session_state("session").control = SimpleNamespace(
        send_prompt=AsyncMock(side_effect=PromptDeliveryFailedError("rejected"))
    )
    with pytest.raises(PromptDeliveryFailedError):
        await service.send_session_prompt("session", "synthetic input")
    assert service.is_busy("session") is already_active


async def test_native_start_during_rejected_delivery_remains_active(runtime):
    service, tracker = runtime

    async def reject(_content):
        tracker.observe(LivenessEvidence(SessionId("session"), LivenessEvidenceKind.TURN_STARTED))
        raise PromptDeliveryFailedError("rejected")

    service._session_state("session").control = SimpleNamespace(send_prompt=reject)
    with pytest.raises(PromptDeliveryFailedError):
        await service.send_session_prompt("session", "synthetic input")
    assert service.is_busy("session")


async def test_concurrent_rejection_preserves_another_pending_prompt(runtime):
    service, tracker = runtime
    entered, release = asyncio.Event(), asyncio.Event()

    async def send(content):
        if content == "rejected":
            raise PromptDeliveryFailedError("rejected")
        entered.set()
        await release.wait()
        return PromptOutcome(PromptState.QUEUED)

    service._session_state("session").control = SimpleNamespace(send_prompt=send)
    first = asyncio.create_task(service.send_session_prompt("session", "accepted"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(PromptDeliveryFailedError):
            await service.send_session_prompt("session", "rejected")
        assert service.is_busy("session")
    finally:
        release.set()
        await first
    assert service.is_busy("session")
    tracker.observe(LivenessEvidence(SessionId("session"), LivenessEvidenceKind.TURN_ENDED))
    assert not service.is_busy("session")


async def test_route_preparation_failure_releases_provisional_activity(runtime):
    service, _ = runtime
    record = SimpleNamespace(
        harness=HarnessKind.CODEX,
        model_source=ModelSource.GATEWAY,
        model="provider/model",
        reasoning_effort=None,
    )
    service._sessions = SimpleNamespace(get_session=AsyncMock(return_value=record))
    service._routes = Mock()
    service._resume_route = AsyncMock(side_effect=RuntimeError("route failed"))
    control = SimpleNamespace(send_prompt=AsyncMock())
    service._session_state("session").control = control
    with pytest.raises(RuntimeError, match="route failed"):
        await service.send_session_prompt("session", "synthetic input")
    assert not service.is_busy("session")
    control.send_prompt.assert_not_awaited()


@pytest.mark.parametrize("error", [ControlTransportError("lost"), asyncio.CancelledError()])
async def test_ambiguous_delivery_stays_busy(runtime, error):
    service, _ = runtime
    service._session_state("session").control = SimpleNamespace(
        send_prompt=AsyncMock(side_effect=error)
    )
    with pytest.raises(type(error)):
        await service.send_session_prompt("session", "synthetic input")
    assert service.is_busy("session")


async def test_rejection_after_identity_retry_releases_provisional_activity(runtime):
    service, _ = runtime
    control = SimpleNamespace(
        send_prompt=AsyncMock(
            side_effect=[HarnessNotInitializedError("wait"), PromptDeliveryFailedError("rejected")]
        ),
        capture_identity=AsyncMock(return_value=None),
    )
    service._session_state("session").control = control
    with pytest.raises(PromptDeliveryFailedError):
        await service.send_session_prompt("session", "synthetic input")
    assert not service.is_busy("session")
    assert control.send_prompt.await_count == 2


async def test_error_outcome_releases_provisional_activity(runtime):
    service, _ = runtime
    service._session_state("session").control = SimpleNamespace(
        send_prompt=AsyncMock(return_value=PromptOutcome(PromptState.ERROR, "rejected"))
    )
    outcome = await service.send_session_prompt("session", "synthetic input")
    assert outcome.state is PromptState.ERROR
    assert not service.is_busy("session")
