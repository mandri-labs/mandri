from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mandri.core.ids import HarnessKind, HarnessSessionId, SessionId
from mandri.core.types.execution import ExecutionBackend, PrivacyMode
from mandri.core.types.model_selection import ModelSource
from mandri.runtime.control.agy import AgyControlAdapter
from mandri.runtime.control.errors import (
    ControlTransportError,
    PromptDeliveryFailedError,
    PromptDeliveryUnknownError,
)
from mandri.runtime.launch_preparation import PreparedLaunch
from mandri.runtime.liveness.tracker import WorkingStateTracker
from mandri.runtime.pump import LinePump
from mandri.runtime.registry import STOPPED
from mandri.runtime.service import RuntimeService
from mandri.sessions.errors import SessionRunningError


@pytest.mark.parametrize("failure_stage", ["identity", "stdin"])
async def test_agy_delivery_preserves_uncertainty_after_write(
    monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        if failure_stage == "stdin":
            yield b'{"event":"init","conversation_id":"synthetic-conversation"}\n'

    tracker = WorkingStateTracker()
    sid = SessionId("synthetic")
    tracker.register(sid)
    service = RuntimeService({"agy": ["agy"]}, liveness=tracker)
    service.registry.mark_live(str(sid), Mock(returncode=None), "agy")
    sink = Mock(write=Mock(side_effect=BrokenPipeError()), drain=AsyncMock())
    control = AgyControlAdapter(
        LinePump(chunks), sink, Mock(aclose=AsyncMock()), AsyncMock(return_value=True)
    )
    service._session_state(str(sid)).control = control
    monkeypatch.setattr(service, "_apply_pending_model", AsyncMock())
    expected = (
        PromptDeliveryFailedError if failure_stage == "identity" else PromptDeliveryUnknownError
    )
    with pytest.raises(expected):
        await service.send_session_prompt(str(sid), "Synthetic prompt")
    assert tracker.working_state(sid).busy == (failure_stage == "stdin")
    if failure_stage == "identity":
        sink.write.assert_not_called()
    else:
        sink.write.assert_called_once()
    await control.aclose()


def resume_service(monkeypatch: pytest.MonkeyPatch):
    service = RuntimeService({"agy": ["agy"]})
    record = SimpleNamespace(
        id=SessionId("synthetic"),
        harness=HarnessKind.AGY,
        model="default",
        model_source=ModelSource.NATIVE,
        execution_backend=ExecutionBackend.HOST,
        privacy_mode=PrivacyMode.NONE,
        privacy_scope_id=None,
        native_id=HarnessSessionId("expected-conversation"),
        parent_session_id=None,
        project_path=str(Path("synthetic-workspace")),
        reasoning_effort=None,
    )
    prepared = PreparedLaunch(["agy"], {}, None)
    resource = Mock(aclose=AsyncMock())

    def prepare(*args):
        service._session_state("synthetic").agy = resource
        return prepared

    monkeypatch.setattr(service, "_load_resumable_record", AsyncMock(return_value=record))
    monkeypatch.setattr(service, "_resume_route", AsyncMock(return_value=None))
    monkeypatch.setattr(service, "_resume_launch_mode", Mock(return_value=None))
    monkeypatch.setattr(service._launch, "prepare", Mock(return_value=prepared))
    monkeypatch.setattr(service, "_prepare_agy", prepare)
    monkeypatch.setattr(service, "_ensure_session_available", AsyncMock())
    return service, resource


@pytest.mark.parametrize("failure", ["ownership", "spawn"])
async def test_resume_failure_before_spawn_releases_profile_resource(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    service, resource = resume_service(monkeypatch)
    spawn = AsyncMock(side_effect=OSError("synthetic spawn failure"))
    monkeypatch.setattr(service, "_spawn_harness", spawn)
    if failure == "ownership":
        monkeypatch.setattr(
            service,
            "_ensure_session_available",
            AsyncMock(side_effect=SessionRunningError("synthetic owner")),
        )
    with pytest.raises((OSError, SessionRunningError)):
        await service.resume_session("synthetic")
    resource.aclose.assert_awaited_once()
    assert service._session_state("synthetic").agy is None
    assert not service._session_state("synthetic").resuming
    if failure == "ownership":
        spawn.assert_not_awaited()


async def test_wrong_resume_identity_stops_process_before_releasing_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, resource = resume_service(monkeypatch)
    order: list[str] = []
    process = Mock(returncode=None, stop=AsyncMock(side_effect=lambda *args: order.append("stop")))
    control = Mock(
        capture_identity=AsyncMock(side_effect=ControlTransportError("different conversation")),
        aclose=AsyncMock(side_effect=lambda: order.append("control")),
    )
    resource.aclose.side_effect = lambda: order.append("profile")
    service._session_state("synthetic").control = control
    monkeypatch.setattr(service, "_spawn_harness", AsyncMock(return_value=process))
    monkeypatch.setattr(service, "_attach_feed", Mock())
    monkeypatch.setattr(service, "_attach_control", Mock())
    with pytest.raises(ControlTransportError, match="different conversation"):
        await service.resume_session("synthetic")
    assert order == ["stop", "control", "profile"]
    assert service.registry.status("synthetic") == STOPPED
    assert service._session_state("synthetic").agy is None
    assert not service._session_state("synthetic").resuming
