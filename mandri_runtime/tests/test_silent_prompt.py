import asyncio
from types import SimpleNamespace

from mandri.core.ids import SessionId
from mandri.core.ports.control import PromptOutcome, PromptState
from mandri.runtime.liveness import LivenessEvidence, LivenessEvidenceKind, WorkingStateTracker
from mandri.runtime.service import RuntimeService


async def test_prompt_remains_busy_before_first_event_and_after_acceptance():
    tracker = WorkingStateTracker()
    tracker.register(SessionId("silent"))
    service = RuntimeService({}, liveness=tracker)
    service.registry.mark_live("silent")
    entered, release = asyncio.Event(), asyncio.Event()

    async def send_prompt(_content):
        entered.set()
        await release.wait()
        return PromptOutcome(state=PromptState.QUEUED)

    service._session_state("silent").control = SimpleNamespace(send_prompt=send_prompt)
    task = asyncio.create_task(service.send_session_prompt("silent", "Hello"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert service.is_busy("silent")
    finally:
        release.set()
        await task
    assert service.is_busy("silent")
    tracker.observe(
        LivenessEvidence(session_id=SessionId("silent"), kind=LivenessEvidenceKind.TURN_ENDED)
    )
    assert not service.is_busy("silent")
