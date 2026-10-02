import asyncio
from unittest.mock import AsyncMock

from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.control.codex_liveness import CodexLivenessSnapshot
from mandri.runtime.liveness import LivenessEvidence, LivenessEvidenceKind, WorkingStateTracker
from mandri.runtime.liveness.codex import CodexLivenessAdapter


def setup_adapter(read_queue=None, read_state=None):
    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = CodexLivenessAdapter(
        Hub(),
        Topic("session.session"),
        tracker,
        session,
        native_identity=lambda: "root",
        read_queue=read_queue,
        read_state=read_state,
    )
    return adapter, tracker, session


def absorb(adapter, method, params=None, identifier=None):
    raw = {"method": method, "params": {"threadId": "root", **(params or {})}}
    if identifier is not None:
        raw["id"] = identifier
    adapter._absorb({"payload": {"source": "codex", "raw": raw}})


def test_active_status_object_preserves_turn_and_pending_prompt():
    adapter, tracker, session = setup_adapter()
    tracker.observe(LivenessEvidence(session, LivenessEvidenceKind.PROMPT_STARTED, "prompt"))
    absorb(adapter, "turn/started", {"turn": {"id": "turn"}})
    absorb(adapter, "thread/status/changed", {"status": {"type": "active", "activeFlags": []}})
    assert tracker.working_state(session).busy
    absorb(adapter, "turn/completed", {"turn": {"id": "turn", "status": "completed"}})
    assert not tracker.working_state(session).busy


async def test_queue_change_reads_authoritative_queue_and_preserves_pending_prompt():
    reader = AsyncMock(return_value=False)
    adapter, tracker, session = setup_adapter(read_queue=reader)
    tracker.observe(LivenessEvidence(session, LivenessEvidenceKind.PROMPT_STARTED, "prompt"))
    absorb(adapter, "thread/queue/changed")
    absorb(adapter, "thread/status/changed", {"status": {"type": "idle"}})
    assert tracker.working_state(session).busy
    await adapter._queue_task
    reader.assert_awaited_once()
    assert tracker.working_state(session).busy
    assert not adapter._queue_open
    absorb(adapter, "turn/started")
    absorb(adapter, "turn/completed")
    assert not tracker.working_state(session).busy


async def test_completion_does_not_release_queued_work():
    reader = AsyncMock(side_effect=[True, False])
    adapter, tracker, session = setup_adapter(read_queue=reader)
    absorb(adapter, "turn/started")
    absorb(adapter, "thread/queue/changed")
    absorb(adapter, "turn/completed")
    await adapter._queue_task
    assert tracker.working_state(session).busy
    absorb(adapter, "thread/queue/changed")
    await adapter._queue_task
    assert not tracker.working_state(session).busy


def test_subagent_activity_items_track_child_lifetime_across_item_completion():
    adapter, tracker, session = setup_adapter()
    for child in ("first", "second"):
        for method in ("item/started", "item/completed"):
            absorb(
                adapter,
                method,
                {"item": {"type": "subAgentActivity", "agentThreadId": child, "kind": "started"}},
            )
    absorb(adapter, "turn/completed")
    for child in ("first", "second"):
        absorb(
            adapter,
            "item/started",
            {"item": {"type": "subAgentActivity", "agentThreadId": child, "kind": "completed"}},
        )
        assert tracker.working_state(session).busy is (child != "second")


async def test_gap_requires_full_snapshot_before_clearing_uncertainty():
    reader = AsyncMock(
        return_value=CodexLivenessSnapshot({"type": "idle"}, False, frozenset(), False)
    )
    adapter, tracker, session = setup_adapter(read_state=reader)
    absorb(adapter, "turn/started")
    absorb(adapter, "item/commandExecution/requestApproval", identifier=1)
    adapter._absorb({"type": "gap"})
    absorb(adapter, "turn/completed")
    assert tracker.working_state(session).uncertain
    await adapter._state_task
    assert not tracker.working_state(session).busy
    assert not tracker.working_state(session).uncertain


async def test_failed_snapshot_retries_once_on_later_native_event():
    reader = AsyncMock(
        side_effect=[
            TimeoutError("native identity not ready"),
            CodexLivenessSnapshot({"type": "idle"}, False, frozenset(), False),
        ]
    )
    adapter, tracker, session = setup_adapter(read_state=reader)
    adapter._absorb({"type": "gap"})
    first_refresh = adapter._state_task
    await first_refresh
    assert tracker.working_state(session).uncertain
    adapter._absorb(
        {
            "payload": {
                "source": "mandri",
                "type": "approval.resolved",
                "raw": {"native_request_ref": "7"},
            }
        }
    )
    adapter._absorb({"payload": {"source": "codex", "raw": {"id": 7, "result": {}}}})
    assert adapter._state_task is first_refresh
    assert reader.await_count == 1
    absorb(adapter, "turn/completed")
    retry = adapter._state_task
    assert retry is not first_refresh
    absorb(adapter, "thread/status/changed", {"status": {"type": "idle"}})
    assert adapter._state_task is retry
    assert tracker.working_state(session).uncertain
    await retry
    assert reader.await_count == 2
    assert not tracker.working_state(session).busy
    absorb(adapter, "turn/completed")
    assert adapter._state_task is retry


async def test_delayed_snapshot_is_discarded_after_local_resolution_and_another_gap():
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def reader():
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            return CodexLivenessSnapshot(
                {"type": "active", "activeFlags": ["waitingOnApproval"]},
                False,
                frozenset(),
                True,
            )
        return CodexLivenessSnapshot({"type": "idle"}, False, frozenset(), False)

    adapter, tracker, session = setup_adapter(read_state=reader)
    absorb(adapter, "item/commandExecution/requestApproval", identifier=1)
    adapter._absorb({"type": "gap"})
    await started.wait()
    adapter._absorb(
        {
            "payload": {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {"native_request_ref": "1"},
            }
        }
    )
    adapter._absorb({"type": "gap"})
    release.set()
    await adapter._state_task
    assert calls == 2
    assert not tracker.working_state(session).busy


def test_malformed_frame_uncertainty_survives_partial_completion_without_reader():
    adapter, tracker, session = setup_adapter()
    adapter._absorb({"payload": {"source": "mandri", "raw": {"error": "parse_error"}}})
    absorb(adapter, "turn/completed")
    assert tracker.working_state(session).busy
    assert tracker.working_state(session).uncertain
    tracker.forget(session)
    tracker.register(session)
    assert not tracker.working_state(session).busy


def test_root_status_and_completion_do_not_absorb_child_turns():
    adapter, tracker, session = setup_adapter()
    absorb(adapter, "turn/started")
    absorb(adapter, "turn/completed", {"threadId": "child"})
    absorb(adapter, "thread/status/changed", {"threadId": "child", "status": {"type": "idle"}})
    assert tracker.working_state(session).busy
    absorb(adapter, "thread/status/changed", {"status": {"type": "idle"}})
    assert not tracker.working_state(session).busy


def test_native_approval_resolutions_preserve_other_pending_requests():
    adapter, tracker, session = setup_adapter()
    for identifier in (11, 12):
        absorb(adapter, "item/commandExecution/requestApproval", identifier=identifier)
    absorb(adapter, "serverRequest/resolved", {"requestId": 11})
    absorb(adapter, "turn/completed")
    assert tracker.working_state(session).busy
    absorb(adapter, "serverRequest/resolved", {"requestId": 12})
    assert not tracker.working_state(session).busy


def test_local_resolution_preserves_typed_request_ids_and_ignores_client_responses():
    adapter, tracker, session = setup_adapter()
    for identifier in (1, "1"):
        absorb(adapter, "item/fileChange/requestApproval", identifier=identifier)
    adapter._absorb(
        {
            "payload": {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {"native_request_ref": "1", "outcome": "expired"},
            }
        }
    )
    assert tracker.working_state(session).busy
    adapter._absorb({"payload": {"source": "codex", "raw": {"id": "1", "result": {}}}})
    assert tracker.working_state(session).busy
    absorb(adapter, "serverRequest/resolved", {"requestId": "1"})
    assert not tracker.working_state(session).busy
