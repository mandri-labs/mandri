import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness import OpencodeLivenessAdapter, WorkingStateTracker


def setup_adapter():
    tracker = WorkingStateTracker()
    session = SessionId("session")
    tracker.register(session)
    adapter = OpencodeLivenessAdapter(Hub(), Topic("session.session"), tracker, session)
    return adapter, lambda: tracker.working_state(session).busy


def absorb(adapter, kind, properties):
    adapter._absorb(
        {"payload": {"source": "opencode", "raw": {"type": kind, "properties": properties}}}
    )


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_terminal_message_ends_work_without_idle_and_late_completed_parts_do_not_restart(finish):
    adapter, busy = setup_adapter()
    absorb(adapter, "session.status", {"sessionID": "root", "status": {"type": "busy"}})
    assert busy()
    absorb(
        adapter,
        "message.updated",
        {
            "info": {
                "sessionID": "root",
                "role": "assistant",
                "finish": finish,
                "time": {"completed": 2},
            }
        },
    )
    assert not busy()
    for part in [
        {"type": "text", "time": {"end": 2}},
        {"type": "reasoning", "time": {"end": 2}},
        {"type": "tool", "state": {"status": "completed"}},
        {"type": "step-finish"},
    ]:
        absorb(adapter, "message.part.updated", {"part": {"sessionID": "root", **part}})
        assert not busy()


@pytest.mark.parametrize(
    "owner,finish,summary",
    [
        ("root", "tool-calls", False),
        ("root", "unknown", False),
        ("root", "stop", True),
        ("child", "stop", False),
    ],
)
def test_intermediate_messages_summary_and_child_completion_do_not_stop_parent(
    owner, finish, summary
):
    adapter, busy = setup_adapter()
    absorb(adapter, "session.status", {"sessionID": "root", "status": {"type": "busy"}})
    absorb(
        adapter,
        "message.updated",
        {
            "info": {
                "sessionID": owner,
                "role": "assistant",
                "finish": finish,
                "summary": summary,
                "time": {"completed": 2},
            }
        },
    )
    assert busy()


def test_local_expiry_and_native_replies_clear_only_matching_pending_approvals():
    adapter, busy = setup_adapter()
    for reference in ("first", "second"):
        absorb(adapter, "permission.asked", {"id": reference, "sessionID": "child"})
        adapter._absorb(
            {
                "payload": {
                    "type": "approval.pending",
                    "approval_id": "approval-" + reference,
                    "source": "opencode",
                    "raw": {"properties": {"id": reference}},
                }
            }
        )
    absorb(adapter, "session.idle", {"sessionID": "root"})
    assert busy()
    adapter._absorb(
        {
            "payload": {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {"approval_id": "approval-first", "outcome": "expired"},
            }
        }
    )
    assert busy()
    absorb(adapter, "permission.replied", {"requestID": "unrelated", "sessionID": "child"})
    assert busy()
    absorb(adapter, "permission.replied", {"requestID": "second", "sessionID": "child"})
    assert not busy()


def test_automatic_resolution_closes_native_approval_without_pending_card_or_native_reply():
    adapter, busy = setup_adapter()
    absorb(adapter, "permission.asked", {"id": "permission", "sessionID": "child"})
    assert busy()
    adapter._absorb(
        {
            "payload": {
                "type": "approval.resolved",
                "source": "mandri",
                "raw": {
                    "approval_id": "automatic",
                    "native_request_ref": "permission",
                    "outcome": "answered",
                },
            }
        }
    )
    assert not busy()
