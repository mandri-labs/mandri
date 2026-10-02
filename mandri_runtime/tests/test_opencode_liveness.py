import pytest
from mandri.core.hub import Hub, Topic
from mandri.core.ids import SessionId
from mandri.runtime.liveness import (
    LivenessEvidence,
    LivenessEvidenceKind,
    OpencodeLivenessAdapter,
    WorkingStateTracker,
)


@pytest.mark.parametrize("started", [False, True])
def test_session_error_releases_pending_prompt(started):
    sid = SessionId("session")
    tracker = WorkingStateTracker()
    tracker.register(sid)
    adapter = OpencodeLivenessAdapter(Hub(), Topic("session"), tracker, sid)
    tracker.observe(LivenessEvidence(sid, LivenessEvidenceKind.PROMPT_STARTED, "prompt"))
    if started:
        adapter._absorb(
            {
                "payload": {
                    "source": "opencode",
                    "raw": {
                        "type": "session.status",
                        "properties": {"sessionID": "root", "status": {"type": "busy"}},
                    },
                }
            }
        )
    assert tracker.working_state(sid).busy
    adapter._absorb(
        {
            "payload": {
                "source": "opencode",
                "raw": {
                    "type": "session.error",
                    "properties": {
                        "sessionID": "root",
                        "error": {"name": "ProviderModelNotFoundError"},
                    },
                },
            }
        }
    )
    assert not tracker.working_state(sid).busy


def test_child_error_does_not_release_parent_prompt():
    sid = SessionId("session")
    tracker = WorkingStateTracker()
    tracker.register(sid)
    adapter = OpencodeLivenessAdapter(Hub(), Topic("session"), tracker, sid)
    for owner in ("root", "child"):
        adapter._absorb(
            {
                "payload": {
                    "source": "opencode",
                    "raw": {
                        "type": "session.status",
                        "properties": {"sessionID": owner, "status": {"type": "busy"}},
                    },
                }
            }
        )
    adapter._absorb(
        {
            "payload": {
                "source": "opencode",
                "raw": {"type": "session.error", "properties": {"sessionID": "child", "error": {}}},
            }
        }
    )
    assert tracker.working_state(sid).busy
    adapter._absorb(
        {
            "payload": {
                "source": "opencode",
                "raw": {"type": "session.idle", "properties": {"sessionID": "root"}},
            }
        }
    )
    assert not tracker.working_state(sid).busy


def test_root_and_first_child_completion_preserve_second_child_activity():
    sid = SessionId("session")
    tracker = WorkingStateTracker()
    tracker.register(sid)
    adapter = OpencodeLivenessAdapter(Hub(), Topic("session"), tracker, sid)
    for owner in ("root", "first", "second"):
        adapter._absorb(
            {
                "payload": {
                    "source": "opencode",
                    "raw": {
                        "type": "session.status",
                        "properties": {"sessionID": owner, "status": {"type": "busy"}},
                    },
                }
            }
        )
    for owner in ("root", "first", "second"):
        adapter._absorb(
            {
                "payload": {
                    "source": "opencode",
                    "raw": {"type": "session.idle", "properties": {"sessionID": owner}},
                }
            }
        )
        assert tracker.working_state(sid).busy is (owner != "second")


@pytest.mark.parametrize("identity_known", [False, True])
def test_child_first_event_does_not_claim_root_or_end_its_pending_prompt(identity_known):
    sid = SessionId("session")
    tracker = WorkingStateTracker()
    tracker.register(sid)
    native_id = "root" if identity_known else None
    adapter = OpencodeLivenessAdapter(
        Hub(), Topic("session"), tracker, sid, native_identity=lambda: native_id
    )
    tracker.observe(LivenessEvidence(sid, LivenessEvidenceKind.PROMPT_STARTED, "prompt"))
    for owner, status in (("child", "busy"), ("child", "idle"), ("root", "busy")):
        adapter._absorb(
            {
                "payload": {
                    "source": "opencode",
                    "raw": {
                        "type": "session.status",
                        "properties": {"sessionID": owner, "status": {"type": status}},
                    },
                }
            }
        )
        assert tracker.working_state(sid).busy
    native_id = "root"
    adapter._absorb(
        {
            "payload": {
                "source": "opencode",
                "raw": {"type": "session.idle", "properties": {"sessionID": "root"}},
            }
        }
    )
    assert not tracker.working_state(sid).busy
