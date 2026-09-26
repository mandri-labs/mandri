import json
from pathlib import Path

from mandri.sessions.transcripts.agy_status import agy_journal_status


def journal(root: Path, records: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mandri-events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def stop(native_id: str, stamp: int, idle: bool = True) -> dict:
    return {
        "event": "hook",
        "hook": "Stop",
        "data": {"conversationId": native_id, "fullyIdle": idle},
        "_mandri_recorded_at_ns": stamp,
    }


def test_idle_marker_requires_matching_conversation_and_fresh_timestamp(tmp_path: Path) -> None:
    journal(tmp_path, [stop("requested", 200), stop("other", 300)])
    assert agy_journal_status(tmp_path, None, "requested", 100) is False
    assert agy_journal_status(tmp_path, None, "requested", 250) is None
    assert agy_journal_status(tmp_path, None, "missing", 100) is None


def test_new_invocation_invalidates_old_idle_marker(tmp_path: Path) -> None:
    journal(
        tmp_path,
        [
            stop("requested", 200),
            {
                "event": "PreInvocation",
                "conversationId": "requested",
                "_mandri_recorded_at_ns": 300,
            },
        ],
    )
    assert agy_journal_status(tmp_path, None, "requested", 100) is True


def test_resumed_process_cannot_inherit_old_idle_marker(tmp_path: Path) -> None:
    journal(
        tmp_path,
        [
            stop("requested", 200),
            {
                "event": "init",
                "conversation_id": "requested",
                "_mandri_recorded_at_ns": 300,
            },
        ],
    )
    assert agy_journal_status(tmp_path, None, "requested", 100) is None


def test_partial_background_completion_stays_busy(tmp_path: Path) -> None:
    journal(tmp_path, [stop("requested", 200, idle=False)])
    assert agy_journal_status(tmp_path, None, "requested", 100) is True
