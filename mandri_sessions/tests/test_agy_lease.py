from pathlib import Path

import pytest
from mandri.sessions.agy_lease import AgyConversationLease
from mandri.sessions.errors import SessionRunningError


def test_conversation_lease_excludes_second_writer_and_releases(tmp_path: Path) -> None:
    first = AgyConversationLease(tmp_path, "conversation")
    second = AgyConversationLease(tmp_path, "conversation")
    first.acquire()
    try:
        with pytest.raises(SessionRunningError):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()
    second.release()


def test_different_conversations_have_independent_leases(tmp_path: Path) -> None:
    first = AgyConversationLease(tmp_path, "first")
    second = AgyConversationLease(tmp_path, "second")
    first.acquire()
    try:
        second.acquire()
        second.release()
    finally:
        first.release()
