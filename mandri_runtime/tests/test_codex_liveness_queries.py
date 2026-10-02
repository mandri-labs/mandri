from unittest.mock import AsyncMock

import pytest
from mandri.runtime.control.codex_liveness import read_queue, read_snapshot
from mandri.runtime.control.errors import ControlTransportError


@pytest.mark.parametrize("entries,expected", [([], False), ([{"id": "queued"}], True)])
async def test_queue_reads_native_list_instead_of_notification_payload(entries, expected):
    call = AsyncMock(return_value={"result": {"data": entries, "nextCursor": None}})
    assert await read_queue(call, "root") is expected
    call.assert_awaited_once_with("thread/queue/list", {"threadId": "root", "limit": 1})


async def test_snapshot_reads_root_queue_and_all_descendant_pages():
    call = AsyncMock(
        side_effect=[
            {"result": {"thread": {"id": "root", "status": {"type": "idle"}}}},
            {"result": {"data": [], "nextCursor": None}},
            {
                "result": {
                    "data": [
                        {
                            "id": "child",
                            "status": {"type": "active", "activeFlags": ["waitingOnApproval"]},
                        }
                    ],
                    "nextCursor": "next",
                }
            },
            {
                "result": {
                    "data": [{"id": "finished", "status": {"type": "notLoaded"}}],
                    "nextCursor": None,
                }
            },
        ]
    )
    state = await read_snapshot(call, "root")
    assert state.children == frozenset({"child"})
    assert state.approval_pending
    assert not state.queued
    assert call.await_args_list[0].args == (
        "thread/read",
        {"threadId": "root", "includeTurns": False},
    )
    assert call.await_args_list[-1].args[1]["cursor"] == "next"
    assert call.await_args_list[-1].args[1]["ancestorThreadId"] == "root"


@pytest.mark.parametrize("status", [{"type": "notLoaded"}, {"type": "systemError"}, None])
async def test_inconclusive_root_status_is_not_an_idle_snapshot(status):
    call = AsyncMock(return_value={"result": {"thread": {"id": "root", "status": status}}})
    with pytest.raises(ControlTransportError):
        await read_snapshot(call, "root")


async def test_invalid_queue_response_is_not_treated_as_empty():
    call = AsyncMock(return_value={"result": {}})
    with pytest.raises(ControlTransportError):
        await read_queue(call, "root")
