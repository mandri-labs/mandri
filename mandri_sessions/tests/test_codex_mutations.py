from unittest.mock import AsyncMock

import pytest
from mandri.core.ids import SessionId, SessionTitle
from mandri.core.version import __version__
from mandri.sessions.adapters.codex_mutations import CodexMutationsAdapter
from mandri.sessions.errors import ServerCommunicationError, SessionDeleteError


@pytest.mark.parametrize(
    ("operation", "method", "params"),
    [
        ("delete", "thread/delete", {"threadId": "test-thread"}),
        ("rename", "thread/name/set", {"threadId": "test-thread", "name": "New title"}),
        ("exists", "thread/resume", {"threadId": "test-thread"}),
    ],
)
async def test_mutations_complete_handshake_before_request(operation, method, params):
    connection = AsyncMock()
    connection.receive.side_effect = [
        {"id": 1, "result": {}},
        {"method": "thread/status/changed", "params": {}},
        {"id": 2, "result": {}},
    ]
    adapter = CodexMutationsAdapter(AsyncMock(return_value=connection))
    args = [SessionId("test-thread")]
    if operation == "rename":
        args.append(SessionTitle("New title"))

    result = await getattr(adapter, f"{operation}_async")(*args)

    assert result is (True if operation == "exists" else None)
    messages = [call.args[0] for call in connection.send.await_args_list]
    assert messages == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"clientInfo": {"name": "mandri", "title": "Mandri", "version": __version__}},
        },
        {"jsonrpc": "2.0", "method": "initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
    ]
    connection.close.assert_awaited_once()


async def test_initialization_rejection_is_a_communication_error_and_closes_connection():
    connection = AsyncMock()
    connection.receive.return_value = {
        "id": 1,
        "error": {"code": -32600, "message": "Invalid request: missing field `version`"},
    }
    adapter = CodexMutationsAdapter(AsyncMock(return_value=connection))

    with pytest.raises(ServerCommunicationError, match=r"initialize failed:.*version"):
        await adapter.delete_async(SessionId("test-thread"))

    connection.send.assert_awaited_once()
    connection.close.assert_awaited_once()


async def test_delete_rejection_keeps_operation_error_and_closes_connection():
    connection = AsyncMock()
    connection.receive.side_effect = [
        {"id": 1, "result": {}},
        {"id": 2, "error": {"code": -32602, "message": "cannot delete thread"}},
    ]
    adapter = CodexMutationsAdapter(AsyncMock(return_value=connection))

    with pytest.raises(SessionDeleteError, match="thread/delete failed: cannot delete thread"):
        await adapter.delete_async(SessionId("test-thread"))

    connection.close.assert_awaited_once()
