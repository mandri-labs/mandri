import asyncio
from unittest.mock import AsyncMock

import pytest
from mandri.runtime.control.codex_commands import CodexCommands
from mandri.runtime.control.errors import ControlError, ControlTransportError


def catalog(skills=None):
    return {"result": {"data": [{"cwd": "/workspace", "errors": [], "skills": skills or []}]}}


def skill(name="inspect", path="/workspace/skills/inspect/SKILL.md", enabled=True):
    return {"name": name, "path": path, "enabled": enabled, "description": "Inspect changes"}


async def test_catalog_is_native_scoped_and_identity_distinguishes_duplicate_names():
    call = AsyncMock(return_value=catalog([skill(), skill(path="/user/inspect", enabled=False)]))
    commands = CodexCommands(call, "/workspace")
    entries = await commands.list_commands()
    call.assert_awaited_once_with("skills/list", {"cwds": ["/workspace"], "forceReload": True})
    assert len(entries) == 2
    assert entries[0]["id"] != entries[1]["id"]
    assert entries[1]["available"] is False
    call.return_value = catalog()
    assert await commands.list_commands() == []


async def test_removed_skill_is_not_sent_as_prompt():
    call = AsyncMock(return_value=catalog([skill()]))
    commands = CodexCommands(call, "/workspace")
    identifier = (await commands.list_commands())[0]["id"]
    call.return_value = catalog()
    with pytest.raises(ControlError, match="no longer available"):
        await commands.execute(identifier, {"threadId": "thread", "input": []})
    assert all(args.args[0] == "skills/list" for args in call.call_args_list)


@pytest.mark.parametrize("early", [False, True])
async def test_command_waits_for_correlated_completion_and_uses_native_skill(early):
    started = asyncio.Event()
    commands = None

    async def call(method, params):
        if method == "skills/list":
            return catalog([skill()])
        assert params["input"] == [
            {"type": "text", "text": "unchanged arguments"},
            {"type": "skill", "name": "inspect", "path": skill()["path"]},
        ]
        started.set()
        if early:
            commands.observe({"id": "turn", "status": "completed"})
        return {"result": {"turn": {"id": "turn", "status": "inProgress"}}}

    commands = CodexCommands(call, "/workspace")
    identifier = (await commands.list_commands())[0]["id"]
    task = asyncio.create_task(
        commands.execute(
            identifier,
            {
                "threadId": "thread",
                "input": [{"type": "text", "text": "unchanged arguments"}],
            },
        )
    )
    await started.wait()
    if not early:
        assert not task.done()
        commands.observe({"id": "other", "status": "completed"})
        assert not task.done()
        commands.observe({"id": "turn", "status": "completed"})
    assert await task == {"kind": "transcript", "message": "Skill completed"}


@pytest.mark.parametrize("terminal", ["failed", "interrupted", "closed"])
async def test_failure_after_ack_is_not_success(terminal):
    started = asyncio.Event()

    async def call(method, params):
        if method == "skills/list":
            return catalog([skill()])
        started.set()
        return {"result": {"turn": {"id": "turn"}}}

    commands = CodexCommands(call, None)
    identifier = (await commands.list_commands())[0]["id"]
    task = asyncio.create_task(commands.execute(identifier, {"input": []}))
    await started.wait()
    if terminal == "closed":
        commands.close()
    else:
        commands.observe({"id": "turn", "status": terminal})
    with pytest.raises(ControlError) as failure:
        await task
    assert isinstance(failure.value, ControlTransportError) is (terminal == "closed")
    assert commands._completion is None


async def test_invalid_or_failed_discovery_is_not_empty_success():
    call = AsyncMock(return_value={"error": {"message": "unsupported method"}})
    commands = CodexCommands(call, None)
    with pytest.raises(ControlError, match="unsupported method"):
        await commands.list_commands()
    call.return_value = {"result": {}}
    with pytest.raises(ControlError, match="invalid skill catalog"):
        await commands.list_commands()
