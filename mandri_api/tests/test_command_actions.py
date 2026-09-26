import asyncio
from typing import Any

from mandri.api.command_actions import register_command_actions
from mandri.core.protocol.commands import CommandCatalogParams
from mandri.core.protocol.frames import RequestFrame
from mandri.core.protocol.registry import ActionRegistry
from mandri.runtime.command_catalogs import CommandCatalogCache
from mandri.runtime.commands import CommandService


class NativeControl:
    def __init__(self) -> None:
        self.finish = asyncio.Event()
        self.calls = 0

    async def list_commands(self) -> list[dict[str, Any]]:
        return [{"id": "inspect", "name": "inspect", "kind": "command"}]

    async def execute_command(self, identity: str, arguments: str) -> dict[str, Any]:
        self.calls += 1
        await self.finish.wait()
        return {"kind": "text", "text": arguments}


async def test_command_actions_ack_before_completion_and_recover_without_replay() -> None:
    control = NativeControl()
    service = CommandService(lambda _: control, lambda _: True, lambda _: False)
    registry = ActionRegistry()
    register_command_actions(registry, service)

    async def call(action: str, **params: Any) -> Any:
        return await registry.handle(
            RequestFrame(
                type="request", op_id=action, action=action, params={"session_id": "s", **params}
            )
        )

    catalog = await call("session.commands")
    assert catalog.ok
    assert catalog.result["commands"][0]["id"] == "inspect"
    first = await call(
        "command.invoke", invocation_id="i", command_id="inspect", arguments="result"
    )
    assert first.ok and first.result["state"] == "running"
    duplicate = await call(
        "command.invoke", invocation_id="i", command_id="inspect", arguments="result"
    )
    assert duplicate.ok
    await asyncio.sleep(0)
    assert control.calls == 1
    recovered = await call("command.list")
    assert recovered.result["invocations"][0]["state"] == "running"
    control.finish.set()
    await asyncio.sleep(0)
    completed = await call("command.get", invocation_id="i")
    assert completed.result["state"] == "succeeded"
    assert completed.result["result"]["text"] == "result"


async def test_startup_catalog_actions_work_without_any_running_session(tmp_path) -> None:
    calls: list[str] = []

    async def discover(scope: CommandCatalogParams) -> list[dict[str, Any]]:
        calls.append(scope.harness)
        return [{"id": "discovered", "name": "discovered"}]

    service = CommandService(lambda _: None, lambda _: False, lambda _: False)
    service.catalogs = CommandCatalogCache(discover, default_cwd=str(tmp_path))
    registry = ActionRegistry()
    register_command_actions(registry, service)
    snapshot = await registry.handle(
        RequestFrame(type="request", op_id="startup", action="command.catalogs", params={})
    )
    assert snapshot.ok
    assert len(snapshot.result["catalogs"]) == 5
    assert snapshot.result["default_cwd"] == str(tmp_path)
    scoped = await registry.handle(
        RequestFrame(
            type="request", op_id="scoped", action="command.catalog", params={"harness": "claude"}
        )
    )
    assert scoped.ok
    assert scoped.result["commands"][0]["id"] == "discovered"
    assert calls.count("claude") == 1
    await service.catalogs.aclose()
