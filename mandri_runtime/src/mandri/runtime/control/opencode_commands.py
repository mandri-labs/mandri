import asyncio
from typing import Any
from urllib.parse import quote

import httpx
from mandri.core.opencode import gateway_model
from mandri.runtime.control.agents.base import HttpCall
from mandri.runtime.control.errors import ControlError, ControlTransportError


def response_data(response: httpx.Response, operation: str) -> Any:
    if not 200 <= response.status_code < 300:
        raise ControlTransportError(f"OpenCode {operation} failed ({response.status_code})")
    try:
        return response.json()
    except ValueError:
        raise ControlTransportError(f"OpenCode returned an invalid {operation} response") from None


class OpencodeCommands:
    def __init__(self, call: HttpCall, session_id: str) -> None:
        self._call = call
        self._session_id = quote(session_id, safe="")

    async def _catalog(self) -> list[dict[str, Any]]:
        async with asyncio.timeout(15):
            response_data(await self._call("GET", "/api/integration", None), "plugin activation")
            result = response_data(
                await self._call("GET", "/api/command", None), "command discovery"
            )
            data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("description", ""), (str, type(None)))
            or not isinstance(item.get("hints", []), list)
            for item in data
        ):
            raise ControlTransportError("OpenCode returned an invalid command catalog")
        return data

    async def list_commands(self) -> list[dict[str, Any]]:
        prompts = [
            {
                "id": item["name"],
                "name": item["name"],
                "description": item.get("description") or "",
                "aliases": [],
                "argument_hint": " ".join(
                    hint for hint in item.get("hints", []) if isinstance(hint, str)
                )
                or None,
                "kind": "prompt",
            }
            for item in await self._catalog()
        ]
        agents = await self._agents()
        return [
            {
                "id": "compact",
                "name": "compact",
                "description": "Compact the conversation context",
                "aliases": [],
                "accepts_arguments": False,
                "kind": "command",
            },
            *[
                {
                    "id": f"agent:{agent['id']}",
                    "name": agent["name"],
                    "description": agent.get("description") or "",
                    "aliases": [],
                    "accepts_arguments": False,
                    "kind": "agent",
                }
                for agent in agents
            ],
            *[item for item in prompts if item["id"] != "compact"],
        ]

    async def _agents(self) -> list[dict[str, Any]]:
        result = response_data(await self._call("GET", "/api/agent", None), "agent discovery")
        agents = result.get("data") if isinstance(result, dict) else None
        if not isinstance(agents, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not isinstance(item.get("name"), str)
            for item in agents
        ):
            raise ControlTransportError("OpenCode returned an invalid agent catalog")
        return [
            item
            for item in agents
            if not item.get("hidden") and item.get("mode") in {"primary", "all"}
        ]

    async def select_agent(self, agent: str) -> None:
        async with asyncio.timeout(15):
            response_data(await self._call("GET", "/api/integration", None), "plugin activation")
            if not any(item["id"] == agent for item in await self._agents()):
                raise ControlError("This OpenCode agent is no longer available")
        result = await self._call(
            "POST", f"/api/session/{self._session_id}/agent", {"agent": agent}
        )
        if result.status_code != 204:
            raise ControlTransportError("OpenCode rejected the agent selection")

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        if command_id.startswith("agent:"):
            if arguments:
                raise ControlError("Agent selection does not accept arguments")
            await self.select_agent(command_id.removeprefix("agent:"))
            return {"kind": "notice", "message": "Agent selected"}
        if command_id == "compact":
            if arguments:
                raise ControlError("OpenCode compaction does not accept arguments")
            selected = await self._call(
                "POST", f"/api/session/{self._session_id}/model", {"model": gateway_model()}
            )
            if selected.status_code != 204:
                raise ControlTransportError("OpenCode rejected the gateway model selection")
            response_data(
                await self._call("POST", f"/api/session/{self._session_id}/compact", {}),
                "compaction submission",
            )
            return {"kind": "transcript", "message": "Compaction submitted"}
        matches = [item for item in await self._catalog() if item["name"] == command_id]
        if len(matches) != 1:
            raise ControlError("This OpenCode command is no longer available")
        selected = await self._call(
            "POST", f"/api/session/{self._session_id}/model", {"model": gateway_model()}
        )
        if selected.status_code != 204:
            raise ControlTransportError("OpenCode rejected the gateway model selection")
        result = await self._call(
            "POST",
            f"/api/session/{self._session_id}/command",
            {"name": command_id, "text": arguments},
        )
        if result.status_code != 204:
            raise ControlTransportError("OpenCode did not confirm the command result")
        return {"kind": "transcript", "message": "Command submitted"}
