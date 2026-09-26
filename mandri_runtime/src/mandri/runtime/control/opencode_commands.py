import asyncio
from typing import Any
from urllib.parse import quote

import httpx
from mandri.core.opencode import GATEWAY_MODEL_REF
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
            data = response_data(await self._call("GET", "/command", None), "command discovery")
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
        return [
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

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        matches = [item for item in await self._catalog() if item["name"] == command_id]
        if len(matches) != 1:
            raise ControlError("This OpenCode command is no longer available")
        command = matches[0]
        model = command.get("model")
        if not model and command.get("agent"):
            async with asyncio.timeout(15):
                agents = response_data(await self._call("GET", "/agent", None), "agent discovery")
            if not isinstance(agents, list):
                raise ControlTransportError("OpenCode returned an invalid agent catalog")
            agent = next(
                (
                    item
                    for item in agents
                    if isinstance(item, dict) and item.get("name") == command["agent"]
                ),
                None,
            )
            if agent is None:
                raise ControlError("The command's OpenCode agent is unavailable")
            model = agent.get("model")
            if isinstance(model, dict):
                model = f"{model.get('providerID')}/{model.get('modelID')}"
        if model and model != GATEWAY_MODEL_REF:
            raise ControlError(
                "This command selects a model outside the Mandri gateway; "
                "update its native configuration before running it"
            )
        result = response_data(
            await self._call(
                "POST",
                f"/session/{self._session_id}/command",
                {"command": command_id, "arguments": arguments, "model": GATEWAY_MODEL_REF},
            ),
            "command execution",
        )
        if not isinstance(result, dict) or not isinstance(result.get("info"), dict):
            raise ControlTransportError("OpenCode did not confirm the command result")
        if result["info"].get("error"):
            raise ControlError("OpenCode could not complete the command")
        return {"kind": "transcript", "message": "Command completed"}
