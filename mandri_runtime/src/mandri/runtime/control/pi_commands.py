import asyncio
from typing import Any

from mandri.runtime.control.agents.base import RpcCall
from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.control.pi_builtins import builtin_commands, execute_builtin


class PiCommands:
    def __init__(self, call: RpcCall, gateway_mode: bool = False) -> None:
        self._call = call
        self._gateway_mode = gateway_mode
        self._settled: asyncio.Event | None = None
        self._observed_run = False
        self._error: str | None = None

    async def list_commands(self) -> list[dict[str, Any]]:
        data = await self._call("get_commands", {})
        rows = data.get("commands")
        if not isinstance(rows, list) or any(
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or any(character.isspace() for character in row["name"])
            or row["name"].startswith("/")
            or row.get("source") not in {"extension", "prompt", "skill"}
            for row in rows
        ):
            raise ControlTransportError("Pi returned an invalid command catalog")
        custom = [
            {
                "id": row["name"],
                "name": row["name"],
                "description": row.get("description") or "",
                "aliases": [],
                "kind": row["source"],
            }
            for row in rows
        ]
        names = {row["id"] for row in custom}
        return custom + [
            row for row in builtin_commands(self._gateway_mode) if row["id"] not in names
        ]

    async def execute_command(self, identifier: str, arguments: str) -> dict[str, Any]:
        row = next((row for row in await self.list_commands() if row["id"] == identifier), None)
        if row is None:
            raise ControlError("This Pi command is no longer available")
        if row.get("available") is False:
            raise ControlError(row["unavailable_reason"])
        if row["kind"] == "command":
            return await execute_builtin(self._call, identifier, arguments)
        if self._settled is not None:
            raise ControlError("A Pi command is already running")
        content = f"/{identifier}" + (f" {arguments}" if arguments else "")
        self._settled = asyncio.Event()
        self._observed_run = False
        self._error = None
        try:
            data = await self._call("prompt", {"message": content, "streamingBehavior": "followUp"})
            pending = False
            if "disposition" not in data:
                state = await self._call("get_state", {})
                pending = bool(
                    state.get("isStreaming")
                    or state.get("isCompacting")
                    or state.get("pendingMessageCount")
                )
            if data.get("disposition") in {"started", "queued"} or self._observed_run or pending:
                await self._settled.wait()
            if self._error is not None:
                raise ControlError(self._error)
            return {
                "kind": "transcript",
                "message": "Command handled"
                if data.get("disposition") == "handled"
                else "Command completed",
            }
        finally:
            self._settled = None

    def observe(self, record: dict[str, Any]) -> None:
        if self._settled is None:
            return
        event = record.get("type")
        if event == "agent_start":
            self._observed_run = True
        elif event == "agent_settled":
            self._settled.set()
        elif event == "message_end":
            message = record.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                self._error = (
                    message.get("errorMessage") or "Pi command did not complete"
                    if message.get("stopReason") in {"error", "aborted"}
                    else None
                )

    def close(self) -> None:
        if self._settled is not None:
            self._error = "Pi control stream closed before command completion"
            self._settled.set()
