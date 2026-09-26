import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

from mandri.runtime.control.errors import ControlError, ControlTransportError


class CodexCommands:
    def __init__(
        self,
        call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        cwd: str | None,
    ) -> None:
        self._call = call
        self._cwd = cwd
        self._skills: dict[str, dict[str, Any]] = {}
        self._completion: asyncio.Future[dict[str, Any]] | None = None
        self._turn_id: str | None = None
        self._early_completions: dict[str, dict[str, Any]] = {}

    async def list_commands(self) -> list[dict[str, Any]]:
        self._skills.clear()
        response = await self._call(
            "skills/list", {"cwds": [self._cwd] if self._cwd else [], "forceReload": True}
        )
        result = self._result(response)
        entries = result.get("data")
        if not isinstance(entries, list):
            raise ControlError("Codex returned an invalid skill catalog")
        commands = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("skills"), list):
                raise ControlError("Codex returned an invalid skill catalog")
            if entry.get("errors"):
                raise ControlError("Codex could not load all skills; check skill configuration")
            for skill in entry["skills"]:
                if not isinstance(skill, dict):
                    raise ControlError("Codex returned invalid skill metadata")
                name, path = skill.get("name"), skill.get("path")
                if not isinstance(name, str) or not isinstance(path, str) or not name or not path:
                    raise ControlError("Codex returned invalid skill metadata")
                identifier = "skill:" + hashlib.sha256(path.encode()).hexdigest()
                if identifier in self._skills:
                    continue
                self._skills[identifier] = skill
                enabled = skill.get("enabled") is True
                commands.append(
                    {
                        "id": identifier,
                        "name": name,
                        "description": skill.get("description", ""),
                        "aliases": [],
                        "argument_hint": "",
                        "kind": "skill",
                        "available": enabled,
                        "unavailable_reason": None if enabled else "Disabled in Codex",
                    }
                )
        return commands

    async def execute(self, identifier: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._completion is not None:
            raise ControlError("A Codex command is already running")
        await self.list_commands()
        if self._completion is not None:
            raise ControlError("A Codex command is already running")
        skill = self._skills.get(identifier)
        if skill is None or skill.get("enabled") is not True:
            raise ControlError("The selected Codex skill is no longer available")
        params["input"].append({"type": "skill", "name": skill["name"], "path": skill["path"]})
        completion: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._completion = completion
        try:
            result = self._result(await self._call("turn/start", params))
            turn = result.get("turn")
            if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
                raise ControlTransportError("Codex did not identify the command turn")
            self._turn_id = turn["id"]
            early = self._early_completions.get(self._turn_id)
            if early is not None and not completion.done():
                completion.set_result(early)
            terminal = await completion
            status = terminal.get("status")
            if status not in ("completed", "failed", "interrupted"):
                raise ControlTransportError("Codex command completion status is unknown")
            if status != "completed":
                error = terminal.get("error")
                detail = error.get("message") if isinstance(error, dict) else None
                raise ControlError(
                    detail
                    if isinstance(detail, str)
                    else f"Codex command ended: {status or 'unknown'}"
                )
            return {"kind": "transcript", "message": "Skill completed"}
        finally:
            if not completion.done():
                completion.cancel()
            elif not completion.cancelled():
                completion.exception()
            self._completion = None
            self._turn_id = None
            self._early_completions.clear()

    def observe(self, turn: dict[str, Any]) -> None:
        completion = self._completion
        identifier = turn.get("id")
        if completion is None or completion.done() or not isinstance(identifier, str):
            return
        if identifier == self._turn_id:
            completion.set_result(turn)
        elif self._turn_id is None:
            self._early_completions[identifier] = turn

    def close(self) -> None:
        if self._completion is not None and not self._completion.done():
            self._completion.set_exception(ControlTransportError("Codex command connection closed"))

    @staticmethod
    def _result(message: dict[str, Any]) -> dict[str, Any]:
        error = message.get("error")
        if isinstance(error, dict):
            text = error.get("message")
            raise ControlError(text if isinstance(text, str) else "Codex request failed")
        result = message.get("result")
        if not isinstance(result, dict):
            raise ControlTransportError("Codex returned an invalid response")
        return result
