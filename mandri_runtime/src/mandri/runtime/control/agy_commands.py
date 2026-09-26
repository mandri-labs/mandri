import asyncio
import contextlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from mandri.runtime.control.errors import ControlError, ControlTransportError
from mandri.runtime.process import ManagedProcess

_NAME = re.compile(r"^[\w][\w:-]*$")


def command_argv(argv: Sequence[str], text: str) -> list[str]:
    result = [argv[0]]
    value_flags = {"--input-format", "--output-format", "--conversation", "--print-timeout"}
    boolean_flags = {"-p", "--print", "--continue", "--dangerously-skip-permissions"}
    index = 1
    while index < len(argv):
        value = argv[index]
        flag = value.split("=", 1)[0]
        if flag in value_flags:
            index += 1 if "=" in value else 2
            continue
        if flag in boolean_flags:
            index += 1
            continue
        result.append(value)
        index += 1
    return [*result, "-p", text, "--output-format", "json"]


def command_catalog(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("commands")
    if not isinstance(rows, list):
        raise ControlTransportError("Antigravity did not return a command catalog")
    commands: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ControlTransportError("Antigravity returned an invalid command catalog")
        name = row.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ControlTransportError("Antigravity returned an invalid command name")
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        commands[name] = {
            "id": name,
            "name": name,
            "description": str(row.get("description") or ""),
            "aliases": [a for a in aliases if isinstance(a, str) and _NAME.fullmatch(a)],
            "argument_hint": None,
            "accepts_arguments": False,
            "kind": "inspection",
        }
    return list(commands.values())


def _label(value: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).replace("_", " ").capitalize()


def _settings_fields(config: dict[str, Any]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for key, value in config.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            fields.append({"label": _label(key), "value": value})
        elif key == "customModelsConfig" and isinstance(value, dict):
            models = value.get("customModels")
            if isinstance(models, dict):
                for name, settings in models.items():
                    if not isinstance(settings, dict):
                        continue
                    model_name = settings.get("modelName")
                    if isinstance(model_name, str):
                        fields.append({"label": f"Custom model · {name}", "value": model_name})
            else:
                fields.append({"label": "Custom models", "value": "Details unavailable"})
        else:
            fields.append({"label": _label(key), "value": "Native details are not supported yet"})
    return fields


def _item_description(row: dict[str, Any]) -> str:
    parts = []
    description = row.get("description")
    if isinstance(description, str) and description:
        parts.append(description)
    aliases = row.get("aliases")
    if isinstance(aliases, list) and all(isinstance(alias, str) for alias in aliases) and aliases:
        parts.append("Aliases: " + ", ".join(f"/{alias}" for alias in aliases))
    path = row.get("path")
    if isinstance(path, str) and path:
        parts.append(f"Location: {path}")
    if isinstance(row.get("builtin"), bool):
        parts.append("Built-in skill" if row["builtin"] else "Workspace or installed skill")
    if isinstance(row.get("model_invocable"), bool):
        parts.append("Model can invoke" if row["model_invocable"] else "Manual invocation only")
    known = {
        "name",
        "label",
        "scope",
        "description",
        "aliases",
        "path",
        "builtin",
        "model_invocable",
    }
    if row.keys() - known:
        parts.append("Additional native details are not supported yet")
    return " · ".join(parts)


def command_result(name: str, data: dict[str, Any], response: Any) -> dict[str, Any]:
    title = _label(name)
    for key in ("commands", "agents", "skills", "hooks", "permissions", "groups"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        items = []
        for row in rows:
            item_title = (
                row.get("name") or row.get("label") or row.get("scope")
                if isinstance(row, dict)
                else None
            )
            if not isinstance(item_title, str):
                return {
                    "kind": "notice",
                    "title": title,
                    "text": "The native command returned entries that are not supported yet.",
                }
            items.append({"title": item_title, "description": _item_description(row)})
        return {
            "kind": "list",
            "title": title,
            "items": items,
            "empty_message": f"No {key} available.",
        }
    for key in ("changelog", "content"):
        content = data.get(key)
        if isinstance(content, str) and content.strip():
            return {"kind": "text", "title": title, "text": content}
    config = data.get("config")
    if isinstance(config, dict):
        return {"kind": "fields", "title": title, "fields": _settings_fields(config)}
    if data and all(
        value is None or isinstance(value, (str, int, float, bool)) for value in data.values()
    ):
        return {"kind": "fields", "title": title, "fields": _settings_fields(data)}
    if data:
        return {
            "kind": "notice",
            "title": title,
            "text": "The native command completed, but this result format is not supported yet.",
        }
    if (
        isinstance(response, str)
        and response.strip()
        and not response.lstrip().startswith(("{", "["))
    ):
        return {"kind": "text", "title": title, "text": response}
    return {
        "kind": "notice",
        "title": title,
        "text": "The command completed without a displayable result.",
    }


class AgyCommandRunner:
    def __init__(
        self,
        argv: Sequence[str],
        spawn: Callable[[list[str]], Awaitable[ManagedProcess]],
        timeout: float = 20,
    ) -> None:
        self._argv = tuple(argv)
        self._spawn = spawn
        self._timeout = timeout
        self._lock = asyncio.Lock()

    async def list_commands(self) -> list[dict[str, Any]]:
        envelope = await self._run("help")
        return command_catalog(envelope["command"]["data"])

    async def execute_command(self, command_id: str, arguments: str) -> dict[str, Any]:
        async with self._lock:
            try:
                catalog = await self.list_commands()
            except ControlTransportError as error:
                raise ControlError(
                    "Antigravity catalog is unavailable. The command was not sent."
                ) from error
            command = next((row for row in catalog if row["id"] == command_id), None)
            if command is None:
                raise ControlError("This Antigravity command is no longer available")
            if arguments.strip():
                raise ControlError(
                    "Antigravity does not expose a supported argument interface for this command"
                )
            envelope = await self._run(command_id)
            return command_result(command_id, envelope["command"]["data"], envelope.get("response"))

    async def _run(self, name: str) -> dict[str, Any]:
        process: ManagedProcess | None = None
        stderr: asyncio.Task[None] | None = None
        try:
            async with asyncio.timeout(self._timeout):
                process = await self._spawn(command_argv(self._argv, f"/{name}"))
                stderr = asyncio.create_task(self._drain(process))
                lines = []
                size = 0
                while line := await process.read_stdout_line():
                    size += len(line)
                    if size > 4 * 1024 * 1024:
                        raise ControlTransportError("Antigravity command output exceeded its limit")
                    lines.append(line)
                code = await process.wait()
                try:
                    envelope = json.loads("\n".join(lines))
                except ValueError as error:
                    raise ControlTransportError(
                        "Antigravity returned an unreadable command result"
                    ) from error
                if isinstance(envelope, dict) and envelope.get("status") == "ERROR":
                    raise ControlError(
                        "Antigravity command failed; check native sign-in and command availability"
                    )
                if not isinstance(envelope, dict) or envelope.get("status") != "SUCCESS" or code:
                    raise ControlTransportError(
                        "Antigravity command failed; check native sign-in and command availability"
                    )
                command = envelope.get("command")
                if (
                    not isinstance(command, dict)
                    or command.get("name") != name
                    or not isinstance(command.get("data"), dict)
                ):
                    raise ControlTransportError("Antigravity did not confirm the native command")
                return envelope
        except TimeoutError as error:
            raise ControlTransportError(
                "Antigravity command timed out; check native sign-in. It was not retried."
            ) from error
        finally:
            try:
                if process is not None:
                    await process.stop(grace=1)
            finally:
                if stderr is not None:
                    stderr.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stderr

    @staticmethod
    async def _drain(process: ManagedProcess) -> None:
        while await process.read_stderr_line():
            pass
