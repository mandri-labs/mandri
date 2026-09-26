import asyncio
import uuid
from typing import Any

from mandri.runtime.control.claude_command_results import command_result
from mandri.runtime.control.errors import ControlError, ControlTransportError


class ClaudeCommands:
    def __init__(self) -> None:
        self.initialized = False
        self._catalog: dict[str, dict[str, Any]] = {}
        self._terminal: set[str] = set()
        self._result: asyncio.Future[dict[str, Any]] | None = None
        self._native_id: str | None = None
        self._output: list[str] = []
        self._structured: dict[str, Any] | None = None
        self._assistant_text: str | None = None
        self._assistant_in_transcript = False

    @property
    def pending(self) -> bool:
        return self._result is not None

    def replace(self, commands: list[Any]) -> None:
        catalog: dict[str, dict[str, Any]] = {}
        for item in commands:
            item = {"name": item} if isinstance(item, str) else item
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name or any(c.isspace() for c in name):
                continue
            name = name.removeprefix("/")
            if not name:
                continue
            aliases = item.get("aliases", [])
            hint = item.get("argumentHint")
            catalog[name] = {
                "id": name,
                "name": name,
                "description": item.get("description") or "",
                "aliases": [alias.removeprefix("/") for alias in aliases if isinstance(alias, str)]
                if isinstance(aliases, list)
                else [],
                "argument_hint": hint if isinstance(hint, str) else None,
                "kind": "command",
            }
        self._catalog = catalog
        self.initialized = True

    def descriptors(self) -> list[dict[str, Any]]:
        return [
            dict(
                item,
                available=name not in self._terminal,
                unavailable_reason="This command requires the Claude terminal"
                if name in self._terminal
                else None,
            )
            for name, item in self._catalog.items()
        ]

    def begin(
        self,
        command_id: str,
        arguments: str,
    ) -> tuple[str, str, asyncio.Future[dict[str, Any]]]:
        if self.pending:
            raise ControlError("A Claude command is already in progress")
        if command_id not in self._catalog:
            raise ControlError("Command catalog changed; select an available command")
        if command_id in self._terminal:
            raise ControlError("This command requires the Claude terminal")
        self._native_id = str(uuid.uuid4())
        self._output = []
        self._structured = None
        self._assistant_text = None
        self._assistant_in_transcript = False
        self._result = asyncio.get_running_loop().create_future()
        content = f"/{command_id}" + (f" {arguments}" if arguments else "")
        return self._native_id, content, self._result

    def observe(self, frame: dict[str, Any]) -> None:
        kind, subtype = frame.get("type"), frame.get("subtype")
        if kind == "system" and subtype == "init":
            terminal = frame.get("terminal_slash_commands")
            if isinstance(terminal, list):
                self._terminal = {
                    name.removeprefix("/") for name in terminal if isinstance(name, str)
                }
            commands = frame.get("slash_commands")
            if isinstance(commands, list) and not self.initialized:
                self.replace(commands)
        if kind == "system" and subtype == "commands_changed":
            commands = frame.get("commands")
            if isinstance(commands, list):
                self.replace(commands)
        if not self.pending:
            return
        if (
            kind == "assistant"
            and not frame.get("parent_tool_use_id")
            and not any(
                frame.get(key) is True for key in ("isMeta", "isSynthetic", "turnCompanion")
            )
        ):
            self._assistant_text = _assistant_text(frame)
            message = frame.get("message")
            self._assistant_in_transcript = (
                isinstance(message, dict) and message.get("model") != "<synthetic>"
            )
        structured = command_result(frame)
        if structured is not None:
            self._structured = structured
        if kind == "system" and subtype == "local_command_output":
            content = frame.get("content")
            if isinstance(content, str) and content:
                self._output.append(content)
        if kind != "result":
            return
        if frame.get("is_error") or str(subtype).startswith("error"):
            errors = frame.get("errors")
            message = frame.get("result")
            if not isinstance(message, str) or not message:
                message = (
                    "; ".join(e for e in errors if isinstance(e, str))
                    if isinstance(errors, list)
                    else "Claude command failed"
                )
            self.fail(message, confirmed=True)
            return
        if subtype != "success":
            self.fail("Claude returned an unrecognized command outcome")
            return
        text = frame.get("result")
        in_transcript = (
            not self._output
            and self._assistant_in_transcript
            and bool(self._assistant_text)
            and text == self._assistant_text
        )
        if isinstance(text, str) and text and text not in self._output:
            self._output.append(text)
        result = self._result
        self._result = None
        if result is not None and not result.done():
            text = "\n\n".join(self._output)
            if self._structured is not None:
                result.set_result(self._structured)
            elif text:
                result.set_result(
                    {
                        "kind": "transcript" if in_transcript else "text",
                        "text": text,
                        "native_id": self._native_id,
                    }
                )
            elif self._assistant_text:
                result.set_result(
                    {
                        "kind": "transcript" if self._assistant_in_transcript else "text",
                        "text": self._assistant_text,
                        "native_id": self._native_id,
                    }
                )
            else:
                result.set_result(
                    {
                        "kind": "notice",
                        "text": "The native command ended without a textual response.",
                        "native_id": self._native_id,
                    }
                )

    def fail(self, message: str, *, confirmed: bool = False) -> None:
        result, self._result = self._result, None
        if result is not None and not result.done():
            result.set_exception(
                ControlError(message) if confirmed else ControlTransportError(message)
            )


def _assistant_text(frame: dict[str, Any]) -> str | None:
    if any(frame.get(key) is True for key in ("isMeta", "isSynthetic", "turnCompanion")):
        return None
    message = frame.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content if content.strip() else None
    if not isinstance(content, list):
        return None
    texts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    return "\n\n".join(texts) or None
