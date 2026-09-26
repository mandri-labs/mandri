import json
from typing import Any

from mandri.runtime.control.agents.base import RpcCall
from mandri.runtime.control.errors import ControlError

_BUILTINS = {
    "compact": ("Compact the current context", "[instructions]"),
    "new": ("Start a new session", None),
    "name": ("Show or set the session name", "[name]"),
    "session": ("Show session information and usage", None),
    "export": ("Export the session as HTML", "[path]"),
    "copy": ("Show the last assistant message", None),
    "model": ("List models or select a provider/model", "[provider/model]"),
    "thinking": ("List or set supported thinking levels", "[level]"),
    "tree": ("Show the session tree", None),
    "fork": ("List fork points or fork from a message", "[entry-id]"),
    "clone": ("Duplicate this session at its current position", None),
    "resume": ("Resume a saved session file", "<session-path>"),
}
_TERMINAL = {
    "settings": "Open Pi settings",
    "scoped-models": "Configure interactive model cycling",
    "login": "Authenticate a provider",
    "logout": "Remove provider authentication",
    "llama": "Manage llama.cpp models",
    "import": "Import a JSONL session",
    "share": "Upload a session to the Pi viewer",
    "bug": "Prepare a Pi bug report",
    "trust": "Save project trust",
    "reload": "Reload Pi resources",
    "hotkeys": "Show terminal shortcuts",
    "changelog": "Show the Pi changelog",
    "quit": "Quit the Pi terminal",
}


def builtin_commands(gateway_mode: bool = False) -> list[dict[str, Any]]:
    rows = [
        {
            "id": name,
            "name": name,
            "description": description,
            "argument_hint": hint,
            "accepts_arguments": hint is not None,
            "kind": "command",
        }
        for name, (description, hint) in _BUILTINS.items()
    ]
    for row in rows:
        if gateway_mode and row["id"] == "model":
            row.update(
                available=False,
                unavailable_reason="Select the model using the session model selector",
            )
        if gateway_mode and row["id"] == "thinking":
            row.update(
                available=False,
                unavailable_reason="Use the session reasoning selector",
            )
    rows.extend(
        {
            "id": name,
            "name": name,
            "description": description,
            "kind": "command",
            "available": False,
            "unavailable_reason": "This command is available in Pi's terminal interface",
        }
        for name, description in _TERMINAL.items()
    )
    return rows


async def execute_builtin(call: RpcCall, identifier: str, arguments: str) -> dict[str, Any]:
    arguments = arguments.strip()
    if identifier == "compact":
        data = await call("compact", {"customInstructions": arguments} if arguments else {})
        return {"kind": "text", "text": data.get("summary", "Context compacted")}
    if identifier == "name":
        if arguments:
            await call("set_session_name", {"name": arguments})
            return {"kind": "notice", "message": "Session name updated"}
        data = await call("get_state", {})
        return {"kind": "text", "text": data.get("sessionName") or "Unnamed session"}
    if identifier == "export":
        data = await call("export_html", {"outputPath": arguments} if arguments else {})
        return {"kind": "text", "text": data.get("path") or "Session exported"}
    if identifier == "copy":
        data = await call("get_last_assistant_text", {})
        return {"kind": "text", "text": data.get("text") or "No assistant message"}
    if identifier == "model" and arguments:
        provider, separator, model = arguments.partition("/")
        if not separator or not provider or not model:
            raise ControlError("Choose a model as provider/model")
        await call("set_model", {"provider": provider, "modelId": model})
        return {"kind": "notice", "message": "Model updated"}
    if identifier == "thinking" and arguments:
        data = await call("get_available_thinking_levels", {})
        if arguments not in data.get("levels", []):
            raise ControlError("This thinking level is unavailable for the selected Pi model")
        await call("set_thinking_level", {"level": arguments})
        return {"kind": "notice", "message": "Thinking level updated"}
    if identifier in {"new", "clone", "resume"} or (identifier == "fork" and arguments):
        if identifier == "resume" and not arguments:
            raise ControlError("Choose a saved Pi session file to resume")
        command, params = {
            "new": ("new_session", {}),
            "clone": ("clone", {}),
            "resume": ("switch_session", {"sessionPath": arguments}),
            "fork": ("fork", {"entryId": arguments}),
        }[identifier]
        data = await call(command, params)
        if data.get("cancelled"):
            raise ControlError("A Pi extension cancelled the session change")
        return {"kind": "notice", "message": "Session changed"}
    query = {
        "model": "get_available_models",
        "thinking": "get_available_thinking_levels",
        "session": "get_session_stats",
        "tree": "get_tree",
        "fork": "get_fork_messages",
    }.get(identifier)
    if query is None:
        raise ControlError("This Pi command requires its terminal interface")
    return {"kind": "text", "text": json.dumps(await call(query, {}), indent=2, ensure_ascii=False)}
