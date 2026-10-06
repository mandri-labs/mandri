from copy import deepcopy
from typing import Any
from urllib.parse import quote

from mandri.core.opencode_messages import message_error, message_events, message_record, part_id
from mandri.runtime.control.agents.base import HttpCall
from mandri.runtime.control.errors import ControlTransportError

_CATALOG_EVENTS = frozenset(
    {
        "project.updated",
        "models-dev.refreshed",
        "provider.updated",
        "model.updated",
        "agent.updated",
        "command.updated",
        "skill.updated",
        "websearch.updated",
        "plugin.updated",
    }
)


def form_event(form: dict[str, Any]) -> dict[str, Any]:
    questions = []
    for field in form["fields"]:
        options = field.get("options", [])
        if field["type"] == "boolean":
            options = [{"label": "true", "value": "true"}, {"label": "false", "value": "false"}]
        questions.append(
            {
                "question": field.get("title") or field["key"],
                "header": form["title"],
                "options": [
                    {"label": item["label"], "description": item.get("description", "")}
                    for item in options
                ],
                "multiple": field["type"] == "multiselect",
                "custom": field.get("custom", not options),
            }
        )
    return {"type": "question.asked", "properties": {**form, "form": form, "questions": questions}}


def permission_event(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "permission.asked",
        "properties": {
            **record,
            "tool": record.get("action", "permission"),
            "permission": record.get("action"),
            "patterns": record.get("resources", []),
        },
    }


class OpencodeEvents:
    def __init__(self, call: HttpCall) -> None:
        self._call = call
        self._messages: dict[str, dict[str, Any]] = {}
        self._parts: dict[str, dict[str, Any]] = {}

    async def convert(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        data = event.get("data")
        if not isinstance(data, dict):
            return [event]
        kind = str(event.get("type", ""))
        owner = data.get("sessionID")
        created = event.get("created", 0)
        if kind in _CATALOG_EVENTS:
            return [{"type": "catalog.updated", "properties": data}]
        if kind == "form.created":
            return [form_event(data["form"])]
        if kind in {"form.replied", "form.cancelled"}:
            return [
                {
                    "type": "question.replied" if kind == "form.replied" else "question.rejected",
                    "properties": {**data, "requestID": data["id"]},
                }
            ]
        if kind == "permission.asked":
            return [permission_event(data)]
        if kind == "session.created":
            return [
                {
                    "type": kind,
                    "properties": {
                        "info": {
                            **data,
                            "id": owner,
                            "time": {"created": created, "updated": created},
                        }
                    },
                }
            ]
        if kind.startswith("session.execution."):
            running = kind.endswith("started")
            if not running:
                self._messages = {
                    key: value
                    for key, value in self._messages.items()
                    if value["sessionID"] != owner
                }
                self._parts = {
                    key: value for key, value in self._parts.items() if value["sessionID"] != owner
                }
            result = [
                {
                    "type": "session.status",
                    "properties": {
                        "sessionID": owner,
                        "status": {"type": "busy" if running else "idle"},
                        "outcome": kind.rsplit(".", 1)[-1],
                    },
                }
            ]
            if kind.endswith("failed"):
                result.append({"type": "session.error", "properties": data})
            return result
        if kind == "session.retry.scheduled":
            return [{"type": "session.status", "properties": {**data, "status": {"type": "retry"}}}]
        if kind in {"session.renamed", "session.updated"}:
            return [{"type": "session.updated", "properties": {"info": {**data, "id": owner}}}]
        if kind == "session.inbox.delivered":
            response = await self._call(
                "GET",
                f"/api/session/{quote(str(owner), safe='')}/message/"
                f"{quote(data['inboxID'], safe='')}",
                None,
            )
            if response.status_code == 404:
                return []
            if response.status_code != 200:
                raise ControlTransportError("OpenCode message recovery failed")
            return message_events(response.json()["data"], str(owner))
        message_id = data.get("assistantMessageID")
        if not isinstance(message_id, str):
            return [] if kind.startswith("session.") else [{"type": kind, "properties": data}]
        if kind.startswith("session.step."):
            info = self._messages.setdefault(
                message_id,
                {
                    "id": message_id,
                    "role": "assistant",
                    "sessionID": owner,
                    "time": {"created": data.get("started", created)},
                },
            )
            if kind.endswith("started"):
                info.update(
                    agent=data["agent"],
                    providerID=data["model"]["providerID"],
                    modelID=data["model"]["id"],
                )
            else:
                info.update(
                    {key: data[key] for key in ("finish", "tokens", "cost", "error") if key in data}
                )
                if isinstance(info.get("error"), dict):
                    info["error"] = message_error(info["error"])
                if kind.endswith(("ended", "failed")):
                    info["time"]["completed"] = created
                    self._messages.pop(message_id, None)
            return [{"type": "message.updated", "properties": {"info": deepcopy(info)}}]
        if kind.startswith(("session.text.", "session.reasoning.")):
            item_kind = kind.split(".")[1]
            identifier = part_id(message_id, item_kind, data["ordinal"])
            part = self._parts.setdefault(
                identifier,
                {
                    "id": identifier,
                    "messageID": message_id,
                    "sessionID": owner,
                    "type": item_kind,
                    "text": "",
                    "time": {"start": created},
                },
            )
            if kind.endswith("delta"):
                part["text"] += data["delta"]
            if kind.endswith("ended"):
                part.update(text=data["text"], time={**part["time"], "end": created})
                self._parts.pop(identifier, None)
            return [{"type": "message.part.updated", "properties": {"part": deepcopy(part)}}]
        if kind.startswith("session.tool."):
            return self._tool(kind, data, message_id, str(owner), created)
        return []

    def _tool(
        self, kind: str, data: dict[str, Any], message_id: str, owner: str, created: Any
    ) -> list[dict[str, Any]]:
        identifier = part_id(message_id, "tool", data["id"])
        part = self._parts.setdefault(
            identifier,
            {
                "id": identifier,
                "messageID": message_id,
                "sessionID": owner,
                "type": "tool",
                "tool": data.get("name", "tool"),
                "callID": data["id"],
                "state": {"status": "pending", "input": {}, "raw": ""},
            },
        )
        state = dict(part["state"])
        if kind.endswith("input.delta"):
            state["raw"] += data["delta"]
        if kind.endswith("input.ended"):
            state["raw"] = data["text"]
        if kind.endswith("called"):
            state.update(status="running", input=data["input"], time={"start": created})
        if kind.endswith("progress"):
            state["metadata"] = data["metadata"]
        if kind.endswith(("success", "failed")):
            item = {
                "id": data["id"],
                "type": "tool",
                "name": part["tool"],
                "time": {
                    "created": state.get("time", {}).get("start", created),
                    "completed": created,
                },
                "state": {
                    **state,
                    "status": "error" if kind.endswith("failed") else "completed",
                    **{key: data[key] for key in ("content", "error", "metadata") if key in data},
                },
            }
            part = message_record(
                {"id": message_id, "type": "assistant", "content": [item]}, owner
            )["parts"][0]
            self._parts.pop(identifier, None)
        else:
            part["state"] = state
        return [{"type": "message.part.updated", "properties": {"part": deepcopy(part)}}]
