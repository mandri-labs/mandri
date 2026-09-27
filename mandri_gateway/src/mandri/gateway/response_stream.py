"""Normalizes translated Responses streams without conflating output identities."""

import copy
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


def _has_output(value: dict[str, Any]) -> bool:
    kind = value.get("type")
    if kind not in {"message", "reasoning"}:
        return True
    parts = value.get("content" if kind == "message" else "summary") or []
    return any(
        part.get("text")
        or part.get("refusal")
        or part.get("type") not in {"output_text", "summary_text", "reasoning_text", "refusal"}
        for part in parts
    ) or bool(value.get("encrypted_content"))


@dataclass
class _Item:
    value: dict[str, Any]
    requested_index: int
    index: int | None = None
    added: dict[str, Any] | None = None
    parts: dict[int, dict[str, Any]] = field(default_factory=dict)
    text: dict[int, str] = field(default_factory=dict)
    text_done: set[int] = field(default_factory=set)
    part_done: set[int] = field(default_factory=set)
    done: bool = False

    @property
    def reasoning(self) -> bool:
        return bool(self.value["type"] == "reasoning")

    @property
    def part_key(self) -> str:
        return "summary_index" if self.reasoning else "content_index"

    @property
    def part_kind(self) -> str:
        return "reasoning_summary_part" if self.reasoning else "content_part"

    @property
    def text_kind(self) -> str:
        return "reasoning_summary_text" if self.reasoning else "output_text"

    def part(self, index: int, text: str = "") -> dict[str, Any]:
        if self.reasoning:
            return {"type": "summary_text", "text": text}
        return {"type": "output_text", "text": text, "annotations": []}

    def event(self, kind: str, index: int, **values: Any) -> dict[str, Any]:
        return {
            "type": f"response.{kind}",
            "item_id": self.value["id"],
            "output_index": self.index,
            self.part_key: index,
            **values,
        }

    def completed(self) -> dict[str, Any]:
        key = "summary" if self.reasoning else "content"
        return {
            **self.value,
            "status": "completed",
            key: [self.part(i, text) for i, text in sorted(self.text.items())],
        }


class _Normalizer:
    def __init__(self) -> None:
        self.items: dict[str, _Item] = {}
        self.indices: dict[int, str] = {}

    def item(self, payload: dict[str, Any], kind: str) -> _Item:
        value = payload.get("item") or {}
        identity = value.get("id") or payload.get("item_id")
        index = int(payload.get("output_index") or 0)
        if not identity:
            identity = next(
                (
                    key
                    for key, item in self.items.items()
                    if item.requested_index == index
                    and item.value["type"] == kind
                    and not item.done
                ),
                None,
            )
        identity = str(identity or uuid.uuid4())
        if identity not in self.items:
            initial: dict[str, Any] = {"id": identity, "type": kind, "status": "in_progress"}
            initial.update(
                {"summary": []} if kind == "reasoning" else {"role": "assistant", "content": []}
            )
            self.items[identity] = _Item(value or initial, index)
        return self.items[identity]

    def open(self, item: _Item) -> list[dict[str, Any]]:
        if item.index is not None:
            return []
        index = item.requested_index
        if index in self.indices:
            index = max(self.indices) + 1
        item.index = index
        self.indices[index] = item.value["id"]
        payload = item.added or {"type": "response.output_item.added", "item": item.value}
        result = [{**payload, "output_index": index}]
        result.extend({**part, "output_index": index} for part in item.parts.values())
        return result

    def close(self, item: _Item, final: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if item.done:
            return []
        result = self.open(item)
        for index, text in item.text.items():
            if index not in item.text_done:
                result.append(item.event(f"{item.text_kind}.done", index, text=text))
            if index not in item.part_done:
                result.append(
                    item.event(f"{item.part_kind}.done", index, part=item.part(index, text))
                )
        result.append(
            {
                "type": "response.output_item.done",
                "output_index": item.index,
                "item": final if final is not None else item.completed(),
            }
        )
        item.done = True
        return result

    def process(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        kind = payload.get("type", "")
        if kind == "response.completed":
            response = payload.get("response") or {}
            if response.get("error") or response.get("status") in {"failed", "incomplete"}:
                status = "failed" if response.get("error") else response["status"]
                return [
                    {
                        **payload,
                        "type": "response." + status,
                        "response": {**response, "status": status},
                    }
                ]
            if not any(_has_output(value) for value in response.get("output") or []) and not any(
                any(item.text.values()) or _has_output(item.value) for item in self.items.values()
            ):
                logger.warning("Provider completed a Responses stream without any output")
                return [
                    {
                        **payload,
                        "type": "response.failed",
                        "response": {
                            **response,
                            "status": "failed",
                            "error": {
                                "code": "server_error",
                                "message": "Provider completed without any output",
                            },
                        },
                    }
                ]
            return self.complete(payload)
        if kind in ("response.failed", "response.incomplete"):
            result = []
            for item in self.items.values():
                result.extend(self.open(item))
            self.order_output(payload)
            return [*result, payload]
        if kind.startswith("response.output_item."):
            value = payload.get("item") or {}
            if value.get("type") == "reasoning" and value.get("summary") is None:
                value["summary"] = []
            item = self.item(payload, value.get("type", "message"))
            if kind.endswith(".added"):
                item.added = payload
                if item.value["type"] == "message" and not item.value.get("content"):
                    return []
                return self.open(item)
            if item.done:
                return []
            result = self.close(item, value)
            result[-1] = {**payload, "output_index": item.index}
            return result
        reasoning = kind.startswith("response.reasoning_summary_")
        text = kind.startswith(("response.output_text.", "response.content_part."))
        if not reasoning and not text:
            identity = payload.get("item_id")
            if identity in self.items:
                item = self.items[identity]
                result = self.open(item)
                payload["output_index"] = item.index
                return [*result, payload]
            return [payload]
        item = self.item(payload, "reasoning" if reasoning else "message")
        index = int(payload.get(item.part_key) or 0)
        payload["item_id"] = item.value["id"]
        payload[item.part_key] = index
        if kind.endswith("part.added"):
            if index in item.parts:
                return []
            item.parts[index] = payload
            if item.index is None:
                return []
            payload["output_index"] = item.index
            return [payload]
        result = []
        if not reasoning:
            for other in self.items.values():
                if other.reasoning and other.index is not None and not other.done:
                    result.extend(self.close(other))
        result.extend(self.open(item))
        payload["output_index"] = item.index
        if index not in item.parts:
            part = item.event(f"{item.part_kind}.added", index, part=item.part(index))
            item.parts[index] = part
            result.append(part)
        if kind.endswith(".delta"):
            item.text[index] = item.text.get(index, "") + str(payload.get("delta", ""))
        elif kind.endswith("text.done"):
            item.text[index] = str(payload.get("text", ""))
            if index in item.text_done:
                return result
            item.text_done.add(index)
        elif kind.endswith("part.done"):
            if index in item.part_done:
                return result
            item.part_done.add(index)
            part = payload.get("part") or {}
            if not reasoning and part.get("type") == "reasoning_text":
                payload["part"] = item.part(index, item.text.get(index, ""))
        result.append(payload)
        return result

    def complete(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        response = payload.get("response") or {}
        output = response.get("output") or []
        result = []
        for index, value in enumerate(output):
            if not isinstance(value, dict) or not value.get("id"):
                continue
            item = self.item({"item": value, "output_index": index}, value.get("type", "message"))
            result.extend(self.close(item, value))
        for item in self.items.values():
            if item.index is not None and not item.done:
                result.extend(self.close(item))
        self.order_output(payload)
        result.append(payload)
        return result

    def order_output(self, payload: dict[str, Any]) -> None:
        response = payload.get("response") or {}
        output = response.get("output") or []
        if output and all(
            isinstance(value, dict) and value.get("id") in self.items for value in output
        ):
            response["output"] = sorted(
                output, key=lambda value: self.items[value["id"]].index or 0
            )


async def normalize_stream(stream: AsyncIterator[Any]) -> AsyncIterator[dict[str, Any]]:
    normalizer = _Normalizer()
    next_sequence = 0
    async for event in stream:
        dump = getattr(event, "model_dump", None)
        payload = dump() if callable(dump) else event
        if not isinstance(payload, dict):
            payload = {"type": str(event)}
        for normalized in normalizer.process(copy.deepcopy(payload)):
            sequence = normalized.get("sequence_number")
            if not isinstance(sequence, int) or sequence < next_sequence:
                sequence = next_sequence
            normalized["sequence_number"] = sequence
            next_sequence = sequence + 1
            yield normalized


def sse_frame(payload: dict[str, Any]) -> bytes:
    name = payload.get("type")
    if isinstance(name, Enum):
        name = name.value
    if not isinstance(name, str) or not name:
        name = "response.event"
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()
