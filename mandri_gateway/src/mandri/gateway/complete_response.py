import json
from collections.abc import Iterator
from typing import Any

from mandri.gateway.privacy_protocol import GatewayProtocol
from mandri.gateway.response_stream import sse_frame


def _responses(value: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if value.get("error"):
        value = {**value, "status": "failed"}
    elif not value.get("output") and value.get("status") not in {"failed", "incomplete"}:
        value = {
            **value,
            "status": "failed",
            "error": {"code": "server_error", "message": "Provider completed without any output"},
        }
    initial = {**value, "status": "in_progress", "output": [], "usage": None}
    yield {"type": "response.created", "response": initial}
    yield {"type": "response.in_progress", "response": initial}
    for index, item in enumerate(value.get("output", [])):
        identity = {"item_id": item.get("id"), "output_index": index}
        initial_item = {**item, "status": "in_progress"}
        for field in ("content", "summary"):
            if field in initial_item:
                initial_item[field] = []
        if item.get("type") == "custom_tool_call":
            initial_item["input"] = ""
        if "arguments" in initial_item:
            initial_item["arguments"] = ""
        yield {"type": "response.output_item.added", "output_index": index, "item": initial_item}
        for field in ("content", "summary"):
            for part_index, part in enumerate(item.get(field) or []):
                summary = field == "summary"
                location = {**identity, "summary_index" if summary else "content_index": part_index}
                part_kind = "reasoning_summary_part" if summary else "content_part"
                text_kind = (
                    "reasoning_summary_text"
                    if summary
                    else (
                        "reasoning_text" if part.get("type") == "reasoning_text" else "output_text"
                    )
                )
                text_field = "refusal" if part.get("type") == "refusal" else "text"
                if text_field == "refusal":
                    text_kind = "refusal"
                yield {
                    "type": f"response.{part_kind}.added",
                    **location,
                    "part": {**part, text_field: ""},
                }
                yield {
                    "type": f"response.{text_kind}.delta",
                    **location,
                    "delta": part.get(text_field, ""),
                }
                yield {
                    "type": f"response.{text_kind}.done",
                    **location,
                    text_field: part.get(text_field, ""),
                }
                yield {"type": f"response.{part_kind}.done", **location, "part": part}
        if item.get("type") == "custom_tool_call":
            yield {
                "type": "response.custom_tool_call_input.delta",
                **identity,
                "delta": item["input"],
            }
            yield {
                "type": "response.custom_tool_call_input.done",
                **identity,
                "input": item["input"],
            }
        if "arguments" in item:
            yield {
                "type": "response.function_call_arguments.delta",
                **identity,
                "delta": item["arguments"],
            }
            yield {
                "type": "response.function_call_arguments.done",
                **identity,
                "arguments": item["arguments"],
            }
        yield {"type": "response.output_item.done", "output_index": index, "item": item}
    status = value.get("status", "completed")
    terminal = status if status in {"failed", "incomplete"} else "completed"
    yield {"type": f"response.{terminal}", "response": value}


def _anthropic(value: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {**value, "content": [], "stop_reason": None, "stop_sequence": None},
    }
    for index, block in enumerate(value.get("content", [])):
        kind = block.get("type")
        field, delta_kind = {
            "text": ("text", "text_delta"),
            "thinking": ("thinking", "thinking_delta"),
            "tool_use": ("input", "input_json_delta"),
        }.get(kind, (None, None))
        initial = dict(block)
        if field is not None:
            initial[field] = {} if field == "input" else ""
        yield {"type": "content_block_start", "index": index, "content_block": initial}
        if field is not None:
            delta = {"type": delta_kind}
            delta["partial_json" if field == "input" else field] = (
                json.dumps(block[field]) if field == "input" else block[field]
            )
            yield {"type": "content_block_delta", "index": index, "delta": delta}
        yield {"type": "content_block_stop", "index": index}
    yield {
        "type": "message_delta",
        "delta": {
            "stop_reason": value.get("stop_reason"),
            "stop_sequence": value.get("stop_sequence"),
        },
        "usage": value.get("usage", {}),
    }
    yield {"type": "message_stop"}


def complete_sse(value: dict[str, Any], protocol: GatewayProtocol) -> bytes:
    if protocol is GatewayProtocol.RESPONSES:
        return b"".join(
            sse_frame({**event, "sequence_number": sequence})
            for sequence, event in enumerate(_responses(value))
        )
    if protocol is GatewayProtocol.ANTHROPIC:
        return b"".join(sse_frame(event) for event in _anthropic(value))
    if protocol is GatewayProtocol.CHAT:
        choices = []
        for choice in value.get("choices", []):
            message = dict(choice.get("message", {}))
            if isinstance(message.get("tool_calls"), list):
                message["tool_calls"] = [
                    {**call, "index": index} for index, call in enumerate(message["tool_calls"])
                ]
            choices.append(
                {
                    **{key: child for key, child in choice.items() if key != "message"},
                    "delta": message,
                }
            )
        payload = {**value, "object": "chat.completion.chunk", "choices": choices}
        return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()
    return f"data: {json.dumps(value)}\n\n".encode()
