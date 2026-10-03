from types import SimpleNamespace
from typing import Any, cast

from litellm import ModelResponse
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.types.llms.openai import ResponsesAPIResponse

_BRIDGE = cast(Any, LiteLLMResponsesTransformationHandler)()


def chat_request(body: dict[str, Any]) -> dict[str, Any]:
    logging: Any = SimpleNamespace()
    result = _BRIDGE.transform_request(
        model=body["model"],
        messages=body.get("messages", []),
        optional_params={
            key: value for key, value in body.items() if key not in {"model", "messages"}
        },
        litellm_params={},
        headers={},
        litellm_logging_obj=logging,
    )
    request = {
        key: value for key, value in result.items() if key not in {"client", "litellm_logging_obj"}
    }
    phases = iter(
        message.get("phase")
        for message in body.get("messages", [])
        if message.get("role") == "assistant" and message.get("content")
    )
    for item in request.get("input", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            phase = next(phases, None)
            if phase in {"commentary", "final_answer"}:
                item["phase"] = phase
    return request


def chat_response(body: dict[str, Any]) -> dict[str, Any]:
    logging: Any = SimpleNamespace()
    response = _BRIDGE.transform_response(
        model=body["model"],
        raw_response=ResponsesAPIResponse.model_validate(body),
        model_response=ModelResponse(id=body["id"], created=body.get("created_at", 0)),
        logging_obj=logging,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    result = dict(response.model_dump(mode="json", exclude_none=True))
    choices = result.get("choices", [])
    if len(choices) > 1:
        result["choices"] = [_merge_choices(choices)]
    messages = [item for item in body.get("output", []) if item.get("type") == "message"]
    if messages and messages[-1].get("phase") in {"commentary", "final_answer"}:
        for choice in result.get("choices", []):
            choice["message"]["phase"] = messages[-1]["phase"]
    return result


def _merge_choices(choices: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    for choice in choices:
        for key, value in choice["message"].items():
            if key in {"content", "reasoning_content", "refusal"} and isinstance(value, str):
                message[key] = "\n\n".join(part for part in (message.get(key), value) if part)
            elif key in {"tool_calls", "annotations", "reasoning_items"} and isinstance(
                value, list
            ):
                message.setdefault(key, []).extend(value)
            else:
                message[key] = value
    finish = choices[-1]["finish_reason"]
    if message.get("tool_calls") and finish not in {"length", "content_filter"}:
        finish = "tool_calls"
    return {**choices[-1], "index": 0, "message": message, "finish_reason": finish}
