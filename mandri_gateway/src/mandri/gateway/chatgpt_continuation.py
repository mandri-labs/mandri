from typing import Any

from mandri.gateway.usage_payload import sum_usage


def commentary_response(response: dict[str, Any]) -> bool:
    if response.get("status") != "completed" or response.get("error"):
        return False
    output = response.get("output", [])
    messages = [item for item in output if item.get("type") == "message"]
    return bool(messages) and all(
        item.get("type") == "reasoning"
        or (item.get("type") == "message" and item.get("phase") == "commentary")
        for item in output
    )


def combine_responses(responses: list[dict[str, Any]]) -> dict[str, Any]:
    if len(responses) == 1:
        return responses[0]
    result = {
        **responses[-1],
        "output": [item for response in responses for item in response["output"]],
    }
    usage: dict[str, Any] = {}
    for response in responses:
        value = response.get("usage")
        if not isinstance(value, dict):
            result["usage"] = None
            return result
        usage = sum_usage(usage, value)
    result["usage"] = usage
    return result
