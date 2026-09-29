import json
from typing import Any

import httpx
from mandri.core.ids import ProviderKind
from mandri.gateway.chatgpt_conversion import chat_request, chat_response
from mandri.gateway.errors.upstream import UpstreamError

_ALLOWED = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "include",
        "text",
        "service_tier",
    }
)


class ChatGptAdapter:
    def __init__(self, base: str) -> None:
        self.base = httpx.URL(base.rstrip("/"))

    def request(self, request: httpx.Request) -> httpx.Request:
        path = request.url.path
        if (
            request.method != "POST"
            or request.url.copy_with(path=self.base.path) != self.base
            or path not in {self.base.path + "/responses", self.base.path + "/chat/completions"}
        ):
            raise UpstreamError(400, "Unsupported ChatGPT endpoint", ProviderKind.CHATGPT)
        body = json.loads(request.content)
        if path.endswith("/chat/completions"):
            body = chat_request(body)
        payload = {key: value for key, value in body.items() if key in _ALLOWED}
        if isinstance(payload.get("input"), str):
            payload["input"] = [{"role": "user", "content": payload["input"]}]
        payload.setdefault("instructions", "")
        payload.update(store=False, stream=True)
        headers = dict(request.headers)
        headers.pop("content-length", None)
        headers["accept"] = "text/event-stream"
        return httpx.Request(
            "POST",
            self.base.copy_with(path=self.base.path + "/responses"),
            json=payload,
            headers=headers,
            extensions=request.extensions,
        )

    async def response(self, response: httpx.Response, request: httpx.Request) -> httpx.Response:
        if response.is_error:
            return response
        try:
            completed = await complete_response(response)
            if request.url.path.endswith("/chat/completions"):
                completed = chat_response(completed)
            headers = {
                name: value
                for name, value in response.headers.items()
                if name
                not in {"content-length", "content-type", "content-encoding", "transfer-encoding"}
            }
            return httpx.Response(200, json=completed, headers=headers, request=request)
        finally:
            await response.aclose()


async def complete_response(response: httpx.Response) -> dict[str, Any]:
    data: list[str] = []
    items: dict[int, dict[str, Any]] = {}
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            completed = _event("\n".join(data), items)
            data.clear()
            if completed is not None:
                return completed
    if data:
        completed = _event("\n".join(data), items)
        if completed is not None:
            return completed
    raise UpstreamError(
        502, "ChatGPT stream ended without a complete response", ProviderKind.CHATGPT
    )


def _event(data: str, items: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if data == "[DONE]":
        return None
    try:
        event = json.loads(data)
    except ValueError:
        raise UpstreamError(502, "Invalid ChatGPT response event", ProviderKind.CHATGPT) from None
    if not isinstance(event, dict):
        raise UpstreamError(502, "Invalid ChatGPT response event", ProviderKind.CHATGPT)
    kind = event.get("type")
    if kind == "response.output_item.done":
        index = event.get("output_index")
        item = event.get("item")
        if type(index) is not int or index < 0 or not isinstance(item, dict):
            raise UpstreamError(502, "Invalid ChatGPT output item", ProviderKind.CHATGPT)
        items[index] = item
        return None
    if kind in {"error", "response.failed"}:
        raise UpstreamError(502, "ChatGPT could not complete the response", ProviderKind.CHATGPT)
    if kind not in {"response.completed", "response.incomplete"}:
        return None
    result = event.get("response")
    if not isinstance(result, dict) or not isinstance(result.get("output"), list):
        raise UpstreamError(502, "Incomplete ChatGPT response object", ProviderKind.CHATGPT)
    if not result["output"] and items:
        if sorted(items) != list(range(len(items))):
            raise UpstreamError(502, "Missing ChatGPT output item", ProviderKind.CHATGPT)
        result = {**result, "output": [items[index] for index in sorted(items)]}
    return result
