import json
from typing import Any

import httpx
from mandri.core.ids import ProviderKind
from mandri.gateway.chatgpt_continuation import combine_responses, commentary_response
from mandri.gateway.chatgpt_conversion import chat_request, chat_response
from mandri.gateway.errors.upstream import UpstreamError
from mandri.gateway.provider_adapter import ProviderSend
from mandri.gateway.usage import CURRENT_USAGE
from mandri.gateway.usage_payload import observe_payload

_MAX_COMMENTARY_CONTINUATIONS = 8

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

    async def response(
        self, response: httpx.Response, request: httpx.Request, send: ProviderSend
    ) -> httpx.Response:
        if response.is_error:
            return response
        upstream = response.request
        payload = json.loads(upstream.content)
        responses: list[dict[str, Any]] = []
        try:
            while True:
                completed = await complete_response(response)
                responses.append(completed)
                if not commentary_response(completed):
                    break
                if len(responses) > _MAX_COMMENTARY_CONTINUATIONS:
                    raise UpstreamError(
                        502,
                        "ChatGPT returned commentary without completing the turn",
                        ProviderKind.CHATGPT,
                    )
                payload["input"] = [*payload.get("input", []), *completed["output"]]
                await response.aclose()
                headers = dict(upstream.headers)
                headers.pop("content-length", None)
                upstream = httpx.Request(
                    "POST",
                    upstream.url,
                    json=payload,
                    headers=headers,
                    extensions=upstream.extensions,
                )
                response = await send(upstream)
                if response.is_error:
                    return response
            completed = combine_responses(responses)
            collector = CURRENT_USAGE.get()
            if collector is not None and len(responses) > 1:
                if completed.get("usage") is None:
                    collector.observation_incomplete = True
                await observe_payload(collector, completed)
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
            if not response.is_error:
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
    if items:
        merged = dict(items)
        indices = {item.get("id"): index for index, item in items.items()}
        for index, item in enumerate(result["output"]):
            if not isinstance(item, dict):
                raise UpstreamError(502, "Invalid ChatGPT output item", ProviderKind.CHATGPT)
            slot = indices.get(item.get("id"), index)
            if slot in merged and merged[slot].get("id") != item.get("id"):
                slot = max(merged) + 1
            merged[slot] = item
        if sorted(merged) != list(range(len(merged))):
            raise UpstreamError(502, "Missing ChatGPT output item", ProviderKind.CHATGPT)
        result = {**result, "output": [merged[index] for index in sorted(merged)]}
    return result
