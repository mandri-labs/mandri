import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import litellm
from litellm.google_genai.adapters.transformation import (
    GoogleGenAIAdapter,
    GoogleGenAIStreamWrapper,
)
from litellm.types.utils import LlmProviders, ModelResponse, ModelResponseStream
from litellm.utils import ProviderConfigManager


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def usage_metadata(response: ModelResponse | ModelResponseStream) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result = {}
    for source, target in (
        ("prompt_tokens", "promptTokenCount"),
        ("completion_tokens", "candidatesTokenCount"),
        ("total_tokens", "totalTokenCount"),
    ):
        value = _count(getattr(usage, source, None))
        if value is not None:
            result[target] = value
    details = getattr(usage, "completion_tokens_details", None)
    thoughts = _count(getattr(details, "reasoning_tokens", None))
    if thoughts is not None:
        result["thoughtsTokenCount"] = thoughts
        output = result.get("candidatesTokenCount")
        if output is not None and thoughts <= output:
            result["candidatesTokenCount"] = output - thoughts
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _count(getattr(details, "cached_tokens", None))
    if cached is not None:
        result["cachedContentTokenCount"] = cached
    return result


def _with_usage(payload: Mapping[str, object], response: Any) -> dict[str, Any]:
    payload = dict(payload)
    payload.pop("usageMetadata", None)
    usage = usage_metadata(response)
    if usage:
        payload["usageMetadata"] = usage
    return payload


async def _stream(response: AsyncIterator[ModelResponseStream]) -> AsyncIterator[bytes]:
    adapter = GoogleGenAIAdapter()
    state = GoogleGenAIStreamWrapper(response)
    try:
        async for chunk in response:
            payload = adapter.translate_streaming_completion_to_generate_content(chunk, state)
            payload = _with_usage(payload or {}, chunk)
            if payload:
                yield ("data: " + json.dumps(payload) + "\n\n").encode()
        async for payload in state:
            yield ("data: " + json.dumps(payload) + "\n\n").encode()
    finally:
        close = getattr(response, "aclose", None)
        if close is not None:
            await close()


async def generate_content(*, model: str, stream: bool, **kwargs: Any) -> Any:
    _, provider, _, _ = litellm.get_llm_provider(model=model, api_base=kwargs.get("api_base"))
    native = ProviderConfigManager.get_provider_google_genai_generate_content_config(
        model=model, provider=LlmProviders(provider)
    )
    if native is not None:
        call = (
            litellm.google_genai.agenerate_content_stream
            if stream
            else litellm.google_genai.agenerate_content
        )
        return await call(model=model, **kwargs)
    adapter = GoogleGenAIAdapter()
    request = adapter.translate_generate_content_to_completion(
        model=model,
        contents=kwargs.pop("contents"),
        config=kwargs.pop("config", None),
        systemInstruction=kwargs.pop("systemInstruction", None),
        tools=kwargs.pop("tools", None),
    )
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    response = await litellm.acompletion(**request, **kwargs, stream=stream)
    if stream and isinstance(response, AsyncIterator):
        return _stream(response)
    return _with_usage(adapter.translate_completion_to_generate_content(response), response)
