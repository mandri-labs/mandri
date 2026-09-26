from typing import Any

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import TokenCountResponse
from mandri.core.ids import ProviderKind
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_egress import EgressGuard, provider_base
from mandri.gateway.privacy_transport import TransportScope, provider_transport


async def protected_count_tokens(guard: EgressGuard, body: dict[str, Any]) -> TokenCountResponse:
    model = str(guard.route.model.model_ref)
    if guard.route.model.provider is not ProviderKind.ANTHROPIC:
        messages = list(body.get("messages") or [])
        if body.get("system"):
            messages.insert(0, {"role": "system", "content": body["system"]})
        return TokenCountResponse(
            total_tokens=litellm.token_counter(
                model=model, messages=messages, tools=body.get("tools")
            ),
            request_model=model,
            model_used=model,
            tokenizer_type="local_tokenizer",
        )
    handler = AsyncHTTPHandler()
    handler.client.follow_redirects = False
    payload = {key: value for key, value in body.items() if key in {"messages", "system", "tools"}}
    payload["model"] = model.removeprefix("anthropic/")
    base = provider_base(guard.route)
    endpoint = base + (
        "/messages/count_tokens" if base.endswith("/v1") else "/v1/messages/count_tokens"
    )
    scope = TransportScope(guard)
    try:
        with provider_transport(scope):
            response = await handler.client.post(
                endpoint,
                json=payload,
                headers={
                    "x-api-key": str(guard.route.model.api_key),
                    "anthropic-version": "2023-06-01",
                },
            )
        response.raise_for_status()
        count = response.json().get("input_tokens")
        if type(count) is not int or count < 0:
            raise ProtectionError(
                "privacy_response_invalid", "Provider returned an invalid token count"
            )
        return TokenCountResponse(
            total_tokens=count,
            request_model=model,
            model_used=model,
            tokenizer_type="anthropic_api",
        )
    finally:
        scope.active = False
        await handler.client.aclose()
