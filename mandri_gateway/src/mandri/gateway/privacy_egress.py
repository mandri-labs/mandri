import hmac
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
from mandri.core.ids import ProviderKind
from mandri.core.types.execution import ProtectionError
from mandri.gateway.privacy_known import KnownValues
from mandri.gateway.privacy_protocol import GatewayProtocol, validate_request, visit_content
from mandri.gateway.provider_identity import identity_headers
from mandri.gateway.route_registry import ResolvedRoute
from mandri.gateway.surrogate import SurrogateEngine
from mandri.gateway.types.model import MODEL_REF_PREFIXES

_BASES = {
    ProviderKind.OPENAI: "https://api.openai.com/v1",
    ProviderKind.CHATGPT: "https://chatgpt.com/backend-api/codex",
    ProviderKind.ANTHROPIC: "https://api.anthropic.com",
    ProviderKind.GEMINI: "https://generativelanguage.googleapis.com",
    ProviderKind.OPENROUTER: "https://openrouter.ai/api/v1",
    ProviderKind.OPENCODE: "https://opencode.ai/zen/v1",
    ProviderKind.OPENCODE_GO: "https://opencode.ai/zen/go/v1",
}
_AUTH_HEADERS = frozenset({"authorization", "x-api-key", "x-goog-api-key", "api-key"})


def provider_base(route: ResolvedRoute) -> str:
    base = str(route.model.api_base or _BASES.get(route.model.provider, ""))
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ProtectionError("privacy_transport_unsupported", "Provider endpoint is unsupported")
    if parsed.query or parsed.fragment:
        raise ProtectionError("privacy_transport_unsupported", "Provider endpoint is unsupported")
    return base.rstrip("/")


def provider_model(route: ResolvedRoute) -> str:
    return str(route.model.model_ref).removeprefix(MODEL_REF_PREFIXES[route.model.provider])


def provider_endpoints(route: ResolvedRoute) -> dict[str, GatewayProtocol]:
    base = httpx.URL(provider_base(route)).path.rstrip("/")
    kind = route.model.provider
    if kind is ProviderKind.GEMINI:
        model = provider_model(route)
        prefixes = (base,) if route.model.api_base else (base + "/v1beta", base + "/v1alpha")
        return {
            f"{prefix}/models/{model}:{operation}": GatewayProtocol.GEMINI
            for prefix in prefixes
            for operation in ("generateContent", "streamGenerateContent", "countTokens")
        }
    if kind is ProviderKind.ANTHROPIC:
        path = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        return {
            path: GatewayProtocol.ANTHROPIC,
            base + "/v1/messages": GatewayProtocol.ANTHROPIC,
            path + "/count_tokens": GatewayProtocol.COUNT_TOKENS,
        }
    if kind is ProviderKind.OLLAMA:
        path = base if base.endswith("/api/chat") else base + "/api/chat"
        return {path: GatewayProtocol.CHAT}
    return {
        base + "/chat/completions": GatewayProtocol.CHAT,
        base + "/responses": GatewayProtocol.RESPONSES,
    }


def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate provider JSON key")
        result[key] = value
    return result


@dataclass
class EgressGuard:
    route: ResolvedRoute
    engine: SurrogateEngine
    headers: dict[str, str] = field(default_factory=dict)
    sends: int = 0

    async def check(self, request: httpx.Request, *, complete_response: bool = False) -> None:
        self._check(request, complete_response=complete_response)
        self.sends += 1

    def _check(self, request: httpx.Request, *, complete_response: bool = False) -> None:
        expected = httpx.URL(provider_base(self.route))
        protocol = provider_endpoints(self.route).get(request.url.path)
        if (
            request.method != "POST"
            or request.url.scheme != expected.scheme
            or request.url.host != expected.host
            or (request.url.port != expected.port)
            or request.url.userinfo
            or request.url.fragment
            or (protocol is None)
        ):
            raise ProtectionError("privacy_egress_blocked", "Unexpected provider destination")
        query_keys = set()
        for key, value in parse_qsl(request.url.query.decode("ascii"), keep_blank_values=True):
            if key in query_keys:
                raise ProtectionError("privacy_egress_blocked", "Duplicate provider query control")
            query_keys.add(key)
            if key == "key" and self.route.model.provider is ProviderKind.GEMINI:
                self._credential(value)
            elif self.route.model.provider is not ProviderKind.GEMINI or (key, value) not in {
                ("alt", "sse"),
                ("$alt", "sse"),
            }:
                raise ProtectionError("privacy_egress_blocked", "Unexpected provider query control")
        known = KnownValues(self.engine)
        identity = {
            name.lower(): value for name, value in identity_headers(self.route.model).items()
        }
        for name, value in request.headers.multi_items():
            if name in _AUTH_HEADERS:
                credential = value.removeprefix("Bearer ") if name == "authorization" else value
                self._credential(credential)
            elif name in identity:
                if not hmac.compare_digest(value, identity[name]):
                    raise ProtectionError("privacy_egress_blocked", "Unexpected provider identity")
            elif known.text(value) != value:
                raise ProtectionError("privacy_egress_blocked", "Unmasked provider header")
        try:
            raw = request.content
        except httpx.RequestNotRead:
            raise ProtectionError(
                "privacy_egress_blocked", "Streaming request bodies are unsupported"
            ) from None
        try:
            payload: Any = json.loads(raw, object_pairs_hook=unique_object)
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise ProtectionError(
                "privacy_egress_blocked", "Provider request must be JSON"
            ) from None
        if not isinstance(payload, dict):
            raise ProtectionError("privacy_egress_blocked", "Invalid provider request")
        if visit_content(payload, known.replace) != payload:
            raise ProtectionError("privacy_egress_blocked", "Unmasked provider content")
        if not complete_response and (
            payload.get("stream") is True or request.url.path.endswith(":streamGenerateContent")
        ):
            raise ProtectionError(
                "privacy_egress_blocked", "Protected provider streaming is disabled"
            )
        if "model" in payload and payload["model"] != provider_model(self.route):
            raise ProtectionError("privacy_egress_blocked", "Unexpected provider model")
        if self.route.model.provider is ProviderKind.OLLAMA:
            common = {
                key: value
                for key, value in payload.items()
                if key not in {"options", "format", "keep_alive", "think"}
            }
            validate_request(common)
        elif self.route.model.provider is ProviderKind.OPENROUTER:
            common = {key: value for key, value in payload.items() if key != "usage"}
            validate_request(common)
        else:
            validate_request(payload)

    def _credential(self, value: str) -> None:
        if not hmac.compare_digest(value, str(self.route.model.api_key)):
            raise ProtectionError("privacy_egress_blocked", "Unexpected provider credentials")
