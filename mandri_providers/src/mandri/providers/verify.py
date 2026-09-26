"""Provider key verification via each provider's models-list endpoint."""

import asyncio

import httpx
from mandri.core.ids import ProviderKind, Url
from mandri.core.ports.provider_verifier import (
    ProviderVerifierPort,
    VerificationResult,
)
from mandri.core.provider_headers import conversation_headers
from mandri.providers.refs import requires_api_base

_TIMEOUT_SECONDS = 10
_AUTH_STATUSES = frozenset({401, 403})
_ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_PROVIDER_BASES: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "https://openrouter.ai/api",
    ProviderKind.OPENCODE: "https://opencode.ai/zen/v1",
    ProviderKind.OPENCODE_GO: "https://opencode.ai/zen/go/v1",
    ProviderKind.OPENAI: "https://api.openai.com/v1",
    ProviderKind.ANTHROPIC: "https://api.anthropic.com",
    ProviderKind.GEMINI: "https://generativelanguage.googleapis.com",
}

_MODELS_PATHS: dict[ProviderKind, str] = {
    ProviderKind.OPENROUTER: "/v1/models",
    ProviderKind.OPENCODE: "/models",
    ProviderKind.OPENCODE_GO: "/models",
    ProviderKind.OLLAMA: "/models",
    ProviderKind.LM_STUDIO: "/models",
    ProviderKind.OPENAI: "/models",
    ProviderKind.ANTHROPIC: "/v1/models",
    ProviderKind.GEMINI: "/v1beta/models",
    ProviderKind.CUSTOM: "/models",
}

_BASES_REQUIRED: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OPENCODE, ProviderKind.OPENCODE_GO}
)


def resolve_api_base(kind: ProviderKind, api_base: Url | None) -> Url | None:
    """Fill in the hosted default base when litellm has none for the kind."""
    if api_base is not None or kind not in _BASES_REQUIRED:
        return api_base
    return Url(DEFAULT_PROVIDER_BASES[kind])


def _redact(api_key: str, reason: str) -> str:
    if api_key:
        return reason.replace(api_key, "[redacted]")
    return reason


def models_endpoint(kind: ProviderKind, api_base: Url | None) -> str:
    base = (api_base if api_base is not None else DEFAULT_PROVIDER_BASES[kind]).rstrip("/")
    return base + _MODELS_PATHS[kind]


def models_headers(kind: ProviderKind, api_key: str) -> dict[str, str]:
    if kind is ProviderKind.ANTHROPIC:
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
        return headers
    if api_key:
        return {
            "Authorization": f"Bearer {api_key}",
            **conversation_headers(kind, "provider-metadata"),
        }
    return conversation_headers(kind, "provider-metadata")


def _status_reason(status: int) -> str:
    if status in _AUTH_STATUSES:
        return f"auth failed (status {status})"
    if status == 404:
        return "provider unreachable (status 404)"
    if status == 429:
        return "rate limited (status 429)"
    return f"http {status}"


class ProviderVerifier(ProviderVerifierPort):
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def verify_async(
        self, kind: ProviderKind, api_base: Url | None, api_key: str
    ) -> VerificationResult:
        if requires_api_base(kind) and api_base is None:
            return self._failure(kind, api_key, f"api_base is required for provider {kind.value}")
        url = models_endpoint(kind, api_base)
        try:
            async with httpx.AsyncClient(
                timeout=_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.get(url, headers=models_headers(kind, api_key))
        except httpx.TimeoutException:
            return self._failure(kind, api_key, f"timeout (GET {url})")
        except httpx.TransportError as exc:
            return self._failure(kind, api_key, f"unreachable ({type(exc).__name__}: {exc})")
        if 200 <= response.status_code < 300:
            return VerificationResult(ok=True, provider=kind, reason=None)
        return self._failure(kind, api_key, _status_reason(response.status_code))

    def verify(self, kind: ProviderKind, api_base: Url | None, api_key: str) -> VerificationResult:
        return asyncio.run(self.verify_async(kind, api_base, api_key))

    def _failure(self, kind: ProviderKind, api_key: str, reason: str) -> VerificationResult:
        return VerificationResult(ok=False, provider=kind, reason=_redact(api_key, reason))
