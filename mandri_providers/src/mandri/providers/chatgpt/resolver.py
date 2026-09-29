"""ChatGPT credential materialization with single-flight token refresh."""

import asyncio

import httpx
from mandri.core.clock import system_now_ms
from mandri.core.ports.provider_credentials import CredentialResolverPort
from mandri.providers.chatgpt.identity import expires_at_ms
from mandri.providers.chatgpt.oauth import ChatGptAuthError, TokenSet, refresh_tokens
from mandri.providers.chatgpt.store import ChatGptTokenStore
from mandri.providers.errors import ProviderInvalidError

REFRESH_SKEW_MS = 120_000


class ChatGptCredentialResolver(CredentialResolverPort):
    """Reads stored access tokens and rotates them before they expire."""

    def __init__(self, store: ChatGptTokenStore, http: httpx.AsyncClient | None = None) -> None:
        self._store = store
        self._http = http
        self._locks: dict[str, asyncio.Lock] = {}

    def access_token(self, provider_name: str) -> str:
        tokens = self._store.load(provider_name)
        if tokens is None:
            raise ProviderInvalidError(f"provider {provider_name!r} has no ChatGPT credentials")
        return tokens.access_token

    async def ensure_fresh(self, provider_name: str) -> None:
        tokens = self._store.load(provider_name)
        if tokens is None:
            raise ProviderInvalidError(f"provider {provider_name!r} has no ChatGPT credentials")
        if not is_stale(tokens):
            return
        async with self._locks.setdefault(provider_name, asyncio.Lock()):
            current = self._store.load(provider_name)
            if current is None:
                raise ProviderInvalidError(f"provider {provider_name!r} has no ChatGPT credentials")
            if not is_stale(current):
                return
            await self._rotate(provider_name, current)

    def clear(self, provider_name: str) -> None:
        self._store.clear(provider_name)

    async def _rotate(self, provider_name: str, tokens: TokenSet) -> None:
        try:
            if self._http is None:
                async with httpx.AsyncClient() as client:
                    refreshed = await refresh_tokens(client, tokens.refresh_token)
            else:
                refreshed = await refresh_tokens(self._http, tokens.refresh_token)
        except ChatGptAuthError as error:
            raise ProviderInvalidError(
                f"ChatGPT credentials for {provider_name!r} are no longer valid: {error}"
            ) from error
        self._store.save(provider_name, refreshed)


def is_stale(tokens: TokenSet) -> bool:
    expires = expires_at_ms(tokens.access_token) or tokens.expires_at_ms
    if expires <= 0:
        return False
    return expires - system_now_ms() <= REFRESH_SKEW_MS
