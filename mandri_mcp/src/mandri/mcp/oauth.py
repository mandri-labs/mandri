import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlsplit

from mandri.mcp.storage import SecretStore
from pydantic import AnyUrl

from mcp.client.auth import OAuthClientProvider
from mcp.client.auth.oauth2 import OAuthContext
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)


class AuthenticationRequired(Exception):
    pass


class OAuthStorage:
    def __init__(self, secrets_store: SecretStore, identity: str) -> None:
        self.secrets = secrets_store
        self.identity = identity
        self.context: OAuthContext | None = None

    def load(self) -> dict[str, Any]:
        value: dict[str, Any] = self.secrets.load(self.identity).get("oauth", {})
        return value

    def save(self, fields: dict[str, Any]) -> None:
        document = self.secrets.load(self.identity)
        document["oauth"] = self.load() | fields
        self.secrets.save(self.identity, document)

    async def get_tokens(self) -> OAuthToken | None:
        fields = self.load()
        if not fields.get("tokens"):
            return None
        token = OAuthToken.model_validate(fields["tokens"])
        expires = fields.get("expires")
        if expires is not None:
            token.expires_in = max(0, int(expires - time.time()))
        return token

    async def set_tokens(self, tokens: OAuthToken) -> None:
        context = self.context
        metadata = {}
        if context is not None:
            metadata = {
                "authorization_server": context.auth_server_url,
                "authorization_metadata": context.oauth_metadata.model_dump(mode="json")
                if context.oauth_metadata
                else None,
                "resource_metadata": context.protected_resource_metadata.model_dump(mode="json")
                if context.protected_resource_metadata
                else None,
            }
        self.save(
            {
                **metadata,
                "tokens": tokens.model_dump(mode="json"),
                "expires": time.time() + tokens.expires_in
                if tokens.expires_in is not None
                else None,
            }
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        info = self.load().get("client")
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.save({"client": client_info.model_dump(mode="json")})

    def clear(self) -> None:
        document = self.secrets.load(self.identity)
        document.pop("oauth", None)
        self.secrets.save(self.identity, document)


@dataclass
class Pending:
    url: str | None
    result: asyncio.Future[AuthorizationCodeResult]


class OAuthManager:
    def __init__(self, store: SecretStore, port: int) -> None:
        self.store = store
        self.port = port
        self.pending: dict[str, Pending] = {}
        self.events: dict[str, asyncio.Event] = {}

    def interactive(self, identity: str) -> bool:
        return identity in self.pending

    def has_tokens(self, identity: str) -> bool:
        return bool(OAuthStorage(self.store, identity).load().get("tokens"))

    def begin(self, identity: str) -> None:
        self.cancel(identity)
        OAuthStorage(self.store, identity).clear()
        self.pending[identity] = Pending(None, asyncio.get_running_loop().create_future())
        self.events.setdefault(identity, asyncio.Event()).set()

    def cancel(self, identity: str) -> None:
        pending = self.pending.pop(identity, None)
        if pending is not None and not pending.result.done():
            pending.result.cancel()
        self.events.setdefault(identity, asyncio.Event()).clear()

    async def wait_for_login(self, identity: str) -> None:
        event = self.events.setdefault(identity, asyncio.Event())
        await event.wait()
        event.clear()

    def url(self, identity: str) -> str | None:
        pending = self.pending.get(identity)
        return pending.url if pending else None

    def provider(self, identity: str, url: str, changed: Any) -> OAuthClientProvider:
        async def redirect(authorize_url: str) -> None:
            pending = self.pending.get(identity)
            if pending is None:
                raise AuthenticationRequired()
            pending.url = authorize_url
            changed(identity)

        async def callback() -> AuthorizationCodeResult:
            pending = self.pending.get(identity)
            if pending is None:
                raise AuthenticationRequired()
            async with asyncio.timeout(300):
                return await pending.result

        storage = OAuthStorage(self.store, identity)
        provider = OAuthClientProvider(
            server_url=url,
            client_metadata=OAuthClientMetadata(
                client_name="Mandri",
                redirect_uris=[
                    AnyUrl(f"http://127.0.0.1:{self.port}/v1/mcp/oauth/callback/{identity}")
                ],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                token_endpoint_auth_method="none",
            ),
            storage=storage,
            redirect_handler=redirect,
            callback_handler=callback,
        )
        context = provider.context
        storage.context = context
        fields = storage.load()
        expires = fields.get("expires")
        if expires is not None:
            context.token_expiry_time = max(1.0, float(expires))
        context.auth_server_url = fields.get("authorization_server")
        if fields.get("authorization_metadata"):
            context.oauth_metadata = OAuthMetadata.model_validate(fields["authorization_metadata"])
        if fields.get("resource_metadata"):
            context.protected_resource_metadata = ProtectedResourceMetadata.model_validate(
                fields["resource_metadata"]
            )
        return provider

    def complete(self, identity: str, code: str, state: str, issuer: str | None) -> None:
        pending = self.pending.get(identity)
        expected = (
            parse_qs(urlsplit(pending.url).query).get("state", [""])[0]
            if (pending and pending.url)
            else ""
        )
        if not pending or not expected or not secrets.compare_digest(expected, state):
            raise ValueError("Invalid authorization state")
        if pending.result.done():
            raise ValueError("Authorization already completed")
        pending.result.set_result(AuthorizationCodeResult(code=code, state=state, iss=issuer))

    def close(self) -> None:
        for identity in list(self.pending):
            self.cancel(identity)
