import re
import secrets
from collections.abc import Sequence

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_CHILD_ENDPOINT = re.compile(r"^/v1/(gateway/llm/[^/]+/.+|runtime/agy/[^/]+/hook)$")


class LocalSecurity:
    def __init__(
        self, app: ASGIApp, hosts: Sequence[str], origins: Sequence[str], token: str | None
    ) -> None:
        self.app = app
        self.hosts = set(hosts)
        self.origins = set(origins)
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = headers.get("host", "").split(":")[0].lower()
        origin = headers.get("origin")
        if host not in self.hosts or (origin is not None and origin not in self.origins):
            await self._reject(scope, receive, send, 403)
            return
        preflight = scope.get("method") == "OPTIONS" and origin in self.origins
        child = scope["type"] == "http" and _CHILD_ENDPOINT.fullmatch(scope["path"])
        if self.token and not preflight and not child:
            authorization = headers.get("authorization", "")
            supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
            if scope["type"] == "websocket":
                if "mandri" not in scope.get("subprotocols", []):
                    await self._reject(scope, receive, send, 401)
                    return
                supplied = next(
                    (
                        p.removeprefix("mandri-token.")
                        for p in scope.get("subprotocols", [])
                        if p.startswith("mandri-token.")
                    ),
                    "",
                )
            if not secrets.compare_digest(supplied.encode(), self.token.encode()):
                await self._reject(scope, receive, send, 401)
                return
            if scope["type"] == "websocket":
                scope["mandri.subprotocol"] = "mandri"
        await self.app(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send, status: int) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"detail": "Local access denied"}, status)(scope, receive, send)
