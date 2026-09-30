"""ChatGPT OAuth authorization-code flow with PKCE and loopback redirect."""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import secrets
import socket
import sys
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from mandri.core.clock import system_now_ms

AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
SCOPE = "openid profile email offline_access"
CALLBACK_PATH = "/auth/callback"
CALLBACK_PORTS = (1455, 1457)
CALLBACK_TIMEOUT_SECONDS = 300.0
_TOKEN_TIMEOUT_SECONDS = 20.0
_DEFAULT_EXPIRES_IN = 3600


class ChatGptAuthError(Exception):
    """The OAuth exchange, refresh, or callback failed."""


class ChatGptLoginCancelled(ChatGptAuthError):
    """No authorization callback arrived before the deadline."""


@dataclass(frozen=True)
class TokenSet:
    access_token: str
    refresh_token: str
    expires_at_ms: int


@dataclass(frozen=True)
class PkcePair:
    verifier: str
    challenge: str


def pkce_pair() -> PkcePair:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return PkcePair(verifier, base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="))


def state_token() -> str:
    return secrets.token_urlsafe(32)


def redirect_uri(port: int) -> str:
    return f"http://localhost:{port}{CALLBACK_PATH}"


def authorize_url(redirect: str, state: str, challenge: str) -> str:
    query = urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def check_state(returned_state: str, expected_state: str) -> None:
    if not hmac.compare_digest(returned_state or "", expected_state):
        raise ChatGptAuthError("authorization state mismatch")


def callback_code(raw_url: str, expected_state: str) -> str:
    query = dict(parse_qsl(urlsplit(raw_url).query, keep_blank_values=True))
    if query.get("error"):
        raise ChatGptAuthError(query.get("error_description") or query["error"])
    check_state(query.get("state") or "", expected_state)
    code = (query.get("code") or "").strip()
    if not code:
        raise ChatGptAuthError("authorization callback carried no code")
    return code


async def exchange_code(
    client: httpx.AsyncClient, code: str, verifier: str, redirect: str
) -> TokenSet:
    return await _token_request(
        client,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
    )


async def refresh_tokens(client: httpx.AsyncClient, refresh_token: str) -> TokenSet:
    if not refresh_token.strip():
        raise ChatGptAuthError("stored credentials have no refresh token")
    return await _token_request(
        client,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
    )


async def _token_request(client: httpx.AsyncClient, data: dict[str, str]) -> TokenSet:
    try:
        response = await client.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=_TOKEN_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:
        raise ChatGptAuthError(f"token endpoint unreachable: {type(error).__name__}") from error
    if response.status_code != 200:
        raise ChatGptAuthError(
            f"token endpoint returned status {response.status_code}: {_reason(response)}"
        )
    try:
        payload = response.json()
    except ValueError:
        raise ChatGptAuthError("token endpoint returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise ChatGptAuthError("token endpoint returned an invalid token response")
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise ChatGptAuthError("token endpoint returned no access token")
    rotated = payload.get("refresh_token")
    refresh = rotated if isinstance(rotated, str) and rotated else data.get("refresh_token", "")
    expires_in = payload.get("expires_in")
    seconds = expires_in if type(expires_in) is int and expires_in > 0 else _DEFAULT_EXPIRES_IN
    return TokenSet(
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=system_now_ms() + seconds * 1000,
    )


def _reason(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        detail = payload.get("error_description") or payload.get("error")
        if isinstance(detail, str):
            return detail
    return response.text[:200]


def _bind(port: int) -> socket.socket | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    except OSError:
        sock.close()
        return None
    sock.listen()
    return sock


class LoopbackCallbackServer:
    """Single-shot localhost listener for the OAuth redirect."""

    def __init__(self, ports: tuple[int, ...] = CALLBACK_PORTS) -> None:
        self._ports = ports
        self._server: asyncio.AbstractServer | None = None
        self._pending: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.port: int | None = None

    @property
    def uri(self) -> str:
        if self.port is None:
            raise ChatGptAuthError("callback listener is not bound")
        return redirect_uri(self.port)

    async def start(self) -> None:
        last_error: OSError | None = None
        for port in self._ports:
            sock = _bind(port)
            if sock is None:
                last_error = OSError(f"port {port} is unavailable")
                continue
            self.port = sock.getsockname()[1]
            self._server = await asyncio.start_server(self._handle, sock=sock)
            return
        raise ChatGptAuthError(f"could not bind the loopback callback port: {last_error}")

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None

    async def wait(self, timeout_seconds: float = CALLBACK_TIMEOUT_SECONDS) -> tuple[str, str]:
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._pending.get()
        except TimeoutError:
            raise ChatGptLoginCancelled(
                "the authorization callback did not arrive in time"
            ) from None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            parts = request_line.decode("latin-1").split(" ")
            target = parts[1] if len(parts) > 1 else ""
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            parsed = urlsplit(target)
            if len(parts) < 2 or parts[0] != "GET" or parsed.path != CALLBACK_PATH:
                _write_response(writer, False)
                await writer.drain()
                return
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            code = query.get("code", "")
            _write_response(writer, bool(code))
            await writer.drain()
            self._pending.put_nowait((code, query.get("state", "")))
        except (asyncio.IncompleteReadError, ConnectionError, ValueError, IndexError):
            pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


def _write_response(writer: asyncio.StreamWriter, ok: bool) -> None:
    body = (
        "<html><body><h1>Authorization complete</h1>"
        "<p>You can close this window and return to Mandri.</p></body></html>"
        if ok
        else "<html><body><h1>Authorization failed</h1>"
        "<p>Return to Mandri and retry.</p></body></html>"
    ).encode("utf-8")
    status = "200 OK" if ok else "400 Bad Request"
    head = (
        f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(head + body)
