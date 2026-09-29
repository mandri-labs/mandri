"""ChatGPT sign-in sessions driving the OAuth flow to completion."""

import asyncio
import contextlib
import dataclasses
import enum
import uuid

import httpx
from mandri.core.ids import ProviderKind
from mandri.providers.chatgpt.oauth import (
    CALLBACK_TIMEOUT_SECONDS,
    ChatGptAuthError,
    ChatGptLoginCancelled,
    LoopbackCallbackServer,
    authorize_url,
    callback_code,
    check_state,
    exchange_code,
    pkce_pair,
    state_token,
)
from mandri.providers.chatgpt.store import ChatGptTokenStore
from mandri.providers.errors import ProviderError, ProviderExistsError, ProviderNotFoundError
from mandri.providers.service import Provider, ProvidersRegistry

DEFAULT_PROVIDER_NAME = "chatgpt"
_MAX_SESSIONS = 32


class ChatGptLoginStatus(enum.StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass
class ChatGptLoginSession:
    id: str
    provider_name: str
    authorize_url: str = ""
    status: ChatGptLoginStatus = ChatGptLoginStatus.PENDING
    provider: Provider | None = None
    error: str | None = None
    error_code: str | None = None
    verifier: str = ""
    state: str = ""
    challenge: str = ""
    redirect: str = ""
    api_base: str | None = None
    exchanging: bool = False

    @property
    def terminal(self) -> bool:
        return self.status is not ChatGptLoginStatus.PENDING


class ChatGptLoginService:
    """Runs browser authorizations and records them as provider credentials."""

    def __init__(
        self,
        providers: ProvidersRegistry,
        store: ChatGptTokenStore,
        http: httpx.AsyncClient,
        timeout_seconds: float = CALLBACK_TIMEOUT_SECONDS,
    ) -> None:
        self._providers = providers
        self._store = store
        self._http = http
        self._timeout = timeout_seconds
        self._sessions: dict[str, ChatGptLoginSession] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._start_lock = asyncio.Lock()
        self._completion_locks: dict[str, asyncio.Lock] = {}

    async def start(
        self, name: str | None = None, api_base: str | None = None
    ) -> ChatGptLoginSession:
        provider_name = (name or DEFAULT_PROVIDER_NAME).strip() or DEFAULT_PROVIDER_NAME
        async with self._start_lock:
            self._check_provider(provider_name)
            pending = self.pending_for(provider_name)
            if pending is not None:
                return pending
            return await self._start(provider_name, api_base)

    async def _start(self, provider_name: str, api_base: str | None) -> ChatGptLoginSession:
        pkce = pkce_pair()
        session = ChatGptLoginSession(
            id=str(uuid.uuid4()),
            provider_name=provider_name,
            verifier=pkce.verifier,
            state=state_token(),
            challenge=pkce.challenge,
            api_base=api_base,
        )
        self._prune()
        self._sessions[session.id] = session
        server = LoopbackCallbackServer()
        try:
            await server.start()
        except ChatGptAuthError as error:
            _fail(session, error, "chatgpt_callback_unavailable")
            return session
        session.redirect = server.uri
        session.authorize_url = authorize_url(server.uri, session.state, session.challenge)
        self._tasks[session.id] = asyncio.create_task(self._await_callback(session, server))
        return session

    def get(self, login_id: str) -> ChatGptLoginSession | None:
        return self._sessions.get(login_id)

    def pending_for(self, provider_name: str) -> ChatGptLoginSession | None:
        """Return the login still awaiting a browser callback for this provider."""
        for session in self._sessions.values():
            if (
                session.provider_name == provider_name
                and session.status is ChatGptLoginStatus.PENDING
            ):
                return session
        return None

    async def cancel(self, login_id: str) -> ChatGptLoginSession | None:
        session = self._sessions.get(login_id)
        if session is None:
            return None
        await self._stop_task(login_id)
        if not session.terminal:
            session.status = ChatGptLoginStatus.CANCELLED
        return session

    async def submit_redirect(self, login_id: str, raw_url: str) -> ChatGptLoginSession | None:
        """Completes a login whose browser cannot reach this machine's loopback listener."""
        session = self._sessions.get(login_id)
        if session is None:
            return None
        async with self._completion_locks.setdefault(login_id, asyncio.Lock()):
            if session.terminal or session.exchanging or not session.redirect:
                return session
            try:
                code = callback_code(raw_url, session.state)
            except ChatGptAuthError as error:
                session.error = str(error)
                session.error_code = "chatgpt_callback_rejected"
                return session
            await self._stop_task(login_id)
            session.status = ChatGptLoginStatus.PENDING
            await self._complete_code(session, code)
        return session

    async def aclose(self) -> None:
        for login_id in list(self._tasks):
            await self.cancel(login_id)

    async def cancel_provider(self, name: str) -> None:
        for session in list(self._sessions.values()):
            if session.provider_name == name and not session.terminal:
                await self.cancel(session.id)

    async def _await_callback(
        self, session: ChatGptLoginSession, server: LoopbackCallbackServer
    ) -> None:
        try:
            callback = await server.wait(self._timeout)
        except ChatGptLoginCancelled as error:
            _fail(session, error, "chatgpt_login_timeout")
            return
        except asyncio.CancelledError:
            session.status = ChatGptLoginStatus.CANCELLED
            raise
        finally:
            await server.aclose()
        if session.terminal:
            return
        try:
            check_state(callback[1], session.state)
        except ChatGptAuthError as error:
            _fail(session, error, "chatgpt_state_mismatch")
            return
        if not callback[0]:
            _fail(
                session, ChatGptAuthError("Authorization was declined"), "chatgpt_callback_rejected"
            )
            return
        await self._complete_code(session, callback[0])

    async def _complete_code(self, session: ChatGptLoginSession, code: str) -> None:
        if session.terminal or session.exchanging:
            return
        session.exchanging = True
        try:
            tokens = await exchange_code(self._http, code, session.verifier, session.redirect)
        except ChatGptAuthError as error:
            _fail(session, error, "chatgpt_exchange_failed")
            return
        if session.terminal:
            return
        previous = self._store.load(session.provider_name)
        try:
            self._check_provider(session.provider_name)
            self._store.save(session.provider_name, tokens)
            session.provider = await self._register(session.provider_name, session.api_base)
        except (ProviderError, OSError) as error:
            if previous is not None:
                self._store.save(session.provider_name, previous)
            else:
                self._store.clear(session.provider_name)
            _fail(session, error, "chatgpt_provider_rejected")
            return
        session.status = ChatGptLoginStatus.COMPLETED
        session.error = None
        session.error_code = None

    async def _register(self, name: str, api_base: str | None) -> Provider:
        self._check_provider(name)
        try:
            return await self._providers.add(name, ProviderKind.CHATGPT, api_base, "", verify=False)
        except ProviderExistsError:
            self._check_provider(name)
            return self._providers.get(name)

    def _check_provider(self, name: str) -> None:
        try:
            provider = self._providers.get(name)
        except ProviderNotFoundError:
            return
        if provider.kind is not ProviderKind.CHATGPT:
            raise ProviderExistsError(f"provider {name!r} already exists")

    async def _stop_task(self, login_id: str) -> None:
        task = self._tasks.pop(login_id, None)
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _prune(self) -> None:
        if len(self._sessions) < _MAX_SESSIONS:
            return
        for login_id, session in list(self._sessions.items()):
            if session.terminal:
                del self._sessions[login_id]
                self._tasks.pop(login_id, None)
                self._completion_locks.pop(login_id, None)
            if len(self._sessions) < _MAX_SESSIONS:
                return


def _fail(session: ChatGptLoginSession, error: BaseException, code: str) -> None:
    session.status = ChatGptLoginStatus.FAILED
    session.error_code = code
    session.error = str(error)
