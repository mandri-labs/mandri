"""ChatGPT OAuth sign-in driven by the provider CLI."""

import asyncio
import contextlib
import sys
from pathlib import Path

import httpx
from mandri.cli.local import build_providers_registry, resolve_base_dir
from mandri.cli.terminal_input import read_terminal_line
from mandri.core.browser import open_browser
from mandri.providers.chatgpt.store import ChatGptTokenStore
from mandri.providers.chatgpt_login import (
    ChatGptLoginService,
    ChatGptLoginSession,
    ChatGptLoginStatus,
)
from mandri.providers.errors import ProviderError

_POLL_SECONDS = 0.5
_STDIN_PROMPT = (
    "If the browser could not open, paste the full redirect URL here and press Enter.\n"
    "Otherwise leave this empty and wait for the automatic callback: "
)


def run_chatgpt_login(name: str | None, base_dir: Path | None, api_base: str | None = None) -> int:
    try:
        return asyncio.run(_login(name, base_dir, api_base))
    except ProviderError as error:
        print(str(error))
        return 1
    except KeyboardInterrupt:
        return 130


async def _login(name: str | None, base_dir: Path | None, api_base: str | None) -> int:
    resolved = resolve_base_dir(base_dir)
    registry, db = await build_providers_registry(resolved)
    try:
        async with httpx.AsyncClient() as http:
            service = ChatGptLoginService(
                providers=registry, store=ChatGptTokenStore(resolved), http=http
            )
            try:
                session = await service.start(name, api_base)
                await asyncio.to_thread(_announce, session)
                paste = asyncio.create_task(_read_redirect(service, session))
                try:
                    await _await_completion(session)
                finally:
                    paste.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await paste
            finally:
                await service.aclose()
    finally:
        await db.close()
    return _report(session)


def _announce(session: ChatGptLoginSession) -> None:
    print()
    print("Sign in to ChatGPT to add your subscription as a Mandri provider.")
    print("(Mandri creates its own session - this does not touch Codex CLI or VS Code)")
    print()
    print("Open this URL to authorize Mandri:")
    print()
    print(f"  {session.authorize_url}")
    print()
    if open_browser(session.authorize_url):
        print("A browser window was opened.")
    else:
        print("Could not open a browser automatically; copy the URL above.")
    print()


async def _await_completion(session: ChatGptLoginSession) -> None:
    while not session.terminal:
        await asyncio.sleep(_POLL_SECONDS)


async def _read_redirect(service: ChatGptLoginService, session: ChatGptLoginSession) -> None:
    if sys.stdin is None or sys.stdin.closed or not sys.stdin.isatty():
        return
    while not session.terminal:
        _prompt()
        raw = await read_terminal_line()
        if raw == "":
            return
        text = (raw or "").strip()
        if not text:
            continue
        if text.lower() in {"q", "quit", "cancel"}:
            await service.cancel(session.id)
            return
        await service.submit_redirect(session.id, text)
        if session.error:
            print(session.error)


def _prompt() -> None:
    print(_STDIN_PROMPT, end="", flush=True)


def _report(session: ChatGptLoginSession) -> int:
    print()
    if session.status is ChatGptLoginStatus.COMPLETED:
        print(f"{session.provider_name}  chatgpt  verified")
        return 0
    if session.status is ChatGptLoginStatus.CANCELLED:
        print("ChatGPT sign-in cancelled.")
        return 130
    print(f"ChatGPT sign-in failed: {session.error or 'unknown error'}")
    return 1
