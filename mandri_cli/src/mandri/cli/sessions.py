"""Sessions CLI commands: local service calls plus daemon-mediated lifecycle."""

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import httpx
from mandri.cli.command import Command
from mandri.cli.daemon_client import error_message, request, resolve_base_url
from mandri.cli.local import build_sessions_service, resolve_base_dir
from mandri.cli.run_errors import DaemonUnreachableError
from mandri.cli.session_continuation import ForkSessionCommand, ResumeSessionCommand
from mandri.core.ids import HarnessKind, SessionId, SessionState, SessionTitle
from mandri.core.types.execution import ProtectionError
from mandri.core.types.sessions import Session
from mandri.sessions.errors import SessionConflictError, SessionNotFoundError, SessionRunningError
from mandri.sessions.service import SessionsService

__all__ = ["run_sessions_command"]

T = TypeVar("T")


class SessionsCommand(Command):
    """Base class for sessions subcommands sharing service access and daemon requests."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._base_dir: Path = resolve_base_dir(args.base_dir)

    async def _with_service(self, action: Callable[[SessionsService], Awaitable[T]]) -> T:
        service, db = await build_sessions_service(self._base_dir)
        try:
            return await action(service)
        finally:
            await db.close()

    def _daemon_post(self, path: str, body: dict[str, str] | None = None) -> httpx.Response | None:
        try:
            return request("POST", f"{resolve_base_url(self._base_dir)}{path}", body=body)
        except DaemonUnreachableError:
            print("daemon unreachable")
            return None

    def _check(self, response: httpx.Response) -> int:
        if response.status_code < 400:
            return 0
        print(error_message(response))
        return 1


class ListSessionsCommand(SessionsCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._harness_raw: str | None = args.harness
        self._state_raw: str | None = args.state

    def run(self) -> int:
        harness = _parse_harness(self._harness_raw)
        if isinstance(harness, str):
            print(harness)
            return 1
        state = _parse_state(self._state_raw)
        if isinstance(state, str):
            print(state)
            return 1
        sessions = asyncio.run(
            self._with_service(lambda service: service.list_sessions(harness=harness, state=state))
        )
        for session in sessions:
            print(
                f"{session.id}  {session.harness.value}  {session.state.value}"
                f"  {session.project_path}  {session.effective_title}"
            )
        return 0


class RenameSessionCommand(SessionsCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._session_id: str = args.session_id
        self._title: str = args.title

    def run(self) -> int:
        async def rename(service: SessionsService) -> Session:
            return await service.rename_session(
                SessionId(self._session_id), SessionTitle(self._title)
            )

        try:
            session = asyncio.run(self._with_service(rename))
        except SessionNotFoundError as error:
            print(str(error))
            return 1
        print(session.effective_title)
        return 0


class DeleteSessionCommand(SessionsCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._session_id: str = args.session_id
        self._purge: bool = args.purge

    def run(self) -> int:
        try:
            response = request(
                "DELETE",
                f"{resolve_base_url(self._base_dir)}/v1/sessions/{self._session_id}",
                params={"purge": str(self._purge).lower()},
            )
        except DaemonUnreachableError:
            response = None
        if response is not None:
            status = self._check(response)
            if status == 0:
                print("deleted" + (" (purged)" if self._purge else ""))
            return status

        async def delete(service: SessionsService) -> None:
            await service.delete_session(SessionId(self._session_id), purge=self._purge)

        try:
            asyncio.run(self._with_service(delete))
        except (
            SessionNotFoundError,
            SessionRunningError,
            SessionConflictError,
            ProtectionError,
        ) as error:
            print(str(error))
            return 1
        print("deleted" + (" (purged)" if self._purge else ""))
        return 0


class StartSessionCommand(SessionsCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._body: dict[str, str] = {
            "harness": args.harness,
            "model": args.model,
            "cwd": args.cwd,
        }
        if args.effort is not None:
            self._body["effort"] = args.effort
        for field in (
            "execution_backend",
            "privacy_mode",
            "mode",
            "model_source",
        ):
            value = getattr(args, field, None)
            if value is not None:
                self._body[field] = value

    def run(self) -> int:
        response = self._daemon_post("/v1/runtime/sessions", self._body)
        if response is None:
            return 3
        status = self._check(response)
        if status != 0:
            return status
        session = response.json()
        print(session["id"])
        print(session["gateway_route_id"])
        return 0


class StopSessionCommand(SessionsCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._session_id: str = args.session_id

    def run(self) -> int:
        response = self._daemon_post(f"/v1/sessions/{self._session_id}/stop")
        if response is None:
            return 3
        status = self._check(response)
        if status != 0:
            return status
        print("stopped")
        return 0


def run_sessions_command(args: argparse.Namespace) -> int:
    if args.sessions_command == "resume":
        return ResumeSessionCommand(args).run()
    if args.sessions_command == "fork":
        return ForkSessionCommand(args).run()
    commands: dict[str, type[SessionsCommand]] = {
        "list": ListSessionsCommand,
        "rename": RenameSessionCommand,
        "delete": DeleteSessionCommand,
        "start": StartSessionCommand,
        "stop": StopSessionCommand,
    }
    return commands[args.sessions_command](args).run()


def _parse_harness(raw: str | None) -> HarnessKind | str | None:
    if raw is None:
        return None
    try:
        return HarnessKind(raw)
    except ValueError:
        return f"Unknown harness {raw!r}"


def _parse_state(raw: str | None) -> SessionState | str | None:
    if raw is None:
        return None
    try:
        return SessionState(raw)
    except ValueError:
        return f"Unknown session state {raw!r}"
