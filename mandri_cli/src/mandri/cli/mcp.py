import argparse
import getpass
import json
import sys
import time
from typing import Any

from mandri.cli.daemon_client import base_url, error_message, request
from mandri.cli.daemon_command import DaemonCommand
from mandri.cli.local import resolve_base_dir
from mandri.cli.run_errors import RunError
from rich.console import Console


def pairs(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        key, separator, content = value.partition("=")
        if not key or not separator:
            raise ValueError("Expected NAME=VALUE")
        result[key] = content
    return result


class McpCommand(DaemonCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self._console = Console(stderr=True)

    def _request(self, method: str, path: str, body: object | None = None) -> Any:
        assert self._address is not None
        response = request(method, f"{base_url(self._address)}/v1/mcp{path}", body=body)
        if response.status_code >= 400:
            raise ValueError(error_message(response))
        return None if response.status_code == 204 else response.json()

    def _changes(self) -> dict[str, Any]:
        args = self.args
        changes: dict[str, Any] = {}
        if args.rename:
            changes["name"] = args.rename
        if args.url or args.sse:
            changes.update(
                transport="sse" if args.sse else "streamable_http",
                url=args.url or args.sse,
                command=None,
                args=[],
                cwd=None,
                env={},
                env_refs={},
            )
        if args.stdio or args.mcp_argv:
            if not args.mcp_argv:
                raise ValueError("Pass a command after --")
            changes.update(
                transport="stdio",
                command=args.mcp_argv[0],
                args=args.mcp_argv[1:],
                url=None,
                headers={},
                auth="none",
            )
        if args.cwd is not None:
            changes["cwd"] = args.cwd
        for field, values in (
            ("env", args.env),
            ("env_refs", args.env_ref),
            ("headers", args.header),
        ):
            if values is not None:
                changes[field] = pairs(values)
        for field, names in (("env", args.secret), ("headers", args.header_secret)):
            if names:
                entries = changes.setdefault(field, {})
                for name in names:
                    entries[name] = getpass.getpass(f"{name}: ")
        if args.oauth is not None:
            changes["auth"] = "oauth" if args.oauth else "none"
        for field in ("startup_timeout", "call_timeout"):
            if getattr(args, field) is not None:
                changes[field] = getattr(args, field)
        for argument, field in (("unset_env", "env_updates"), ("unset_header", "header_updates")):
            names = getattr(args, argument, None)
            if names:
                changes[field] = dict.fromkeys(names)
        if args.allow_protected is not None:
            changes["allow_protected"] = args.allow_protected
        return changes

    def run(self) -> int:
        args = self.args
        try:
            self._ensure_running(resolve_base_dir(args.base_dir))
            command = args.mcp_command
            if command == "add":
                body = {"name": args.name, **self._changes()}
                result = self._request("POST", "/servers", body)
            else:
                snapshot = self._request("GET", "/servers")
                rows = snapshot["servers"]
                if command == "list":
                    result = rows
                else:
                    row = next((row for row in rows if args.name in (row["id"], row["name"])), None)
                    if row is None:
                        raise ValueError("Server not found")
                    path = f"/servers/{row['id']}"
                    if command == "show":
                        result = row
                    elif command in ("enable", "disable", "update"):
                        changes = (
                            self._changes()
                            if command == "update"
                            else {"enabled": command == "enable"}
                        )
                        if command == "update" and not (
                            args.url or args.sse or args.stdio or args.mcp_argv
                        ):
                            for source, target in (
                                ("env", "env_updates"),
                                ("headers", "header_updates"),
                            ):
                                if source in changes:
                                    changes.setdefault(target, {}).update(changes.pop(source))
                        result = self._request(
                            "PATCH", path, {"revision": row["revision"], "changes": changes}
                        )
                    elif command == "remove":
                        result = self._request("DELETE", path)
                    else:
                        result = self._request("POST", f"{path}/{command}")
                        if command == "login":
                            deadline = time.monotonic() + 15
                            while not result.get("authorize_url") and time.monotonic() < deadline:
                                if result["state"] in ("ready", "error", "reconnecting"):
                                    break
                                time.sleep(0.2)
                                snapshot = self._request("GET", "/servers")
                                result = next(
                                    row for row in snapshot["servers"] if row["id"] == result["id"]
                                )
            if args.json:
                print(json.dumps(result, indent=2))
            elif isinstance(result, list):
                for row in result:
                    self._print(row)
            elif isinstance(result, dict):
                self._print(result)
                if result.get("authorize_url"):
                    print(result["authorize_url"])
            return 0
        except (RunError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1

    @staticmethod
    def _print(row: dict[str, Any]) -> None:
        print(f"{row['name']}  {row['transport']}  {row['state']}  {row['tools']}")
        if row.get("error"):
            print(f"  {row['error']}")


def run_mcp_command(args: argparse.Namespace) -> int:
    return McpCommand(args).run()
