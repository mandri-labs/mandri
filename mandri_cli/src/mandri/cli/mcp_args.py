import argparse
from pathlib import Path


def add_mcp_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    mcp = subparsers.add_parser("mcp", help="manage shared MCP servers")
    commands = mcp.add_subparsers(dest="mcp_command", required=True)
    for command in (
        "add",
        "update",
        "list",
        "show",
        "remove",
        "enable",
        "disable",
        "reconnect",
        "login",
        "logout",
    ):
        parser = commands.add_parser(command)
        parser.add_argument("--base-dir", type=Path, default=None)
        parser.add_argument("--json", action="store_true")
        if command != "list":
            parser.add_argument("name")
        if command in ("add", "update"):
            transport = parser.add_mutually_exclusive_group()
            transport.add_argument("--stdio", action="store_true")
            transport.add_argument("--url")
            transport.add_argument("--sse")
            parser.add_argument("--rename")
            parser.add_argument("--cwd")
            parser.add_argument("--env", action="append", default=None, metavar="NAME=VALUE")
            parser.add_argument("--env-ref", action="append", default=None, metavar="NAME=VARIABLE")
            parser.add_argument("--secret", action="append", default=None, metavar="NAME")
            parser.add_argument("--header", action="append", default=None, metavar="NAME=VALUE")
            parser.add_argument("--header-secret", action="append", default=None, metavar="NAME")
            parser.add_argument("--oauth", action=argparse.BooleanOptionalAction, default=None)
            parser.add_argument(
                "--allow-protected", action=argparse.BooleanOptionalAction, default=None
            )
            parser.add_argument("--startup-timeout", type=float, default=None)
            parser.add_argument("--call-timeout", type=float, default=None)
            if command == "update":
                parser.add_argument("--unset-env", action="append", default=None, metavar="NAME")
                parser.add_argument("--unset-header", action="append", default=None, metavar="NAME")
            parser.set_defaults(mcp_argv=[])
