"""mandri-daemon entry point: start, stop, status, and serve."""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import litellm
import uvicorn
from mandri.config.toml_adapter import DEFAULT_BASE_DIR, TomlConfigAdapter
from mandri.core.version import __version__
from mandri.daemon import daemonctl
from mandri.daemon.desktop import backup_profile, serve_owned
from mandri.daemon.interrupts import InterruptController
from mandri.daemon.serve import RuntimeResources, build_server, load_config, serve


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve_command(args)
    if args.command == "start":
        return _start_command(args)
    if args.command == "status":
        return _status_command(args)
    return _stop_command(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mandri-daemon", description="Mandri daemon control")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="start the daemon")
    start.add_argument("--host", default=None, help="host override")
    start.add_argument("--port", type=int, default=None, help="port override")
    start.add_argument("--base-dir", type=Path, default=None, help="base directory")
    _add_logging_arguments(start)
    stop = subparsers.add_parser("stop", help="stop the daemon")
    stop.add_argument("--base-dir", type=Path, default=None, help="base directory")
    status = subparsers.add_parser("status", help="show daemon status")
    status.add_argument("--base-dir", type=Path, default=None, help="base directory")
    serve = subparsers.add_parser("serve", help="run the daemon in the foreground")
    serve.add_argument("--host", default=None, help="host override")
    serve.add_argument("--port", type=int, default=None, help="port override")
    serve.add_argument("--base-dir", type=Path, default=None, help="base directory")
    serve.add_argument("--desktop", action="store_true", help="run as an owned desktop daemon")
    _add_logging_arguments(serve)
    return parser


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="logging level (default: LOG_LEVEL or INFO)",
    )
    parser.add_argument(
        "--enable-litellm-debug",
        action="store_true",
        default=_env_flag("ENABLE_LITELLM_DEBUG"),
        help="enable LiteLLM debug logging",
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _serve_command(args: argparse.Namespace) -> int:
    log_level = args.log_level.upper()
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)
    if args.enable_litellm_debug:
        litellm._turn_on_debug()
    base_dir = args.base_dir if args.base_dir is not None else DEFAULT_BASE_DIR
    token = os.environ.pop("MANDRI_DESKTOP_TOKEN", None) if args.desktop else None
    if args.desktop:
        if not token or len(token) < 32 or args.host != "127.0.0.1":
            raise ValueError("Desktop mode requires a loopback host and a private token")
        backup_profile(base_dir)
    config = load_config(base_dir, args.host, args.port, persist_overrides=not args.desktop)
    server, resources = build_server(
        config.server.host,
        config.server.port,
        config.server.cors_origins,
        token,
        log_level=log_level.lower(),
    )

    def kill_children() -> None:
        if resources.runtime is not None:
            resources.runtime.kill_all_now()

    try:
        with InterruptController(server, kill_children).installed():
            asyncio.run(_run_server(server, resources, base_dir, args.desktop))
    except KeyboardInterrupt:
        return 0
    return 0


async def _run_server(
    server: uvicorn.Server, resources: RuntimeResources, base_dir: Path, desktop: bool
) -> None:
    task = asyncio.create_task(serve(server, resources, base_dir))
    if desktop:
        await serve_owned(server, task)
    else:
        await task


def _start_command(args: argparse.Namespace) -> int:
    base_dir = args.base_dir if args.base_dir is not None else DEFAULT_BASE_DIR
    load_config(base_dir, args.host, args.port)
    try:
        address = daemonctl.ensure_running(
            base_dir,
            log_level=args.log_level.lower(),
            enable_litellm_debug=args.enable_litellm_debug,
        )
    except daemonctl.ForeignServiceError as error:
        print(str(error), file=sys.stderr)
        return 4
    except daemonctl.ReadinessTimeoutError as error:
        print(str(error), file=sys.stderr)
        return 3
    print(f"running at http://{address.host}:{address.port}")
    return 0


def _status_command(args: argparse.Namespace) -> int:
    base_dir = args.base_dir if args.base_dir is not None else DEFAULT_BASE_DIR
    adapter = TomlConfigAdapter(base_dir)
    config_path = adapter.config_path
    db_path = base_dir / "mandri.db"
    if not config_path.is_file() and not db_path.is_file():
        print("not running")
        return 3
    config = adapter.load()
    print(f"config: {config_path}")
    print(f"database: {db_path}")
    address = daemonctl.Address(host=config.server.host, port=config.server.port)
    if daemonctl.probe(address) is daemonctl.ProbeStatus.RUNNING:
        print(f"running at http://{config.server.host}:{config.server.port}")
        return 0
    print("not running")
    return 3


def _stop_command(args: argparse.Namespace) -> int:
    base_dir = args.base_dir if args.base_dir is not None else DEFAULT_BASE_DIR
    record = daemonctl.read_pid_file(base_dir)
    if record is None or not daemonctl.pid_alive(record.pid):
        daemonctl.remove_pid_file(base_dir)
        print("not running")
        return 3
    try:
        daemonctl.terminate_pid(record.pid)
    except daemonctl.StopError as error:
        print(str(error), file=sys.stderr)
        return 1
    daemonctl.remove_pid_file(base_dir)
    print(f"stopped (pid {record.pid})")
    return 0
