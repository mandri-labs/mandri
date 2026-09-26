"""Command-line interface for Mandri."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from mandri.cli.providers import run_provider_command
from mandri.cli.providers_args import add_providers_parser
from mandri.cli.run import RunCommand
from mandri.cli.run_errors import (
    DaemonUnreachableError,
    ForeignServiceError,
    HarnessBinaryNotFoundError,
    ModelResolutionError,
    ReadinessTimeoutError,
    RunError,
)
from mandri.cli.sessions import run_sessions_command
from mandri.cli.sessions_args import add_sessions_parser
from mandri.cli.types import RunSpec
from mandri.config.toml_adapter import DEFAULT_BASE_DIR
from mandri.core.ids import HarnessKind
from mandri.core.version import __version__


def main(argv: Sequence[str] | None = None) -> int:
    return dispatch_args(build_parser().parse_args(argv))


def dispatch_args(args: argparse.Namespace) -> int:
    if args.command == "run":
        return _run_command(args)
    if args.command == "sessions":
        return run_sessions_command(args)
    return run_provider_command(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mandri", description="Mandri CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="run a harness against the daemon gateway",
        usage="mandri run --model <provider>/<model-id> <harness> [-- <harness args>]",
    )
    run.add_argument(
        "harness",
        choices=[kind.value for kind in HarnessKind],
        help="harness kind (claude, codex, opencode, agy, pi)",
    )
    run.add_argument("--model", required=True, help="model as <provider>/<model-id>")
    run.add_argument("--effort", default=None, help="Reasoning effort level for the session")
    run.add_argument(
        "--cwd", type=Path, default=None, help="working directory for the harness process"
    )
    run.add_argument("--base-dir", type=Path, default=None, help="base directory")
    run.add_argument(
        "passthrough", nargs=argparse.REMAINDER, help="arguments passed to the harness"
    )
    add_sessions_parser(subparsers)
    add_providers_parser(subparsers)
    return parser


def _run_command(args: argparse.Namespace) -> int:
    base_dir = _base_dir(args)
    spec = RunSpec(
        harness=HarnessKind(args.harness),
        model_arg=args.model,
        base_dir=base_dir,
        cwd=args.cwd,
        effort=args.effort,
        passthrough_args=_passthrough(args),
    )
    try:
        return RunCommand(spec).run()
    except RunError as error:
        print(str(error), file=sys.stderr)
        return _error_exit_code(error)


def _base_dir(args: argparse.Namespace) -> Path:
    return args.base_dir if args.base_dir is not None else DEFAULT_BASE_DIR


def _passthrough(args: argparse.Namespace) -> tuple[str, ...]:
    passthrough = tuple(args.passthrough)
    if passthrough[:1] == ("--",):
        return passthrough[1:]
    return passthrough


def _error_exit_code(error: RunError) -> int:
    if isinstance(error, ModelResolutionError):
        return 5
    if isinstance(error, HarnessBinaryNotFoundError):
        return 6
    if isinstance(error, ForeignServiceError):
        return 4
    if isinstance(error, (DaemonUnreachableError, ReadinessTimeoutError)):
        return 3
    return 2
