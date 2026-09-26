"""Argument definitions for the sessions CLI commands."""

import argparse
from pathlib import Path


def add_sessions_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    sessions = subparsers.add_parser("sessions", help="manage sessions via the daemon")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    listing = sessions_sub.add_parser("list", help="list sessions")
    listing.add_argument("--harness", default=None, help="filter by harness")
    listing.add_argument("--state", default=None, help="filter by state")
    listing.add_argument("--base-dir", type=Path, default=None, help="base directory")
    rename = sessions_sub.add_parser("rename", help="rename a session")
    rename.add_argument("session_id")
    rename.add_argument("title")
    rename.add_argument("--base-dir", type=Path, default=None, help="base directory")
    delete = sessions_sub.add_parser("delete", help="delete a session")
    delete.add_argument("session_id")
    delete.add_argument("--purge", action="store_true", help="also delete from the harness")
    delete.add_argument("--base-dir", type=Path, default=None, help="base directory")
    start = sessions_sub.add_parser("start", help="start a live harness session")
    start.add_argument(
        "--harness", required=True, help="harness kind (codex, claude, opencode, agy, pi)"
    )
    start.add_argument("--model", required=True, help="model id to bind the session route to")
    start.add_argument("--effort", default=None, help="Reasoning effort level for the session")
    start.add_argument("--cwd", required=True, help="working directory for the harness process")
    start.add_argument("--base-dir", type=Path, default=None, help="base directory")
    start.add_argument("--execution-backend", choices=("host", "docker"), default=None)
    start.add_argument("--privacy-mode", choices=("none", "surrogate"), default=None)
    start.add_argument("--mode", default=None, help="native harness permission mode")
    start.add_argument("--model-source", choices=("gateway", "native"), default=None)
    stop = sessions_sub.add_parser("stop", help="stop a daemon-started session")
    stop.add_argument("session_id")
    stop.add_argument("--base-dir", type=Path, default=None, help="base directory")
    resume = sessions_sub.add_parser("resume", help="resume a session with its persisted policy")
    resume.add_argument("session_id")
    resume.add_argument("--mode", default=None, help="native harness permission mode")
    resume.add_argument("--base-dir", type=Path, default=None, help="base directory")
    fork = sessions_sub.add_parser("fork", help="fork a stopped session into a selected policy")
    fork.add_argument("session_id")
    fork.add_argument("--execution-backend", choices=("host", "docker"), required=True)
    fork.add_argument("--privacy-mode", choices=("none", "surrogate"), required=True)
    fork.add_argument("--mode", default=None, help="native harness permission mode")
    fork.add_argument("--base-dir", type=Path, default=None, help="base directory")
