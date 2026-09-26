"""Smoke tests for the CLI entry point surface."""

import argparse

import pytest
from mandri.cli.main import build_parser, main


def test_lifecycle_commands_absent_from_parser() -> None:
    for command in ("start", "stop", "status"):
        with pytest.raises((SystemExit, argparse.ArgumentError)):
            build_parser().parse_args([command])
        with pytest.raises((SystemExit, argparse.ArgumentError)):
            main([command])


def test_parser_accepts_run_shape() -> None:
    args = build_parser().parse_args(["run", "--model", "prov/model", "claude"])
    assert args.harness == "claude"
    assert args.model == "prov/model"


def test_parser_strips_passthrough_marker() -> None:
    args = build_parser().parse_args(["run", "--model", "p/m", "codex", "--", "--flag"])
    assert tuple(args.passthrough) == ("--flag",)


def test_parser_accepts_sessions_commands() -> None:
    listing = build_parser().parse_args(["sessions", "list"])
    assert listing.command == "sessions"
    assert listing.sessions_command == "list"
    rename = build_parser().parse_args(["sessions", "rename", "s1", "new title"])
    assert rename.session_id == "s1"
    assert rename.title == "new title"


def test_parser_accepts_provider_commands() -> None:
    listing = build_parser().parse_args(["provider", "list"])
    assert listing.command == "provider"
    assert listing.provider_command == "list"
    add = build_parser().parse_args(
        ["provider", "add", "openrouter", "--kind", "openrouter", "--key", "k"]
    )
    assert add.name == "openrouter"
    assert add.no_verify is False
