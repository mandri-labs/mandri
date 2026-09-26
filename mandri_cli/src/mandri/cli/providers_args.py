"""Argument definitions for the provider CLI commands."""

import argparse
from pathlib import Path


def add_providers_parser(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    provider = subparsers.add_parser("provider", help="manage provider credentials via the daemon")
    provider_sub = provider.add_subparsers(dest="provider_command", required=True)
    add = provider_sub.add_parser("add", help="register provider credentials")
    add.add_argument("name", help="provider name")
    add.add_argument(
        "--kind",
        default=None,
        help="provider kind (inferred from the name when it matches a known kind)",
    )
    add.add_argument("--base", default=None, help="provider API base URL")
    add.add_argument("--key", default=None, help="provider API key (else interactive prompt)")
    add.add_argument("--no-verify", dest="no_verify", action="store_true", help="skip verification")
    add.add_argument("--base-dir", type=Path, default=None, help="base directory")
    listing = provider_sub.add_parser("list", help="list configured providers")
    listing.add_argument("--base-dir", type=Path, default=None, help="base directory")
    remove = provider_sub.add_parser("remove", help="remove a provider")
    remove.add_argument("name")
    remove.add_argument("--base-dir", type=Path, default=None, help="base directory")
    verify = provider_sub.add_parser("verify", help="re-verify a provider")
    verify.add_argument("name")
    verify.add_argument("--base-dir", type=Path, default=None, help="base directory")
    models = provider_sub.add_parser("models", help="list the provider's available model ids")
    models.add_argument("name")
    models.add_argument("--base-dir", type=Path, default=None, help="base directory")
