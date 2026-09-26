"""Provider credential CLI commands: local service calls."""

import argparse
import asyncio
import getpass
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from mandri.cli.command import Command
from mandri.cli.local import build_providers_registry, resolve_base_dir
from mandri.core.ids import ProviderKind
from mandri.providers.errors import (
    ProviderExistsError,
    ProviderInUseError,
    ProviderInvalidError,
    ProviderNotFoundError,
    ProviderVerificationError,
)
from mandri.providers.refs import requires_api_base
from mandri.providers.service import Provider, ProvidersRegistry
from mandri.providers.verify import models_endpoint, models_headers

MODELS_TIMEOUT_S = 10.0
MAX_ERROR_BODY_CHARS = 500

__all__ = ["run_provider_command"]

T = TypeVar("T")


class ProvidersCommand(Command):
    """Base class for provider subcommands sharing the local registry bootstrap."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._base_dir = resolve_base_dir(args.base_dir)

    async def _with_registry(self, action: Callable[[ProvidersRegistry], Awaitable[T]]) -> T:
        registry, db = await build_providers_registry(self._base_dir)
        try:
            return await action(registry)
        finally:
            await db.close()

    def _print_provider(self, provider: Provider) -> None:
        print(
            f"{provider.name}  {provider.kind.value}  {provider.api_base}  {provider.state.value}"
        )


class AddProviderCommand(ProvidersCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._name: str = args.name
        self._kind_raw: str | None = args.kind
        self._api_base: str | None = args.base
        self._key_raw: str | None = args.key
        self._verify: bool = not args.no_verify

    def run(self) -> int:
        kind = self._resolve_kind()
        if kind is None:
            print(f"cannot infer provider kind from {self._name!r} (pass --kind)")
            return 2
        key = self._resolve_key()
        if not key:
            print("no api key provided (pass --key)")
            return 2

        async def add(registry: ProvidersRegistry) -> Provider:
            return await registry.add(
                self._name,
                ProviderKind(kind),
                self._api_base,
                key,
                verify=self._verify,
            )

        try:
            provider = asyncio.run(self._with_registry(add))
        except (ProviderExistsError, ProviderInvalidError, ProviderVerificationError) as error:
            print(str(error))
            return 1
        print(f"{provider.name}  {provider.kind.value}  {provider.state.value}")
        return 0

    def _resolve_kind(self) -> str | None:
        if self._kind_raw:
            return self._kind_raw
        try:
            ProviderKind(self._name)
        except ValueError:
            return None
        return self._name

    def _resolve_key(self) -> str:
        if self._key_raw:
            return self._key_raw
        return str(getpass.getpass("provider API key: "))


class ListProvidersCommand(ProvidersCommand):
    def run(self) -> int:
        async def listing(registry: ProvidersRegistry) -> list[Provider]:
            return registry.list()

        for provider in asyncio.run(self._with_registry(listing)):
            self._print_provider(provider)
        return 0


class RemoveProviderCommand(ProvidersCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._name: str = args.name

    def run(self) -> int:
        async def remove(registry: ProvidersRegistry) -> None:
            await registry.remove(self._name)

        try:
            asyncio.run(self._with_registry(remove))
        except (ProviderNotFoundError, ProviderInUseError) as error:
            print(str(error))
            return 1
        print("removed")
        return 0


class VerifyProviderCommand(ProvidersCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._name: str = args.name

    def run(self) -> int:
        async def verify(registry: ProvidersRegistry) -> Provider:
            return await registry.verify(self._name)

        try:
            provider = asyncio.run(self._with_registry(verify))
        except (ProviderNotFoundError, ProviderVerificationError) as error:
            print(str(error))
            return 1
        self._print_provider(provider)
        return 0


class ProviderModelsCommand(ProvidersCommand):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._name: str = args.name

    def run(self) -> int:
        async def fetch(registry: ProvidersRegistry) -> Provider:
            return registry.get(self._name)

        try:
            provider = asyncio.run(self._with_registry(fetch))
        except ProviderNotFoundError as error:
            print(str(error))
            return 1
        if requires_api_base(provider.kind) and provider.api_base is None:
            print(f"api_base is required for provider {provider.kind.value}")
            return 1
        url = models_endpoint(provider.kind, provider.api_base)
        headers = models_headers(provider.kind, str(provider.api_key))
        try:
            response = httpx.get(url, headers=headers, timeout=MODELS_TIMEOUT_S)
        except httpx.HTTPError:
            print("provider unreachable")
            return 1
        if not 200 <= response.status_code < 300:
            body = _redact(str(provider.api_key), response.text)[:MAX_ERROR_BODY_CHARS]
            print(f"provider models request failed (status {response.status_code}): {body}")
            return 1
        try:
            payload: Any = response.json()
        except ValueError:
            print("provider returned invalid JSON")
            return 1
        for model_id in _model_ids(provider.kind, payload):
            print(model_id)
        return 0


def run_provider_command(args: argparse.Namespace) -> int:
    commands: dict[str, type[ProvidersCommand]] = {
        "add": AddProviderCommand,
        "list": ListProvidersCommand,
        "remove": RemoveProviderCommand,
        "verify": VerifyProviderCommand,
        "models": ProviderModelsCommand,
    }
    return commands[args.provider_command](args).run()


def _redact(secret: str, text: str) -> str:
    if secret:
        return text.replace(secret, "[redacted]")
    return text


def _model_ids(kind: ProviderKind, payload: Any) -> list[str]:
    if kind is ProviderKind.GEMINI:
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return [
            str(entry["name"]).removeprefix("models/")
            for entry in models
            if isinstance(entry, dict) and entry.get("name")
        ]
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return [str(entry["id"]) for entry in data if isinstance(entry, dict) and entry.get("id")]
