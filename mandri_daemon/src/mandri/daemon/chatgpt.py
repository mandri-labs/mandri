"""Daemon wiring for the ChatGPT subscription provider."""

from pathlib import Path

import httpx
from mandri.api.deps import ChatGptWiring
from mandri.config.toml_adapter import TomlConfigAdapter
from mandri.providers.chatgpt.resolver import ChatGptCredentialResolver
from mandri.providers.chatgpt.store import ChatGptTokenStore
from mandri.providers.chatgpt_login import ChatGptLoginService
from mandri.providers.service import ProvidersRegistry


def build_chatgpt(
    http: httpx.AsyncClient, base_dir: Path, providers: ProvidersRegistry
) -> ChatGptWiring:
    return ChatGptWiring(
        login=ChatGptLoginService(
            providers=providers, store=ChatGptTokenStore(base_dir), http=http
        ),
    )


def build_resolver(http: httpx.AsyncClient, base_dir: Path) -> ChatGptCredentialResolver:
    store = ChatGptTokenStore(base_dir)
    store.migrate([entry.name for entry in TomlConfigAdapter(base_dir).load().providers])
    return ChatGptCredentialResolver(store, http)
