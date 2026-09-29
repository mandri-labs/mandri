"""Provider credential registry, model argument parsing, and model ref helpers."""

import dataclasses
import enum

from mandri.core.ids import ModelRef, ProviderKind, SecretRef, Url
from mandri.core.ports.config import ConfigPort
from mandri.core.ports.provider_credentials import CredentialResolverPort
from mandri.core.ports.provider_verifier import ProviderVerifierPort
from mandri.core.ports.routes import RouteLookupPort
from mandri.core.types.config import ProviderConfig
from mandri.providers.errors import (
    ProviderExistsError,
    ProviderInUseError,
    ProviderInvalidError,
    ProviderNotFoundError,
    ProviderVerificationError,
)
from mandri.providers.refs import MODEL_REF_PREFIXES
from mandri.providers.verify import ProviderVerifier, resolve_api_base

_LOCAL_KINDS: frozenset[ProviderKind] = frozenset(
    {ProviderKind.OLLAMA, ProviderKind.LM_STUDIO, ProviderKind.CUSTOM}
)

_TOKEN_KINDS: frozenset[ProviderKind] = frozenset({ProviderKind.CHATGPT})


class _Unset:
    """Sentinel type distinguishing omitted arguments from explicit None."""


_UNSET = _Unset()


def parse_model_arg(arg: str) -> tuple[str, str]:
    provider_name, separator, model_id = arg.partition("/")
    if not separator or not provider_name.strip() or not model_id.strip():
        raise ProviderInvalidError(f"model must be '<provider>/<model_id>', got {arg!r}")
    return provider_name.strip(), model_id.strip()


def provider_model_ref(kind: ProviderKind, model_id: str) -> ModelRef:
    return ModelRef(MODEL_REF_PREFIXES[kind] + model_id)


def split_model_ref(kind: ProviderKind, model_ref: str) -> str:
    prefix = MODEL_REF_PREFIXES[kind]
    if model_ref.startswith(prefix):
        return model_ref[len(prefix) :]
    return model_ref


class ProviderState(enum.StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    DEGRADED = "degraded"
    PENDING_AUTH = "pending_auth"


@dataclasses.dataclass(frozen=True)
class Provider:
    name: str
    kind: ProviderKind
    api_base: Url | None
    api_key: SecretRef
    state: ProviderState


def _coerce_kind(kind: ProviderKind) -> ProviderKind:
    try:
        return ProviderKind(kind)
    except ValueError as error:
        raise ProviderInvalidError(f"unknown provider kind {kind!r}") from error


class ProvidersRegistry:
    def __init__(
        self,
        config: ConfigPort,
        routes: RouteLookupPort,
        verifier: ProviderVerifierPort | None = None,
        token_resolver: CredentialResolverPort | None = None,
    ) -> None:
        self._config = config
        self._routes = routes
        if verifier is None:
            verifier = ProviderVerifier()
        self._verifier: ProviderVerifierPort = verifier
        self._token_resolver = token_resolver

    def list(self) -> list[Provider]:
        return [self._materialize(entry) for entry in self._config.load().providers]

    def get(self, name: str) -> Provider:
        for entry in self._config.load().providers:
            if entry.name == name:
                return self._materialize(entry)
        raise ProviderNotFoundError(f"unknown provider {name!r}")

    async def ensure_fresh(self, provider_name: str) -> None:
        """Rotate stored credentials that are about to expire."""
        if self._token_resolver is None:
            return
        entries = self._config.load().providers
        entry = next((item for item in entries if item.name == provider_name), None)
        if entry is None or ProviderKind(entry.kind) not in _TOKEN_KINDS:
            return
        await self._token_resolver.ensure_fresh(provider_name)

    async def add(
        self,
        name: str,
        kind: ProviderKind,
        api_base: str | None,
        api_key: str,
        verify: bool = True,
    ) -> Provider:
        if not name.strip():
            raise ProviderInvalidError("provider name must be a non-empty string")
        resolved_kind = _coerce_kind(kind)
        if resolved_kind in _LOCAL_KINDS and not api_base:
            raise ProviderInvalidError(f"provider kind {resolved_kind.value} requires api_base")
        if name in {entry.name for entry in self._config.load().providers}:
            raise ProviderExistsError(f"provider {name!r} already exists")
        base = resolve_api_base(resolved_kind, Url(api_base) if api_base else None)
        state = ProviderState.UNVERIFIED
        if verify:
            result = await self._verifier.verify_async(resolved_kind, base, api_key)
            if not result.ok:
                raise ProviderVerificationError(resolved_kind, result.reason)
            state = ProviderState.VERIFIED
        stored_key = "" if resolved_kind in _TOKEN_KINDS else api_key
        entry = ProviderConfig(
            name=name, kind=resolved_kind.value, api_base=api_base or None, api_key=stored_key
        )
        self._persist_append(entry)
        return Provider(
            name=name, kind=resolved_kind, api_base=base, api_key=SecretRef(api_key), state=state
        )

    async def update(
        self,
        name: str,
        api_base: str | _Unset | None = _UNSET,
        api_key: str | _Unset | None = _UNSET,
        verify: bool = True,
    ) -> Provider:
        current = self.get(name)
        if isinstance(api_base, _Unset):
            base = current.api_base
        else:
            base = Url(api_base) if api_base else None
        if current.kind in _LOCAL_KINDS and base is None:
            raise ProviderInvalidError(f"provider kind {current.kind.value} requires api_base")
        base = resolve_api_base(current.kind, base)
        key = current.api_key if isinstance(api_key, _Unset) else SecretRef(api_key or "")
        if base == current.api_base and key == current.api_key:
            state = current.state
        else:
            if verify:
                result = await self._verifier.verify_async(current.kind, base, str(key))
                if not result.ok:
                    raise ProviderVerificationError(current.kind, result.reason)
                state = ProviderState.VERIFIED
            else:
                state = ProviderState.UNVERIFIED
        self._persist_replace(
            ProviderConfig(
                name=name,
                kind=current.kind.value,
                api_base=str(base) if base is not None else None,
                api_key="" if current.kind in _TOKEN_KINDS else str(key),
            )
        )
        return Provider(name=name, kind=current.kind, api_base=base, api_key=key, state=state)

    async def remove(self, name: str) -> None:
        self.get(name)
        in_use = await self._routes.route_ids_for_provider(name)
        if in_use:
            raise ProviderInUseError(name, [str(route_id) for route_id in in_use])
        config = self._config.load()
        config = dataclasses.replace(
            config, providers=[entry for entry in config.providers if entry.name != name]
        )
        self._config.save(config)
        if self._token_resolver is not None:
            self._token_resolver.clear(name)

    async def verify(self, name: str) -> Provider:
        await self.ensure_fresh(name)
        provider: Provider = self.get(name)
        result = await self._verifier.verify_async(
            provider.kind, provider.api_base, str(provider.api_key)
        )
        state = ProviderState.VERIFIED if result.ok else ProviderState.DEGRADED
        return dataclasses.replace(provider, state=state)

    def _materialize(self, entry: ProviderConfig) -> Provider:
        provider = _provider_from_entry(entry)
        if self._token_resolver is None or provider.kind not in _TOKEN_KINDS:
            return provider
        try:
            api_key = SecretRef(self._token_resolver.access_token(entry.name))
        except ProviderInvalidError:
            return dataclasses.replace(
                provider, api_key=SecretRef(""), state=ProviderState.PENDING_AUTH
            )
        return dataclasses.replace(provider, api_key=api_key)

    def _persist_append(self, entry: ProviderConfig) -> None:
        config = self._config.load()
        config = dataclasses.replace(config, providers=[*config.providers, entry])
        self._config.save(config)

    def _persist_replace(self, entry: ProviderConfig) -> None:
        config = self._config.load()
        providers = [entry if item.name == entry.name else item for item in config.providers]
        config = dataclasses.replace(config, providers=providers)
        self._config.save(config)


def _provider_from_entry(entry: ProviderConfig) -> Provider:
    kind = ProviderKind(entry.kind)
    return Provider(
        name=entry.name,
        kind=kind,
        api_base=resolve_api_base(kind, Url(entry.api_base) if entry.api_base else None),
        api_key=SecretRef(entry.api_key),
        state=ProviderState.VERIFIED,
    )
