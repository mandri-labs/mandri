"""Provider connectivity verification port."""

import dataclasses

from mandri.core.ids import ProviderKind, Url


@dataclasses.dataclass(frozen=True)
class VerificationResult:
    ok: bool
    provider: ProviderKind
    reason: str | None


class ProviderVerifierPort:
    def verify(self, kind: ProviderKind, api_base: Url | None, api_key: str) -> VerificationResult:
        raise NotImplementedError

    async def verify_async(
        self, kind: ProviderKind, api_base: Url | None, api_key: str
    ) -> VerificationResult:
        raise NotImplementedError
