from typing import Protocol


class PrivacyKeyProvider(Protocol):
    def load(self, *, create: bool = False) -> bytes: ...
