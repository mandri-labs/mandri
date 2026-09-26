"""Session fetch port."""

from mandri.core.types.sessions import Session


class FetchSessionsPort:
    def fetch(self) -> list[Session]:
        raise NotImplementedError
