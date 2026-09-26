"""Session existence check port."""

from mandri.core.ids import SessionId


class CheckSessionExistsPort:
    def exists(self, session_id: SessionId) -> bool:
        raise NotImplementedError
