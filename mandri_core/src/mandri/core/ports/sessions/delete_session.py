"""Session delete port."""

from mandri.core.ids import SessionId


class DeleteSessionPort:
    def delete(self, session_id: SessionId) -> None:
        raise NotImplementedError
