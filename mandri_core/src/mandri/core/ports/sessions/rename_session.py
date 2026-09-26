"""Session rename port."""

from mandri.core.ids import SessionId, SessionTitle


class RenameSessionPort:
    def rename(self, session_id: SessionId, title: SessionTitle) -> None:
        raise NotImplementedError
