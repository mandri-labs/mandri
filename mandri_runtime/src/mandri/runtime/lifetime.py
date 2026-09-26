"""Port consumed by the feed layer to report viewer-count transitions."""

import typing


class SessionLifetimePort(typing.Protocol):
    def viewer_joined(self, session_id: str) -> None: ...

    def viewers_zero(self, session_id: str) -> None: ...
