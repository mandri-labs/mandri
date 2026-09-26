"""Transcript reader port and its value objects."""

import dataclasses
from typing import Protocol

from mandri.core.ids import FsPath, HarnessKind, HarnessSessionId, PageToken, ProjectPath, RawEvent


@dataclasses.dataclass(frozen=True)
class SessionRef:
    harness: HarnessKind
    native_id: HarnessSessionId
    project_path: ProjectPath | None = None
    transcript_path: FsPath | None = None


@dataclasses.dataclass(frozen=True)
class TranscriptPage:
    entries: list[RawEvent]
    next_token: PageToken | None
    has_more: bool


class TranscriptReader(Protocol):
    def page(self, session: SessionRef, cursor: PageToken | None, limit: int) -> TranscriptPage:
        raise NotImplementedError
