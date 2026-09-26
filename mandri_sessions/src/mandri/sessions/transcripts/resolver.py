"""Per-harness transcript reader resolution."""

from collections.abc import Mapping

from mandri.core.ids import HarnessKind
from mandri.core.ports.transcripts import TranscriptReader
from mandri.core.types.execution import ExecutionBackend
from mandri.core.types.sessions import Session
from mandri.sessions.transcripts.docker_readers import docker_reader


class TranscriptResolver:
    def __init__(self, readers: Mapping[HarnessKind, TranscriptReader]) -> None:
        self._readers = dict(readers)

    def reader(self, kind: HarnessKind) -> TranscriptReader | None:
        return self._readers.get(kind)

    def for_session(self, session: Session) -> TranscriptReader | None:
        if session.execution_backend is ExecutionBackend.DOCKER:
            return docker_reader(session)
        return self.reader(session.harness)
