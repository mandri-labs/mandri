from dataclasses import replace
from pathlib import Path, PurePosixPath

from mandri.core.ids import FsPath, HarnessKind
from mandri.core.ports.transcripts import SessionRef, TranscriptReader
from mandri.core.types.execution import ProtectionError
from mandri.core.types.sessions import Session
from mandri.sessions.execution_context import DockerSessionContext
from mandri.sessions.pi_store import PiSessionStore
from mandri.sessions.transcripts.agy_transcripts import AgyTranscriptReader
from mandri.sessions.transcripts.claude_transcripts import ClaudeTranscriptReader
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader
from mandri.sessions.transcripts.pi_transcripts import PiTranscriptReader


class DockerCodexReader(CodexTranscriptReader):
    def __init__(self, context: DockerSessionContext) -> None:
        self.context = context
        home = context.state_path(context.state_root / ".codex")
        database = home / "state_5.sqlite"
        if database.exists() or database.is_symlink():
            database = context.state_path(database)
        super().__init__(home / "sessions", database)

    def _resolve(self, session: SessionRef) -> Path:
        return self.context.state_path(super()._resolve(session))


class DockerClaudeReader(ClaudeTranscriptReader):
    def __init__(self, context: DockerSessionContext) -> None:
        self.context = context
        super().__init__(context.state_root / ".claude/projects")

    def _resolve(self, session: SessionRef) -> Path:
        return self.context.state_path(super()._resolve(session))


class DockerAgyReader(AgyTranscriptReader):
    def __init__(self, context: DockerSessionContext) -> None:
        self.context = context
        super().__init__(context.state_root / ".gemini")

    def _resolve(self, session: SessionRef) -> Path:
        return self.context.state_path(super()._resolve(session))


class DockerPiReader(PiTranscriptReader):
    def __init__(self, context: DockerSessionContext) -> None:
        self.context = context
        super().__init__(
            store=PiSessionStore(
                context.state_root / ".pi/agent/sessions", path_validator=self._validate_path
            )
        )

    def _validate_path(self, path: Path) -> Path:
        posix = PurePosixPath(path.as_posix())
        if posix.is_relative_to(self.context.native_home):
            path = self.context.state_root.joinpath(
                *posix.relative_to(self.context.native_home).parts
            )
        elif posix.is_relative_to(self.context.container_root):
            path = self.context.workspace_root.joinpath(
                *posix.relative_to(self.context.container_root).parts
            )
        resolved = path.resolve()
        if not (
            resolved.is_relative_to(self.context.state_root)
            or resolved.is_relative_to(self.context.workspace_root)
        ):
            raise ProtectionError(
                "native_state_incompatible", "Pi state references an external path"
            )
        return resolved

    def _resolve(self, session: SessionRef) -> Path:
        if session.transcript_path is not None:
            path = self._validate_path(Path(str(session.transcript_path)))
            session = replace(session, transcript_path=FsPath(str(path)))
        return self._validate_path(super()._resolve(session))


def docker_reader(session: Session) -> TranscriptReader:
    context = DockerSessionContext.from_session(session)
    if session.harness is HarnessKind.CODEX:
        return DockerCodexReader(context)
    if session.harness is HarnessKind.CLAUDE:
        return DockerClaudeReader(context)
    if session.harness is HarnessKind.AGY:
        return DockerAgyReader(context)
    if session.harness is HarnessKind.PI:
        return DockerPiReader(context)
    return OpencodeTranscriptReader(
        context.state_path(context.state_root / ".local/share/opencode/opencode.db")
    )
