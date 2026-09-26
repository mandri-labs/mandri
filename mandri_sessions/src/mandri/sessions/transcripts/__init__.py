"""Per-harness transcript readers with lazy pagination."""

from mandri.core.ports.transcripts import SessionRef, TranscriptPage, TranscriptReader
from mandri.sessions.transcripts.claude_transcripts import ClaudeTranscriptReader
from mandri.sessions.transcripts.codex_transcripts import CodexTranscriptReader
from mandri.sessions.transcripts.errors import (
    HarnessStoreUnavailableError,
    PageTokenInvalidError,
    PageTokenStaleError,
    TranscriptError,
    TranscriptNotFoundError,
    TranscriptStoreError,
)
from mandri.sessions.transcripts.opencode_transcripts import OpencodeTranscriptReader
from mandri.sessions.transcripts.pi_transcripts import PiTranscriptReader
from mandri.sessions.transcripts.resolver import TranscriptResolver
from mandri.sessions.transcripts.tokens import (
    PageTokenData,
    decode_page_token,
    encode_page_token,
)

__all__ = [
    "ClaudeTranscriptReader",
    "CodexTranscriptReader",
    "HarnessStoreUnavailableError",
    "OpencodeTranscriptReader",
    "PageTokenData",
    "PageTokenInvalidError",
    "PageTokenStaleError",
    "PiTranscriptReader",
    "SessionRef",
    "TranscriptError",
    "TranscriptNotFoundError",
    "TranscriptPage",
    "TranscriptReader",
    "TranscriptResolver",
    "TranscriptStoreError",
    "decode_page_token",
    "encode_page_token",
]
