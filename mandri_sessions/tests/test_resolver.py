"""Tests for per-harness transcript reader resolution."""

from mandri.core.ids import HarnessKind
from mandri.sessions.transcripts import TranscriptResolver

from mandri_sessions.tests.substitutes import FakeReader


def test_reader_returns_mapped_reader() -> None:
    reader = FakeReader()
    resolver = TranscriptResolver({HarnessKind.CLAUDE: reader})
    assert resolver.reader(HarnessKind.CLAUDE) is reader


def test_reader_unknown_harness_returns_none() -> None:
    resolver = TranscriptResolver({HarnessKind.CLAUDE: FakeReader()})
    assert resolver.reader(HarnessKind.CODEX) is None
