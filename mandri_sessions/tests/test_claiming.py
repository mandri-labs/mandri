"""Tests for the verified claim evidence rules."""

import ntpath

import pytest
from mandri.core.fs import paths
from mandri.core.ids import ClaimEvidence, EpochMs, HarnessKind, HarnessSessionId, ProjectPath
from mandri.sessions.claiming import NativeSessionRecord, claim_evidence_matches

_PATH = "C:/work/proj"
_WINDOW = (EpochMs(0), EpochMs(10_000))


def _evidence(
    harness: HarnessKind = HarnessKind.CLAUDE,
    project_path: str = _PATH,
    captured_native_id: str | None = None,
    time_window: tuple[EpochMs, EpochMs] = _WINDOW,
) -> ClaimEvidence:
    return ClaimEvidence(
        harness=harness,
        project_path=ProjectPath(project_path),
        captured_native_id=(
            None if captured_native_id is None else HarnessSessionId(captured_native_id)
        ),
        time_window=time_window,
    )


def _candidate(
    harness: HarnessKind = HarnessKind.CLAUDE,
    native_id: str = "n1",
    project_path: str = _PATH,
    created_at: int = 5_000,
) -> NativeSessionRecord:
    return NativeSessionRecord(
        harness=harness,
        native_id=HarnessSessionId(native_id),
        project_path=ProjectPath(project_path),
        created_at=EpochMs(created_at),
    )


def test_matching_evidence_accepts_candidate() -> None:
    assert claim_evidence_matches(_evidence(), _candidate()) is True


def test_harness_mismatch_rejects_candidate() -> None:
    assert claim_evidence_matches(_evidence(harness=HarnessKind.CODEX), _candidate()) is False


def test_captured_native_id_match_short_circuits_other_fields() -> None:
    evidence = _evidence(captured_native_id="n1", project_path="C:/elsewhere")
    candidate = _candidate(native_id="n1", project_path="C:/work/proj", created_at=999_999)
    assert claim_evidence_matches(evidence, candidate) is True


def test_captured_native_id_mismatch_rejects_candidate() -> None:
    evidence = _evidence(captured_native_id="n1")
    assert claim_evidence_matches(evidence, _candidate(native_id="n2")) is False


def test_windows_extended_prefix_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "path", ntpath)
    evidence = _evidence(project_path=r"\\?\C:\work\proj")
    assert claim_evidence_matches(evidence, _candidate(project_path="C:\\work\\proj")) is True


def test_created_at_before_window_rejects_candidate() -> None:
    assert claim_evidence_matches(_evidence(), _candidate(created_at=-1)) is False


def test_created_at_after_window_rejects_candidate() -> None:
    assert claim_evidence_matches(_evidence(), _candidate(created_at=10_001)) is False


def test_window_boundaries_accept_candidate() -> None:
    assert claim_evidence_matches(_evidence(), _candidate(created_at=0)) is True
    assert claim_evidence_matches(_evidence(), _candidate(created_at=10_000)) is True
