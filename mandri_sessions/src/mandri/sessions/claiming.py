"""Verified claim evidence rules for native session records."""

import dataclasses

from mandri.core.fs.paths import normalize_fs_path
from mandri.core.ids import ClaimEvidence, EpochMs, HarnessKind, HarnessSessionId, ProjectPath


@dataclasses.dataclass(frozen=True)
class NativeSessionRecord:
    harness: HarnessKind
    native_id: HarnessSessionId
    project_path: ProjectPath
    created_at: EpochMs


def normalize_project_path(path: ProjectPath) -> str:
    return str(normalize_fs_path(str(path)))


def claim_evidence_matches(evidence: ClaimEvidence, candidate: NativeSessionRecord) -> bool:
    if evidence.harness is not candidate.harness:
        return False
    if evidence.captured_native_id is not None:
        return evidence.captured_native_id == candidate.native_id
    if normalize_project_path(evidence.project_path) != normalize_project_path(
        candidate.project_path
    ):
        return False
    window_start, window_end = evidence.time_window
    return window_start <= candidate.created_at <= window_end
